# -*- coding: utf-8 -*-
"""
批量执行器 —— 任务队列 + 限流 + 成本记账 + 合规门禁

关键约束（来自平台实测/文档）：
  - 百炼 Qwen 图像接口 RPM = 5 → 用 60s 滑动窗口限 5 次，间隔至少 12s
  - 结果 URL 24h 过期 → generate() 内部当场落盘
  - 429 / 5xx 退避重试，4xx 直接失败（参数或权限问题重试无意义）

每条运行产出：
  runs/<run_id>/meta.json     参数、耗时、usage、成本、合规报告
  runs/<run_id>/final.png     合成后的交付图
  runs/<run_id>/raw.png       模型原始输出
  runs/cost_ledger.csv        累加成本台账

用法：
  python scripts/run_batch.py --task T1 --skus JK-ORANGE-03,CR-MATTE-05
  python scripts/run_batch.py --task T1 --all --dry-run
  # 换 seed 重跑（run_id 自动加 __s{seed} 后缀，不覆盖原结果）
  python scripts/run_batch.py --task T1 --skus LS-BLACK-06 --seeds 20261117,20261118
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from postprocess_canvas import SPECS, compose            # noqa: E402
from qwen_image_client import generate                   # noqa: E402

AD = ROOT / "assets" / "p0_testset"
CONFIG = ROOT / "config" / "prompt_locks.json"
RUNS = ROOT / "runs"
LEDGER = RUNS / "cost_ledger.csv"

# 单价（人民币/张，1K 输出）。公开价，需与月度账单核对后再定稿。
CNY_PER_IMAGE = 0.18
USD_CNY = 7.15

MODEL = "qwen-image-3.0"
TASK_SPEC = {"T1": ("T1_white_pack", "white_pack_1x1")}


class RateLimiter:
    """60s 滑动窗口限流。"""

    def __init__(self, max_calls: int = 5, window_s: int = 60):
        self.max_calls = max_calls
        self.window = window_s
        self.calls: deque[float] = deque()

    def acquire(self):
        now = time.time()
        while self.calls and now - self.calls[0] > self.window:
            self.calls.popleft()
        if len(self.calls) >= self.max_calls:
            wait = self.window - (now - self.calls[0]) + 0.5
            print(f"    [限流] 窗口已满 {len(self.calls)}/{self.max_calls}，等待 {wait:.1f}s")
            time.sleep(max(0.0, wait))
            return self.acquire()
        if self.calls:
            gap = 12.5 - (now - self.calls[-1])
            if gap > 0:
                time.sleep(gap)
        self.calls.append(time.time())


def build_prompt(task_key: str, sku: str, cfg: dict) -> str:
    tpl = cfg["task_templates"][task_key]
    locks = cfg["skus"][sku]
    return tpl.format(product=locks["product"], locks=locks["locks"])


def append_ledger(row: dict):
    RUNS.mkdir(parents=True, exist_ok=True)
    new = not LEDGER.exists()
    with LEDGER.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def run_one(sku: str, task: str, task_key: str, spec_name: str,
            cfg: dict, limiter: RateLimiter, dry: bool = False,
            seed: int | None = None, slug: str = "qwen30") -> dict:
    base_id = f"{sku}__{task}__{slug}" + (f"__s{seed}" if seed is not None else "")
    run_id, k = base_id, 2
    # 同一 seed 重复跑（确定性检查）时自动加 __r2/__r3，不覆盖已有结果
    while (RUNS / run_id / "meta.json").exists():
        run_id = f"{base_id}__r{k}"
        k += 1
    out_dir = RUNS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt = build_prompt(task_key, sku, cfg)
    ref = AD / sku / "ref_front.png"
    spec = SPECS[spec_name]

    rec = {
        "run_id": run_id, "sku": sku, "task": task, "model": MODEL,
        "seed": seed,
        "prompt": prompt, "ref": str(ref.relative_to(ROOT)),
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    if dry:
        print(f"    [dry-run] prompt 长度 {len(prompt)} 字")
        rec.update(status="dry-run")
        return rec

    limiter.acquire()
    t0 = time.time()
    try:
        res = generate(prompt, ref_images=[ref], model=MODEL, size="1024*1024",
                       out_dir=out_dir, tag="raw", prompt_extend=False, watermark=False,
                       seed=seed)
    except Exception as e:  # noqa: BLE001
        rec.update(status="failed", error=str(e)[:400], elapsed_s=round(time.time() - t0, 1))
        print(f"    [失败] {e}")
        (out_dir / "meta.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        return rec

    raw = Path(res.images[0])
    final = out_dir / "final.png"
    try:
        report = compose(raw, spec, final)
    except Exception as e:  # noqa: BLE001
        report = {"pass": False, "issues": [f"合成失败: {e}"]}

    cost_cny = CNY_PER_IMAGE
    rec.update(
        status="ok",
        elapsed_s=res.elapsed_s,
        usage=res.usage,
        raw=str(raw.relative_to(ROOT)),
        final=str(final.relative_to(ROOT)) if final.exists() else "",
        compliance=report,
        cost_cny=cost_cny,
        cost_usd=round(cost_cny / USD_CNY, 4),
    )
    (out_dir / "meta.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")

    status = "PASS" if report.get("pass") else "FAIL"
    print(f"    [完成] {res.elapsed_s}s  合规 {status}  "
          f"高占比 {report.get('product_h_ratio')} 顶留 {report.get('top_safe_ratio')}")
    if report.get("issues"):
        for i in report["issues"]:
            print(f"      ! {i}")

    append_ledger({
        "run_id": run_id, "model": MODEL, "task": task,
        "cost_cny": f"{cost_cny:.2f}", "cost_usd": f"{cost_cny / USD_CNY:.4f}",
        "compliance": "PASS" if report.get("pass") else "FAIL",
        "ts": rec["started_at"],
    })
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="T1", choices=list(TASK_SPEC))
    ap.add_argument("--skus", help="逗号分隔")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seeds", help="逗号分隔的 seed 列表；给定时每个 SKU 按每个 seed 各跑一次，"
                                    "run_id 加 __s{seed} 后缀。重复同一 seed 会生成 __r2 副本（确定性检查）")
    args = ap.parse_args()

    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    sheet = json.loads((AD / "identity_sheet.json").read_text(encoding="utf-8"))
    all_skus = [s["sku"] for s in sheet["skus"]]

    skus = all_skus if args.all else [s.strip() for s in (args.skus or "").split(",") if s.strip()]
    if not skus:
        ap.error("需指定 --skus 或 --all")
    for s in skus:
        if s not in all_skus:
            ap.error(f"未知 SKU: {s}")

    seeds: list[int | None] = (
        [int(x) for x in args.seeds.split(",") if x.strip()] if args.seeds else [None]
    )

    task_key, spec_name = TASK_SPEC[args.task]
    limiter = RateLimiter()

    print(f"任务 {args.task} | 模型 {MODEL} | {len(skus)} 个 SKU × {len(seeds)} 个 seed "
          f"= {len(skus) * len(seeds)} 次 | 预估成本 ¥{CNY_PER_IMAGE * len(skus) * len(seeds):.2f}")
    results = []
    for i, sku in enumerate(skus, 1):
        for seed in seeds:
            tag = f"seed={seed}" if seed is not None else "seed=随机"
            print(f"\n[{i}/{len(skus)}] {sku}  {tag}")
            results.append(run_one(sku, args.task, task_key, spec_name, cfg, limiter,
                                   dry=args.dry_run, seed=seed))

    ok = [r for r in results if r.get("status") == "ok"]
    bad = [r for r in results if r.get("status") == "failed"]
    passed = [r for r in ok if r.get("compliance", {}).get("pass")]
    print(f"\n汇总：成功 {len(ok)} / 失败 {len(bad)} / 合成合规 {len(passed)}")
    if ok:
        print(f"本轮成本 ¥{sum(r['cost_cny'] for r in ok):.2f} "
              f"(约 ${sum(r['cost_usd'] for r in ok):.3f})")


if __name__ == "__main__":
    main()
