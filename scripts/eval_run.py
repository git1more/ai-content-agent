# -*- coding: utf-8 -*-
"""评测与稳定性复跑：给每一个对外数字配一个"能重跑、能比对"的样本。

为什么需要它（2026-11-19）：
  项目里所有对外数字都只有一个样本——88%（42/48）、4/7 过线、均值 0.904、
  返工率 27%、并发 7.52×。没有一个是"连跑几次都得到同一个数"验证过的。
  规则层有回归（`manju_lint --selftest` 19/19），指标层一条都没有。
  于是换模型、改提示词模板、调 VLM 阈值时，自检照样全绿，通过率掉了没人知道。

  这个脚本补的就是那一层。设计上刻意分成两类，成本差两个数量级：

  ① 尺子稳不稳（几毛钱）
     - judge_stability：同一张图让 VLM 判 N 轮，看结论会不会变
     - cost_determinism：同 stage 同 duration 的单价是否恒定（读台账，零成本）
     - lint：规则自检（零成本）
  ② 东西稳不稳（几块钱）
     - seed_variance：换 seed 重跑同一套，看通过率波动
     - 只有在①稳定时，②的数字才有意义

用法：
  python scripts/eval_run.py --all                    # 跑全部能零成本/低成本的
  python scripts/eval_run.py --suite judge_stability  # 单跑
  python scripts/eval_run.py --baseline                # 把本次结果冻结为基线
退出码：有指标相对基线退化超阈值则返回 1。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

EVAL_DIR = ROOT / "runs" / "eval"
BASELINE = EVAL_DIR / "baseline.json"

# 退化阈值：超过就判红。数值取自"这些指标的测量精度"——
# VLM 判定本来就允许 ±1 张的摆动，一致性分数刻度是 0.05，
# 所以阈值设成"比测量精度小一档"，只抓真正的漂移。
TOL = {
    "judge_image_agree": 0.05,     # 图级判定一致率掉 5 个点
    "judge_label_agree": 0.08,     # 项级
    "lint_pass_rate": 0.01,        # 自检掉一条就红
    "unit_price_dev": 0.02,        # 单价偏差 2%
}


# --------------------------------------------------------------------------- #
# ① 尺子：VLM 判定抖动
# --------------------------------------------------------------------------- #
def suite_judge_stability(cfg_path: Path, rounds: int = 3,
                          only: list[str] | None = None) -> dict[str, Any]:
    """同一张图判 N 轮，看结论会不会变。

    为什么这条排在第一位：通过率的分母是 VLM 判定。如果判定本身抖动 10%，
    那么"42/48"与"44/48"的差别就没有意义，所有基于它的结论都要打折。
    temperature=0 不等于确定性——服务端批处理、图像编码、max_tokens 截断
    都可能让同一张图得到不同结论。这件事必须实测，不能假设。
    """
    from manju_vlm_check import judge

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    ep = cfg_path.stem.replace("manju_", "")
    out_dir = ROOT / "assets" / "manju" / ep

    targets = []
    for shot in cfg["shots"]:
        sid = shot["id"]
        if only and sid not in only:
            continue
        p = out_dir / f"{sid}_keyframe.png"
        if p.exists():
            targets.append((sid, p, shot))

    if not targets:
        return {"skipped": f"{cfg_path.name} 无可判的关键帧"}

    per_image: dict[str, Any] = {}
    label_hits: dict[str, list[bool]] = defaultdict(list)

    for sid, path, shot in targets:
        rounds_out = []
        for _ in range(rounds):
            r = judge(path, cfg, shot)
            # 只统计 hard 项：soft 项（景别、文字、道具朝向）本来就是"提示"，
            # 把它们算进一致率，会让这个指标自己先抖起来 —— 指标不能含噪声项。
            rounds_out.append({
                "ok": r["ok"],
                "got": {c["label"]: c["got"] for c in r.get("checks", [])
                        if not c.get("soft")},
            })
        oks = [x["ok"] for x in rounds_out]
        labels = sorted({k for x in rounds_out for k in x["got"]})
        unstable = []
        for lb in labels:
            vals = [x["got"].get(lb) for x in rounds_out]
            same = len(set(map(str, vals))) == 1
            label_hits[lb].append(same)
            if not same:
                unstable.append({"label": lb, "values": vals})
        per_image[sid] = {
            "verdict_stable": len(set(oks)) == 1,
            "ok_rounds": oks,
            "unstable_labels": unstable,
            "n_labels": len(labels),
        }
        flag = "✓ 稳" if not unstable and len(set(oks)) == 1 else f"⚠ 抖动 {len(unstable)} 项"
        print(f"  {sid}: {flag}  ok={oks}")

    n = len(per_image)
    img_agree = sum(1 for v in per_image.values() if v["verdict_stable"]) / n
    label_agree = (
        sum(sum(v) for v in label_hits.values()) / sum(len(v) for v in label_hits.values())
        if label_hits else 1.0
    )
    worst = sorted(
        ((lb, sum(v) / len(v)) for lb, v in label_hits.items()),
        key=lambda x: x[1],
    )[:5]

    return {
        "config": cfg_path.name,
        "images": n,
        "rounds": rounds,
        "judge_image_agree": round(img_agree, 4),
        "judge_label_agree": round(label_agree, 4),
        "unstable_images": [k for k, v in per_image.items() if not v["verdict_stable"]],
        "label_stability": {lb: round(sum(v) / len(v), 3) for lb, v in label_hits.items()},
        "worst_labels": [{"label": lb, "agree": round(a, 3)} for lb, a in worst],
        "detail": per_image,
    }


# --------------------------------------------------------------------------- #
# ② 尺子：成本确定性（读台账，零成本）
# --------------------------------------------------------------------------- #
def suite_cost_determinism() -> dict[str, Any]:
    """同 stage 同单位的单价是否恒定。

    这条能验证一件很实际的事：**给客户报价能不能报死**。
    如果同一档位（4 秒视频）在不同时间跑出来单价不同，那所有"单集 ¥9.07"的
    说法都只是平均值，不能当承诺。
    """
    prices: dict[str, set[float]] = defaultdict(set)
    counts: dict[str, int] = defaultdict(int)

    def eat(path: Path) -> None:
        if not path.exists():
            return
        # 两套台账格式不同：漫剧 cost.csv 有 stage/qty/unit，电商 cost_ledger.csv
        # 一行就是一张图、qty 隐含为 1。按表头判断，不按文件名猜。
        with path.open(encoding="utf-8-sig") as f:
            rd = csv.DictReader(f)
            cols = set(rd.fieldnames or [])
            for r in rd:
                try:
                    cost = float(r.get("cost_cny") or 0)
                except (TypeError, ValueError):
                    continue
                if "qty" in cols and "stage" in cols:
                    try:
                        qty = float(r["qty"] or 0)
                    except (TypeError, ValueError):
                        continue
                    if qty <= 0:
                        continue
                    key = f"{r['stage']}|{r['model']}|{qty:g}{r['unit']}"
                    prices[key].add(round(cost / qty, 4))
                elif cost > 0:
                    key = f"ecom|{r.get('model','?')}|1image"
                    prices[key].add(round(cost, 4))
                counts[key] += 1

    for p in sorted((ROOT / "assets" / "manju").glob("*/cost.csv")):
        eat(p)
    eat(ROOT / "runs" / "cost_ledger.csv")

    rows = []
    max_dev = 0.0
    for key, vals in sorted(prices.items()):
        lo, hi = min(vals), max(vals)
        dev = (hi - lo) / lo if lo else 0.0
        max_dev = max(max_dev, dev)
        rows.append({"key": key, "n": counts[key], "min": lo, "max": hi,
                     "dev_pct": round(dev * 100, 2)})

    return {
        "rows": rows,
        "unit_price_dev": round(max_dev, 4),
        "violations": [r for r in rows if r["dev_pct"] > TOL["unit_price_dev"] * 100],
    }


# --------------------------------------------------------------------------- #
# ③ 规则自检
# --------------------------------------------------------------------------- #
def suite_lint(cfg_path: Path) -> dict[str, Any]:
    import subprocess
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "manju_lint.py"),
         "--config", str(cfg_path), "--selftest"],
        capture_output=True, text=True, encoding="utf-8",
    )
    tail = [ln for ln in r.stdout.strip().splitlines() if "/" in ln and "通过" in ln]
    passed = total = 0
    if tail:
        try:
            a, b = tail[-1].split("通过")[0].strip().split("/")
            passed, total = int(a), int(b)
        except ValueError:
            pass
    return {
        "config": cfg_path.name,
        "passed": passed,
        "total": total,
        "lint_pass_rate": round(passed / total, 4) if total else 0.0,
    }


# --------------------------------------------------------------------------- #
# ④ 东西：换 seed 的通过率波动（要花钱，默认不跑）
# --------------------------------------------------------------------------- #
def suite_seed_variance(cfg_path: Path, seeds: list[int], out_tag: str) -> dict[str, Any]:
    """同配置换 seed 重跑关键帧，看通过率波动。

    注意：同 seed 的图像是完全确定的（7.9 节已证），所以"同配置连跑 3 次"
    在不换 seed 的前提下波动恒为 0 —— 那种测法等于没测。
    真正要测的是"换一组随机种子，这个指标会飘到哪里"。
    """
    from manju_vlm_check import judge
    from manju_run import build_keyframe_prompt, negative_of
    from qwen_image_client import generate as gen_image

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    sheet = ROOT / "assets" / "manju" / cfg_path.stem.replace("manju_", "") / "character_sheet.png"
    out_dir = EVAL_DIR / out_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in seeds:
        ok_n, cost, n = 0, 0.0, 0
        fails: list[dict] = []
        for i, shot in enumerate(cfg["shots"]):
            prompt = build_keyframe_prompt(cfg, shot)
            r = gen_image(
                prompt, ref_images=[sheet], model="qwen-image-3.0",
                size=cfg["format"]["keyframe_size"], out_dir=out_dir,
                tag=f"{shot['id']}_s{seed}", negative_prompt=negative_of(cfg, shot),
                prompt_extend=False, seed=seed + i * 101,
            )
            p = Path(r.images[0])
            res = judge(p, cfg, shot)
            n += 1
            cost += 0.18
            ok_n += 1 if res["ok"] else 0
            if not res["ok"]:
                fails.append({"shot": shot["id"], "reasons": res["fails"]})
            p.unlink(missing_ok=True)       # 只留数字，不留图
        rows.append({"seed": seed, "passed": ok_n, "total": n,
                     "rate": round(ok_n / n, 4), "cost_cny": round(cost, 2),
                     "fails": fails})
        print(f"  seed {seed}: {ok_n}/{n}  通过率 {ok_n/n:.0%}  ¥{cost:.2f}")

    rates = [r["rate"] for r in rows]
    return {
        "config": cfg_path.name,
        "runs": rows,
        "rate_min": min(rates),
        "rate_max": max(rates),
        "rate_span": round(max(rates) - min(rates), 4),
        "cost_total": round(sum(r["cost_cny"] for r in rows), 2),
    }


# --------------------------------------------------------------------------- #
def compare(cur: dict, base: dict | None) -> list[str]:
    """与基线比，只报超过容差的项。"""
    if not base:
        return []
    notes = []
    b, c = base.get("suites", {}), cur.get("suites", {})
    for name, cv in c.items():
        bv = b.get(name) or {}
        for k, tol in TOL.items():
            if k in cv and k in bv:
                d = cv[k] - bv[k]
                if d < -tol:
                    notes.append(f"✗ {name}.{k}: {bv[k]} → {cv[k]}（掉 {abs(d):.3f}，容差 {tol}）")
                elif d > tol:
                    notes.append(f"↑ {name}.{k}: {bv[k]} → {cv[k]}")
    return notes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="跑全部零/低成本 suite（lint + cost_determinism + judge_stability）")
    ap.add_argument("--suite", action="append",
                    choices=["judge_stability", "cost_determinism", "lint", "seed_variance"],
                    help="可重复指定；缺省跑前三个（零/低成本）")
    ap.add_argument("--config", default="config/manju_ch04.json")
    ap.add_argument("--judge-config", default="config/manju_ch04.json")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seeds", nargs="*", type=int, default=[])
    ap.add_argument("--baseline", action="store_true", help="把本次结果冻结为基线")
    ap.add_argument("--no-write", action="store_true", help="只打印，不落盘")
    args = ap.parse_args()

    suites = args.suite or ["lint", "cost_determinism", "judge_stability"]
    if args.all:
        suites = ["lint", "cost_determinism", "judge_stability"]
    cfg = ROOT / args.config
    jcfg = ROOT / args.judge_config
    result: dict[str, Any] = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                              "commit_note": "", "suites": {}}

    for s in suites:
        print(f"[{s}] ...")
        t0 = time.time()
        if s == "lint":
            result["suites"]["lint"] = suite_lint(cfg)
        elif s == "cost_determinism":
            result["suites"]["cost_determinism"] = suite_cost_determinism()
        elif s == "judge_stability":
            result["suites"]["judge_stability"] = suite_judge_stability(jcfg, rounds=args.rounds)
        elif s == "seed_variance":
            if not args.seeds:
                print("  跳过：seed_variance 需要 --seeds，例如 --seeds 777001 777002")
                continue
            result["suites"]["seed_variance"] = suite_seed_variance(jcfg, args.seeds, "seedvar")
        print(f"  用时 {time.time()-t0:.1f}s")

    # ---- 打印摘要 ----
    print("\n" + "=" * 72)
    for name, v in result["suites"].items():
        if name == "lint":
            print(f"规则自检      {v['passed']}/{v['total']}  通过率 {v['lint_pass_rate']:.1%}")
        elif name == "judge_stability":
            print(f"VLM 判定      {v['images']} 图 × {v['rounds']} 轮")
            print(f"  图级一致    {v['judge_image_agree']:.1%}"
                  f"{'  不稳：' + ','.join(v['unstable_images']) if v['unstable_images'] else ''}")
            print(f"  项级一致    {v['judge_label_agree']:.1%}")
            for w in v["worst_labels"]:
                print(f"    最抖：{w['label']}  {w['agree']:.0%}")
        elif name == "cost_determinism":
            print(f"单价确定性    {len(v['rows'])} 档，最大偏差 {v['unit_price_dev']:.2%}")
            for x in v["violations"]:
                print(f"  ⚠ {x['key']}  {x['min']}~{x['max']}（{x['dev_pct']}%）")
        elif name == "seed_variance":
            print(f"换 seed 波动  通过率 {v['rate_min']:.0%}~{v['rate_max']:.0%}"
                  f"（跨度 {v['rate_span']:.0%}），花费 ¥{v['cost_total']}")

    base = json.loads(BASELINE.read_text(encoding="utf-8")) if BASELINE.exists() else None
    notes = compare(result, base)
    if notes:
        print("\n相对基线：")
        for n in notes:
            print("  " + n)
    elif base:
        print("\n相对基线：无退化")

    if not args.no_write:
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out = EVAL_DIR / f"eval_{stamp}.json"
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n→ {out.relative_to(ROOT)}")
        if args.baseline:
            BASELINE.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"→ 基线已冻结 {BASELINE.relative_to(ROOT)}")

    sys.exit(1 if any(n.startswith("✗") for n in notes) else 0)


if __name__ == "__main__":
    main()
