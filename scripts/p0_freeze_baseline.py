# -*- coding: utf-8 -*-
"""
P0 基线冻结 / 退化检测。

用途：
  --freeze  把当前 P0 结果固化成 baseline_v1.json（regression baseline）
  --check   拿一份新的 runs_matrix.csv 与基线比对，报告退化项

为什么需要它：
  P0 的价值不在"这 6 张图好不好看"，而在于**换模型 / 改 Prompt 模板 / 改后处理时，
  能不能立刻知道有没有变差**。没有基线，前面的校准成果在第一次换模型时就全丢了。

用法：
  python scripts/p0_freeze_baseline.py --freeze
  python scripts/p0_freeze_baseline.py --check path/to/new_runs_matrix.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
TS = ROOT / "assets" / "p0_testset"
MATRIX = TS / "runs_matrix.csv"
SCORES = TS / "scores.json"
RECALIB = TS / "recalib_metrics.json"
LEDGER = ROOT / "runs" / "cost_ledger.csv"
OUT = TS / "baseline_v1.json"

# 当前口径：保真 >= 0.85 且无关键项 FAIL 判"可用"
USABLE_THRESHOLD = 0.85

# 冻结的模型。稳定性只统计这个模型的运行——早期用内置生成能力跑的 baseline 行
# 是 A/B 对照记录，混进来会把"换了模型"误算成"同一模型不稳定"。
FROZEN_MODEL = "qwen-image-3.0"

# 每个 SKU 的"记账运行"（run of record）。
# 不是简单取最高分——必须是：当前 prompt_lock 版本 + 当前后处理版本 + 参考图未被作废。
RUN_OF_RECORD = {
    "JK-ORANGE-03": "JK-ORANGE-03__T1__qwen30",
    "TS-WHITE-01": "TS-WHITE-01__T1__qwen30",
    "SR-GLASS-04": "SR-GLASS-04__T1__qwen30",
    "CR-MATTE-05": "CR-MATTE-05__T1__qwen30",
    "LS-BLACK-06": "LS-BLACK-06__T1__qwen30",
    # 连衣裙取 s20261118：它是唯一过线的一次，且已复核非口径漂移
    "DR-NAVY-02": "DR-NAVY-02__T1__qwen30__s20261118",
}

DEFECTS = {
    "JK-ORANGE-03": [],
    "TS-WHITE-01": ["织纹略平滑（竹节肌理被压平）"],
    "SR-GLASS-04": ["泵头偏高大"],
    "CR-MATTE-05": ["材质层次（哑光罐身缺乏细腻转折）"],
    "LS-BLACK-06": ["膏体旋出高度 +9.9%"],
    "DR-NAVY-02": ["色相偏移 8°", "裙摆外扩（h/w −7%）", "材质偏哑"],
}


def read_matrix(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [r for r in csv.DictReader(f) if (r.get("run_id") or "").strip()]


def to_float(v: Any) -> float | None:
    try:
        s = str(v).strip()
        return float(s) if s else None
    except (TypeError, ValueError):
        return None


def cost_summary() -> dict[str, Any]:
    if not LEDGER.exists():
        return {}
    rows = read_matrix(LEDGER)
    total = sum(float(r.get("cost_cny") or 0) for r in rows)
    return {"calls_logged": len(rows), "cost_cny": round(total, 4)}


def freeze() -> dict[str, Any]:
    rows = read_matrix(MATRIX)
    scores = json.loads(SCORES.read_text(encoding="utf-8"))
    recalib = json.loads(RECALIB.read_text(encoding="utf-8")) if RECALIB.exists() else {}

    by_sku: dict[str, list[dict]] = {}
    for r in rows:
        if (r.get("fidelity_score") or "").strip() and r.get("model") == FROZEN_MODEL:
            by_sku.setdefault(r["sku"], []).append(r)

    skus: dict[str, Any] = {}
    for sku, run_id in RUN_OF_RECORD.items():
        rec = next((r for r in by_sku.get(sku, []) if r["run_id"] == run_id), None)
        if rec is None:
            print(f"[warn] {sku}: 找不到记账运行 {run_id}，跳过", file=sys.stderr)
            continue
        cal = recalib.get(run_id, {})
        runs = sorted(by_sku.get(sku, []), key=lambda x: x["run_id"])
        n_run = len(runs)
        n_ok = sum(1 for r in runs if (r.get("usable") or "").strip().upper() == "Y")
        # 稳定性：不看某一次是否过线，看多次运行是否稳定过线。
        # borderline = 有运行过线但也有运行掉线，说明接近能力边界，换 seed 会翻车。
        if n_ok == n_run:
            status = "stable"
        elif n_ok > 0:
            status = "borderline"
        else:
            status = "fail"
        skus[sku] = {
            "run_of_record": run_id,
            "fidelity": to_float(rec.get("fidelity_score")),
            "usable": (rec.get("usable") or "").strip().upper() == "Y",
            "critical_fail": (rec.get("critical_fail") or "").strip().upper() == "Y",
            "stability": status,
            "stability_confidence": "multi_run" if n_run >= 3 else ("two_run" if n_run == 2 else "single_run"),
            "runs_usable_ratio": f"{n_ok}/{n_run}",
            "image": rec.get("out_path"),
            "defects": DEFECTS.get(sku, []),
            "structure": {
                "aspect_deviation_pct": cal.get("aspect_deviation_pct"),
                "aspect_verdict": cal.get("aspect_verdict"),
                "hue_delta_deg": cal.get("hue_delta_deg"),
                "hue_verdict": cal.get("hue_verdict"),
            },
            "all_runs": [
                {
                    "run_id": r["run_id"],
                    "fidelity": to_float(r.get("fidelity_score")),
                    "usable": (r.get("usable") or "").strip().upper() == "Y",
                }
                for r in runs
            ],
        }

    usable = [s for s in skus.values() if s["stability"] == "stable"]
    borderline = [s for s in skus.values() if s["stability"] == "borderline"]
    all_runs = [r for lst in by_sku.values() for r in lst]
    baseline = {
        "schema_version": "1.0",
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": "P1/P2 换模型、改 Prompt 模板、改后处理时的退化检测基线。"
                   "本文件由 scripts/p0_freeze_baseline.py 生成，不手工编辑。",
        "frozen_scope": {
            "task": "T1_white_pack",
            "model": "qwen-image-3.0",
            "provider": "dashscope 北京区",
            "size": "1024*1024",
            "prompt_extend": False,
            "watermark": False,
            "postprocess": "postprocess_canvas.py（保留全部显著连通域）",
            "scores_file": "scores.json@v2.0",
        },
        "scoring": {
            "usable_threshold": USABLE_THRESHOLD,
            "critical_fail_vetoes": True,
            "rubric": "critical 3 / major 2 / minor 1，按权重归一化",
            "reference_principle": scores.get("rubric_basis"),
            "reference_caveat": "参考图为 AI 生成，与输出同族模型风格先验接近；"
                                "分数只可横向比（同参考同模型），不可外推为真实商品还原度。",
        },
        "sku_baseline": skus,
        "aggregate": {
            "sku_total": len(skus),
            "sku_stable": len(usable),
            "sku_borderline": len(borderline),
            "stable_pass_rate": round(len(usable) / len(skus), 4) if skus else None,
            "mean_fidelity": round(
                sum(s["fidelity"] for s in skus.values() if s["fidelity"] is not None) / len(skus), 4
            ) if skus else None,
            "runs_scored": len(all_runs),
            "runs_usable": sum(1 for r in all_runs if (r.get("usable") or "").strip().upper() == "Y"),
            "note": "通过率按「稳定可用」计（所有运行均过线）。连衣裙 3 次中 1 次过线，"
                    "记为 borderline，不计入通过率——否则等于拿最好的一次当基线。",
        },
        "cost_ledger": cost_summary(),
        "known_gaps": [
            "T2 场景图、T3 局部编辑两条任务线未跑（runs_matrix 中为占位行）",
            "文字 / Logo 保真未单独度量，是电商图唯一「一眼废」的硬伤",
            "无真实手机拍摄素材，绝对保真度不可外推",
        ],
        "explicitly_stopped": [
            "连衣裙继续刷 seed（s20261118 已达 0.850，继续投入边际收益趋零）",
            "面霜罐「材质层次」单独攻关（归入「材质表现」品类级课题）",
            "逐 SKU 四法交叉几何精测（降级为换模型时的抽样体检）",
        ],
    }
    OUT.write_text(json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"已冻结 -> {OUT.relative_to(ROOT)}")
    print(f"  SKU {baseline['aggregate']['sku_total']} 个：稳定可用 {baseline['aggregate']['sku_stable']}、"
          f"边缘 {baseline['aggregate']['sku_borderline']}；稳定通过率 "
          f"{baseline['aggregate']['stable_pass_rate']:.0%}；平均保真 {baseline['aggregate']['mean_fidelity']}")
    print(f"  运行 {baseline['aggregate']['runs_scored']} 次，可用 "
          f"{baseline['aggregate']['runs_usable']} 次；累计调用成本 "
          f"¥{baseline['cost_ledger'].get('cost_cny')}")
    return baseline


def check(matrix_path: Path) -> int:
    if not OUT.exists():
        sys.exit("基线不存在，先跑 --freeze")
    base = json.loads(OUT.read_text(encoding="utf-8"))["sku_baseline"]
    rows = read_matrix(matrix_path)
    regressions = 0

    print(f"{'SKU':<14}{'基线':>8}{'新值':>8}{'Δ':>8}  判定")
    print("-" * 62)
    for sku, b in base.items():
        cand = [r for r in rows if r["sku"] == sku and (r.get("fidelity_score") or "").strip()]
        if not cand:
            print(f"{sku:<14}{b['fidelity']:>8.3f}{'--':>8}{'--':>8}  未跑")
            continue
        best = max(cand, key=lambda r: to_float(r.get("fidelity_score")) or 0)
        new = to_float(best.get("fidelity_score")) or 0
        delta = new - b["fidelity"]
        usable = (best.get("usable") or "").strip().upper() == "Y"

        flags = []
        if delta <= -0.02:
            flags.append(f"保真退化 {delta:+.3f}")
        if b["usable"] and not usable:
            flags.append("可用性翻转 Y→N")
        if flags:
            regressions += 1
            verdict = "✗ " + "；".join(flags)
        else:
            verdict = "✓"
        print(f"{sku:<14}{b['fidelity']:>8.3f}{new:>8.3f}{delta:>+8.3f}  {verdict}")

    print("-" * 62)
    print("有退化。" if regressions else "无退化，与基线一致。")
    return 1 if regressions else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true", help="冻结基线")
    ap.add_argument("--check", metavar="MATRIX", help="用新矩阵做退化检测")
    args = ap.parse_args()
    if args.freeze:
        freeze()
    elif args.check:
        sys.exit(check(Path(args.check)))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
