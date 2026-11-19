# -*- coding: utf-8 -*-
"""
把实测运行结果写入评测表，并输出 seed 方差报告。

数据源分层（2026-11-17 重构）：
  - identity_sheet.json  特征与权重（唯一真值定义）
  - scores.json          逐次运行的人工评分（与代码解耦，便于审计）
  - recalib_metrics.json 比例偏差 / 色相偏移（客观指标）
  - runs_matrix.csv      汇总表（本脚本维护）

评分口径：
  - 保真分（fidelity）：模型原始输出 vs **参考图**，不是 vs 设计书。
  - 任务合规（task_compliance）：最终交付图是否满足版式规范。
  - 可用（usable）= 保真 ≥0.85 且无关键项 0 分 且 版式合规 且 参考图现实性过关。

用法： python scripts/p0_fill_scores.py
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AD = ROOT / "assets" / "p0_testset"
RUNS = ROOT / "runs"

SHEET = AD / "identity_sheet.json"
SCORES = AD / "scores.json"
METRICS = AD / "recalib_metrics.json"
MATRIX = AD / "runs_matrix.csv"
FEATURES = AD / "feature_scores.csv"
VARIANCE = AD / "variance_report.json"

MATRIX_FIELDS = ["run_id", "sku", "task", "model", "provider", "params_json",
                 "unit_cost_usd", "out_path", "fidelity_score", "usable",
                 "critical_fail", "note"]
FEATURE_FIELDS = ["run_id", "sku", "task", "model", "feature_id", "feature_name",
                  "weight", "score", "evidence"]


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return list(csv.DictReader(path.open(encoding="utf-8-sig")))


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    sheet = json.loads(SHEET.read_text(encoding="utf-8"))
    scores = json.loads(SCORES.read_text(encoding="utf-8"))
    metrics = json.loads(METRICS.read_text(encoding="utf-8")) \
        if METRICS.exists() else {}

    wmap = sheet["scoring"]["weights"]
    skus = {s["sku"]: s for s in sheet["skus"]}
    notes = scores.get("sku_notes", {})

    matrix = {r["run_id"]: r for r in read_csv(MATRIX)}
    feats = {(r["run_id"], r["feature_id"]): r for r in read_csv(FEATURES)}

    # 1) 为 scores.json 中新增的运行补齐行
    for rid, rec in scores["runs"].items():
        sku, task = rid.split("__")[0], rid.split("__")[1]
        if rid not in matrix:
            matrix[rid] = {k: "" for k in MATRIX_FIELDS}
            matrix[rid].update({"run_id": rid, "sku": sku, "task": task,
                                "model": "qwen-image-3.0" if "qwen30" in rid else "baseline-internal-imagegen"})
        for f in skus[sku]["features"]:
            key = (rid, f["id"])
            if key not in feats:
                feats[key] = {
                    "run_id": rid, "sku": sku, "task": task,
                    "model": matrix[rid]["model"], "feature_id": f["id"],
                    "feature_name": f["cat"], "weight": wmap[f["weight"]],
                    "score": "", "evidence": "",
                }

    # 2) 逐次计算保真分
    for rid, rec in scores["runs"].items():
        sku = rid.split("__")[0]
        num = den = 0.0
        crit_fail = []
        for f in skus[sku]["features"]:
            sc = rec["features"].get(f["id"], (None,))[0]
            if sc is None:
                continue
            wgt = wmap[f["weight"]]
            num += wgt * sc
            den += wgt
            if f["weight"] == "critical" and sc == 0.0:
                crit_fail.append(f"{f['id']}({f['cat']})")
        fid = num / den if den else 0.0
        task_ok, task_note = rec["task_compliance"]
        realism = notes.get(sku, {}).get("realism_gate", "PASS")
        usable = (fid >= 0.85 and not crit_fail and task_ok and realism == "PASS")

        row = matrix[rid]
        mi = rec["meta"]
        row["provider"] = mi.get("src", "")
        row["params_json"] = json.dumps(
            {"size": "1024*1024", "prompt_extend": False, "watermark": False,
             "ref_images": 1, "seed": rid.split("__s")[1].split("__")[0] if "__s" in rid else None},
            ensure_ascii=False)
        row["unit_cost_usd"] = mi.get("cost", "")
        row["out_path"] = mi.get("out", "")
        row["fidelity_score"] = f"{fid:.3f}"
        row["critical_fail"] = "Y" if crit_fail else "N"
        row["usable"] = "Y" if usable else "N"
        reasons = [task_note]
        if crit_fail:
            reasons.append("关键项0分: " + "、".join(crit_fail))
        if realism != "PASS":
            reasons.append(f"参考图现实性 {realism}: {notes[sku]['reason']}")
        if mi.get("note"):
            reasons.append(mi["note"])
        row["note"] = "；".join(x for x in reasons if x)

    # 3) 写回逐特征打分
    for rid, rec in scores["runs"].items():
        for fid_, (sc, ev) in rec["features"].items():
            key = (rid, fid_)
            if key in feats:
                feats[key]["score"] = f"{sc:.1f}"
                feats[key]["evidence"] = ev

    write_csv(MATRIX, [matrix[k] for k in sorted(matrix)], MATRIX_FIELDS)
    write_csv(FEATURES, [feats[k] for k in sorted(feats)], FEATURE_FIELDS)

    # 4) seed 方差分析
    by_sku = {}
    for rid in scores["runs"]:
        if "__qwen30" not in rid or "__baseline" in rid:
            continue
        by_sku.setdefault(rid.split("__")[0], []).append(rid)

    report = {}
    for sku, rids in sorted(by_sku.items()):
        # 同 seed 复跑不计入方差（它是确定性对照）
        rids = [r for r in rids if not r.endswith("__r2")]
        fids = [f["id"] for f in skus[sku]["features"]]
        per_feature = {}
        for fid in fids:
            vals = [scores["runs"][r]["features"].get(fid, (None,))[0] for r in rids]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            per_feature[fid] = {
                "scores": vals,
                "mean": round(statistics.fmean(vals), 2),
                "spread": round(max(vals) - min(vals), 2),
            }
        fids_all = [float(matrix[r]["fidelity_score"]) for r in rids]
        report[sku] = {
            "runs": rids,
            "fidelity": [round(v, 3) for v in fids_all],
            "fidelity_mean": round(statistics.fmean(fids_all), 3),
            "fidelity_std": round(statistics.pstdev(fids_all), 3) if len(fids_all) > 1 else 0.0,
            "fidelity_range": round(max(fids_all) - min(fids_all), 3),
            "aspect_dev_pct": [metrics.get(r, {}).get("aspect_deviation_pct") for r in rids],
            "hue_delta_deg": [metrics.get(r, {}).get("hue_delta_deg") for r in rids],
            "unstable_features": {k: v for k, v in per_feature.items() if v["spread"] > 0},
            "stable_features": [k for k, v in per_feature.items() if v["spread"] == 0],
            "realism_gate": notes.get(sku, {}).get("realism_gate", "PASS"),
        }
    VARIANCE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 5) 输出
    print(f"{'run_id':<44}{'保真':<8}{'关键FAIL':<9}{'可用':<5}")
    print("-" * 68)
    for r in sorted(matrix.values(), key=lambda x: x["run_id"]):
        if not r["fidelity_score"]:
            continue
        print(f"{r['run_id']:<44}{r['fidelity_score']:<8}{r['critical_fail']:<9}{r['usable']:<5}")

    n_eval = sum(1 for r in matrix.values() if r["fidelity_score"])
    n_ok = sum(1 for r in matrix.values() if r["usable"] == "Y")
    print(f"\n可用 {n_ok} / 已评测 {n_eval}")

    print("\nseed 方差：")
    for sku, v in report.items():
        print(f"  {sku:<14} 保真 {v['fidelity']}  极差 {v['fidelity_range']:.3f}  "
              f"比例 {v['aspect_dev_pct']}  波动特征 {sorted(v['unstable_features']) or '无'}")

    print(f"\n写入 {MATRIX.name} / {FEATURES.name} / {VARIANCE.name}")


if __name__ == "__main__":
    main()
