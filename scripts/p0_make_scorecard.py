# -*- coding: utf-8 -*-
"""
由 identity_sheet.json 生成人工评分用的派生产物，保证单一数据源：
  1. 身份特征清单.md   —— 人工打分时对着看的表
  2. runs_matrix.csv    —— 6 SKU x 3 任务的运行登记表
  3. feature_scores.csv —— 逐特征打分的长表（模板）

用法： python scripts/p0_make_scorecard.py
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSET_DIR = ROOT / "assets" / "p0_testset"
SHEET = ASSET_DIR / "identity_sheet.json"

WEIGHT_LABEL = {"critical": "🔴 关键", "major": "🟠 重要", "minor": "⚪ 次要"}

# 纳入评测的模型：run_id = {sku}__{task}__{slug}
# 加入新模型只需在这里加一行，重跑本脚本 + p0_fill_scores.py
MODELS = [
    ("baseline-internal-imagegen", "baseline"),
    ("qwen-image-3.0", "qwen30"),
]



def build_markdown(data: dict) -> str:
    sc = data["scoring"]
    lines = []
    lines.append("# P0 测试集 · 身份特征清单（评分地面真值）")
    lines.append("")
    lines.append(f"> 数据集：`{data['dataset']}`　生成日：{data['created']}")
    lines.append(f"> {data['purpose']}")
    lines.append("")
    lines.append("## 评分规则")
    lines.append("")
    lines.append("| 判定 | 分值 | 含义 |")
    lines.append("|---|---|---|")
    for k, v in sc["levels"].items():
        meaning = {"PASS": "与清单完全一致", "PARTIAL": "可辨识但存在偏差", "FAIL": "缺失/错误/被替换"}[k]
        lines.append(f"| {k} | {v} | {meaning} |")
    lines.append("")
    lines.append("**权重**：" + "　".join(f"{WEIGHT_LABEL[k]} = {v}" for k, v in sc["weights"].items()))
    lines.append("")
    lines.append(f"**保真分** = `{sc['fidelity_formula']}`")
    lines.append("")
    lines.append("**判定阈值**")
    lines.append("")
    for k, v in sc["pass_thresholds"].items():
        lines.append(f"- `{k}`：{v}")
    lines.append("")
    lines.append("**注意事项**")
    lines.append("")
    for n in sc["notes"]:
        lines.append(f"- {n}")
    lines.append("")

    lines.append("## 三类任务")
    lines.append("")
    lines.append("| 任务 | 定义 | 主要风险 |")
    lines.append("|---|---|---|")
    for tid, t in data["tasks"].items():
        lines.append(f"| `{tid}` | {t['desc']} | {t['primary_risk']} |")
    lines.append("")

    for sku in data["skus"]:
        crit = sum(1 for f in sku["features"] if f["weight"] == "critical")
        lines.append("---")
        lines.append("")
        lines.append(f"## {sku['sku']} · {sku['name']}")
        lines.append("")
        lines.append(f"- 品类：{sku['category']}　参考图：`{sku['ref_image']}`　分辨率：{sku['resolution']}")
        lines.append(f"- 特征数：{len(sku['features'])}（关键 {crit} 项）")
        lines.append(f"- 难点：{'；'.join(sku['difficulty_tags'])}")
        if sku.get("shooting_attrs"):
            lines.append(f"- 拍摄属性（不计分）：{'；'.join(sku['shooting_attrs'])}")
        ra = sku.get("reference_audit")
        if ra:
            flag = "✅ 可用" if ra["realism_gate"] == "PASS" else "❌ 不合格"
            lines.append(f"- 参考图审计：{flag} —— {ra['note']}")
        lines.append("")
        has_ref_actual = any("ref_actual" in f for f in sku["features"])
        if has_ref_actual:
            lines.append("| 编号 | 类别 | 权重 | 特征描述（设计意图） | 参考图实际 | 校验方法 | PASS | PARTIAL | FAIL |")
            lines.append("|---|---|---|---|---|---|---|---|---|")
        else:
            lines.append("| 编号 | 类别 | 权重 | 特征描述 | 校验方法 | PASS | PARTIAL | FAIL |")
            lines.append("|---|---|---|---|---|---|---|---|")
        for f in sku["features"]:
            desc = f["desc"].replace("|", "\\|")
            verify = f["verify"].replace("|", "\\|")
            ref_actual = f.get("ref_actual", "—").replace("|", "\\|")
            if has_ref_actual:
                lines.append(
                    f"| {f['id']} | {f['cat']} | {WEIGHT_LABEL[f['weight']]} | {desc} | {ref_actual} "
                    f"| {verify} | ☐ | ☐ | ☐ |"
                )
            else:
                lines.append(
                    f"| {f['id']} | {f['cat']} | {WEIGHT_LABEL[f['weight']]} | {desc} | {verify} | ☐ | ☐ | ☐ |"
                )
        lines.append("")
    return "\n".join(lines)


def build_runs_matrix(data: dict) -> list:
    rows = []
    for model, slug in MODELS:
        for sku in data["skus"]:
            for tid in data["tasks"].keys():
                rows.append({
                    "run_id": f"{sku['sku']}__{tid.split('_')[0]}__{slug}",
                    "sku": sku["sku"],
                    "task": tid,
                    "model": model,
                    "provider": "",
                    "params_json": '{"size":"TBD","seed":null,"ref_images":1}',
                    "unit_cost_usd": "",
                    "out_path": "",
                    "fidelity_score": "",
                    "usable": "",
                    "critical_fail": "",
                    "note": "",
                })
    return rows


def build_feature_scores(data: dict) -> list:
    rows = []
    for model, slug in MODELS:
        for sku in data["skus"]:
            for tid in data["tasks"].keys():
                for f in sku["features"]:
                    rows.append({
                        "run_id": f"{sku['sku']}__{tid.split('_')[0]}__{slug}",
                        "sku": sku["sku"],
                        "task": tid,
                        "model": model,
                        "feature_id": f["id"],
                        "feature_name": f["cat"],
                        "weight": data["scoring"]["weights"][f["weight"]],
                        "score": "",
                        "evidence": "",
                    })
    return rows


def write_csv(path: Path, rows: list):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    data = json.loads(SHEET.read_text(encoding="utf-8"))

    md = build_markdown(data)
    md_path = ASSET_DIR / "身份特征清单.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"[输出] {md_path.name}  ({len(md.splitlines())} 行)")

    runs = build_runs_matrix(data)
    write_csv(ASSET_DIR / "runs_matrix.csv", runs)
    print(f"[输出] runs_matrix.csv  ({len(runs)} 行)")

    feats = build_feature_scores(data)
    write_csv(ASSET_DIR / "feature_scores.csv", feats)
    print(f"[输出] feature_scores.csv  ({len(feats)} 行)")

    total_feat = sum(len(s["features"]) for s in data["skus"])
    crit = sum(1 for s in data["skus"] for f in s["features"] if f["weight"] == "critical")
    print(f"\n统计：{len(data['skus'])} SKU / {total_feat} 特征（关键 {crit} 项）/ "
          f"{len(runs)} 次运行 / {len(feats)} 条打分记录")


if __name__ == "__main__":
    main()
