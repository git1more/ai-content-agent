# -*- coding: utf-8 -*-
"""
漫剧角色一致性计分器。

与 P0 电商评分同一套方法学：
  - 评分数据落独立 JSON（可审计、可回放），代码只做计算
  - 权重 critical 3 / major 2 / minor 1
  - 关键项 0 分母量不加权否决，只降分（角色一致性是连续的，不是硬失败）
  - 全局阈值 0.85

用法：
  python scripts/manju_score.py
  python scripts/manju_score.py --json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCORES = ROOT / "assets" / "manju" / "ch01" / "consistency_scores.json"
WEIGHT = {"critical": 3, "major": 2, "minor": 1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="输出机器可读结果")
    ap.add_argument("--file", default=None, help="指定评分文件（用于回评历史版本）")
    args = ap.parse_args()

    src = Path(args.file) if args.file else SCORES
    data = json.loads(src.read_text(encoding="utf-8"))
    crit = data["criteria"]
    th = data.get("threshold", 0.85)

    rows = []
    for sid, shot in data["shots"].items():
        # 逐镜可判项：背身/远景镜头看不到脸，评"眼型"是在量一个不存在的东西。
        # 排除项同时从分子和分母里扣掉，否则背身镜会凭空掉分。
        na = set(shot.get("na") or [])
        total_w = sum(WEIGHT[c["level"]] for cid, c in crit.items() if cid not in na)
        if total_w <= 0:
            continue
        got = 0.0
        fails = []
        for cid, (v, why) in shot["scores"].items():
            if cid in na:
                continue
            w = WEIGHT[crit[cid]["level"]]
            got += v * w
            if v < 1.0 and crit[cid]["level"] == "critical":
                fails.append(f"{crit[cid]['part']}({v:.1f})")
        score = got / total_w
        rows.append({
            "shot": sid,
            "shot_size": shot["shot_size"],
            "score": round(score, 3),
            "usable": score >= th,
            "critical_deductions": fails,
        })

    rows.sort(key=lambda r: r["shot"])
    if args.json:
        print(json.dumps({"threshold": th, "shots": rows}, ensure_ascii=False, indent=2))
        return

    print(f"{'镜头':<6}{'景别':<28}{'一致率':>8}  {'判定':<6} 扣分项")
    print("-" * 96)
    for r in rows:
        mark = "可用" if r["usable"] else "不可用"
        print(f"{r['shot']:<6}{r['shot_size'][:26]:<28}{r['score']:>8.3f}  {mark:<6} "
              f"{'、'.join(r['critical_deductions']) or '无'}")
    print("-" * 96)
    n = sum(1 for r in rows if r["usable"])
    mean = sum(r["score"] for r in rows) / len(rows)
    print(f"通过 {n}/{len(rows)}；平均一致率 {mean:.3f}；阈值 {th}")

    worst = min(rows, key=lambda r: r["score"])
    print(f"最差镜头 {worst['shot']}（{worst['shot_size']}）— {worst['score']:.3f}")


if __name__ == "__main__":
    main()
