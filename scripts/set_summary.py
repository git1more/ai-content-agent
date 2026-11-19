"""跨 SKU 汇总：把各套图报告合成一张品类对照表，并按失败类型归并。

为什么单独做：单套报告只回答「这一套过了几张」，回答不了
「哪个品类难、难在哪一项、是不是系统性问题」。后者才是决定下一步该改什么的依据。

用法：
  python scripts/set_summary.py
  python scripts/set_summary.py --md      # 输出 markdown 表格
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "ecom_set"

# 失败项归并：把「主体高度占比 0.33 超出 0.6–0.92」这类具体描述归到大类
BUCKETS = [
    ("主体完整", "完整性"),
    ("可辨认文字", "文字"),
    ("槽位专属检查", "景别/构图"),
    ("主体高度占比", "主体占比"),
]


def bucket(fail: str) -> str:
    for key, name in BUCKETS:
        if key in fail:
            return name
    return "其他"


def load() -> list[dict]:
    rows = []
    for f in sorted(OUT.glob("*/set_report.json")):
        try:
            rows.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            continue
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", action="store_true", help="输出 markdown 表格")
    args = ap.parse_args()

    data = load()
    if not data:
        raise SystemExit("没有找到任何 set_report.json")

    if args.md:
        print("| SKU | 品类 | 出图 | 自动通过 | 通过率 | 主要失败项 |")
        print("|---|---|---|---|---|---|")
    else:
        print(f"{'SKU':<15}{'品类':<8}{'出图':>4}{'通过':>5}{'通过率':>8}  主要失败项")
        print("-" * 96)

    total_gen = total_pass = 0
    all_buckets: Counter = Counter()

    for d in data:
        sku = d.get("sku", "?")
        name = d.get("name", "")
        prof = d.get("profile", "")
        gen = d.get("generated") or len(d.get("rows", []))
        passed = d.get("passed") or 0
        total_gen += gen
        total_pass += passed

        cnt: Counter = Counter()
        for r in d.get("rows", []):
            for fail in (r.get("check") or {}).get("fails") or []:
                cnt[bucket(fail)] += 1
        all_buckets.update(cnt)
        top = "、".join(f"{k}×{v}" for k, v in cnt.most_common(3)) or "—"
        rate = f"{passed / gen:.0%}" if gen else "—"

        if args.md:
            print(f"| {sku} | {prof or name[:6]} | {gen} | {passed} | {rate} | {top} |")
        else:
            print(f"{sku:<15}{(prof or name)[:6]:<8}{gen:>4}{passed:>5}{rate:>8}  {top}")

    rate = f"{total_pass / total_gen:.0%}" if total_gen else "—"
    print()
    if args.md:
        print(f"\n合计：{total_gen} 张，自动通过 {total_pass} 张（{rate}）")
        print("\n失败项分布：" + "、".join(f"{k} {v}" for k, v in all_buckets.most_common()))
    else:
        print(f"合计：{total_gen} 张，自动通过 {total_pass} 张（{rate}）")
        print("失败项分布：" + "、".join(f"{k} {v}" for k, v in all_buckets.most_common()))


if __name__ == "__main__":
    main()
