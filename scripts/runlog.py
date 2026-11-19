# -*- coding: utf-8 -*-
"""结构化运行日志：把"失败原因"从终端输出变成可查询的事件。

为什么需要它（2026-11-19）：
  在这之前，全项目的失败信息只存在于 `print` 里。想知道"最近 100 次运行为什么坏"，
  只能翻终端滚动缓冲区——翻不回去就重跑一遍（视频 ¥0.6~0.9/镜）。
  题三、题四都会问可观测性，而当时的答案是"有 cost.csv 和 manifest.json"，
  这两个都只记"成了什么"，不记"为什么没成"。

事件格式（一行一条 JSON，按月分文件，避免无限增长）：
  {"ts","run","stage","item","level","rule","reason","cost_cny","retry"}

  level: ok / fail / warn
  rule:  规则名（lint 的 SYMBOL_MOTION、VLM 的检查项标签），没有规则就留空
  reason:人读的一句话，足够定位，不需要完整堆栈

用法：
  python scripts/runlog.py --summary           # 各 stage 成败与花费
  python scripts/runlog.py --top rule          # 哪条规则触发最多
  python scripts/runlog.py --top reason --stage video
  python scripts/runlog.py --tail 20
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "runs" / "runlog"

RUN_ID = time.strftime("%Y%m%d_%H%M%S")


def log(stage: str, item: str = "-", level: str = "ok", *,
        rule: str = "", reason: str = "", cost: float = 0.0,
        retry: int = 0, run: str = "", **extra: Any) -> None:
    """记一条事件。任何异常都必须被吞掉——日志不能把业务流程弄挂。"""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        rec = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "run": run or RUN_ID,
            "stage": stage,
            "item": item,
            "level": level,
            "rule": rule,
            "reason": (reason or "")[:300],
            "cost_cny": round(float(cost), 4),
            "retry": retry,
        }
        if extra:
            rec["extra"] = extra
        path = LOG_DIR / f"{time.strftime('%Y-%m')}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:                       # noqa: BLE001
        pass


def read_all() -> list[dict]:
    out: list[dict] = []
    if not LOG_DIR.exists():
        return out
    for p in sorted(LOG_DIR.glob("*.jsonl")):
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--top")
    ap.add_argument("--stage")
    ap.add_argument("--level", choices=["ok", "fail", "warn"])
    ap.add_argument("--tail", type=int, default=0)
    args = ap.parse_args()

    rows = read_all()
    if args.stage:
        rows = [r for r in rows if r["stage"] == args.stage]
    if args.level:
        rows = [r for r in rows if r["level"] == args.level]

    if not rows:
        print("（暂无日志）")
        return

    if args.tail:
        for r in rows[-args.tail:]:
            print(f"{r['ts']}  {r['stage']:<11}{r['item']:<8}{r['level']:<5}"
                  f"{r['rule']:<20}{r['reason'][:60]}")
        return

    if args.top:
        key = args.top
        c = Counter((r.get(key) or "（空）") for r in rows if r["level"] != "ok")
        total = sum(c.values())
        print(f"按 {key} 统计（仅 fail/warn，共 {total} 条）")
        print("-" * 68)
        for k, v in c.most_common(15):
            bar = "▇" * min(30, int(v / max(1, total) * 60))
            print(f"{v:>4}  {k:<32}{bar}")
        return

    # 默认按 stage 汇总
    agg: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"ok": 0, "fail": 0, "warn": 0, "cost": 0.0}
    )
    for r in rows:
        a = agg[r["stage"]]
        a[r["level"]] = a.get(r["level"], 0) + 1
        a["cost"] += r["cost_cny"]
    print(f"{'stage':<13}{'ok':>6}{'fail':>6}{'warn':>6}{'花费':>10}{'成功率':>9}")
    print("-" * 52)
    tot_cost = 0.0
    for k, v in sorted(agg.items()):
        n = v["ok"] + v["fail"]
        rate = f"{v['ok']/n:.0%}" if n else "-"
        print(f"{k:<13}{v['ok']:>6}{v['fail']:>6}{v['warn']:>6}"
              f"{'¥'+format(v['cost'], '.2f'):>10}{rate:>9}")
        tot_cost += v["cost"]
    print("-" * 52)
    print(f"合计花费 ¥{tot_cost:.2f}，事件 {len(rows)} 条")


if __name__ == "__main__":
    main()
