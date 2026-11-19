# -*- coding: utf-8 -*-
"""文档一致性自检：防止"做了但没记"和"记了但版本号没跟"。

为什么需要它：
  2026-11-18 用户指出方案文档"好像很久没更新"——实际正文一直在追加 7.13/7.14/7.15，
  但标题停在 v1.7、变更记录停在 v1.6。文档头是第一眼看到的地方，它不同步，
  读起来就像整份文档没更新。
  这类问题靠自觉一定会再犯，必须让机器查。

检查四件事：
  1. 标题版本号 与 变更记录里的最大版本号 是否一致
  2. 每个 7.x 章节是否有日期标注
  3. scripts/ 与 config/ 下的文件是否在文档里被提到（孤儿检测）
  4. 独立报告 .md 是否在文档里被引用

用法：
  python scripts/doc_check.py
  python scripts/doc_check.py --strict   # 有 WARN 也返回非零，可接 CI
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "AI内容生成Agent系统_实施方案_v1.md"

# 这些是工具脚本，不要求主文档逐个提及；只要至少被一份报告或文档提到即可
IGNORE_FILES = {"__init__.py", "get_ffmpeg.py", "probe_provider.py", "probe_video.py"}
IGNORE_DIRS = {"tools", "__pycache__"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    if not DOC.exists():
        sys.exit(f"找不到 {DOC}")

    text = DOC.read_text(encoding="utf-8")
    lines = text.splitlines()
    problems: list[str] = []
    warns: list[str] = []

    # ---- 1) 版本号一致性 ----
    m = re.search(r"^#\s+.*?v(\d+)\.(\d+)", lines[0])
    if not m:
        problems.append("标题里找不到版本号")
        title_ver = None
    else:
        title_ver = (int(m.group(1)), int(m.group(2)))

    rec_vers = [(int(a), int(b)) for a, b in re.findall(r"\*\*v(\d+)\.(\d+) 变更", text)]
    if rec_vers:
        max_rec = max(rec_vers)
        if title_ver and max_rec != title_ver:
            problems.append(
                f"标题写 v{title_ver[0]}.{title_ver[1]}，但变更记录最新只到 v{max_rec[0]}.{max_rec[1]}"
            )
        # 变更记录里是否有断号
        seq = sorted(set(v[1] for v in rec_vers if v[0] == title_ver[0]))
        if seq and seq != list(range(seq[0], seq[-1] + 1)):
            misses = [n for n in range(seq[0], seq[-1] + 1) if n not in seq]
            if misses:
                warns.append(f"变更记录缺 v{title_ver[0]}.{', v%d.%s' % (title_ver[0], ', v%d.' % title_ver[0]).join(map(str, misses))}")

    # ---- 2) 7.x 章节日期标注 ----
    secs = re.findall(r"^### (7\.\d+)\s+(.+)$", text, re.M)
    # 7.2–7.5 是 v1.1 时代的规划章节，不是实测日志，不要求日期
    undated = [f"{n} {t[:24]}" for n, t in secs
               if n not in ("7.2", "7.3", "7.4", "7.5")
               and not re.search(r"（?20\d\d-\d\d", t)]
    if undated:
        warns.append(f"{len(undated)} 个 7.x 章节没有日期标注：" + "；".join(undated[:3]))

    # ---- 3) 孤儿文件检测（做了但没记）----
    all_text = text
    for rep in ROOT.glob("*.md"):
        if rep.name != DOC.name:
            all_text += rep.read_text(encoding="utf-8", errors="ignore")
    # 注意：这里读的是**仓库外**的工作区记忆。
    # 后果是从 clone 出来的仓库跑，这两份日志不存在，孤儿检测会比本地更严格。
    # 那不是误报——恰恰说明有的脚本只被本地日志记过、没进 README。
    log_dirs = [ROOT / ".workbuddy" / "memory", ROOT.parent / ".workbuddy" / "memory"]
    for log in (f for d in log_dirs if d.exists() for f in d.glob("*.md")):
        all_text += log.read_text(encoding="utf-8", errors="ignore")

    orphans: list[str] = []
    for sub in ("scripts", "config"):
        for f in sorted((ROOT / sub).iterdir()):
            if f.is_dir() or f.name in IGNORE_FILES or f.suffix not in (".py", ".json"):
                continue
            if f.name not in all_text:
                orphans.append(f"{sub}/{f.name}")
    if orphans:
        problems.append(f"{len(orphans)} 个文件从未在任何文档/日志里出现：" + "、".join(orphans))

    # ---- 4) 独立报告是否被引用 ----
    # README 是项目入口，不是待引用的报告；
    # 面试答辩对照是求职材料（对外表达层），不属于项目技术报告，不要求主文档引用
    SKIP_REPORTS = {"README.md", "面试答辩对照_四题证据盘点.md"}
    reports = [f.name for f in ROOT.glob("*.md")
               if f.name != DOC.name and f.name not in SKIP_REPORTS]
    unref = [r for r in reports if r not in text]
    if unref:
        warns.append("以下报告未在主文档中被引用：" + "、".join(unref))

    # ---- 输出 ----
    print(f"文档 {DOC.name}")
    print(f"  标题版本 v{title_ver[0]}.{title_ver[1]}" if title_ver else "  标题版本 未知")
    print(f"  变更记录 {len(rec_vers)} 条，最新 v{max(rec_vers)[0]}.{max(rec_vers)[1]}" if rec_vers else "  变更记录 无")
    print(f"  7.x 章节 {len(secs)} 节，共 {len(lines)} 行")
    print()

    if not problems and not warns:
        print("✓ 一致：版本号同步、章节有日期、无孤儿文件、报告均已引用")
        return

    for p in problems:
        print(f"✗ {p}")
    for w in warns:
        print(f"⚠ {w}")

    if problems:
        print(f"\n{len(problems)} 个问题需要修")
        sys.exit(1)
    if args.strict and warns:
        sys.exit(1)


if __name__ == "__main__":
    main()
