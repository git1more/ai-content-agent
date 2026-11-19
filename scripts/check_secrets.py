"""密钥自检：确认真实 API Key 没有进到工作区、Git 索引或 Git 历史里。

为什么要单独写一个脚本：`.gitignore` 能挡住"没被显式 add 的文件"，
但挡不住三种情况——① `git add -f`；② 把 Key 粘贴进代码或文档；③ 在别的目录
clone 后改了配置再提交回来。前两种是最常见的泄漏方式。

只做检测，不做修复。**发现命中时不要 print 命中的内容**——那会把 Key 打印到终端
和 CI 日志里，等于二次泄漏。

用法：
  python scripts/check_secrets.py            # 查工作区 + 索引 + 全历史
  python scripts/check_secrets.py --worktree # 只查工作区（CI 里跑这个就够）

退出码：0 干净，1 有命中。
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 已知的密钥形态。只匹配"长得像真 Key"的串，不算太激进，避免把占位符当泄漏。
PATTERNS = [
    ("DashScope/OpenAI 风格", re.compile(r"sk-[A-Za-z0-9_\-]{20,}")),
    ("阿里云 AccessKey", re.compile(r"LTAI[A-Za-z0-9]{12,}")),
    ("GitHub Token", re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}")),
    ("AWS AccessKey", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("Bearer 硬编码", re.compile(r"Bearer\s+[A-Za-z0-9_\-\.]{30,}")),
]

# 这些是文档里的占位符，允许出现
ALLOW = ("REPLACE_WITH", "xxxxxxxx", "your-key", "YOUR_KEY", "<待填>")

SKIP_DIRS = {".git", "_tmp", "__pycache__", "venv", ".venv", "node_modules"}
SKIP_SUFFIX = {".png", ".jpg", ".jpeg", ".mp4", ".wav", ".gif", ".exe", ".ttf",
               ".zip", ".pdf", ".psd"}


def real_key() -> str:
    """.env 里的真实 Key（仅用于比对，绝不打印）。"""
    env = ROOT / ".env"
    if not env.exists():
        return ""
    for line in env.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("PROVIDER_API_KEY="):
            return line.split("=", 1)[1].strip()
    return ""


def scan_text(text: str, where: str, key: str, hits: list[str]) -> None:
    if key and len(key) >= 16 and key in text:
        hits.append(f"{where}：命中了 .env 里的真实 PROVIDER_API_KEY")
        return          # 不继续报模式，避免重复计数
    for name, pat in PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        frag = m.group(0)
        if any(a.lower() in frag.lower() for a in ALLOW):
            continue
        # 只报前 6 位，够定位、不够复现
        hits.append(f"{where}：疑似{name}（{frag[:6]}…，长度 {len(frag)}）")


def scan_worktree(key: str, hits: list[str]) -> tuple[int, int]:
    files = 0
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT)
        if set(rel.parts) & SKIP_DIRS or p.suffix.lower() in SKIP_SUFFIX:
            continue
        if rel.name == ".env":
            continue          # 它本来就该有 Key，但已在 .gitignore 里
        files += 1
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        scan_text(text, str(rel).replace("\\", "/"), key, hits)
    return files, 0


def scan_git_history(key: str, hits: list[str]) -> int:
    """遍历仓库里所有 blob（含未被任何提交引用的悬空对象）。"""
    try:
        revs = subprocess.run(["git", "cat-file", "--batch-all-objects",
                               "--batch-check=%(objectname) %(objecttype)"],
                              cwd=ROOT, capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 0
    blobs = [ln.split()[0] for ln in revs.stdout.splitlines() if ln.endswith(" blob")]
    for oid in blobs:
        raw = subprocess.run(["git", "cat-file", "blob", oid], cwd=ROOT,
                             capture_output=True)
        if raw.returncode != 0:
            continue
        try:
            text = raw.stdout.decode("utf-8", errors="ignore")
        except Exception:  # noqa: BLE001
            continue
        scan_text(text, f"git 对象 {oid[:10]}", key, hits)
    return len(blobs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worktree", action="store_true", help="只查工作区，不查 Git 历史")
    a = ap.parse_args()

    key = real_key()
    hits: list[str] = []

    files, _ = scan_worktree(key, hits)
    print(f"工作区：扫描 {files} 个文本文件")

    if not a.worktree:
        n = scan_git_history(key, hits)
        print(f"Git 历史：扫描 {n} 个 blob（含未引用对象）")

    if not key:
        print("提示：没有读到 .env 里的 PROVIDER_API_KEY，只按通用模式检查。")

    if hits:
        print(f"\n✗ {len(hits)} 处命中：")
        for h in hits:
            print("  -", h)
        print("\n处理方式：从源码里删掉，改成读环境变量；若已提交，改写历史后再推送。")
        sys.exit(1)

    print("\n✓ 干净：未发现真实 Key 或可疑密钥串")


if __name__ == "__main__":
    main()
