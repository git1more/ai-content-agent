"""生成用于对外展示的压缩资产（docs/showcase/）。

为什么需要这一步：`assets/` 下是**证据**（原始 PNG/MP4、成本台账、版本归档），
总体积 859 MB，不能也不该进 Git 仓库。但面试演示需要能直接看的成品。

所以分成两层：
  · `assets/`     = 原始证据，本地保留，gitignore
  · `docs/showcase/` = 压缩后的展示件，入库

**压缩只用于展示，不替代原图。** 所有量化结论仍以 assets 下的原文件为准。

用法：
  python scripts/make_showcase.py            # 全量重建
  python scripts/make_showcase.py --only ecom
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "showcase"

FG = (26, 26, 26)
MUTED = (110, 110, 110)
BG = (255, 255, 255)
ACCENT = (198, 64, 44)

FONT_PATH = None
for cand in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
             "/System/Library/Fonts/PingFang.ttc"):
    if Path(cand).exists():
        FONT_PATH = cand
        break


def font(size: int) -> ImageFont.ImageFont:
    if FONT_PATH:
        return ImageFont.truetype(FONT_PATH, size)
    return ImageFont.load_default()


def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def contact_sheet(ep: str, title: str, cols: int = 4, thumb_w: int = 300) -> Path | None:
    """把一集 12 镜的关键帧拼成一张四联图，带镜号与台词摘要。

    关键帧原图 928×1664、单张 1.6 MB，12 张近 20 MB；拼成一张后约 1.5 MB，
    既能一眼看完整集节奏，也不占仓库。
    """
    d = ROOT / "assets" / "manju" / ep
    cfg = load_json(ROOT / "config" / f"manju_{ep}.json")
    if not d.exists() or not cfg:
        return None
    shots = cfg.get("shots") or []
    ims: list[tuple[str, str, Image.Image]] = []
    for s in shots:
        p = d / f"{s['id']}_keyframe.png"
        if not p.exists():
            continue
        im = Image.open(p).convert("RGB")
        h = int(im.height * thumb_w / im.width)
        im = im.resize((thumb_w, h), Image.LANCZOS)
        voice = (s.get("voice_line") or {}).get("text") or ""
        label = f"{s['id']} · {s.get('shot_size', '')}"
        ims.append((label, voice, im))
    if not ims:
        return None

    rows = (len(ims) + cols - 1) // cols
    cell_h = max(i.height for _, _, i in ims)
    cap_h = 46
    pad, gap = 18, 12
    head = 62
    W = pad * 2 + cols * thumb_w + (cols - 1) * gap
    H = head + pad + rows * (cell_h + cap_h) + (rows - 1) * gap + pad

    cv = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(cv)
    dr.text((pad, 18), title, font=font(24), fill=FG)

    f_lbl, f_cap = font(15), font(13)
    for idx, (label, voice, im) in enumerate(ims):
        r, c = divmod(idx, cols)
        x = pad + c * (thumb_w + gap)
        y = head + pad + r * (cell_h + cap_h + gap)
        cv.paste(im, (x, y))
        dr.text((x, y + im.height + 5), label, font=f_lbl, fill=ACCENT)
        if voice:
            v = voice if len(voice) <= 16 else voice[:15] + "…"
            dr.text((x, y + im.height + 24), v, font=f_cap, fill=MUTED)

    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / f"{ep}_keyframes.jpg"
    cv.save(dst, quality=86, optimize=True)
    print(f"  {dst.relative_to(ROOT)}  {cv.size[0]}x{cv.size[1]}  {dst.stat().st_size/1024:.0f} KB")
    return dst


def ecom_overview() -> None:
    """每品类一张 12 格总览（已有「整套一览.png」的压缩版）。"""
    root = ROOT / "assets" / "ecom_set"
    dst_dir = OUT / "ecom"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        src = d / "整套一览.png"
        if not src.exists():
            continue
        im = Image.open(src).convert("RGB")
        if im.width > 1400:
            h = int(im.height * 1400 / im.width)
            im = im.resize((1400, h), Image.LANCZOS)
        dst = dst_dir / f"{d.name}.jpg"
        im.save(dst, quality=86, optimize=True)
        print(f"  {dst.relative_to(ROOT)}  {dst.stat().st_size/1024:.0f} KB")


def copy_sheets() -> None:
    """角色卡 / 环境卡 / 证据拼图：压缩后另存，不动原件。"""
    pairs = [
        ("assets/manju/ch04/character_sheet.png", "character_sheet.jpg"),
        ("assets/manju/ch04/env_living_room.png", "env_living_room.jpg"),
        ("assets/manju/ch04/env_entry.png", "env_entry.jpg"),
        ("assets/manju/_envtest/envcard_4shots.png", "envcard_4shots.jpg"),
    ]
    for src, name in pairs:
        p = ROOT / src
        if not p.exists():
            continue
        im = Image.open(p).convert("RGB")
        if im.width > 1400:
            h = int(im.height * 1400 / im.width)
            im = im.resize((1400, h), Image.LANCZOS)
        dst = OUT / name
        im.save(dst, quality=86, optimize=True)
        print(f"  {dst.relative_to(ROOT)}  {dst.stat().st_size/1024:.0f} KB")


def copy_films() -> None:
    """成片不压缩（画质就是成果），只统计。真正的入库规则在 .gitignore 里放行。"""
    for p in [
        ROOT / "assets/manju/ch03/versions/v2/EP01_雨夜第七单_v2.mp4",
        ROOT / "assets/manju/ch04/versions/v4/EP02_雨夜第七单_v4.mp4",
    ]:
        if p.exists():
            print(f"  {p.relative_to(ROOT)}  {p.stat().st_size/1024/1024:.1f} MB（不压缩）")
        else:
            print(f"  [缺] {p.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["keyframes", "ecom", "sheets", "films"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if a.only in (None, "keyframes"):
        print("关键帧总览：")
        contact_sheet("ch03", "EP01 · 雨夜第七单 · 12 镜 46 秒 · 关键帧")
        contact_sheet("ch04", "EP02 · 雨夜第七单（续）· 12 镜 46 秒 · 关键帧")
    if a.only in (None, "ecom"):
        print("电商五品类总览：")
        ecom_overview()
    if a.only in (None, "sheets"):
        print("角色卡 / 环境卡 / 证据拼图：")
        copy_sheets()
    if a.only in (None, "films"):
        print("成片：")
        copy_films()


if __name__ == "__main__":
    main()
