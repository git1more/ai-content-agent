# -*- coding: utf-8 -*-
"""
关键帧结构门禁：出图后立刻把「高危区」裁出来，供人 10 秒过一遍。

为什么需要它：
  EP00 的 S1 连续两次栽在同一处 —— 一手持手机 + 另一手抱头盔。
    v1：握手机的手多出一根朝上的手指
    v2：出现两只右手（一只拿手机、一只抱头盔）
  这两次都是**关键帧就已经错了**，但我们直到整片出来才发现，
  等于白花了 ¥0.60/镜 去做视频。视频是最贵的环节，必须后置。

  机器判不了"两只右手"这种结构错误（没有骨骼先验），但可以把它放大到
  一眼就能看出来。人花 10 秒，省掉一次返工。

高危区定义（按镜头类型）：
  含手/持物  → 下半身 60% 放大（手出现概率最高）
  近景/特写  → 面部 40% 放大（五官一致性）

用法：
  python scripts/manju_gate.py
  python scripts/manju_gate.py --out assets/manju/ch01/_gate.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
CFG = ROOT / "config" / "manju_ch01.json"
OUT = ROOT / "assets" / "manju" / "ch01"
FONT = Path("C:/Windows/Fonts/msyh.ttc")

HAND_WORDS = ("手", "握", "抱", "持", "拿", "扶", "把")


def regions_for(shot: dict) -> list[tuple[str, tuple[float, float, float, float]]]:
    """按镜头内容决定要查哪些区域。"""
    text = (shot.get("keyframe_prompt") or "") + (shot.get("subject") or "")
    out = []
    if any(w in text for w in HAND_WORDS):
        out.append(("手部", (0.00, 0.40, 1.00, 1.00)))
    size = shot.get("shot_size") or ""
    if "近景" in size or "特写" in size:
        out.append(("面部", (0.15, 0.05, 0.90, 0.55)))
    if not out:
        out.append(("全图", (0.0, 0.0, 1.0, 1.0)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT / "_gate.png"))
    args = ap.parse_args()

    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    f_title = ImageFont.truetype(str(FONT), 24)
    f_sub = ImageFont.truetype(str(FONT), 17)

    tiles: list[tuple[str, str, Image.Image]] = []
    for shot in cfg["shots"]:
        sid = shot["id"]
        p = OUT / f"{sid}_keyframe.png"
        if not p.exists():
            continue
        im = Image.open(p).convert("RGB")
        W, H = im.size
        for name, (x0, y0, x1, y1) in regions_for(shot):
            c = im.crop((int(W * x0), int(H * y0), int(W * x1), int(H * y1)))
            h = 620
            c = c.resize((max(1, int(c.width * h / c.height)), h), Image.LANCZOS)
            if c.width > 460:
                c = c.resize((460, int(c.height * 460 / c.width)), Image.LANCZOS)
            tiles.append((sid, name, c))

    if not tiles:
        raise SystemExit("没有关键帧，先跑 --stage keyframe")

    pad, label_h = 16, 74
    W = sum(t[2].width for t in tiles) + pad * (len(tiles) + 1)
    H = max(t[2].height for t in tiles) + label_h + pad * 2
    cv = Image.new("RGB", (W, H), (250, 250, 248))
    d = ImageDraw.Draw(cv)
    x = pad
    for sid, name, img in tiles:
        cv.paste(img, (x, pad))
        d.text((x, pad + img.height + 8), f"{sid} · {name}", font=f_title, fill=(24, 24, 24))
        # 副标题按格宽截断，避免压到相邻格子
        sub = "查手指 / 左右手 / 肢体连贯"
        while sub and d.textlength(sub, font=f_sub) > img.width:
            sub = sub[:-1]
        d.text((x, pad + img.height + 40), sub, font=f_sub, fill=(120, 120, 120))
        x += img.width + pad

    Image.Image.save(cv, args.out)
    print(f"门禁检查图 -> {args.out}")
    print("人工确认项：① 手指数目对不对 ② 有没有两只同侧手 ③ 手臂是否从身体连贯长出 ④ 五官是否是同一人")
    print("任何一项不过，就不要进视频阶段（¥0.60/镜）。")


if __name__ == "__main__":
    main()
