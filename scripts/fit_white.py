"""白底主图归一化：裁到主体包围盒 → 按比例缩放 → 居中贴到纯白画布。

为什么需要它：主体占比不该靠提示词求模型"占画面 65%"。实测提示词写 65%，
服饰能做到，矮圆柱的罐体和细长的瓶子只有 25–28%（见 P1 品类验证）。
模型对"占多大"没有稳定控制力，这件事必须由确定性的后处理保证。

与 postprocess_canvas.py 的分工：
- postprocess_canvas.py 用 cv2 做连通域抠图，力量更强，但本机装不上 cv2。
- 本脚本只用 PIL + numpy，前提是「源图已经是白底」——主图生成就是这个前提。
  不需要抠图，只要包围盒。而且刻意取所有前景连通域的并集包围盒，
  不用最大连通域：早先那次「口红盖子整件消失」就是最大连通域造成的。

用法：
  python scripts/fit_white.py --input a.png --out b.png --size 1200x1200 --ratio 0.70
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def bg_color(arr: np.ndarray, strip: int = 8) -> np.ndarray:
    edges = np.concatenate([
        arr[:strip].reshape(-1, 3), arr[-strip:].reshape(-1, 3),
        arr[:, :strip].reshape(-1, 3), arr[:, -strip:].reshape(-1, 3),
    ])
    return np.median(edges, axis=0)


def foreground_bbox(arr: np.ndarray, tol: int | None = None) -> tuple[int, int, int, int] | None:
    """返回所有非背景像素的并集包围盒 (x0, y0, x1, y1)。"""
    bg = bg_color(arr)
    spread = float(np.percentile(np.abs(arr.astype(np.int16) - bg.astype(np.int16)).max(axis=2), 99))
    if tol is None:
        tol = int(max(10, min(40, spread + 8)))
    diff = np.abs(arr.astype(np.int16) - bg.astype(np.int16)).max(axis=2)
    fg = diff > tol
    if fg.mean() < 0.004:
        return None
    ys, xs = np.where(fg)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def whiten(arr: np.ndarray, tol: int = 18) -> np.ndarray:
    """把接近背景色的像素推成纯白，避免画布拼接处出现色差。"""
    bg = bg_color(arr)
    diff = np.abs(arr.astype(np.int16) - bg.astype(np.int16)).max(axis=2)
    out = arr.copy()
    out[diff <= tol] = (255, 255, 255)
    return out


def feather_mask(w: int, h: int, feather: int = 14) -> Image.Image:
    """边缘羽化遮罩：往里 feather 像素内从 0 渐变到 255。

    用途：裁切区的背景色与画布底色可能有细微色差（生成图背景是渐变的），
    直接硬贴会看到一圈方块。羽化之后边界消失，且不影响商品本体——
    因为包围盒的四周本来就是背景。
    """
    yy, xx = np.indices((h, w))
    d = np.minimum.reduce([yy, h - 1 - yy, xx, w - 1 - xx]).astype(np.float32)
    a = np.clip(d / max(1, feather), 0.0, 1.0) * 255
    return Image.fromarray(a.astype(np.uint8), "L")


def fit(src: Path, dst: Path, size: tuple[int, int], ratio: float,
        max_w_ratio: float = 0.86, shadow: bool = False) -> dict:
    """裁到主体包围盒 → 等比缩放 → 居中贴到画布。

    画布底色取源图的背景色，不做「接近背景色就刷纯白」。
    注意两件事：
    1. 尺度按「最长边」量（scale_ratio）。矮胖商品的高度会低于目标——面霜罐目标 70%，
       高度实测 0.679，但宽度 0.73，长边是达标的。别用高度去卡矮胖件。
    2. 羽化会让边界被算作背景，外测值比目标低 1–2%。判断阈值留 ±0.03 容差。

    2026-11-18 踩过：磨砂白罐贴在浅灰底上，刷白把商品内部的高光也刷掉了，
    罐身出现两大块白斑。刷白只对「主图」这种真纯白底才有意义，而那种图本来
    不需要刷——生成图四角实测 251–255。浅底详情图（205–221）用同色画布更稳。
    """
    im = Image.open(src).convert("RGB")
    arr = np.asarray(im)
    box = foreground_bbox(arr)
    if box is None:
        return {"ok": False, "why": "找不到主体（前景占比过低）"}

    bg = bg_color(arr)
    canvas_bg = tuple(int(v) for v in bg)
    x0, y0, x1, y1 = box
    crop = Image.fromarray(arr[y0:y1, x0:x1])

    cw, ch = size
    target_h = int(ch * ratio)
    scale = min(target_h / crop.height, (cw * max_w_ratio) / crop.width)
    new = (max(1, round(crop.width * scale)), max(1, round(crop.height * scale)))
    crop = crop.resize(new, Image.LANCZOS)

    canvas = Image.new("RGB", (cw, ch), canvas_bg)
    x = (cw - new[0]) // 2
    y = int((ch - new[1]) / 2)
    if shadow:
        sh = Image.new("L", (cw, ch), 0)
        d = ImageDraw.Draw(sh)
        ew, eh = int(new[0] * 0.62), max(6, int(new[1] * 0.045))
        d.ellipse([x + (new[0] - ew) // 2, y + new[1] - eh // 2,
                   x + (new[0] + ew) // 2, y + new[1] + eh // 2], fill=70)
        sh = sh.filter(ImageFilter.GaussianBlur(max(4, eh // 2)))
        canvas.paste(Image.new("RGB", (cw, ch), (150, 150, 150)), (0, 0), sh)
    if crop.width > 2 and crop.height > 2:
        canvas.paste(crop, (x, y), feather_mask(crop.width, crop.height))
    else:
        canvas.paste(crop, (x, y))
    canvas.save(dst, quality=95)

    got = new[1] / ch
    return {"ok": True, "height_ratio": round(got, 3), "target": ratio,
            "size": f"{cw}x{ch}", "bg": canvas_bg, "bbox": [x0, y0, x1, y1]}


def measure(path: Path) -> dict:
    """量主体在画面里的真实尺寸。不调用任何视觉模型。

    为什么要单独量：VLM 回答的是「面积占比」，而提示词要求的是「高度占比」。
    细长的瓶子按 65% 高度摆好时面积只有 0.20 左右，会被面积口径误判成「偏小」。
    2026-11-18 实测：玻璃瓶 M1 高度 85.6% 被判「占比异常 0.28」——口径错，不是图错。
    """
    arr = np.asarray(Image.open(path).convert("RGB"))
    h, w = arr.shape[:2]
    box = foreground_bbox(arr)
    if box is None:
        return {"ok": False, "why": "找不到主体"}
    x0, y0, x1, y1 = box
    hr = (y1 - y0) / h
    wr = (x1 - x0) / w
    return {"ok": True,
            "height_ratio": round(hr, 3),
            "width_ratio": round(wr, 3),
            # 尺度取最长边：矮胖的罐子要占够高度就必然超出画布宽度，
            # 高度口径会把「已经撑满」误判成「偏小」（面霜罐 D3 实测 53%）。
            "scale_ratio": round(max(hr, wr), 3),
            "area_ratio": round((y1 - y0) * (x1 - x0) / (h * w), 3),
            "bbox": [x0, y0, x1, y1], "canvas": [w, h]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", help="输出路径；--measure 时不需要")
    ap.add_argument("--size", default="1200x1200")
    ap.add_argument("--ratio", type=float, default=0.70)
    ap.add_argument("--shadow", action="store_true")
    ap.add_argument("--measure", action="store_true", help="只测量主体尺寸，不输出文件")
    a = ap.parse_args()
    if a.measure:
        print(measure(Path(a.input)))
        return
    if not a.out:
        ap.error("需要 --out（或改用 --measure 只测量）")
    w, h = (int(v) for v in a.size.lower().split("x"))
    r = fit(Path(a.input), Path(a.out), (w, h), a.ratio, shadow=a.shadow)
    print(r)
    raise SystemExit(0 if r["ok"] else 1)


if __name__ == "__main__":
    main()
