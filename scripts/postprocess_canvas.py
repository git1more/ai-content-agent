# -*- coding: utf-8 -*-
"""
画布合成 / 版面规范化 —— 确定性后处理层
=====================================

为什么必须有这一层（实测依据）：
  内置生成能力与 qwen-image-3.0 两次 T1 实测，均无视「商品占画面 60-70%、顶部留 15%
  安全区」，一律把商品撐到 88-93%。生成模型没有"画布"和"留白"概念，提示词改不动。
  且模型生成的阴影是不规则投影，不符合规范。

这一层的职责（全部可复现、无随机性）：
  1. 从生成图中抠出商品主体（背景色驱动 + 连通域，不依赖外部 matting 模型）
  2. 按画布规格等比缩放、水平居中、控制顶部安全区
  3. 叠加规范接触阴影（固定形状与强度，不随模型变）
  4. 输出合规报告，不合规直接报错 —— 用于流水线门禁

用法：
  python scripts/postprocess_canvas.py --input in.png --spec white_pack_1x1 --out out.png
  python scripts/postprocess_canvas.py --check out.png --spec white_pack_1x1
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# 画布规格
# --------------------------------------------------------------------------- #
@dataclass
class Spec:
    name: str
    canvas: tuple[int, int]          # (w, h)
    height_ratio: float              # 商品高度占画布高度
    top_ratio: float                 # 顶部安全区占画布高度
    bg: tuple[int, int, int] | None  # None = 保留原背景（场景图用）
    shadow: bool
    shadow_strength: float = 0.12
    tile_ratio: float = 1.35         # 商品宽/高比上限，超过则按宽适配


SPECS: dict[str, Spec] = {
    # 白底主图：1:1，主体 65%，顶部 15% 安全区
    "white_pack_1x1": Spec("white_pack_1x1", (1200, 1200), 0.65, 0.15, (255, 255, 255), True),
    "white_pack_1x1_800": Spec("white_pack_1x1_800", (800, 800), 0.65, 0.15, (255, 255, 255), True),
    # 详情页竖图：3:4
    "detail_3x4": Spec("detail_3x4", (1200, 1600), 0.68, 0.12, (255, 255, 255), True),
    # 场景图：3:4，不换背景、不加人造阴影
    "scene_3x4": Spec("scene_3x4", (1200, 1600), 0.70, 0.08, None, False),
    # 主图横版：16:9
    "white_pack_16x9": Spec("white_pack_16x9", (1920, 1080), 0.62, 0.14, (255, 255, 255), True),
}


# --------------------------------------------------------------------------- #
# 主体提取
# --------------------------------------------------------------------------- #
def estimate_bg(arr: np.ndarray, strip: int = 10) -> tuple[np.ndarray, float]:
    """取四边条带的中位色作为背景色，并给出该区域的离散程度。"""
    h, w, _ = arr.shape
    edges = np.concatenate([
        arr[:strip, :, :].reshape(-1, 3),
        arr[-strip:, :, :].reshape(-1, 3),
        arr[:, :strip, :].reshape(-1, 3),
        arr[:, -strip:, :].reshape(-1, 3),
    ])
    bg = np.median(edges, axis=0)
    spread = float(np.percentile(np.abs(edges - bg).max(axis=1), 95))
    return bg, spread


def foreground_mask(arr: np.ndarray, tol: int = 12, min_area_ratio: float = 0.01) -> np.ndarray:
    """背景色驱动的连通域抠图。

    步骤：与背景色的最大通道差 → 阈值 → 取与边框相连的背景连通域 → 前景 = 其余
    → 保留最大前景连通域 → 填内部孔洞（例如袖子与衣身之间的封闭区域）
    """
    h, w, _ = arr.shape
    bg, _ = estimate_bg(arr)
    diff = np.abs(arr.astype(np.int16) - bg.astype(np.int16)).max(axis=2)
    bg_candidate = (diff <= tol).astype(np.uint8)

    # 背景连通域：从边框种子出发的连通标记
    n, labels = cv2.connectedComponents(bg_candidate, connectivity=4)
    border_labels = set(labels[0, :].tolist()) | set(labels[-1, :].tolist()) \
                  | set(labels[:, 0].tolist()) | set(labels[:, -1].tolist())
    bg_mask = np.isin(labels, list(border_labels)).astype(np.uint8)

    fg = (1 - bg_mask).astype(np.uint8)

    # 保留所有显著前景连通域（不能只留最大的：口红+盖子、套装、鞋+鞋盒等
    # 多件并排构图会被截掉次要件。2026-11-17 实测：口红盖子整件丢失）
    n2, lab2, stats2, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n2 <= 1:
        raise RuntimeError("未能提取到前景，检查输入图背景是否足够干净（本层要求近纯色背景）")
    areas = stats2[1:, cv2.CC_STAT_AREA]
    # 主件阈值沿用 min_area_ratio；配件阈值放宽到 1/10，滤掉噪点但保住小件
    part_thr = max(h * w * min_area_ratio * 0.1, 300)
    keep = [i + 1 for i, a in enumerate(areas) if a >= part_thr]
    if not keep or areas.max() < h * w * min_area_ratio:
        raise RuntimeError("前景面积过小，抠图失败")
    mask = np.isin(lab2, keep).astype(np.uint8)

    # 填内部孔洞：从外部 flood fill 不可达的区域即孔洞
    ff = mask.copy()
    hh, ww = ff.shape
    ff_mask = np.zeros((hh + 2, ww + 2), np.uint8)
    cv2.floodFill(ff, ff_mask, (0, 0), 1)
    holes = ((ff == 0) & (mask == 0)).astype(np.uint8)
    mask = np.clip(mask + holes, 0, 1)

    # 闭运算补小缺口
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask


def content_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


# --------------------------------------------------------------------------- #
# 合成
# --------------------------------------------------------------------------- #
def make_shadow(canvas_w: int, canvas_h: int, prod_w: int, prod_h: int,
                strength: float) -> Image.Image:
    """规范接触阴影：居中的柔和椭圆，固定相对尺寸与强度。"""
    sw = int(prod_w * 0.62)
    sh = max(8, int(canvas_h * 0.030))
    layer = Image.new("L", (canvas_w, canvas_h), 0)
    d = ImageDraw.Draw(layer)
    cx = canvas_w // 2
    # 椭圆中心落在商品底边稍上方，模拟接触面
    cy = 0
    d.ellipse([cx - sw // 2, cy - sh // 2, cx + sw // 2, cy + sh // 2], fill=255)
    layer = layer.filter(ImageFilter.GaussianBlur(radius=max(4, sh // 3)))
    alpha = layer.point(lambda v: int(v * strength))
    return alpha


def compose(src_path: str | Path, spec: Spec, out_path: str | Path,
            tol: int = 12) -> dict:
    img = Image.open(src_path).convert("RGB")
    arr = np.array(img)
    mask = foreground_mask(arr, tol=tol)
    x0, y0, x1, y1 = content_bbox(mask)

    prod = img.crop((x0, y0, x1, y1))
    pmask = Image.fromarray((mask[y0:y1, x0:x1] * 255).astype(np.uint8), mode="L")

    # 轻微腐蚀 + 羽化，消除背景色晕边
    pmask = pmask.filter(ImageFilter.MinFilter(3))
    pmask = pmask.filter(ImageFilter.GaussianBlur(1.2))

    cw, ch = spec.canvas
    target_h = ch * spec.height_ratio
    scale = target_h / prod.height
    new_w = int(round(prod.width * scale))
    new_h = int(round(prod.height * scale))

    # 宽度也要留边，避免长条商品撑出画布
    max_w = cw * (1 - 0.10)
    if new_w > max_w:
        s2 = max_w / new_w
        new_w = int(new_w * s2)
        new_h = int(new_h * s2)

    prod_r = prod.resize((new_w, new_h), Image.LANCZOS)
    pmask_r = pmask.resize((new_w, new_h), Image.LANCZOS)

    bg_color = spec.bg or (255, 255, 255)
    canvas = Image.new("RGB", (cw, ch), bg_color)

    px = (cw - new_w) // 2
    py = int(round(ch * spec.top_ratio))

    if spec.shadow and spec.bg is not None:
        sh_alpha = make_shadow(cw, ch, new_w, new_h, spec.shadow_strength)
        # 把阴影层平移到商品底部
        sh_full = Image.new("L", (cw, ch), 0)
        sh_full.paste(sh_alpha, (0, py + new_h - int(ch * 0.012)))
        shadow_layer = Image.new("RGB", (cw, ch), (70, 70, 75))
        canvas = Image.composite(shadow_layer, canvas, sh_full)

    canvas.paste(prod_r, (px, py), pmask_r)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, optimize=True)

    report = dict(spec=spec.name)
    report.update(check_layout(out_path, spec))
    return report


# --------------------------------------------------------------------------- #
# 合规校验（流水线门禁）
# --------------------------------------------------------------------------- #
def check_layout(path: str | Path, spec: Spec, tol: int = 12) -> dict:
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    h, w, _ = arr.shape

    try:
        mask = foreground_mask(arr, tol=tol)
        x0, y0, x1, y1 = content_bbox(mask)
        prod_h_ratio = (y1 - y0) / h
        prod_w_ratio = (x1 - x0) / w
        top_ratio = y0 / h
        left_ratio = x0 / w
        right_ratio = (w - x1) / w
    except RuntimeError:
        prod_h_ratio = prod_w_ratio = top_ratio = left_ratio = right_ratio = 0.0

    bg, bg_spread = estimate_bg(arr)
    bg_ok = bool(spec.bg is None or (bg.min() >= 250 and bg_spread <= 3))

    issues = []
    if abs(prod_h_ratio - spec.height_ratio) > 0.03:
        issues.append(f"商品高度占比 {prod_h_ratio:.3f} 偏离目标 {spec.height_ratio:.2f}（±0.03）")
    if top_ratio + 1e-6 < spec.top_ratio - 0.02:
        issues.append(f"顶部安全区 {top_ratio:.3f} 低于要求 {spec.top_ratio:.2f}")
    if abs(left_ratio - right_ratio) > 0.02:
        issues.append(f"左右留白不对称（{left_ratio:.3f} vs {right_ratio:.3f}）")
    if not bg_ok:
        issues.append(f"背景不纯白（中位色 {bg.astype(int).tolist()}，离散度 {bg_spread:.1f}）")
    if prod_w_ratio < 0.05 or prod_h_ratio < 0.05:
        issues.append("未能识别商品主体")

    return {
        "pass": len(issues) == 0,
        "canvas": [w, h],
        "product_h_ratio": round(prod_h_ratio, 3),
        "product_w_ratio": round(prod_w_ratio, 3),
        "top_safe_ratio": round(top_ratio, 3),
        "side_balance": [round(left_ratio, 3), round(right_ratio, 3)],
        "bg_median": [int(v) for v in bg],
        "bg_spread": round(bg_spread, 2),
        "issues": issues,
    }


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input")
    ap.add_argument("--out")
    ap.add_argument("--spec", default="white_pack_1x1", choices=list(SPECS))
    ap.add_argument("--check", help="只做合规校验，不合成")
    ap.add_argument("--tol", type=int, default=12)
    ap.add_argument("--json", action="store_true", help="只输出 JSON")
    args = ap.parse_args()

    spec = SPECS[args.spec]

    if args.check:
        rep = check_layout(args.check, spec, tol=args.tol)
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        sys.exit(0 if rep["pass"] else 1)

    if not args.input or not args.out:
        ap.print_help()
        return

    rep = compose(args.input, spec, args.out, tol=args.tol)
    if args.json:
        print(json.dumps(rep, ensure_ascii=False))
    else:
        flag = "PASS" if rep["pass"] else "FAIL"
        print(f"[{flag}] {args.out}")
        print(f"  画布 {rep['canvas']}  商品高占比 {rep['product_h_ratio']}  "
              f"顶部安全区 {rep['top_safe_ratio']}  左右 {rep['side_balance']}")
        for i in rep["issues"]:
            print(f"  ! {i}")
    sys.exit(0 if rep["pass"] else 1)


if __name__ == "__main__":
    main()
