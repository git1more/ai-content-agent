# -*- coding: utf-8 -*-
"""
P0 测试集素材预处理
------------------
做三件事：
1. 去掉生成图右下角的水印（逐行取左侧参考列填充，适配平滑影棚背景）
2. 统一重命名为 ref_front.png，原始文件归档到 _raw/
3. 生成 6 SKU 的拼版预览图，便于人工比对

用法： python scripts/p0_prepare_assets.py
"""

from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
ASSET_DIR = ROOT / "assets" / "p0_testset"

# SKU -> 源文件名（生成时自动命名的长文件名，显式列出避免误匹配）
SOURCES = {
    "TS-WHITE-01": "专业电商商品摄影_一件纯白色纯棉圆领短袖T恤_穿在隐形人台上_2026-11-17T04-53-35.png",
    "DR-NAVY-02": "专业电商商品摄影_一件藏青色_navy_约_1B2A4A_无_2026-11-17T04-53-45.png",
    "JK-ORANGE-03": "专业电商商品摄影_一件荧光橙色_约_FF5A1F_轻薄运动风_2026-11-17T04-54-10.png",
    "SR-GLASS-04": "专业美妆产品摄影_一瓶透明玻璃精华液_园柱形瓶身_30ml__2026-11-17T04-54-25.png",
    "CR-MATTE-05": "只修改罐身正面的烫印文字_把现在的_HVDRA_CREAM__2026-11-17T04-55-36.png",
    "LS-BLACK-06": "只修复口红的醬体_把醬体表面那个不自然的凹陷_压痕痕迹去除__2026-11-17T04-55-50.png",
}

# 需要保留的"负样本"：修文字之前的 HVDRA 版本，作为文字失败样例留档
NEGATIVE_SAMPLES = {
    "CR-MATTE-05": (
        "专业美妆产品摄影_一个磨砂哑光面霜罐_园柱形_直径约7cm高_2026-11-17T04-54-53.png",
        "negative_text_HVDRA.png",
    )
}


def strip_watermark(img: Image.Image, x_frac=0.850, y_frac=0.900, pad=6) -> Image.Image:
    """抹掉右下角水印。

    做法：水印所在区域的每一行，取该区域左侧 pad~2*pad 列的像素均值作为该行填充色。
    影棚背景在水平方向近乎均匀、垂直方向才有渐变，所以逐行填充不会破坏渐变。
    """
    arr = np.array(img.convert("RGB")).astype(np.float32)
    h, w, _ = arr.shape
    x0 = int(w * x_frac)
    y0 = int(h * y_frac)

    # 参考列：水印左侧 pad 距离外的一小段（避开可能的水印抗锯齿边缘）
    ref_lo = max(0, x0 - 3 * pad)
    ref_hi = max(1, x0 - pad)

    patch = arr[y0:h, ref_lo:ref_hi, :]
    if patch.size == 0:
        return img

    row_color = patch.mean(axis=1, keepdims=True)          # (rows, 1, 3)
    target = arr[y0:h, x0:w, :]

    # 横向做一点线性微调：若该行左右两端有细微亮度差，按比例过渡，避免出现垂直硬边
    left_val = row_color[:, 0, :]                           # (rows, 3)
    right_ref = arr[y0:h, max(0, w - 4):w, :].mean(axis=1)  # (rows, 3)
    n = target.shape[1]
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)[None, :, None]
    fill = left_val[:, None, :] * (1 - ramp) + right_ref[:, None, :] * ramp

    arr[y0:h, x0:w, :] = fill
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def contact_sheet(paths, out_path, cols=3, cell=420):
    """拼版预览：每格按比例缩放后居中贴在浅灰底上。"""
    rows = (len(paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * cell), (238, 238, 238))
    for i, p in enumerate(paths):
        im = Image.open(p).convert("RGB")
        im.thumbnail((cell - 20, cell - 20), Image.LANCZOS)
        x = (i % cols) * cell + (cell - im.width) // 2
        y = (i // cols) * cell + (cell - im.height) // 2
        sheet.paste(im, (x, y))
    sheet.save(out_path, quality=95)
    return out_path


def main():
    if not ASSET_DIR.exists():
        sys.exit(f"目录不存在: {ASSET_DIR}")

    done = []
    for sku, fname in SOURCES.items():
        src = ASSET_DIR / sku / fname
        if not src.exists():
            print(f"[跳过] 找不到源文件 {sku}: {fname}")
            continue

        raw_dir = ASSET_DIR / sku / "_raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        # 归档原始文件（保留带水印版本，便于溯源）
        raw_copy = raw_dir / "front_raw.png"
        if not raw_copy.exists():
            shutil.copy2(src, raw_copy)

        img = Image.open(src)
        clean = strip_watermark(img)
        out = ASSET_DIR / sku / "ref_front.png"
        clean.save(out, optimize=True)
        done.append(out)

        w, h = clean.size
        note = "" if w == h else f"  非 1:1 ({w}x{h})"
        print(f"[完成] {sku:14s} -> ref_front.png  {w}x{h}{note}")

    # 留档负样本
    for sku, (fname, newname) in NEGATIVE_SAMPLES.items():
        src = ASSET_DIR / sku / fname
        if src.exists():
            dst = (ASSET_DIR / sku / "_raw" / newname)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(src, dst)
            print(f"[负样本] {sku} -> _raw/{newname}")

    if done:
        sheet = ASSET_DIR / "contact_sheet.png"
        contact_sheet(done, sheet)
        print(f"[拼版] {sheet}")

    print(f"\n共处理 {len(done)} 个 SKU")


if __name__ == "__main__":
    main()
