# -*- coding: utf-8 -*-
"""
重新校准用的客观指标：结构比例偏差 + 色相偏移。

存在的原因：2026-11-17 的初轮评分把「看起来不像真商品」当成「与参考图不一致」，
导致 7 项误判。本脚本只保留两个可复算、且已验证可靠的维度：

  1. aspect_dev_pct —— Canny 外框高宽比相对参考图的偏差（CR 用人工量取值覆盖）
  2. hue_delta_deg  —— 高饱和像素的中位色相角差（仅对有彩色商品计算）

明度/材质类不做自动判定：这类差异在本批次里无法与光照归一化分离，
仍由人工对照参考图判定，并在证据里写明参照的是参考图。

用法： python scripts/p0_recalib.py
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
AD = ROOT / "assets" / "p0_testset"
RUNS = ROOT / "runs"
OUT = AD / "recalib_metrics.json"

SKUS = ["TS-WHITE-01", "DR-NAVY-02", "JK-ORANGE-03",
        "SR-GLASS-04", "CR-MATTE-05", "LS-BLACK-06"]

# 自动测量不可靠时的参考值覆盖（必须记录原因，禁止静默失真）
# 2026-11-17 二次校准：原值 0.800 系按设计书 7:5 填写，非参考图实测，已作废。
# 该 SKU 为低对比度白罐，Canny/Sobel/局部方差/GrabCut 四法给出 0.394–0.779，全部不可用；
# 唯一有效口径为网格人工标定（900px 网格，不确定度约 ±3%）。
REF_ASPECT_OVERRIDE = {
    "CR-MATTE-05": (0.940, "网格人工标定：罐体 364×342（网格读数 x268/x632, y358/y700）"),
    "LS-BLACK-06": (2.162, "网格人工标定：两部件外接框 340×735（网格读数 x280/x620, y95/y830）；"
                           "浅灰渐变背景导致 Canny 把整幅画面当轮廓"),
}
# 参考与输出必须同一口径；参考已降级为人工时，输出也须人工，否则是两把尺子比数
OUT_ASPECT_OVERRIDE = {
    "CR-MATTE-05": (0.977, "网格人工标定：罐体 375×365（网格读数 x265/x640, y350/y715）"),
    "LS-BLACK-06": (2.222, "纯白背景，色差分割可靠（379×842，检出 2 个部件）；与参考同为外接框 h/w"),
}


def imread(p: Path):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def edge_bbox(path: Path, close_k: int = 15):
    """Canny + 闭运算取最大轮廓外接框。cv2 读不了中文路径，用 imdecode。"""
    a = imread(path)
    if a is None:
        return None
    g = cv2.GaussianBlur(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    e = cv2.Canny(g, 30, 90)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    e = cv2.morphologyEx(e, cv2.MORPH_CLOSE, k, iterations=3)
    cnts, _ = cv2.findContours(e, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    return dict(w=w, h=h, aspect=round(h / w, 3))


def median_hue(path: Path, s_min: int = 60, v_min: int = 30) -> float | None:
    """高饱和像素的中位色相角（0-180，OpenCV 口径）。无彩色返回 None。"""
    a = imread(path)
    if a is None:
        return None
    hsv = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)
    hch, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = (s >= s_min) & (v >= v_min)
    if m.sum() < 2000:
        return None
    return float(np.median(hch[m]))


def hue_delta(h1: float, h2: float) -> float:
    """色相角环形距离（度）。OpenCV 色相 0-180 对应 0-360°。"""
    d = abs(h1 - h2) * 2
    return min(d, 360 - d)


def verdict(v: float, pass_t: float, fail_t: float) -> str:
    return "PASS" if v < pass_t else ("PARTIAL" if v < fail_t else "FAIL")


def main():
    refs = {}
    for sku in SKUS:
        p = AD / sku / "ref_front.png"
        if sku in REF_ASPECT_OVERRIDE:
            aspect, src = REF_ASPECT_OVERRIDE[sku][0], "manual"
        else:
            aspect, src = edge_bbox(p)["aspect"], "canny"
        refs[sku] = dict(aspect=aspect, src=src, hue=median_hue(p))

    out = {}
    for sku in SKUS:
        r = refs[sku]
        for d in sorted(RUNS.glob(f"{sku}__T1__*")):
            raw = d / "raw.png"
            if not raw.exists():
                cand = sorted(d.glob("raw_*.png"))
                if cand:
                    raw = cand[0]
            if not raw.exists():
                continue
            o = edge_bbox(raw)
            if not o:
                continue
            manual_note = None
            aspect_val = o["aspect"]
            if sku in OUT_ASPECT_OVERRIDE:
                # 人工口径只标定主运行；seed 变体不假装已测，比例项标 N/A（色相仍自动测）
                if "__s" in d.name:
                    aspect_val = None
                    manual_note = "seed 变体未做人工标定，比例项不参与判定"
                else:
                    aspect_val = OUT_ASPECT_OVERRIDE[sku][0]
                    manual_note = OUT_ASPECT_OVERRIDE[sku][1]
            dev = None if aspect_val is None else (aspect_val - r["aspect"]) / r["aspect"] * 100
            hue = median_hue(raw)
            rec = {
                "sku": sku,
                "ref_aspect": r["aspect"],
                "ref_aspect_source": r["src"],
                "out_aspect": None if aspect_val is None else round(aspect_val, 3),
                "aspect_deviation_pct": None if dev is None else round(dev, 1),
                "aspect_verdict": "N/A" if dev is None else verdict(abs(dev), 5, 12),
                "ref_hue": None if r["hue"] is None else round(r["hue"], 1),
                "out_hue": None if hue is None else round(hue, 1),
                "hue_delta_deg": None if (hue is None or r["hue"] is None)
                else round(hue_delta(r["hue"], hue), 1),
            }
            if manual_note:
                rec["aspect_note"] = manual_note
            if rec["hue_delta_deg"] is not None:
                rec["hue_verdict"] = verdict(rec["hue_delta_deg"], 4, 10)
            else:
                rec["hue_verdict"] = "N/A"
            out[d.name] = rec

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{'run':<40}{'h/w偏差':<11}{'比例':<9}{'色相Δ°':<9}{'色相':<6}")
    print("-" * 76)
    for k, v in out.items():
        hd = "-" if v["hue_delta_deg"] is None else f"{v['hue_delta_deg']:.1f}"
        av = v["aspect_deviation_pct"]
        atxt = "  N/A  " if av is None else f"{av:>+7.1f}%"
        print(f"{k:<40}{atxt}   {v['aspect_verdict']:<9}"
              f"{hd:<9}{v['hue_verdict']:<6}")
    print(f"\n写入 {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
