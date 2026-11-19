# -*- coding: utf-8 -*-
"""
唇形运动检测 —— 判断「这句台词到底该不该让角色动嘴」。

为什么需要它：
  漫剧的台词分两类，处理方式完全相反。
    · 角色开口（as = 角色 ID）  → 必须有口型对齐，否则一眼假
    · 画外旁白（as = narrator） → 角色必须闭嘴，动了反而像配音没对上
  这两类在一份剧本里混着写，肉眼很难逐镜核对，于是做成可测量的：
  量「相邻帧唇部灰度剖面相似度」。唇形稳定 → 相似度接近 1；唇形在动 → 掉到 0.9 以下。

方法（为什么不用帧差/面积）：
  帧差被全局亮度抖动污染——实测额头（本应静止）的变化量比嘴部还大；
  阈值面积法是循环论证——用百分位定阈值，选出的像素数恒定。
  取嘴部每一行的平均亮度当剖面，再做「相邻帧剖面相关」，只对形状敏感、对整体亮度不敏感。

用法：
  python scripts/mouth_check.py --config config/manju_ch03.json
  python scripts/mouth_check.py --config config/manju_ch03.json --shots S2 S12

需要镜头在配置里写 `mouth_box: [x0, y0, x1, y1]`（原分辨率像素坐标），
并且 clips/ 与 voice/ 下已有产物。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
FFMPEG = ROOT / "tools" / "ffmpeg" / "ffmpeg.exe"
FPS = 10                 # 采样率；10 帧/秒足够看清唇形变化，又不至于被抖动淹没
CHURN_SIM = 0.90         # 相邻帧相似度低于此值算「唇形发生明显变化」


def _frames(clip: Path, tmp: Path) -> list[Path]:
    tmp.mkdir(parents=True, exist_ok=True)
    for f in tmp.glob("*.png"):
        f.unlink()
    subprocess.run([str(FFMPEG), "-y", "-hide_banner", "-loglevel", "error",
                    "-i", str(clip), "-vf", f"fps={FPS}", str(tmp / "f%04d.png")],
                   check=True, capture_output=True)
    return sorted(tmp.glob("f*.png"))


def _profiles(frames: list[Path], box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    rows = []
    for f in frames:
        g = np.asarray(Image.open(f).convert("L"), dtype=np.float32)
        p = g[y0:y1, x0:x1].mean(axis=1)
        rows.append((p - p.mean()) / (p.std() + 1e-6))
    return np.array(rows)


def _envelope(wav: Path, n: int) -> np.ndarray:
    with wave.open(str(wav)) as w:
        sr, raw = w.getframerate(), w.readframes(w.getnframes())
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    hop = max(1, sr // FPS)
    env = [float(np.sqrt((a[i:i + hop] ** 2).mean())) if a[i:i + hop].size else 0.0
           for i in range(0, len(a), hop)]
    return np.array((env + [0.0] * n)[:n])


def measure(clip: Path, wav: Path, box: tuple[int, int, int, int]) -> dict:
    with tempfile.TemporaryDirectory() as td:
        frames = _frames(clip, Path(td))
        prof = _profiles(frames, box)
    sim = np.array([1.0] + [float(np.corrcoef(prof[i], prof[i - 1])[0, 1])
                            for i in range(1, len(prof))])
    env = _envelope(wav, len(sim))
    n = min(len(sim), len(env))
    sim, env = sim[:n], env[:n]
    voiced = env > env.mean()
    # 相似度越低 = 唇形变化越大，所以「变化量」取 1-sim
    r = float(np.corrcoef(1.0 - sim, env)[0, 1]) if sim.std() > 1e-9 else 0.0
    return {
        "frames": n,
        "sim_mean": float(sim.mean()),
        "sim_min": float(sim.min()),
        "churn_pct": float((sim < CHURN_SIM).mean() * 100),
        "voiced_sim": float(sim[voiced].mean()) if voiced.any() else float("nan"),
        "silent_sim": float(sim[~voiced].mean()) if (~voiced).any() else float("nan"),
        "r_audio": r,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--shots", nargs="*")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    ep = cfg_path.stem.replace("manju_", "")
    base = ROOT / "assets" / "manju" / ep

    rows, missing = [], []
    for shot in cfg["shots"]:
        sid = shot["id"]
        if args.shots and sid not in args.shots:
            continue
        box = shot.get("mouth_box")
        if not box:
            missing.append(sid)
            continue
        clip = base / "clips" / f"{sid}.mp4"
        wav = base / "voice" / f"{sid}_fit.wav"
        if not clip.exists() or not wav.exists():
            missing.append(f"{sid}(产物缺)")
            continue
        vl = shot.get("voice_line") or {}
        rows.append((sid, str(vl.get("as") or "—"), measure(clip, wav, tuple(box))))

    if not rows:
        print("没有可测镜头：需要配置里的 mouth_box + clips/voice 产物")
        sys.exit(1)

    print(f"{'镜头':<6}{'说话人':<10}{'唇形相似度':>10}{'最低':>8}"
          f"{'变化帧%':>9}{'有声段':>9}{'静音段':>9}{'r(音频)':>9}  判定")
    print("-" * 92)
    for sid, who, m in rows:
        offscreen = who in ("narrator", "unknown")
        # 画外音却在高频动唇 = 观众会误读成「他在说话」
        if offscreen:
            ok = m["churn_pct"] < 20
            verdict = "✓ 画外音，唇静" if ok else "✗ 画外音但唇在动"
        else:
            ok = m["churn_pct"] >= 20 or abs(m["r_audio"]) >= 0.4
            verdict = "✓ 角色开口，有唇动" if ok else "✗ 角色开口但唇不动"
        print(f"{sid:<6}{who:<10}{m['sim_mean']:>10.3f}{m['sim_min']:>8.3f}"
              f"{m['churn_pct']:>8.0f}%{m['voiced_sim']:>9.3f}{m['silent_sim']:>9.3f}"
              f"{m['r_audio']:>+9.3f}  {verdict}")

    if missing:
        print(f"\n未测：{', '.join(missing)}（缺 mouth_box 或产物）")


if __name__ == "__main__":
    main()
