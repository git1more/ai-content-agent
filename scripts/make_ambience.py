# -*- coding: utf-8 -*-
"""
环境音：确定性合成，不依赖任何外部素材。

为什么合成，而不是找素材：
  1. 手上没有可商用的雨声/BGM 素材，从网上找有版权风险（漫剧要能商用）；
  2. 漫剧的环境音只需要「存在感」，不需要可辨识的具体音效 —— 压在 -26dB 下面，
     观众感知到的是「这个空间有空气」，而不是「这是哪一场雨」；
  3. 确定性合成可复现、零成本、零版权，与 fit_white（占比靠后处理）、
     PIL 字幕（文字靠渲染）是同一条哲学：**能算准的就不交给模型/素材**。

声音设计（不是随便加噪声）：
  雨声 ≈ 粉红噪声（低频厚重、高频自然衰减，频谱接近真实雨）
       + 白噪声高通层（雨点打在铁皮/水面上的「沙沙」）
       + 缓慢音量起伏（0.12Hz 级别）——没有起伏就是机器噪声，一听就假；
         有了起伏才像「雨势有大小」，这是合成雨声能否以假乱真的关键。

  室内底噪 ≈ 极低音量粉红噪声（空调/电器/远处车流），只提供「房间不是真空」这一信息。

BGM 不在这里做，但原因要写准：百炼的 fun-music-v1（¥0.002/秒）**存在**，
只是处于邀测、且它是「自动写词并演唱」的歌曲模型，没有纯器乐模式——
当台词下的 BGM 会和人声撞。程序化合成旋律必然难听，所以也不自己搓。
config 里留 `ambience.bgm` 字段，拿到素材或邀测权限时直接接。

用法：
  python scripts/make_ambience.py --list
  python scripts/make_ambience.py --type rain_night_alley --duration 16 --out _tmp/rain.wav
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FFMPEG = ROOT / "tools" / "ffmpeg" / "ffmpeg.exe"

SR = 48000

# 每层：噪声颜色 + 振幅 + 滤波链。振幅是相对值，最终整体电平由 target_db 决定。
TYPES: dict[str, dict] = {
    "rain_night_alley": {
        "desc": "暴雨夜窄巷：厚雨幕 + 雨点打在地面与铁皮上",
        # 配比经过实测校准：首版白噪声层 amp 0.16 太强，频谱质心 5339Hz、
        # 低/高频能量比 0.2，听感是"嘶嘶"的雪花声而不是雨。
        # 雨的能量主体在 200–3000Hz，高频只该是薄薄一层"沙沙"。
        "layers": [
            {"color": "pink",  "amp": 0.60, "chain": "highpass=f=40,lowpass=f=3000"},
            {"color": "white", "amp": 0.06, "chain": "highpass=f=1800,lowpass=f=7000"},
            {"color": "brown", "amp": 0.45, "chain": "highpass=f=45,lowpass=f=280"},  # 雨幕低频 + 远处闷响
        ],
        "swell": {"depth": 0.16, "period_s": 9.0},
        "target_db": -34.0,
    },
    "rain_heavy": {
        "desc": "纯暴雨，无街景低频（空旷处）",
        "layers": [
            {"color": "pink",  "amp": 0.62, "chain": "highpass=f=50,lowpass=f=3600"},
            {"color": "white", "amp": 0.09, "chain": "highpass=f=2200,lowpass=f=8000"},
        ],
        "swell": {"depth": 0.14, "period_s": 7.0},
        "target_db": -33.0,
    },
    "indoor_quiet": {
        "desc": "安静室内：空调/电器底噪，几乎不可闻但填空",
        "layers": [
            {"color": "pink", "amp": 0.35, "chain": "highpass=f=120,lowpass=f=1800"},
            {"color": "brown", "amp": 0.18, "chain": "lowpass=f=120"},
        ],
        "swell": {"depth": 0.06, "period_s": 12.0},
        "target_db": -44.0,
    },
    "street_night": {
        "desc": "夜间街道：远处车流 + 城市低频轰鸣（无雨）",
        "layers": [
            {"color": "pink",  "amp": 0.40, "chain": "highpass=f=150,lowpass=f=3000"},
            {"color": "brown", "amp": 0.45, "chain": "lowpass=f=200"},
        ],
        "swell": {"depth": 0.12, "period_s": 11.0},
        "target_db": -36.0,
    },
}


def build(type_key: str, duration: float, out: Path,
          target_db: float | None = None) -> dict:
    """合成 → 测电平 → 按差值补增益。

    为什么不是直接写 volume=-26dB：
      ffmpeg 的 volume 是**相对**增益，`volume=-26dB` 是"衰减 26dB"，
      不是"设为 -26dB"。第一版就是这么写错的，目标 -26 实测只有 -50.8。
      噪声类信号又没法靠 amplitude 反推（多层 amix 后电平不可预测），
      所以只能量一遍再补。
    """
    spec = TYPES[type_key]
    db = spec["target_db"] if target_db is None else target_db
    n = len(spec["layers"])
    raw = out.with_name(out.stem + "_raw" + out.suffix)

    inputs: list[str] = []
    parts: list[str] = []
    for i, layer in enumerate(spec["layers"]):
        inputs += ["-f", "lavfi", "-i",
                   f"anoisesrc=color={layer['color']}:d={duration:.3f}:r={SR}:a={layer['amp']}"]
        parts.append(f"[{i}:a]{layer['chain']}[l{i}]")

    mix = "".join(f"[l{i}]" for i in range(n)) + f"amix=inputs={n}:normalize=0[m]"

    # 起伏：让雨势有大小。没有这一层，合成噪声会立刻被听成"电视雪花声"。
    sw = spec["swell"]
    d = sw["depth"]
    t = sw["period_s"]
    swell = (f"[m]volume='1-{d}+{d}*sin(2*PI*t/{t})':eval=frame"
             f",afade=t=in:st=0:d=1.2"
             f",afade=t=out:st={max(0.0, duration - 1.6):.3f}:d=1.6"
             f",aformat=sample_fmts=s16:channel_layouts=stereo[aout]")

    cmd = [str(FFMPEG), "-y", "-hide_banner", "-loglevel", "error",
           *inputs, "-filter_complex", ";".join(parts + [mix, swell]),
           "-map", "[aout]", "-ar", str(SR), "-ac", "2",
           "-c:a", "pcm_s16le", str(raw)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if r.returncode != 0:
        raise RuntimeError(f"合成失败:\n{r.stderr[-1200:]}")

    lv = measure(raw)
    gain = db - lv.get("mean_volume", db)
    # 补增益：PCM 无损，不会二次损伤
    r2 = subprocess.run(
        [str(FFMPEG), "-y", "-hide_banner", "-loglevel", "error",
         "-i", str(raw), "-af", f"volume={gain:.2f}dB",
         "-ar", str(SR), "-ac", "2", "-c:a", "pcm_s16le", str(out)],
        capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if r2.returncode != 0:
        raise RuntimeError(f"补增益失败:\n{r2.stderr[-800:]}")
    raw.unlink(missing_ok=True)

    return {"type": type_key, "desc": spec["desc"], "path": str(out),
            "duration_s": round(duration, 3), "target_db": db,
            "gain_applied_db": round(gain, 2),
            "raw_mean_db": lv.get("mean_volume")}


def measure(path: Path) -> dict:
    """实际电平自检。合成出来的东西必须量一遍，不能"听着差不多"。"""
    r = subprocess.run([str(FFMPEG), "-hide_banner", "-i", str(path),
                        "-af", "volumedetect", "-f", "null", "-"],
                       capture_output=True, text=True, encoding="utf-8", errors="ignore")
    out: dict[str, float] = {}
    for line in r.stderr.splitlines():
        for key in ("mean_volume", "max_volume"):
            if key in line:
                try:
                    out[key] = float(line.split(key + ":")[1].split("dB")[0].strip())
                except (IndexError, ValueError):
                    pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", default="rain_night_alley")
    ap.add_argument("--duration", type=float, default=16.0)
    ap.add_argument("--out", default="_tmp/ambience.wav")
    ap.add_argument("--target-db", type=float, default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for k, v in TYPES.items():
            print(f"  {k:<18} {v['target_db']:>6.1f}dB  {v['desc']}")
        return

    if args.type not in TYPES:
        sys.exit(f"未知类型 {args.type}，可选：{', '.join(TYPES)}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    info = build(args.type, args.duration, out, args.target_db)
    lv = measure(out)
    info["measured"] = lv
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
