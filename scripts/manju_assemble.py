# -*- coding: utf-8 -*-
"""
漫剧成片：字幕叠加 → 拼接 → 音轨混合 → 封装。

为什么字幕不用模型生成：
  图像模型渲染中文的失败率极高，而电商图/漫剧里"文字错了"是唯一一眼废的硬伤。
  所以字幕一律用 PIL 确定性渲染，模型只负责画面。

音轨架构（2026-11-18 重构：从"每镜混音"改为"整条独立音轨"）

    voice/S1_fit.wav ─┐
    voice/S3_fit.wav ─┼→ voice_full.wav ─┐
    voice/S4_fit.wav ─┘                   ├→ mix.wav ─┐
    ambience.wav ──ducking──→ amb_ducked.wav┘         ├→ 成片
    画面（逐镜字幕 + 拼接）───────────────────────────┘

  好处是每一环都能单独试听、单独验证、单独替换：
    - 人声轨错了，不用重跑视频
    - 环境音太吵，只换 ambience.wav
    - 以后接 BGM，只是往 mix 里多一路，画面不动

为什么环境音要 ducking：
  人声出来时环境音自动压低 6–8dB，人声走后慢慢回来。这是影视/广播的标准做法，
  否则 -26dB 的厚雨幕会持续盖住台词。用 ffmpeg 的 sidechaincompress 以人声为侧链，
  不需要手工画音量包络（包络要跟着每镜台词改，改一次错一次）。

字幕与配音同步：
  字幕淡入起点不再是固定 0.25s，而是 timing.json 里的 head_pad_s ——
  也就是"人物开口的那一帧"。原来是画面先演 0.25s、字再出、声再晚 0.35s，三层错位。

用法：
  python scripts/manju_assemble.py --config config/manju_ch01.json --ver v5
  python scripts/manju_assemble.py --config config/manju_ch01.json --no-ambience
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

FFMPEG = ROOT / "tools" / "ffmpeg" / "ffmpeg.exe"
CFG = ROOT / "config" / "manju_ch01.json"
OUT = ROOT / "assets" / "manju" / "ch01"
TMP = OUT / "_sub"
VOICE = OUT / "voice"
FONT = Path("C:/Windows/Fonts/msyh.ttc")

AUDIO_SR = 48000
AUDIO_BITRATE = "192k"


def set_episode(cfg_path: str | Path) -> None:
    global CFG, OUT, TMP, VOICE
    CFG = Path(cfg_path)
    stem = CFG.stem.replace("manju_", "")
    OUT = ROOT / "assets" / "manju" / stem
    TMP = OUT / "_sub"
    VOICE = OUT / "voice"


def episode_filename(cfg: dict, ver: str | None = None) -> str:
    """EP00 + series -> EP00_雨夜第七单.mp4；带版本 -> EP00_雨夜第七单_v4.mp4

    台账规则（config/ep_versions.json）：根目录不放无标识成片，
    正式版一律 versions/v<N>/ 下带版本后缀。
    """
    head = str(cfg.get("episode", "EP00"))
    for sep in ("·", "《"):
        head = head.split(sep)[0]
    head = head.strip()
    series = str(cfg.get("series") or "").strip()
    base = f"{head}_{series}" if series else head
    if ver:
        base += f"_{ver}"
    return f"{base}.mp4"


def load_cfg() -> dict:
    return json.loads(CFG.read_text(encoding="utf-8"))


def load_timing() -> dict:
    p = VOICE / "timing.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def probe_size(clip: Path) -> tuple[int, int]:
    r = subprocess.run([str(FFMPEG), "-hide_banner", "-i", str(clip)],
                       capture_output=True, text=True, encoding="utf-8", errors="ignore")
    for line in r.stderr.splitlines():
        if "Video:" in line and "x" in line:
            part = line.split("Video:")[1]
            for tok in part.replace(",", " ").split():
                if "x" in tok and tok.replace("x", "").isdigit():
                    w, h = tok.split("x")
                    return int(w), int(h)
    raise RuntimeError(f"无法解析 {clip} 分辨率")


def probe_duration(path: Path) -> float:
    r = subprocess.run([str(FFMPEG), "-hide_banner", "-i", str(path)],
                       capture_output=True, text=True, encoding="utf-8", errors="ignore")
    for line in r.stderr.splitlines():
        if "Duration:" in line:
            hh, mm, ss = line.split("Duration:")[1].split(",")[0].strip().split(":")
            return round(int(hh) * 3600 + int(mm) * 60 + float(ss), 3)
    return 0.0


def render_caption(text: str, sub: str | None, w: int, h: int, cfg: dict) -> Image.Image:
    st = cfg["caption_style"]
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    f_main = ImageFont.truetype(str(FONT), int(h * st["main_size_ratio"]))
    f_sub = ImageFont.truetype(str(FONT), int(h * st["sub_size_ratio"]))
    stroke = max(2, int(h * st["main_size_ratio"] * st["stroke_ratio"]))
    y = int(h * st["position_y_ratio"])

    def draw_center(txt: str, font: ImageFont.FreeTypeFont, yy: int, sw: int) -> int:
        """把一行字按"墨迹框"水平居中，返回墨迹高度。

        注意必须用 y - bb[1] 对齐墨迹上沿：bb[1] 是字形相对绘制原点的上沿偏移，
        直接用 yy 会把字画低，并让下一行叠上来（踩过这个坑）。
        """
        bb = d.textbbox((0, 0), txt, font=font, stroke_width=sw)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        d.text(((w - tw) / 2 - bb[0], yy - bb[1]), txt, font=font, fill=st["fill"],
               stroke_width=sw, stroke_fill=st["stroke"])
        return th

    hh = draw_center(text, f_main, y, stroke)
    if sub:
        gap = int(h * 0.026)
        draw_center(sub, f_sub, y + hh + gap, max(2, stroke - 1))
    return img


def make_gradient(w: int, h: int, top_ratio: float = 0.66, max_alpha: int = 235) -> Image.Image:
    """底部渐变压暗条。

    两个作用：
      1. 字幕在任何画面上都读得清（漫剧标准做法）
      2. **盖掉模型自作主张画出来的东西** —— S3 车尾被画上了带拉丁字符的车牌，
         而否定式提示词对这类模型无效（写"没有车牌"反而让车牌更清楚）。
         确定性后期比继续改提示词可靠，且零成本。
    """
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    y0 = int(h * top_ratio)
    for y in range(y0, h):
        t = (y - y0) / max(1, h - y0)
        a = int(max_alpha * (t ** 1.6))
        d.line([(0, y), (w, y)], fill=(0, 0, 0, a))
    return img


def build_shot(clip: Path, cfg_shot: dict, cfg: dict, dur: float, cap_start: float) -> Path:
    """逐镜叠字幕：底部渐变（必加）+ 字幕（有则加）。纯视频，不带音轨。"""
    sid = cfg_shot["id"]
    w, h = probe_size(clip)
    dst = TMP / f"{sid}_sub.mp4"

    grad = TMP / f"gradient_{w}x{h}.png"
    if not grad.exists():
        make_gradient(w, h).save(grad)

    has_cap = bool(cfg_shot.get("subtitle"))
    cap = TMP / f"{sid}_cap.png"
    if has_cap:
        render_caption(cfg_shot["subtitle"], cfg_shot.get("subtitle_sub"), w, h, cfg).save(cap)

    st = cfg["caption_style"]
    fade_out = st["fade_out_s"]

    inputs = ["-i", str(clip), "-loop", "1", "-t", f"{dur:.2f}", "-i", str(grad)]
    if has_cap:
        inputs += ["-loop", "1", "-t", f"{dur:.2f}", "-i", str(cap)]

    parts = ["[0:v][1:v]overlay=0:0:format=auto[g]"]
    last = "g"
    if has_cap:
        start = max(0.0, cap_start)
        t_out = max(start + 0.1, dur - fade_out - 0.2)
        parts.append(
            f"[2:v]format=rgba,"
            f"fade=t=in:st={start:.2f}:d=0.25:alpha=1,"
            f"fade=t=out:st={t_out:.2f}:d={fade_out:.2f}:alpha=1[cap]"
        )
        parts.append("[g][cap]overlay=0:0:format=auto[c]")
        last = "c"
    parts.append(f"[{last}]format=yuv420p[v]")

    cmd = [
        str(FFMPEG), "-y", "-hide_banner", "-loglevel", "error",
        *inputs, "-filter_complex", ";".join(parts), "-map", "[v]",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-r", "30", "-pix_fmt", "yuv420p", str(dst),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if r.returncode != 0:
        raise RuntimeError(f"{sid} 叠加失败:\n{r.stderr[-1500:]}")
    return dst


def concat(parts: list[Path], out: Path, total_s: float) -> None:
    lst = TMP / "concat.txt"
    lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
    cmd = [
        str(FFMPEG), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(lst),
        "-vf", f"fade=t=in:st=0:d=0.25,fade=t=out:st={total_s - 0.45:.2f}:d=0.45",
        "-t", f"{total_s:.2f}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-an", str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if r.returncode != 0:
        raise RuntimeError(f"拼接失败:\n{r.stderr[-1500:]}")


# --------------------------------------------------------------------------- #
# 音轨：人声 → 环境音 → ducking → 混音
# --------------------------------------------------------------------------- #
def build_voice_track(cfg: dict, timing: dict) -> Path | None:
    """把各镜 _fit.wav 顺序拼成整条人声轨。无台词镜补等长静音，保证时间轴不漂。"""
    voiced = timing.get("shots") or {}
    if not any(s["id"] in voiced and (VOICE / f"{s['id']}_fit.wav").exists()
               for s in cfg["shots"]):
        return None

    inputs: list[str] = []
    labels: list[str] = []
    for i, shot in enumerate(cfg["shots"]):
        fit = VOICE / f"{shot['id']}_fit.wav"
        dur = float(shot["duration"])
        if shot["id"] in voiced and fit.exists():
            inputs += ["-i", str(fit)]
        else:
            inputs += ["-f", "lavfi", "-t", f"{dur:.3f}", "-i",
                       f"anullsrc=r={AUDIO_SR}:cl=stereo"]
        # 统一格式，否则 concat filter 会因采样格式不同而拒绝
        labels.append(f"[{i}:a]aformat=sample_fmts=fltp:sample_rates={AUDIO_SR}"
                      f":channel_layouts=stereo[a{i}]")

    dst = VOICE / "voice_full.wav"
    cmd = [str(FFMPEG), "-y", "-hide_banner", "-loglevel", "error", *inputs,
           "-filter_complex",
           ";".join(labels) + ";" + "".join(f"[a{i}]" for i in range(len(labels)))
           + f"concat=n={len(labels)}:v=0:a=1[aout]",
           "-map", "[aout]", "-ar", str(AUDIO_SR), "-ac", "2",
           "-c:a", "pcm_s16le", str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if r.returncode != 0:
        raise RuntimeError(f"人声轨拼接失败:\n{r.stderr[-1200:]}")
    return dst


def build_ambience(cfg: dict, total_s: float, force: bool = False) -> Path | None:
    """生成环境音（未生成或时长不符时重建）。配置见 cfg['ambience']。"""
    amb = cfg.get("ambience") or {}
    kind = amb.get("type")
    if not kind:
        return None

    from make_ambience import build as build_amb
    dst = VOICE / "ambience.wav"
    if dst.exists() and not force and abs(probe_duration(dst) - total_s) < 0.05:
        return dst
    VOICE.mkdir(parents=True, exist_ok=True)
    build_amb(kind, total_s, dst, amb.get("target_db"))
    return dst


def duck(amb: Path, voc: Path) -> Path:
    """以人声为侧链压缩环境音：人声一来，雨声自动让位。

    threshold 0.03 ≈ -30dB；人声平均在 -20dB 附近，超出 10dB，
    ratio=6 把它压掉约 8dB。attack 25ms 够快（不让人声开头被盖），
    release 420ms 让雨声慢慢回来 —— 少了这个参数，雨声会"啪"地弹回。
    """
    dst = VOICE / "ambience_ducked.wav"
    fc = ("[0:a][1:a]sidechaincompress="
          "threshold=0.022:ratio=9:attack=25:release=420:makeup=1[aout]")
    cmd = [str(FFMPEG), "-y", "-hide_banner", "-loglevel", "error",
           "-i", str(amb), "-i", str(voc), "-filter_complex", fc,
           "-map", "[aout]", "-ar", str(AUDIO_SR), "-ac", "2",
           "-c:a", "pcm_s16le", str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if r.returncode != 0:
        raise RuntimeError(f"环境音侧链压缩失败:\n{r.stderr[-1000:]}")
    return dst


def mix(voice: Path, amb: Path, total_s: float) -> Path:
    """人声 + 已 ducking 的环境音。

    normalize=0 很关键：amix 默认会把每路除以输入数，人声会被莫名压低 6dB，
    表现为"加了环境音后人声变小了"。淡入淡出必须写进 filter_complex，
    不能再用 -af（两者共存会被 ffmpeg 拒绝）。
    """
    dst = VOICE / "mix.wav"
    fc = ("[0:a][1:a]amix=inputs=2:duration=first:normalize=0[m];"
          f"[m]afade=t=in:st=0:d=0.25,afade=t=out:st={total_s - 0.55:.2f}:d=0.55[aout]")
    cmd = [str(FFMPEG), "-y", "-hide_banner", "-loglevel", "error",
           "-i", str(voice), "-i", str(amb), "-filter_complex", fc,
           "-map", "[aout]", "-t", f"{total_s:.2f}",
           "-ar", str(AUDIO_SR), "-ac", "2", "-c:a", "pcm_s16le", str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if r.returncode != 0:
        raise RuntimeError(f"混音失败:\n{r.stderr[-1000:]}")
    return dst


def mux(video: Path, audio: Path, out: Path) -> None:
    """画面直接 copy，避免二次编码；音轨转 AAC。"""
    cmd = [str(FFMPEG), "-y", "-hide_banner", "-loglevel", "error",
           "-i", str(video), "-i", str(audio),
           "-map", "0:v:0", "-map", "1:a:0",
           "-c:v", "copy", "-c:a", "aac", "-b:a", AUDIO_BITRATE,
           "-ar", str(AUDIO_SR), "-ac", "2",
           "-movflags", "+faststart", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if r.returncode != 0:
        raise RuntimeError(f"封装失败:\n{r.stderr[-1200:]}")


def normalize_loudness(src: Path, dst: Path,
                       target_lufs: float = -16.0, tp: float = -1.5,
                       lra: float = 11.0) -> dict:
    """响度标准化到短视频平台可用的电平。两遍法。

    为什么必须做：
      首版成片实测 -26.0 LUFS、真峰值 -7.9 dBTP。手机外放会明显偏轻，
      发给别人看的第一反应是"声音怎么这么小"。平台标准约 -14 ~ -16 LUFS。

    为什么不能简单加增益：
      -26 → -16 要 +10dB，峰值 -7.9 会顶到 +2.1，直接削波。
      必须用 loudnorm 的 true-peak limiting（TP=-1.5）兜住。

    为什么用两遍 + linear：
      单遍 loudnorm 是动态模式，会把雨声的起伏压平（实测 LRA 10.4 → 7.5），
      环境音的"呼吸感"就没了。两遍法先测量再用 measured_* 线性应用，
      只做整体增益，动态保留。
    """
    import re
    r = subprocess.run(
        [str(FFMPEG), "-hide_banner", "-i", str(src),
         "-af", f"loudnorm=I={target_lufs}:TP={tp}:LRA={lra}:print_format=json",
         "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="ignore")
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", r.stderr, re.S)
    if not m:
        raise RuntimeError(f"loudnorm 测量失败:\n{r.stderr[-800:]}")
    meas = json.loads(m.group(0))

    af = (f"loudnorm=I={target_lufs}:TP={tp}:LRA={lra}"
          f":measured_I={meas['input_i']}:measured_TP={meas['input_tp']}"
          f":measured_LRA={meas['input_lra']}:measured_thresh={meas['input_thresh']}"
          f":offset={meas['target_offset']}:linear=true")
    r2 = subprocess.run(
        [str(FFMPEG), "-y", "-hide_banner", "-loglevel", "error",
         "-i", str(src), "-af", af, "-ar", str(AUDIO_SR), "-ac", "2",
         "-c:a", "pcm_s16le", str(dst)],
        capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if r2.returncode != 0:
        raise RuntimeError(f"响度标准化失败:\n{r2.stderr[-800:]}")

    return {"in_lufs": float(meas["input_i"]), "in_tp": float(meas["input_tp"]),
            "gain_db": round(target_lufs - float(meas["input_i"]), 1)}


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="剧本配置路径")
    ap.add_argument("--ver", default=None, help="版本号，如 v5 → 输出到 versions/v5/")
    ap.add_argument("--no-ambience", action="store_true", help="不铺环境音")
    ap.add_argument("--force-ambience", action="store_true", help="强制重新合成环境音")
    args = ap.parse_args()
    if args.config:
        set_episode(args.config)

    if not FFMPEG.exists():
        sys.exit("缺 ffmpeg，先跑 python scripts/get_ffmpeg.py")
    TMP.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg()
    timing = load_timing()
    head_pad = float(timing.get("head_pad_s", cfg["caption_style"]["fade_in_s"]))
    voiced = timing.get("shots") or {}
    total_s = sum(float(s["duration"]) for s in cfg["shots"])

    # 1) 画面
    parts: list[Path] = []
    for shot in cfg["shots"]:
        clip = OUT / "clips" / f"{shot['id']}.mp4"
        if not clip.exists():
            sys.exit(f"缺片段 {clip}")
        has_voice = shot["id"] in voiced and (VOICE / f"{shot['id']}_fit.wav").exists()
        p = build_shot(clip, shot, cfg, float(shot["duration"]),
                       head_pad if has_voice else cfg["caption_style"]["fade_in_s"])
        parts.append(p)
        print(f"  {shot['id']} 字幕合成 -> {p.name}")

    video = TMP / "_video.mp4"
    concat(parts, video, total_s)
    print(f"  画面拼接 -> {video.name}  ({total_s:.1f}s)")

    # 2) 人声
    voice = build_voice_track(cfg, timing)
    if voice:
        print(f"  人声轨 -> {voice.name}  ({probe_duration(voice):.2f}s)")

    # 3) 环境音 → ducking → 混音
    audio: Path | None
    amb = None if args.no_ambience else build_ambience(cfg, total_s, args.force_ambience)
    if amb:
        duck_db = float((cfg.get("ambience") or {}).get("duck_db", 7.0))
        if voice:
            amb_d = duck(amb, voice)
            print(f"  环境音 -> {amb.name} → 侧链压缩（目标 -{duck_db:.0f}dB）→ {amb_d.name}")
            audio = mix(voice, amb_d, total_s)
        else:
            audio = amb
        print(f"  音轨 -> {audio.name}  ({probe_duration(audio):.2f}s)")
    else:
        audio = voice

    # 4) 封装
    if args.ver:
        out_dir = OUT / "versions" / args.ver
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / episode_filename(cfg, args.ver)
    else:
        out = OUT / episode_filename(cfg)

    if audio:
        # 响度标准化：只对"有人声"的混音做。纯环境音不需要提到 -16，
        # 那会把本来就是底噪的东西抬成主角。
        if voice:
            master = cfg.get("audio_master") or {}
            norm = normalize_loudness(
                audio, VOICE / "mix_norm.wav",
                target_lufs=float(master.get("lufs", -16.0)),
                tp=float(master.get("true_peak_db", -1.5)),
            )
            print(f"  响度 {norm['in_lufs']:.1f} → "
                  f"{master.get('lufs', -16.0)} LUFS（+{norm['gain_db']}dB，峰值限 -1.5dBTP）")
            audio = VOICE / "mix_norm.wav"
        mux(video, audio, out)
    else:
        subprocess.run([str(FFMPEG), "-y", "-hide_banner", "-loglevel", "error",
                        "-i", str(video), "-c", "copy", str(out)], check=True)

    print(f"\n成片 -> {out.relative_to(ROOT)}")
    r = subprocess.run([str(FFMPEG), "-hide_banner", "-i", str(out)],
                       capture_output=True, text=True, encoding="utf-8", errors="ignore")
    for line in r.stderr.splitlines():
        if "Duration" in line or "Stream #0:" in line:
            print("  " + line.strip())


if __name__ == "__main__":
    main()
