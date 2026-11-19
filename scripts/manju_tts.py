# -*- coding: utf-8 -*-
"""
漫剧配音：剧本台词 → 逐镜 TTS → 对齐到镜头时长 → 供成片混音。

核心难点不是"调 TTS"，是**时长对齐**。
  视频模型按整秒出片（4s / 5s），TTS 按字出音（"第 7 单，收件人：林夏" = 2.40s）。
  两者对不上是常态，不是异常。对齐策略分三层，从便宜到贵：

    1. 生成前：lint 按 4.0 字/秒算预算，超了就报 ERROR（零成本，最优）
    2. 生成后短了：adelay 补头 + apad 补齐到镜头时长（零成本，无损）
    3. 生成后长了：atempo 不变调加速，上限 1.2 倍；仍超长则硬截断并标 overrun
       —— 超过 1.2 倍人耳已能听出"赶"，宁可截断告知，也不静默变速毁掉语气

  为什么不用"让 TTS 时长决定镜头时长"：
  视频已经生成了，改镜头时长要重跑图生视频（¥0.6-0.75/镜），成本是 TTS 的 100 倍。
  顺序反了：**先定镜头时长，再让台词适配它**。

台词来源优先级（避免字幕和配音各写一份、后面必然不同步）：
  shot.voice_line.text  >  shot.subtitle + shot.subtitle_sub
  回退时会打印提示 —— 字幕卡（"第 7 单"）读出来是旁白味，拟声词/时间戳不该念，
  所以正式剧本建议显式写 voice_line。

输出：
  assets/manju/<ep>/voice/<SID>.wav        原始 TTS（24k 单声道）
  assets/manju/<ep>/voice/<SID>_fit.wav    对齐后（48k 立体声，长度严格等于镜头时长）
  assets/manju/<ep>/voice/timing.json      每镜文本/音色/字数/原始时长/变速比/标记
  assets/manju/<ep>/cost.csv               成本台账（与 manju_run.py 同表）

用法：
  python scripts/manju_tts.py --config config/manju_ch01.json
  python scripts/manju_tts.py --config config/manju_ch01.json --shots S4 --force
  python scripts/manju_tts.py --config config/manju_ch01.json --dry-run   # 只算预算不调用
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from qwen_tts_client import synthesize, probe_duration, PRICE_CNY_PER_CHAR  # noqa: E402

FFMPEG = ROOT / "tools" / "ffmpeg" / "ffmpeg.exe"

# 默认对齐参数。head_pad 给画面 0.35s 的"入戏"时间，tail_pad 留 0.45s 收尾，
# 避免台词顶着镜头切换开口/收口（听感上"被切掉"）。
DEFAULT_HEAD_PAD = 0.35
DEFAULT_TAIL_PAD = 0.45
DEFAULT_MAX_TEMPO = 1.2
OUT_SR = 48000


def set_episode(cfg_path: str | Path) -> None:
    global CFG_PATH, OUT, VOICE, TIMING, COST
    CFG_PATH = Path(cfg_path)
    stem = CFG_PATH.stem.replace("manju_", "")
    OUT = ROOT / "assets" / "manju" / stem
    VOICE = OUT / "voice"
    TIMING = VOICE / "timing.json"
    COST = OUT / "cost.csv"


CFG_PATH = ROOT / "config" / "manju_ch01.json"
OUT = ROOT / "assets" / "manju" / "ch01"
VOICE = OUT / "voice"
TIMING = VOICE / "timing.json"
COST = OUT / "cost.csv"


def load_cfg() -> dict[str, Any]:
    return json.loads(CFG_PATH.read_text(encoding="utf-8"))


def log_cost(stage: str, item: str, model: str, unit: str, qty: float, cny: float) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    new = not COST.exists()
    with COST.open("a", encoding="utf-8") as f:
        if new:
            f.write("ts,stage,item,model,unit,qty,cost_cny\n")
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')},{stage},{item},{model},{unit},{qty},{cny:.6f}\n")


# --------------------------------------------------------------------------- #
# 台词解析
# --------------------------------------------------------------------------- #
def voice_of(cfg: dict, vl: dict) -> str:
    """音色解析：本镜显式 > 角色表 > 旁白 > Cherry。"""
    if vl.get("voice"):
        return str(vl["voice"])
    cast = cfg.get("voice_cast") or {}
    entry = cast.get(vl.get("as") or "narrator") or cast.get("narrator") or "Cherry"
    if isinstance(entry, dict):
        return str(entry.get("voice", "Cherry"))
    return str(entry)


def resolve_line(shot: dict) -> tuple[str | None, bool]:
    """返回（要念的文本, 是否来自字幕回退）。"""
    vl = shot.get("voice_line") or {}
    text = vl.get("text")
    if text is not None:
        return (str(text).strip() or None), False

    main = (shot.get("subtitle") or "").strip()
    sub = (shot.get("subtitle_sub") or "").strip()
    if not main and not sub:
        return None, False
    # 主副标题用逗号连读："第 7 单，收件人：林夏"
    return "，".join(x for x in (main, sub) if x), True


def speech_budget(shot: dict, cfg: dict) -> tuple[float, float]:
    """本镜可用于念白的天数（秒），以及台词字数预算。给 lint 复用。"""
    fmt = cfg.get("format") or {}
    head = float(fmt.get("voice_head_pad_s", DEFAULT_HEAD_PAD))
    tail = float(fmt.get("voice_tail_pad_s", DEFAULT_TAIL_PAD))
    avail = max(0.3, float(shot.get("duration", 4)) - head - tail)
    return avail, avail * float(fmt.get("speech_rate_chars_per_s", 4.0))


# --------------------------------------------------------------------------- #
# 时长对齐
# --------------------------------------------------------------------------- #
def fit_to_shot(src: Path, dst: Path, target_s: float, head_pad: float,
                tail_pad: float, max_tempo: float) -> dict[str, Any]:
    """把 TTS 音频加工成"长度严格等于 target_s 的 48k 立体声轨"。

    短了：前置 head_pad 静音 + 尾部 apad 补齐。
    长了：atempo 不变调加速（上限 max_tempo），仍超长则被 -t 截断并标 overrun。
    """
    tts_s = probe_duration(src)
    avail = max(0.3, target_s - head_pad - tail_pad)

    tempo = 1.0
    if tts_s > avail:
        tempo = min(max_tempo, tts_s / avail)

    chain: list[str] = []
    if tempo > 1.001:
        # atempo 单次支持 0.5–2.0，本处上限 1.2，无需级联
        chain.append(f"atempo={tempo:.4f}")
    delay_ms = int(head_pad * 1000)
    chain.append(f"adelay={delay_ms}|{delay_ms}")
    chain.append("apad")

    cmd = [
        str(FFMPEG), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src), "-af", ",".join(chain),
        "-t", f"{target_s:.3f}", "-ar", str(OUT_SR), "-ac", "2",
        "-c:a", "pcm_s16le", str(dst),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if r.returncode != 0:
        raise RuntimeError(f"对齐失败 {src.name}:\n{r.stderr[-800:]}")

    fitted_speech = tts_s / tempo
    flags: list[str] = []
    if fitted_speech > avail + 0.02:
        flags.append(f"overrun(超 {fitted_speech - avail:.2f}s 被截断)")
    if tts_s < 0.5:
        flags.append("too_short(台词过短，听感像音效)")
    if tempo >= max_tempo - 0.001 and tts_s > avail:
        flags.append("tempo_capped(已到变速上限)")

    return {
        "src": str(src), "dst": str(dst),
        "tts_s": tts_s, "tempo": round(tempo, 4), "speech_s": round(fitted_speech, 3),
        "avail_s": round(avail, 3), "fit_s": target_s, "flags": flags,
    }


# --------------------------------------------------------------------------- #
def run(cfg: dict | None = None, only: list[str] | None = None,
        force: bool = False, dry_run: bool = False) -> dict[str, Any]:
    cfg = cfg or load_cfg()
    fmt = cfg.get("format") or {}
    head_pad = float(fmt.get("voice_head_pad_s", DEFAULT_HEAD_PAD))
    tail_pad = float(fmt.get("voice_tail_pad_s", DEFAULT_TAIL_PAD))
    max_tempo = float(fmt.get("voice_max_tempo", DEFAULT_MAX_TEMPO))

    VOICE.mkdir(parents=True, exist_ok=True)
    prev = json.loads(TIMING.read_text(encoding="utf-8")) if TIMING.exists() else {}
    shots_out: dict[str, Any] = dict(prev.get("shots") or {})

    rows: list[dict[str, Any]] = []
    total_chars = 0
    total_cost = 0.0

    for shot in cfg["shots"]:
        sid = shot["id"]
        if only and sid not in only:
            continue
        text, from_subtitle = resolve_line(shot)
        if not text:
            print(f"[voice] {sid} 无台词，跳过")
            continue

        voice = voice_of(cfg, shot.get("voice_line") or {})
        avail, budget = speech_budget(shot, cfg)
        dur = float(shot.get("duration", 4))
        est_s = len(text) / float(fmt.get("speech_rate_chars_per_s", 4.0))

        src = VOICE / f"{sid}.wav"
        fit = VOICE / f"{sid}_fit.wav"

        if dry_run:
            print(f"[dry] {sid} {len(text)}字 预估 {est_s:.2f}s / 可用 {avail:.2f}s "
                  f"{'✗ 超预算' if est_s > avail else '✓'}  「{text}」")
            continue

        if from_subtitle:
            print(f"  ⚠ {sid} 未写 voice_line，回退用字幕朗读：「{text}」")

        if src.exists() and not force:
            print(f"[skip] {sid} 配音已存在 {src.name}")
            tts_s = probe_duration(src)
            chars = len(text)
            cost = round(chars * PRICE_CNY_PER_CHAR, 6)
        else:
            print(f"[voice] {sid} {voice} 「{text}」")
            r = synthesize(text, src, voice=voice)
            tts_s, chars, cost = r.duration_s, r.chars, r.cost_cny
            log_cost("voice", sid, "qwen3-tts-flash", "char", chars, cost)
            print(f"  -> {src.name} {tts_s:.2f}s / {chars}字 ¥{cost:.4f}")

        info = fit_to_shot(src, fit, dur, head_pad, tail_pad, max_tempo)
        info.update({"text": text, "voice": voice, "chars": chars,
                     "from_subtitle": from_subtitle, "est_s": round(est_s, 2),
                     "cost_cny": cost, "wav": str(src.relative_to(ROOT)),
                     "fit": str(fit.relative_to(ROOT))})
        shots_out[sid] = info
        rows.append(info)
        total_chars += chars
        total_cost += cost

        mark = "  ⚠ " + "；".join(info["flags"]) if info["flags"] else ""
        print(f"    对齐 {info['tts_s']:.2f}s → {dur:.1f}s  变速 {info['tempo']:.2f}x{mark}")

    if dry_run:
        return {}

    timing = {
        "schema_version": "1.0",
        "episode": cfg.get("episode"),
        "model": "qwen3-tts-flash",
        "sample_rate": OUT_SR,
        "head_pad_s": head_pad,
        "tail_pad_s": tail_pad,
        "max_tempo": max_tempo,
        "speech_rate_chars_per_s": float(fmt.get("speech_rate_chars_per_s", 4.0)),
        "shots": shots_out,
        "total_chars": total_chars,
        "total_cost_cny": round(total_cost, 6),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    TIMING.write_text(json.dumps(timing, ensure_ascii=False, indent=2), encoding="utf-8")

    if rows:
        print(f"\n{len(rows)} 镜配音完成  合计 {total_chars} 字  ¥{total_cost:.4f}")
        bad = [r for r in rows if any("overrun" in f for f in r["flags"])]
        if bad:
            names = ", ".join(Path(r["dst"]).name.replace("_fit.wav", "") for r in bad)
            print(f"⚠ {len(bad)} 镜台词超长被截断：{names}")
            print("  改法（按成本从低到高）：删字 / 调高 voice_max_tempo / 加长镜头时长后重跑视频")
    print(f"时间轴 -> {TIMING.relative_to(ROOT)}")
    return timing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CFG_PATH))
    ap.add_argument("--shots", nargs="*", help="只配指定镜头，如 S1 S4")
    ap.add_argument("--force", action="store_true", help="覆盖已有配音")
    ap.add_argument("--dry-run", action="store_true", help="只算字数预算，不调用接口")
    args = ap.parse_args()

    set_episode(args.config)
    if not FFMPEG.exists():
        sys.exit("缺 ffmpeg，先跑 python scripts/get_ffmpeg.py")
    run(load_cfg(), only=args.shots, force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
