# -*- coding: utf-8 -*-
"""
Qwen3-TTS 客户端（DashScope 百炼，同步接口）。

已验证（2026-11-18，本账号实测）：
  - 端点：POST /api/v1/services/aigc/multimodal-generation/generation（同步，非异步任务）
  - 模型：qwen3-tts-flash
  - 入参：input.text + input.voice + input.language_type
  - 出参：output.audio.url（24h 过期，必须立刻落盘）+ usage.characters
  - 音频格式：wav / pcm_s16le / 24000 Hz / 单声道

实测语速（Cherry，中文，含标点）：
  13 字 -> 2.40s ；11 字 -> 2.32s ；5 字 -> 1.04s
  纯汉字约 4.6 字/秒。lint 的台词预算按 4.0 字/秒（留停顿余量）。

计费：qwen-tts 系列约 ¥0.8/万字符 = ¥0.00008/字符（按 usage.characters 计，
包含空格与标点）。非权威报价，仅用于成本台账。

为什么不用视频模型自带的 audio=True：
  wan2.6-i2v-flash 的 audio 是"音效/环境声"生成，不能念指定台词，且 720P 从
  ¥0.15/s 涨到 ¥0.30/s（翻倍）。配音必须是独立的 TTS：可改、可重配、可按角色分轨，
  成本还低两个数量级（一集台词 60 字 ≈ ¥0.005）。

用法：
  python scripts/qwen_tts_client.py --selftest
  python scripts/qwen_tts_client.py --text "你迟到了。" --voice Cherry --out _tmp/a.wav
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
FFMPEG = ROOT / "tools" / "ffmpeg" / "ffmpeg.exe"
ENV_PATH = ROOT / ".env"

BASE = "https://dashscope.aliyuncs.com"
TTS_PATH = "/api/v1/services/aigc/multimodal-generation/generation"
MODEL = "qwen3-tts-flash"

# 元/字符。qwen-tts 系列 ¥0.8/万字符，按 usage.characters 计（含标点）。
PRICE_CNY_PER_CHAR = 0.00008

# 本账号实测可用的音色（同一句话三个音色均返回 200）
VOICES = {
    "Cherry": "女声，中性偏明亮，语速中等 —— 通用旁白",
    "Serena": "女声，柔和偏慢（同句 2.96s vs Cherry 2.40s）—— 抒情/回忆",
    "Ethan": "男声，偏低沉，语速略快（同句 2.24s）—— 成年男性角色",
}


def load_env(path: Path = ENV_PATH) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    return env


def api_key() -> str:
    """延迟取 key：模块可被 import（lint / 编排器）而不因缺 key 直接崩。"""
    import os
    k = load_env().get("PROVIDER_API_KEY", "") or os.environ.get("PROVIDER_API_KEY", "")
    if not k:
        raise SystemExit("缺少 PROVIDER_API_KEY（见 .env）")
    return k


def probe_duration(path: str | Path) -> float:
    """用 ffmpeg 读时长。本机只有 ffmpeg.exe，没有 ffprobe。"""
    r = subprocess.run([str(FFMPEG), "-hide_banner", "-i", str(path)],
                       capture_output=True, text=True, encoding="utf-8", errors="ignore")
    for line in r.stderr.splitlines():
        if "Duration:" in line:
            hh, mm, ss = line.split("Duration:")[1].split(",")[0].strip().split(":")
            return round(int(hh) * 3600 + int(mm) * 60 + float(ss), 3)
    raise RuntimeError(f"无法解析音频时长: {path}")


@dataclass
class TtsResult:
    path: str
    text: str
    voice: str
    duration_s: float      # TTS 原始时长（未做任何对齐）
    chars: int
    cost_cny: float
    elapsed_s: float
    remote_url: str
    request_id: str


def synthesize(
    text: str,
    out_path: str | Path,
    voice: str = "Cherry",
    language_type: str = "Chinese",
    retries: int = 3,
    timeout: int = 90,
) -> TtsResult:
    """一段文字 → 一个 wav 文件（24kHz 单声道，原样落盘，不改采样率）。"""
    text = (text or "").strip()
    if not text:
        raise ValueError("text 为空")
    if voice not in VOICES:
        # 不硬拦：账号可能有更多音色，但提示一句，避免打错字静默换音色
        print(f"  ⚠ 未登记的音色 {voice}，按原样提交", file=sys.stderr)

    payload = {
        "model": MODEL,
        "input": {"text": text, "voice": voice, "language_type": language_type},
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"}

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    last: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(BASE + TTS_PATH, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                resp: dict[str, Any] = json.loads(r.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "ignore")
            last = RuntimeError(f"HTTP {e.code}: {detail[:500]}")
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2 ** attempt * 2)
                continue
            raise last from e
        except Exception as e:  # noqa: BLE001
            last = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt * 2)
                continue
            raise
    else:
        raise last or RuntimeError("TTS 未知失败")

    audio = (resp.get("output") or {}).get("audio") or {}
    url = audio.get("url")
    if not url:
        raise RuntimeError(f"TTS 无 audio.url: {json.dumps(resp, ensure_ascii=False)[:500]}")

    # 24h 过期，立刻落盘
    urllib.request.urlretrieve(url, out)
    chars = int((resp.get("usage") or {}).get("characters") or len(text))

    return TtsResult(
        path=str(out), text=text, voice=voice,
        duration_s=probe_duration(out), chars=chars,
        cost_cny=round(chars * PRICE_CNY_PER_CHAR, 6),
        elapsed_s=round(time.time() - t0, 1),
        remote_url=url, request_id=resp.get("request_id", ""),
    )


def selftest() -> None:
    r = synthesize("你迟到了。", ROOT / "_tmp" / "tts_selftest.wav", voice="Cherry")
    print(json.dumps({
        "path": r.path, "duration_s": r.duration_s, "chars": r.chars,
        "cost_cny": r.cost_cny, "elapsed_s": r.elapsed_s,
    }, ensure_ascii=False, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--text")
    ap.add_argument("--voice", default="Cherry")
    ap.add_argument("--out", default="_tmp/tts_out.wav")
    ap.add_argument("--list-voices", action="store_true")
    args = ap.parse_args()

    if args.list_voices:
        for v, desc in VOICES.items():
            print(f"  {v:<8} {desc}")
        return
    if args.selftest:
        selftest()
        return
    if not args.text:
        ap.print_help()
        return
    r = synthesize(args.text, ROOT / args.out, voice=args.voice)
    print(json.dumps({"path": r.path, "duration_s": r.duration_s, "chars": r.chars,
                      "cost_cny": r.cost_cny}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
