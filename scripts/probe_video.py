# -*- coding: utf-8 -*-
"""
探测百炼账号是否开通视频生成（图生视频 / 文生视频）。

策略：发一个**参数必然不合法**的请求（input 为空）。
  - 模型不存在 / 未开通 -> 报 Model not exist 类错误（不产生费用）
  - 模型存在           -> 报参数缺失类错误（同样不产生费用）
据错误码区分两者，避免为了探测真花一次视频钱。

用法： python scripts/probe_video.py
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "https://dashscope.aliyuncs.com"
VIDEO_PATH = "/api/v1/services/aigc/video-generation/video-synthesis"

CANDIDATES = [
    "wan2.7-i2v",
    "wan2.7-t2v",
    "wan2.6-i2v",
    "wan2.6-i2v-flash",
    "wan2.6-t2v",
    "wan2.5-i2v-preview",
    "wan2.2-i2v-flash",
    "wan2.2-i2v-plus",
    "wanx2.1-i2v-turbo",
    "wanx2.1-i2v-plus",
    "wanx2.1-t2v-turbo",
    "wanx2.1-kf2v-plus",
    "kling-v2",
    "kling-v1-6",
    "MiniMax-Hailuo-02",
    "MiniMax-Hailuo-2.3",
    "doubao-seedance-1-0-pro",
    "seedance-1-0-pro",
    "veo3",
    "veo3-fast",
]


def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    p = ROOT / ".env"
    if not p.exists():
        return env
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    return env


ENV = load_env()
KEY = ENV.get("PROVIDER_API_KEY", "")


def probe(model: str) -> tuple[int | None, str, str]:
    """返回 (http_code, error_code, message)。"""
    payload = {
        "model": model,
        "input": {},  # 故意为空 -> 必定参数校验失败
        "parameters": {},
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        BASE + VIDEO_PATH,
        data=body,
        headers={
            "Authorization": f"Bearer {KEY}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            txt = r.read().decode("utf-8", "ignore")
            return r.status, "OK", txt[:200]
    except urllib.error.HTTPError as e:
        txt = e.read().decode("utf-8", "ignore")
        try:
            j = json.loads(txt)
            return e.code, str(j.get("code", "")), str(j.get("message", ""))[:200]
        except Exception:  # noqa: BLE001
            return e.code, "", txt[:200]
    except Exception as e:  # noqa: BLE001
        return None, type(e).__name__, str(e)[:200]


def main() -> None:
    if not KEY:
        raise SystemExit("缺少 PROVIDER_API_KEY")
    print(f"{'model':<26} {'http':<6} {'code':<24} message")
    print("-" * 110)
    ok: list[str] = []
    for m in CANDIDATES:
        code, err, msg = probe(m)
        # 判定规则（实测校准过）：
        #   200                    -> 异步队列已受理，说明模型存在
        #   InvalidParameter 且 message 含 "Model not exist" -> 不存在
        #   InvalidParameter 其他   -> 模型存在，只是参数不对
        if code == 200:
            exists, flag = True, "  <= 存在（已受理，任务会因参数为空而失败，不计费）"
        elif "Model not exist" in msg or "Model not exist" in err:
            exists, flag = False, "  <= 不存在"
        elif err in ("InvalidParameter", "MissingParameter", "DataInspection"):
            exists, flag = True, "  <= 存在"
        else:
            exists, flag = False, ""
        if exists:
            ok.append(m)
        print(f"{m:<26} {str(code):<6} {err[:24]:<24} {msg[:52]}{flag}")
    print()
    print("存在的视频模型：", ok or "无")


if __name__ == "__main__":
    main()
