# -*- coding: utf-8 -*-
"""
探测 .env 里的 API Key 属于哪个平台、能调用哪些模型。

做法：对一批候选 OpenAI 兼容 base_url 发 GET /models（带 Bearer），
看谁返回 200。返回 200 的那个就是该 key 归属平台，顺便列出可用模型。

用法： python scripts/probe_provider.py
"""

import json
import re
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV = ROOT / ".env"

CANDIDATES = [
    ("阿里云百炼-北京", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    ("阿里云百炼-新加坡", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
    ("硅基流动-国内", "https://api.siliconflow.cn/v1"),
    ("硅基流动-国际", "https://api.siliconflow.com/v1"),
    ("302.AI", "https://api.302.ai/v1"),
    ("火山方舟", "https://ark.cn-beijing.volces.com/api/v3"),
    ("WaveSpeed", "https://api.wavespeed.ai/api/v1"),
    ("OpenRouter", "https://openrouter.ai/api/v1"),
    ("Together", "https://api.together.xyz/v1"),
    ("Fireworks", "https://api.fireworks.ai/inference/v1"),
    ("MiniMax", "https://api.minimax.chat/v1"),
    ("Moonshot", "https://api.moonshot.cn/v1"),
    ("DeepSeek", "https://api.deepseek.com/v1"),
    ("智谱 BigModel", "https://open.bigmodel.cn/api/paas/v4"),
    ("百度千帆", "https://qianfan.baidubce.com/v2"),
    ("阶跃 StepFun", "https://api.stepfun.com/v1"),
    ("OpenAI", "https://api.openai.com/v1"),
]

# 只关心图像相关模型
IMG_PAT = re.compile(
    r"(image|flux|seedream|seededit|qwen-image|wanx|wan-|kolors|dall|stable|sd3|midjourney|niji|imagen|banana|gpt-image|kling|seedance|veo|hunyuan-image|doubao-seedream)",
    re.I,
)


def load_env(path: Path) -> dict:
    env = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    return env


def mask(s: str) -> str:
    if len(s) <= 12:
        return "***"
    return f"{s[:7]}...{s[-4:]}"


def probe(base: str, key: str, timeout=12):
    url = base.rstrip("/") + "/models"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "p0-probe/1.0",
        },
    )
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
            body = r.read().decode("utf-8", "ignore")
            return r.status, body
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "ignore")[:300]
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def main():
    env = load_env(ENV)
    key = env.get("PROVIDER_API_KEY", "")
    if not key:
        sys.exit("未在 .env 找到 PROVIDER_API_KEY")

    print(f"Key: {mask(key)}  长度 {len(key)}\n")
    print(f"{'平台':<20} {'状态':<8} 说明")
    print("-" * 78)

    hits = []
    for name, base in CANDIDATES:
        status, body = probe(base, key)
        if status == 200:
            try:
                data = json.loads(body)
                items = data.get("data") or data.get("models") or []
                ids = [m.get("id") or m.get("name") or "" for m in items if isinstance(m, dict)]
            except Exception:  # noqa: BLE001
                ids = []
            print(f"{name:<20} 200      ✓ 可用，模型数 {len(ids)}")
            hits.append((name, base, ids))
        elif status is None:
            print(f"{name:<20} --       {body[:50]}")
        elif status in (401, 403):
            print(f"{name:<20} {status:<8} 认证不通过")
        else:
            print(f"{name:<20} {status:<8} {body[:60].replace(chr(10), ' ')}")

    if not hits:
        print("\n未找到可用平台。可能需要换 base_url 或该 key 非 OpenAI 兼容接口。")
        return

    for name, base, ids in hits:
        img = [i for i in ids if IMG_PAT.search(i)]
        print(f"\n=== {name}  ({base}) ===")
        print(f"图像类模型 {len(img)} 个：")
        for i in sorted(img)[:60]:
            print(f"  - {i}")
        if len(img) > 60:
            print(f"  ... 还有 {len(img) - 60} 个")
        if not img:
            print("  （无匹配，前 30 个模型如下）")
            for i in sorted(ids)[:30]:
                print(f"  - {i}")


if __name__ == "__main__":
    main()
