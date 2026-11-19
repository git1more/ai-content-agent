# -*- coding: utf-8 -*-
"""
Qwen Image (DashScope 百炼) 最小调用客户端 —— 适配层的第一个实现。

已验证事实（2026-11-17，本账号实测）：
  - 地域：仅北京区（dashscope.aliyuncs.com）可用，新加坡区 401
  - 模型：qwen-image-3.0 / qwen-image-3.0-pro / qwen-image-2.x / wan2.7-image / z-image-turbo
  - **无独立 -edit 端点**，编辑 = 同一 ID + 传入参考图
  - 端点：POST /api/v1/services/aigc/multimodal-generation/generation（同步）

用法：
  python scripts/qwen_image_client.py --selftest
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"

BASE_URL = "https://dashscope.aliyuncs.com"
GEN_PATH = "/api/v1/services/aigc/multimodal-generation/generation"

# 本账号可用的图像模型（实测 /models 返回）
AVAILABLE_MODELS = [
    "qwen-image-3.0",
    "qwen-image-3.0-pro",
    "qwen-image-2.1-pro",
    "qwen-image-2.0",
    "qwen-image-2.0-pro",
    "wan2.7-image",
    "wan2.7-image-pro",
    "z-image-turbo",
]


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


ENV = load_env()
API_KEY = os.environ.get("PROVIDER_API_KEY") or ENV.get("PROVIDER_API_KEY", "")


def require_key() -> str:
    """惰性检查，不在 import 时 raise。

    模块级 raise 会让所有 import 它的**纯本地**脚本一起崩——`manju_lint --selftest`、
    `doc_check`、`ecom_set --dry-run` 都不发请求，但都会因为缺 key 直接退出。
    从仓库 clone 下来的第一件事通常是跑自检，不该卡在这里。
    """
    if not API_KEY:
        raise SystemExit("缺少 PROVIDER_API_KEY（复制 .env.example 为 .env 后填写）")
    return API_KEY


# --------------------------------------------------------------------------- #
# 图像编码
# --------------------------------------------------------------------------- #
def encode_image(path: str | Path, max_side: int = 1024, quality: int = 92) -> str:
    """把本地图片转成 data URL。

    超过 max_side 会等比缩小（控制请求体积，也避免模型被大图分辨率带偏）。
    PNG 保留 PNG（无损，避免 JPEG 振铃污染小字边缘），其余转 JPEG。
    """
    from PIL import Image

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    img = Image.open(p)
    img = img.convert("RGB") if img.mode not in ("RGB",) else img
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)

    buf = io.BytesIO()
    fmt = "PNG" if p.suffix.lower() == ".png" else "JPEG"
    if fmt == "PNG":
        img.save(buf, format="PNG", optimize=True)
        mime = "image/png"
    else:
        img.save(buf, format="JPEG", quality=quality)
        mime = "image/jpeg"
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:{mime};base64,{b64}"


# --------------------------------------------------------------------------- #
# 调用
# --------------------------------------------------------------------------- #
@dataclass
class GenResult:
    images: list[str] = field(default_factory=list)          # 本地落盘路径
    raw: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0


def _post(payload: dict, timeout: int = 180, retries: int = 3) -> dict:
    url = BASE_URL + GEN_PATH
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last_err: Exception | None = None

    for attempt in range(retries):
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {require_key()}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "ignore")
            last_err = RuntimeError(f"HTTP {e.code}: {detail[:600]}")
            # 429 / 5xx 退避重试；4xx 其他错误直接抛出（参数或权限问题，重试无意义）
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2 ** attempt * 2)
                continue
            raise last_err from e
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt * 2)
                continue
            raise
    raise last_err or RuntimeError("unknown error")


def generate(
    prompt: str,
    ref_images: list[str | Path] | None = None,
    model: str = "qwen-image-3.0",
    size: str = "1024*1024",
    n: int = 1,
    out_dir: str | Path | None = None,
    negative_prompt: str = "",
    prompt_extend: bool = False,
    watermark: bool = False,
    seed: int | None = None,
    tag: str = "out",
) -> GenResult:
    """生成或编辑一张图。

    ref_images 为空 => 文生图；非空（1-3 张）=> 编辑/参考图驱动。
    prompt_extend 默认关闭：否则平台会自动改写你的提示词，破坏可复现性。
    """
    content: list[dict[str, str]] = []
    for p in (ref_images or []):
        content.append({"image": encode_image(p)})
    content.append({"text": prompt})

    parameters: dict[str, Any] = {
        "size": size,
        "n": n,
        "prompt_extend": prompt_extend,
        "watermark": watermark,
    }
    if negative_prompt:
        parameters["negative_prompt"] = negative_prompt
    if seed is not None:
        parameters["seed"] = seed

    payload = {
        "model": model,
        "input": {"messages": [{"role": "user", "content": content}]},
        "parameters": parameters,
    }

    t0 = time.time()
    resp = _post(payload)
    elapsed = time.time() - t0

    if "output" not in resp:
        raise RuntimeError(f"响应异常: {json.dumps(resp, ensure_ascii=False)[:800]}")

    urls: list[str] = []
    for choice in resp["output"].get("choices", []):
        for item in choice.get("message", {}).get("content", []):
            if isinstance(item, dict) and item.get("image"):
                urls.append(item["image"])

    if not urls:
        raise RuntimeError(f"未返回图片: {json.dumps(resp, ensure_ascii=False)[:800]}")

    # 结果 URL 24h 过期，立刻下载落盘
    local_paths: list[str] = []
    if out_dir:
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        for i, u in enumerate(urls):
            suffix = ".png"
            guess = mimetypes.guess_extension(
                urllib.request.urlopen(u, timeout=60).headers.get_content_type() or ""
            )
            if guess in (".png", ".jpg", ".jpeg", ".webp"):
                suffix = guess
            name = f"{tag}_{i}{suffix}"
            dest = d / name
            urllib.request.urlretrieve(u, dest)
            local_paths.append(str(dest))

    return GenResult(
        images=local_paths,
        raw=resp,
        usage=resp.get("usage", {}),
        elapsed_s=round(elapsed, 2),
    )


# --------------------------------------------------------------------------- #
# 自检
# --------------------------------------------------------------------------- #
def selftest():
    sku = "TS-WHITE-01"
    ref = ROOT / "assets" / "p0_testset" / sku / "ref_front.png"
    out_dir = ROOT / "assets" / "p0_testset" / sku

    prompt = (
        "把图中的T恤制作成电商白底主图：把浅灰背景换成纯白色 #FFFFFF 无缝背景；"
        "T恤本体的颜色、面料纹理、罗纹领口、胸前黑色几何印花（形状与位置不得变）、"
        "袖口与下摆缝线全部保持完全一致；添加自然、轻微、方向一致的接触阴影。"
    )

    print(f"模型: qwen-image-3.0   参考图: {ref.name}")
    r = generate(
        prompt,
        ref_images=[ref],
        size="1024*1024",
        out_dir=out_dir,
        tag="qwen_t1",
        prompt_extend=False,
    )
    print(f"耗时 {r.elapsed_s}s")
    print(f"usage: {json.dumps(r.usage, ensure_ascii=False)}")
    for p in r.images:
        print(f"落盘: {p}  ({Path(p).stat().st_size / 1024:.0f} KB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="跑一次 T1 白底自检")
    ap.add_argument("--prompt", help="直接指定 prompt")
    ap.add_argument("--ref", action="append", default=[], help="参考图路径，可多次")
    ap.add_argument("--model", default="qwen-image-3.0")
    ap.add_argument("--size", default="1024*1024")
    ap.add_argument("--out", help="输出目录")
    ap.add_argument("--list-models", action="store_true")
    args = ap.parse_args()

    if args.list_models:
        for m in AVAILABLE_MODELS:
            print(m)
        return
    if args.selftest:
        selftest()
        return
    if not args.prompt:
        ap.print_help()
        return

    r = generate(args.prompt, ref_images=args.ref, model=args.model,
                 size=args.size, out_dir=args.out or (ROOT / "runs"))
    print(json.dumps({"elapsed_s": r.elapsed_s, "usage": r.usage, "images": r.images},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
