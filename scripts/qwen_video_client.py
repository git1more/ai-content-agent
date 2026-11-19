# -*- coding: utf-8 -*-
"""
Qwen / 万相 视频生成客户端（DashScope 异步任务）。

已验证（2026-11-17，本账号）：
  - 端点：POST /api/v1/services/aigc/video-generation/video-synthesis
  - 必须带 header X-DashScope-Async: enable
  - 提交返回 task_id，需轮询 GET /api/v1/tasks/{task_id}
  - 可用模型：wan2.6-i2v-flash / wan2.2-i2v-flash / wan2.7-i2v 等
  - 结果 video_url 24h 过期，必须立刻落盘

计费（华北2 北京，输出按秒）：
  wan2.6-i2v-flash 静音 720P ¥0.15/s，1080P ¥0.25/s；audio=true 720P ¥0.30/s
  wan2.2-i2v-flash 480P ¥0.10/s，720P ¥0.20/s（固定 5s）
  wan2.7-i2v 720P ¥0.60/s，1080P ¥1.00/s

用法：
  python scripts/qwen_video_client.py --selftest
"""

from __future__ import annotations

import argparse
import base64
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
BASE = "https://dashscope.aliyuncs.com"
SUBMIT_PATH = "/api/v1/services/aigc/video-generation/video-synthesis"
TASK_PATH = "/api/v1/tasks/{task_id}"

# 每（模型, 分辨率, 是否有声）的单价，元/秒。用于成本记账，非权威报价。
PRICE_CNY_PER_S: dict[tuple[str, str, bool], float] = {
    ("wan2.6-i2v-flash", "720P", False): 0.15,
    ("wan2.6-i2v-flash", "1080P", False): 0.25,
    ("wan2.6-i2v-flash", "720P", True): 0.30,
    ("wan2.6-i2v-flash", "1080P", True): 0.50,
    ("wan2.2-i2v-flash", "480P", False): 0.10,
    ("wan2.2-i2v-flash", "720P", False): 0.20,
    ("wan2.2-i2v-flash", "1080P", False): 0.48,
    ("wan2.7-i2v", "720P", True): 0.60,
    ("wan2.7-i2v", "1080P", True): 1.00,
}


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


API_KEY = load_env().get("PROVIDER_API_KEY", "") or __import__("os").environ.get("PROVIDER_API_KEY", "")
if not API_KEY:
    raise SystemExit("缺少 PROVIDER_API_KEY")


def _request(url: str, payload: dict | None = None, timeout: int = 90,
             extra_headers: dict | None = None, retries: int = 3) -> dict:
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    headers.update(extra_headers or {})
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None

    last: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=body, headers=headers,
                                     method="POST" if body else "GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "ignore")
            last = RuntimeError(f"HTTP {e.code}: {detail[:600]}")
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2 ** attempt * 3)
                continue
            raise last from e
        except Exception as e:  # noqa: BLE001
            last = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt * 3)
                continue
            raise
    raise last or RuntimeError("unknown")


def image_to_data_url(path: str | Path, max_side: int = 1280) -> str:
    """把本地关键帧编码成 data URL。视频接口接受 base64 以支持本地文件。"""
    from PIL import Image
    import io

    img = Image.open(path).convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


@dataclass
class VideoResult:
    task_id: str
    path: str | None
    duration_s: float
    elapsed_s: float
    cost_cny: float
    raw: dict[str, Any]


def generate_video(
    image: str | Path,
    prompt: str,
    out_path: str | Path,
    model: str = "wan2.6-i2v-flash",
    resolution: str = "720P",
    duration: int = 4,
    audio: bool = False,
    negative_prompt: str = "",
    seed: int | None = None,
    prompt_extend: bool = False,
    poll_interval: int = 10,
    max_wait_s: int = 900,
) -> VideoResult:
    """关键帧 → 视频片段，落盘到 out_path。"""
    payload_in: dict[str, Any] = {
        "prompt": prompt[:1500],
        "img_url": image_to_data_url(image),
    }
    if negative_prompt:
        payload_in["negative_prompt"] = negative_prompt[:500]

    params: dict[str, Any] = {
        "resolution": resolution,
        "duration": duration,
        "prompt_extend": prompt_extend,
        "watermark": False,
    }
    if model.startswith("wan2.6"):
        params["audio"] = audio
    if seed is not None:
        params["seed"] = seed

    payload = {"model": model, "input": payload_in, "parameters": params}

    t0 = time.time()
    resp = _request(BASE + SUBMIT_PATH, payload, extra_headers={"X-DashScope-Async": "enable"})
    task_id = (resp.get("output") or {}).get("task_id")
    if not task_id:
        raise RuntimeError(f"提交失败: {json.dumps(resp, ensure_ascii=False)[:600]}")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    while time.time() - t0 < max_wait_s:
        time.sleep(poll_interval)
        st = _request(BASE + TASK_PATH.format(task_id=task_id))
        o = st.get("output") or {}
        status = o.get("task_status")
        if status == "SUCCEEDED":
            url = o.get("video_url")
            if not url:
                raise RuntimeError(f"任务成功但无 video_url: {json.dumps(st, ensure_ascii=False)[:600]}")
            urllib.request.urlretrieve(url, out)
            usage = st.get("usage") or {}
            dur = float(usage.get("output_video_duration") or usage.get("duration") or duration)
            unit = PRICE_CNY_PER_S.get((model, resolution, audio), 0.0)
            return VideoResult(
                task_id=task_id, path=str(out), duration_s=dur,
                elapsed_s=round(time.time() - t0, 1),
                cost_cny=round(dur * unit, 4), raw=st,
            )
        if status in ("FAILED", "CANCELED", "UNKNOWN"):
            raise RuntimeError(f"任务 {status}: {json.dumps(st, ensure_ascii=False)[:600]}")

    raise TimeoutError(f"任务 {task_id} 超时 {max_wait_s}s")


def selftest() -> None:
    ref = ROOT / "assets" / "p0_testset" / "TS-WHITE-01" / "ref_front.png"
    r = generate_video(
        ref,
        "镜头缓慢推近，衣物表面轻微起伏",
        ROOT / "runs" / "_video_selftest.mp4",
        duration=2,
    )
    print(json.dumps({"task_id": r.task_id, "path": r.path, "dur": r.duration_s,
                      "cost": r.cost_cny, "elapsed": r.elapsed_s}, ensure_ascii=False, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--image")
    ap.add_argument("--prompt")
    ap.add_argument("--out", default="runs/out.mp4")
    ap.add_argument("--model", default="wan2.6-i2v-flash")
    ap.add_argument("--resolution", default="720P")
    ap.add_argument("--duration", type=int, default=4)
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.image or not args.prompt:
        ap.print_help()
        return
    r = generate_video(args.image, args.prompt, ROOT / args.out, model=args.model,
                       resolution=args.resolution, duration=args.duration)
    print(json.dumps({"path": r.path, "cost_cny": r.cost_cny, "elapsed_s": r.elapsed_s},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
