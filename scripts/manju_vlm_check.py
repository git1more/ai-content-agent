# -*- coding: utf-8 -*-
"""关键帧自动质检：用 VLM 按表单自动生成检查项，判 pass/fail。

为什么要有这一层：
  EP00 和 ch02 加起来，我人工看图下错过 4 次结论（裁错区域、缩略图误判）。
  人眼验收不只是慢，是准确率不够。而"咖啡从杯中流出""手机屏幕朝镜头""车牌乱码"
  这类硬伤，本质是可以用 5-6 个结构化问题判出来的。

设计原则（保证换剧本能用）：
  检查项不写死，全部从 shot["form"] 自动生成：
    hands.visible_count  -> 问手臂数量
    props[].facing       -> 问画面是否符合该朝向描述
    weather.type         -> 问天气是否持续
    一贯项（文字/比例）  -> 固定问，因为任何片子都不想要

输出是结构化 JSON，可存档、可统计、可追溯到具体镜头。

用法：
  python scripts/manju_vlm_check.py --image <path> --config config/manju_ch02.json --shot S2
  python scripts/manju_vlm_check.py --batch   # 检查某集全部关键帧
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
MODEL = "qwen3-vl-plus"

# 压到 720 高，控制 token；判定硬伤够用
MAX_SIDE = 1024


def _key() -> str:
    env = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.strip().split("=", 1)
                env[k] = v
    k = os.environ.get("PROVIDER_API_KEY") or env.get("PROVIDER_API_KEY", "")
    if not k:
        raise SystemExit("缺少 PROVIDER_API_KEY")
    return k


def _img_b64(path: Path) -> str:
    im = Image.open(path).convert("RGB")
    im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()



FFMPEG = ROOT / "tools" / "ffmpeg" / "ffmpeg.exe"

# 视频阶段只查"一眼废"的硬伤。景别、道具朝向在运动中本就会变，不适用。
VIDEO_LABELS = {"无可辨认文字", "无物理异常"}


def frames_of(video: Path, duration: float, n: int = 3) -> list[Path]:
    """从片段抽 n 帧。视频模型会在运动过程中加关键帧没有的东西（泪痕、液体流出），
    只查关键帧会漏掉一整类缺陷。"""
    import subprocess
    tmp = video.parent / "_vlmframes"
    tmp.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(n):
        t = duration * (i + 0.5) / n
        dst = tmp / f"{video.stem}_{i}.jpg"
        subprocess.run(
            [str(FFMPEG), "-y", "-hide_banner", "-loglevel", "error",
             "-ss", f"{t:.2f}", "-i", str(video), "-frames:v", "1", str(dst)],
            check=True,
        )
        if dst.exists():
            out.append(dst)
    return out

def ask(image: Path, prompt: str, model: str = MODEL, max_tokens: int = 400) -> str:
    """通用问句接口：给图 + 任意问题，返回模型原文。

    电商线的质检不走 shot.form 那套（没有分镜表单），直接用它。
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_img_b64(image)}"}},
            {"type": "text", "text": prompt},
        ]}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"Authorization": f"Bearer {_key()}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def build_questions(cfg: dict, shot: dict) -> tuple[str, list[dict]]:
    """从表单生成检查清单。返回（提示词正文，判据列表）。

    判据列表里每一项形如：
      {"key": "arms", "kind": "int_eq", "expect": 1, "label": "手臂数量"}
      {"key": "prop_手机", "kind": "bool", "expect": True, "label": "手机朝向"}
    """
    form = shot.get("form") or {}
    qs: list[str] = []
    rules: list[dict] = []
    n = 1

    # 1) 手部/手臂数量（结构类最强信号）
    hands = form.get("hands") or {}
    if "visible_count" in hands:
        # 问的必须是"手"，不能是"手臂或手"。
        # 实测（11-19 评测首次运行）：visible_count 的语义是"手"（配 `which` 写"无手入画"），
        # 近景镜「双臂在画内、手在画框外」是完全正常的构图。
        # 题面写成"手臂或手"之后，VLM 数了 2 只手臂，把 S4/S5/S7 三镜全判成不合规——
        # 尺子错了，不是画面错了。这一条不改，通过率永远比真实值低 25%。
        qs.append(
            f'{n}. 画面里能看到几只「手」（手掌或手指）？填整数，看不到填 0。'
            f'注意：只数手，不数手臂、不数衣袖。'
        )
        rules.append({"key": f"q{n}", "kind": "int_eq",
                      "expect": int(hands["visible_count"]), "label": "手的数量"})
        n += 1

    # 2) 可辨认文字（任何片子的一眼废硬伤）
    qs.append(
        f'{n}. 画面里有没有可辨认的文字、字母、数字、水印？'
        f'注意：招牌或封面上的乱码笔画也算"有"。填 true/false。'
    )
    rules.append({"key": f"q{n}", "kind": "bool", "expect": False, "label": "无可辨认文字",
                  # 软项：只提示，不计入通过率。
                  # 依据（2026-11-18，EP01 12 镜实测）：12 张全部被判"有可辨认文字"，
                  # 逐张看图确认全是霓虹招牌的模糊光块与手机 UI 色块，没有一张真有可读字。
                  # 同一问题在电商侧也出现过（玻璃瓶的 "VITAMIN C 15%" 是商品自己的标签）。
                  # 结论：这个问法在 VLM 上不可靠，宁可降级也不能让误报毁掉门禁的可信度。
                  "soft": True})
    n += 1

    # 3) 天气是否持续
    #    室内镜不问这一题。原因：雨已经被 rain_scope 明确限制在窗外/门外，
    #    画面里本来就不该有雨丝。拿这道题去问室内空镜（S3 玄关地砖上的鞋），
    #    它如实答 false，于是被判不合规——**假警报**。
    #    这条题原本是为了抓 EP00「雨只在部分镜头出现」，那说的是**室外**连续性。
    weather = form.get("weather") or {}
    wtype = str(weather.get("type", ""))
    from compose_prompt import requires_spatial_scope
    _indoor = requires_spatial_scope(cfg, str(form.get("env_variant") or "").strip())
    if wtype and not ("雨" in wtype and _indoor):
        if "雨" in wtype:
            qs.append(f'{n}. 画面里有没有正在下落的雨丝或雨幕？填 true/false。')
            rules.append({"key": f"q{n}", "kind": "bool", "expect": True, "label": "雨持续"})
        else:
            qs.append(f'{n}. 画面是白天还是夜晚？填 "白天" 或 "夜晚"。')
            rules.append({"key": f"q{n}", "kind": "enum", "expect": "白天" if "晴" in wtype or "昼" in wtype else None,
                          "label": "昼夜"})
        n += 1

    # 4) 每个入画道具的朝向
    for p in (form.get("props") or []):
        if not p.get("in_frame"):
            continue
        desc = str(p.get("facing", "")).strip()
        if not desc:
            continue
        qs.append(
            f'{n}. 道具「{p.get("name")}」应当满足：{desc}。'
            f'画面是否符合这个描述？填 true/false。'
        )
        rules.append({"key": f"q{n}", "kind": "bool", "expect": True,
                      # 软项：VLM 判不了空间朝向（"朝前还是朝后""是不是微距"在电商侧已三次判错）。
                      # 它只可靠地回答"有没有某个东西"，问"朝向是否符合描述"必然虚报。
                      "label": f"道具朝向·{p.get('name')}", "soft": True})
        n += 1

    # 5) 物理异常（能抓"咖啡从杯中流出""肢体断裂"这类一眼废）
    #    软项：只提示，不计入通过率。
    #    依据（2026-11-19 评测实测）：同一张 S9，仅改动第 1 题的措辞，
    #    这一项的结论就从"有异常"翻成"无异常"——它的判断受整段提示词影响，
    #    不是对画面的独立观察。hard 项若会随机翻转，通过率就不可信。
    #    而且它问的是"悬空/穿模/肢体"，判不了"挂钟跨在墙角上"这种空间合理性——
    #    那类要靠人看，或者靠 SURFACE_ANCHOR 在提示词层提前拦住。
    qs.append(
        f'{n}. 逐项看这三件事并回答有/无：(a) 画面里有没有液体流到容器外面、'
        f'从容器边缘挂下来、或悬在半空成一条线；(b) 有没有物体悬空不接触任何支撑面；'
        f'(c) 有没有肢体断裂或凭空多出。只要出现任意一项就填 true（有异常），都没有填 false。'
    )
    rules.append({"key": f"q{n}", "kind": "bool", "expect": False,
                  "label": "无物理异常", "soft": True})
    n += 1

    # 6) 景别
    qs.append(f'{n}. 这张图的景别属于哪一种？只能填：大特写、特写、近景、中近景、中景、全景、远景。')
    rules.append({"key": f"q{n}", "kind": "in_set",
                  # 软项：景别判断依赖尺度推理，VLM 做不到（EP01 实测把中景判成远景、
                  # 特写判成中景共 3 例，逐张看图确认实际景别与配置相符或接近）。
                  "expect": str(shot.get("shot_size", "")).replace("（侧面）", "").strip(),
                  "label": "景别", "soft": True})
    n += 1

    # 7) 人体比例（只在有人脸/头入画的镜头问）
    subj = str(form.get("subject", ""))
    # 不能用「面」单字：会误伤「积水路面」「后视镜面」，把空镜判成有人脸。
    if any(k in subj for k in ("脸", "眼", "瞳", "眉", "唇", "五官", "面部", "面孔", "脸颊")):
        qs.append(
            f'{n}. 人物的头身比看起来正常吗？头明显偏大、肩明显偏窄、脖子明显偏长都算不正常。'
            f'填 true（正常）/false（异常）。'
        )
        rules.append({"key": f"q{n}", "kind": "bool", "expect": True, "label": "人体比例"})
        n += 1

    keys = [r["key"] for r in rules]
    prompt = (
        "你在做动画关键帧的画面质检。请严格按下面问题逐项回答，只依据你实际看到的，"
        "不要推测创作者的意图。\n\n"
        + "\n".join(qs)
        + "\n\n只输出一个 JSON 对象，键名必须依次是："
        + ", ".join(f'"{k}"' for k in keys)
        + "。不要输出任何解释文字，不要用 markdown 代码块。"
    )
    return prompt, rules


def _parse_json(txt: str) -> dict | None:
    t = txt.strip()
    if t.startswith("```"):
        t = t.split("```")[1] if len(t.split("```")) > 1 else t
        t = t.replace("json", "", 1).strip()
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        return json.loads(t[a:b + 1])
    except json.JSONDecodeError:
        return None


def judge(image: Path, cfg: dict, shot: dict, model: str = MODEL,
          only_labels: set[str] | None = None) -> dict:
    prompt, rules = build_questions(cfg, shot)
    if only_labels:
        rules = [r for r in rules if r["label"] in only_labels]
        if not rules:
            return {"ok": True, "checks": [], "fails": [], "image": image.name}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_img_b64(image)}"}},
            {"type": "text", "text": prompt},
        ]}],
        "temperature": 0,
        "max_tokens": 300,
    }
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"Authorization": f"Bearer {_key()}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read().decode("utf-8"))
    txt = data["choices"][0]["message"]["content"]
    obj = _parse_json(txt)

    if obj is None:
        return {"ok": False, "raw": txt[:200], "fails": ["VLM 未返回可解析 JSON"],
                "checks": [], "image": image.name}

    checks, fails, warns = [], [], []
    for rule in rules:
        got = obj.get(rule["key"])
        kind = rule["kind"]
        if kind == "bool":
            ok = (str(got).lower() == str(rule["expect"]).lower())
        elif kind == "int_eq":
            try:
                ok = int(got) == rule["expect"]
            except (TypeError, ValueError):
                ok = False
        elif kind == "enum":
            ok = (rule["expect"] is None) or (str(got) == rule["expect"])
        elif kind == "in_set":
            exp = rule["expect"]
            ok = (exp == "") or (str(got) in exp) or (exp in str(got))
        else:
            ok = True
        checks.append({"label": rule["label"], "got": got, "expect": rule["expect"],
                       "ok": ok, "soft": bool(rule.get("soft"))})
        if not ok:
            msg = f"{rule['label']}：期望 {rule['expect']}，实际 {got}"
            (warns if rule.get("soft") else fails).append(msg)

    out = {"ok": not fails, "checks": checks, "fails": fails, "warns": warns,
           "image": image.name}
    # 只记"没过去"的部分：通过了不需要日志，失败了必须能查。
    if fails or warns:
        try:
            from runlog import log as _rlog
            for f in fails:
                _rlog("vlm", str(shot.get("id") or "?"), "fail",
                      rule=str(f).split("：")[0], reason=str(f))
            for w in warns:
                _rlog("vlm", str(shot.get("id") or "?"), "warn",
                      rule=str(w).split("：")[0], reason=str(w))
        except Exception:      # noqa: BLE001
            pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--config", default=str(ROOT / "config" / "manju_ch01.json"))
    ap.add_argument("--shot")
    ap.add_argument("--batch", action="store_true", help="检查该集全部关键帧")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--video", action="store_true", help="检查视频片段（抽 3 帧，只查硬伤）")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    stem = Path(args.config).stem.replace("manju_", "")
    outdir = ROOT / "assets" / "manju" / stem

    targets: list[tuple[Path, dict]] = []
    if args.batch:
        for shot in cfg["shots"]:
            p = outdir / f"{shot['id']}_keyframe.png"
            if p.exists():
                targets.append((p, shot))
    else:
        if not args.image or not args.shot:
            raise SystemExit("需要 --image 和 --shot，或用 --batch")
        shot = next(s for s in cfg["shots"] if s["id"] == args.shot)
        if args.video:
            targets += [(f, shot) for f in frames_of(Path(args.image), float(shot["duration"]))]
        else:
            targets.append((Path(args.image), shot))

    results = []
    for p, shot in targets:
        r = judge(p, cfg, shot,
                  only_labels=VIDEO_LABELS if args.video else None)
        r["shot"] = shot["id"]
        results.append(r)
        mark = "✓" if r["ok"] else "✗"
        print(f"{mark} {shot['id']}  {p.name}")
        for c in r.get("checks", []):
            if not c["ok"]:
                tag = "提示" if c.get("soft") else "不符"
                print(f"    - [{tag}] {c['label']}：期望 {c['expect']}，实际 {c['got']}")

    passed = sum(1 for r in results if r["ok"])
    n_warn = sum(1 for r in results for c in r.get("checks", []) if not c["ok"] and c.get("soft"))
    print(f"\n通过 {passed}/{len(results)}" + (f"（另有 {n_warn} 条软项提示，不计入通过率）" if n_warn else ""))
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
