# -*- coding: utf-8 -*-
"""
提示词写作模型横测。

要回答两个问题：
  A 组（自由写作）：换个更强的文本模型，提示词层的缺陷会不会减少？
  B 组（填表）：     换成结构化 schema 之后，模型差异还剩多少？

设计要点：
  - brief 里**不透露任何已知坑**（雨要写进行时、比例要锁、手机朝向、单手构图）。
    一旦透露，测的就是"照抄能力"，不是"会不会自己想到"。
  - brief 故意复刻真实失败构图（一手举手机 + 臂弯夹头盔），让风险自己冒出来。
  - 评分按 7 项已知失效模式逐条判，不评文采。

用法：
  python scripts/prompt_bench.py
  python scripts/prompt_bench.py --models deepseek-v4-pro qwen3.8-max
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
OUT = ROOT / "assets" / "bench"

DEFAULT_MODELS = [
    "deepseek-v4.1-flash",   # 对照组：当前使用的模型
    "deepseek-v4-pro",
    "qwen3.8-max",
    "glm-5.3-prime",
    "kimi-k3.1-pro",
]

BRIEF = """你是一名 AI 视频/图像的提示词导演。请为下面这个镜头写一条**用于 AI 图像生成**的提示词。

【剧本设定】
暴雨夜，南方城市的窄巷口。主角林夏，22 岁女性外卖骑手：齐耳短发、米白色薄冲锋衣（胸前一道横向黄色反光条、立领拉链到顶）、左眉尾外侧一颗小痣。

【这一镜的内容】
中近景，胸部以上。她低头看着手里的手机，屏幕上跳出订单——收件人是她自己的名字。
手机屏幕是画面唯一强光源。她的臂弯里还夹着一顶黑色半盔。

【风格】
2.5D 国漫电影质感，赛璐璐平涂配写实光影，冷蓝夜色主调，霓虹暖橙作点光源，浅景深。
画幅 9:16 竖屏。

【输出要求】
只输出提示词正文，中文，一段，不要解释、不要分点、不要 markdown 代码块。
"""

SCHEMA_HINT = """请把这条镜头填成下面这个 JSON（只输出 JSON，不要解释）：

{
  "subject": "主体（含年龄/身份/关键外观）",
  "action_moment": "动作的瞬间（可拍的一刻，不是过程叙述）",
  "space": "可识别的具体空间类型",
  "light": "光源与方向（谁照亮谁）",
  "camera": "机位来源与景别（固定/手持/推轨…）",
  "proportions": "人体比例约束",
  "weather": "天气及其持续状态",
  "negatives": ["需要排除的具体风险项"]
}
"""


def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


KEY = load_env().get("PROVIDER_API_KEY", "")
if not KEY:
    raise SystemExit("缺少 PROVIDER_API_KEY")


def call(model: str, prompt: str, timeout: int = 120) -> tuple[str, dict]:
    def _once(with_temp: bool) -> tuple[str, dict]:
        payload: dict = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if with_temp:
            payload["temperature"] = 0.3
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            BASE, data=body, method="POST",
            headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            return "", {"error": f"HTTP {e.code}: {e.read().decode('utf-8','ignore')[:200]}"}
        except Exception as e:  # noqa: BLE001
            return "", {"error": f"{type(e).__name__}: {e}"}
        try:
            txt = data["choices"][0]["message"]["content"]
        except Exception:  # noqa: BLE001
            return "", {"error": f"响应异常: {json.dumps(data, ensure_ascii=False)[:200]}"}
        return txt.strip(), {"elapsed_s": round(time.time() - t0, 1), "usage": data.get("usage", {})}

    txt, meta = _once(True)
    # 部分模型（如 kimi-k3 系列）不接受 temperature，去掉重试一次
    if not txt and "temperature" in str(meta.get("error", "")):
        txt, meta = _once(False)
        meta["note"] = "已去掉 temperature 重试"
    return txt, meta


# --------------------------------------------------------------------------- #
# 评分：7 项已知失效模式
# --------------------------------------------------------------------------- #
def has(t: str, *words: str) -> bool:
    return any(w in t for w in words)


def score(t: str) -> dict[str, bool | str]:
    r: dict[str, bool | str] = {}

    # C1 雨必须是"正在下"，而不是"被雨打湿"
    r["C1_雨进行时"] = has(t, "雨") and has(t, "正在", "持续", "不断", "不停", "雨丝", "雨幕", "下着", "倾盆", "瓢泼")

    # C2 比例约束：必须是具体的量（肩宽/头高/颈长），"精致写实比例"这类泛词不算
    r["C2_比例约束"] = has(t, "肩宽", "头高", "颈长", "颈部", "头身比", "七头身", "八头身", "肩线")

    # C3 道具朝向
    r["C3_道具朝向"] = has(t, "手机", "屏幕") and has(t, "背面", "屏幕朝", "朝向", "背对", "面向她", "正对她")

    # C4 单手构图（显式分配了两条手臂）
    r["C4_单手构图"] = has(t, "一只手臂", "只有一只手", "单手持", "另一只手臂", "另一只手", "在画框外", "不入画", "出画", "移出画面")

    # C5 否定词点名可画实物（反效果写法）
    trap = re.search(r"(不得|不要|禁止|没有|无)[^，。；]{0,8}(泪水|泪|车牌|牌照|血迹|纹身)", t)
    r["C5_无否定陷阱"] = trap is None

    # C6 机位来源
    r["C6_机位来源"] = has(t, "固定机位", "机位", "手持", "推轨", "跟拍", "摇镜", "横移", "升降", "固定镜头", "镜头固定")

    # C7 是否显式禁止文字
    r["C7_禁文字约束"] = bool(
        re.search(r"(不得出现|不允许|不出现|无|没有|不要|禁止|避免)[^，。；]{0,12}(文字|字母|水印|logo|LOGO|字符)", t)
    )

    # C8 是否反而要求模型渲染可读中文（模型画中文必乱码，这是最严重的硬伤）
    # 只判"是否点名要画出名字/订单文字"，不判 JSON 里的普通引号（否则字段名会被误伤）
    r["C8_不要求画中文"] = not (
        re.search(r"[“「]林夏[”」]", t)
        or re.search(r"(?<![不没])(显示|写着|写有|出现)[^，。；]{0,8}(名字|收件人|订单信息|文字内容)", t)
    )

    r["命中项"] = ""
    return r


CHECKS = ["C1_雨进行时", "C2_比例约束", "C3_道具朝向", "C4_单手构图",
          "C5_无否定陷阱", "C6_机位来源", "C7_禁文字约束", "C8_不要求画中文"]


def extract_json(t: str) -> dict | None:
    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return None


RICH_SCHEMA = """请把这条镜头填成下面这个 JSON（只输出 JSON，不要解释）：

{
  "subject": "主体（年龄/身份/关键外观）",
  "action_moment": "动作的瞬间（可拍的一刻）",
  "space": "可识别的具体空间类型",
  "light": {"source": "唯一主光源是什么", "direction": "光从哪照向哪", "lights_what": "照亮了哪些部位"},
  "camera": {"mount": "机位来源（固定/手持/推轨…）", "height": "机位高度", "lens": "焦段或视角"},
  "proportions": {"head_ratio": "人物几头身", "shoulder_width": "肩宽相对头高的倍数", "neck": "颈部露出长度", "note": "其他比例约束"},
  "weather": {"type": "天气", "ongoing_evidence": "画面里可见的'正在发生'的证据"},
  "props": [{"name": "道具名", "held_by": "由哪只手/哪个部位持有", "facing": "朝谁/朝哪", "in_frame": true}],
  "hands": {"visible_count": 1, "which": "入画的是哪只手", "other_limb": "另一条手臂在哪"},
  "text_policy": "画面内文字的处理方式（应明确不出现可读文字）",
  "negatives": ["需要排除的具体风险项"]
}
"""


def print_table(title: str, rows: list[dict], key: str) -> None:
    print("\n" + "=" * 104)
    print(title)
    print("=" * 104)
    print(f"{'模型':<22}" + "".join(f"{c.split('_')[0]:>5}" for c in CHECKS) + f"{'分':>7}{'字数':>7}")
    print("-" * 104)
    for r in rows:
        if not r.get(key):
            print(f"{r['model']:<22} 调用失败: {r.get(key + '_err')}")
            continue
        line = f"{r['model']:<22}"
        for c in CHECKS:
            line += f"{'✓' if r[key][c] else '·':>5}"
        n = sum(1 for c in CHECKS if r[key][c])
        print(line + f"{n:>6}/8{r.get(key + '_len', 0):>7}")


def rescore() -> None:
    """不重新调用 API，直接对已保存的原始输出重打分。"""
    rows = []
    for f in sorted(OUT.glob("*__A.txt")):
        model = f.name.replace("__A.txt", "")
        a = f.read_text(encoding="utf-8")
        b_path = OUT / f"{model}__B.txt"
        b = b_path.read_text(encoding="utf-8") if b_path.exists() else ""
        c_path = OUT / f"{model}__C.txt"
        c = c_path.read_text(encoding="utf-8") if c_path.exists() else ""
        rows.append({
            "model": model,
            "A": score(a) if a else None, "A_len": len(a),
            "B": score(b) if b else None, "B_len": len(b),
            "C": score(c) if c and not c.startswith("(空)") else None, "C_len": len(c),
        })
    print_table("A 组 · 自由写作（8 项检查）", rows, "A")
    print_table("B 组 · 基础填表（8 项检查）", rows, "B")
    print_table("C 组 · 加字段填表（8 项检查）", rows, "C")


def phase_c(models: list[str]) -> None:
    """加字段对照：同一批模型，换成把'朝向/持物/文字'都做成字段的 schema。"""
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in models:
        print(f"[{m}] C 组 加字段填表 ...", flush=True)
        c, meta = call(m, BRIEF + "\n\n" + RICH_SCHEMA)
        (OUT / f"{m}__C.txt").write_text(c or f"(空) {meta}", encoding="utf-8")
        rows.append({
            "model": m,
            "C": score(c) if c else None,
            "C_len": len(c),
            "C_err": meta.get("error"),
        })
    print_table("C 组 · 加字段填表（把朝向/持物/文字做成必填字段）", rows, "C")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--rescore", action="store_true", help="只对已保存输出重打分")
    ap.add_argument("--phase-c", action="store_true", help="跑加字段对照")
    args = ap.parse_args()

    if args.rescore:
        rescore()
        return
    if args.phase_c:
        phase_c(args.models)
        return

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    for m in args.models:
        print(f"[{m}] A 组 自由写作 ...", flush=True)
        a, meta_a = call(m, BRIEF)
        print(f"[{m}] B 组 填表 ...", flush=True)
        b, meta_b = call(m, BRIEF + "\n\n" + SCHEMA_HINT)

        (OUT / f"{m}__A.txt").write_text(a or f"(空) {meta_a}", encoding="utf-8")
        (OUT / f"{m}__B.txt").write_text(b or f"(空) {meta_b}", encoding="utf-8")

        rows.append({
            "model": m,
            "A": score(a) if a else None, "A_len": len(a), "A_err": meta_a.get("error"),
            "B": score(b) if b else None, "B_len": len(b), "B_err": meta_b.get("error"),
        })

    (OUT / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print_table("A 组 · 自由写作（8 项检查）", rows, "A")
    print_table("B 组 · 基础填表（8 项检查）", rows, "B")
    print("\n原始输出已存 assets/bench/")


if __name__ == "__main__":
    main()
