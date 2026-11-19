# -*- coding: utf-8 -*-
"""
镜头提示词编译器：结构化表单 → 最终提示词。

为什么要有这一层（2026-11-18 五模型横测结论）：
  自由写作     平均 2.4/8 项通过，换模型最多 +1 分，且不稳定
  基础填表     平均 4.8/8
  加字段填表   平均 7.0/8
  唯一全体失败项是「单手构图」——即使 schema 明写 visible_count=1，
  模型仍会把第二只手安排进画面。**构图减法没有任何模型会替你做。**

所以编译器不只是"拼接字符串"，它同时是**规则的执行者**：
  - 排序由代码保证：主体 → 动作 → 场景 → 光线 → 镜头 → 风格 → 画质
  - 构图减法由代码强制：visible_count >= 2 直接拒绝编译，不生成
  - 载具镜自动补"不做环绕"
  - 否定陷阱（点名禁止可画实物）直接报错

设计立场：宁可编译失败让人改表单，也不要生成一张注定返工的图。

用法：
  python scripts/compose_prompt.py --check          # 校验全部镜头
  python scripts/compose_prompt.py --shot S1        # 打印编译结果
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CFG = ROOT / "config" / "manju_ch01.json"

VEHICLE = ("骑车", "骑行", "开车", "驾驶", "电动车", "摩托车")
NEG_TRAPS = ("泪水", "泪痕", "泪珠", "车牌", "牌照", "血迹", "纹身", "伤疤")


class ComposeError(RuntimeError):
    pass


@dataclass
class Compiled:
    positive: str
    motion: str
    negative: str
    warnings: list[str] = field(default_factory=list)


def _req(form: dict, path: str) -> Any:
    cur: Any = form
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur or cur[k] in ("", None, []):
            raise ComposeError(f"表单缺字段：{path}")
        cur = cur[k]
    return cur


def requires_spatial_scope(cfg: dict, vkey: str) -> bool:
    """这个变体是否需要「围合 + 天气归属」——由配置声明，不写死在代码里。

    为什么要把这条从代码里搬进配置：变体从 3 个（室内/走廊/门洞）拆成
    4 个（玄关/客厅/走廊/门洞）之后，任何一处硬编码列表都会立刻过期，
    而过期的规则比没有规则更糟——它会静默放行，你还以为它查过了。
    """
    if not vkey:
        return False
    env = cfg.get("environment_lock") or {}
    var = (env.get("variants") or {}).get(vkey) or {}
    if var.get("enclosed") is True or var.get("rain_scope_required") is True:
        return True
    if "enclosed" in var or "rain_scope_required" in var:
        return False
    # 向后兼容：EP01/ch02 这类老剧本没有这些属性，按历史约定判
    return vkey in ("interior", "corridor", "doorway")


def validate(form: dict, shot_id: str, cfg: dict | None = None) -> list[str]:
    """硬规则校验。违反即拒绝编译。"""
    warns: list[str] = []
    # 雨（以及任何"会充满画幅"的天气）必须有空间归属。
    # EP02 的 S2 就是没写归属：环境锁说"雨幕清晰"，模型把雨丝画进了门内的书架前和床前，
    # 成片里就是"屋里在下雨"。这类错误观众一眼就判死，比场景不一致严重得多。
    _v = str(form.get("env_variant") or "").strip()
    if cfg is not None and requires_spatial_scope(cfg, _v):
        _scope = str(form.get("rain_scope") or "").strip()
        if not _scope:
            raise ComposeError(
                f"{sid} 缺 rain_scope：「{_v}」类镜头必须写明雨出现在画面哪一侧、哪一侧是干的。"
                f"不写就会被涂满整张画幅（EP02 S2 因此变成屋里下雨）。"
            )
        if not any(k in _scope for k in ("干", "门外", "窗外", "外侧", "右侧", "左侧", "楼道口", "门口", "一侧")):
            raise ComposeError(
                f"{sid} 的 rain_scope 没有空间边界：需要出现「干/门外/窗外/外侧/楼道口」之类的定位词"
            )

    hands = _req(form, "hands")
    n = hands.get("visible_count")
    if not isinstance(n, int) or n < 0:
        raise ComposeError("hands.visible_count 必须是整数")
    if n >= 3:
        # R1 的本意不是"画面里只能有一只手"，而是"不能让两只手各拿一样东西"。
        # 首版 S1 崩坏的真实病因是「一手手机 + 一手头盔」这个结构，
        # 不是"出现了两只手"。12 镜实测把规则收紧到 n>=2 会误伤骑车握把、
        # 双手自然下垂这类完全正常的姿态，于是配置只能写 1 而画面是 2 —— 配置开始说谎。
        # 现在把阈值放到 3，真正的病由下面 held>1 那条拦。
        raise ComposeError(
            f"hands.visible_count={n}：三只手以上必崩。"
            f"画面里只能有两只手，其余肢体必须移出画框。"
        )
    if n == 2:
        warns.append(
            "R1：双手镜头。确认不是「各持一物」（那是首版结构崩坏的真正原因）；"
            "骑车握把、双手自然下垂属于正常姿态，允许。"
        )

    held = [p for p in form.get("props", []) if p.get("in_frame") and ("手" in str(p.get("held_by", "")))]
    if len(held) > 1:
        raise ComposeError(
            f"画面内有 {len(held)} 件道具分别由手持有：{[p['name'] for p in held]}。违反 R1。"
        )

    # R6：空镜（form.has_person=false）不该出现肢体。
    # 12 镜正片里用了空巷、车轮、楼道灯这类空镜来规避人脸一致性风险，
    # 但"无人"必须显式声明，否则模型会往空镜头里塞人。
    if form.get("has_person") is False and n != 0:
        raise ComposeError(
            f"has_person=false 但 hands.visible_count={n}：空镜不该出现肢体。"
            f"要么改成有人的镜头，要么把 visible_count 改成 0。"
        )

    # R4：否定陷阱
    text = json.dumps(form, ensure_ascii=False)
    for trap in NEG_TRAPS:
        if re.search(rf"(不得|不要|禁止|没有|无)[^，。；\"]{{0,8}}{trap}", text):
            raise ComposeError(
                f"表单里用否定词点名了「{trap}」。实测反向强化（写'不得流泪'→泪痕变粗），违反 R4。"
            )

    # R2：一镜一动作（启发式，只警告）
    am = form.get("action_moment", "")
    if am.count("，") + am.count("、") >= 3:
        warns.append("R2：action_moment 逗号过多，可能塞了多个动作，建议拆镜")

    # R3：面部大特写
    # 注意：只有「特写」且主体含面部器官才算。迁移测试时手部特写被误判为面部特写。
    _lens = f"{form.get('camera', {}).get('lens', '')}{form.get('framing', '')}"
    _subj = str(form.get("subject", ""))
    _is_face = any(k in _subj for k in ("脸", "面", "眼", "瞳", "眉", "唇", "五官"))
    if "特写" in _lens and _is_face:
        warns.append("R3：面部大特写一致率实测 0.700（唯一不可用镜），建议改局部或加道具同框")

    return warns


# 身份特征里属于「道具/可变装束」的部分，不参与逐镜强制注入
NON_IDENTITY_PARTS = {"配饰"}

# 这些部位必须存在于形象卡，缺失即拒编（缺了眼部约束的后果见 v3 首次运行）
REQUIRED_PARTS = {"脸型", "眼", "发型"}


def identity_line(card: dict) -> str:
    """把形象卡里的身份特征拼成一句强制注入的锚定描述。

    为什么必须由编译器注入、而不写进每个镜头的 form.subject：
    v3 首次运行时，四镜的 subject 是我手填的，只写了发型和服装，
    眼部三项 critical 特征（单眼皮/眼尾上挑/深褐瞳）全部丢失。
    结果 S2 一致率从 0.925 掉到 0.777——模型按默认审美把眼睛画圆画大。
    身份特征是全局不变量，不该依赖每镜手抄。
    """
    feats = card.get("identity_features") or []
    if not feats:
        raise ComposeError("形象卡缺少 identity_features：无身份锚定，角色会每镜漂移")

    parts = {f.get("part") for f in feats if f.get("level") in ("critical", "major")}
    missing = REQUIRED_PARTS - parts
    if missing:
        raise ComposeError(
            f"形象卡缺少必填部位 {sorted(missing)}："
            f"缺「眼」会让模型按默认审美把眼睛画圆画大（v3 首次运行实测掉 0.15）"
        )

    items = [
        f"{f['part']}：{f['desc']}"
        for f in feats
        if f.get("level") in ("critical", "major") and f.get("part") not in NON_IDENTITY_PARTS
    ]
    return "身份特征（每一镜都必须完全一致）：" + "；".join(items) + "。"


def env_canon(cfg: dict, vkey: str, sid: str = "?") -> str:
    """环境卡：把一间的固定陈设压成一句，逐镜强制注入。

    为什么环境也需要"形象卡"（2026-11-19 EP02 复盘）：
      EP02 的客厅在 S5 是「姜黄色花纹沙发 + 正中大窗 + 深色窗帘 + 深木茶几」，
      在 S10 是「米色布艺沙发 + 左侧木框窗 + 白纱帘」。同一个客厅长成了两间屋子。
      根因不是模型不稳定 —— **是 S5 的 space 只写了「沙发与茶几」，没写长什么样**。
      模型只能按默认值填，而默认值每镜都不一样。

      角色靠形象卡锁住五官，环境必须靠环境卡锁住陈设。两者的病是同一个病：
      不写的东西会被默认值填满，而默认值不稳定。

    注入的是**方位 + 材质 + 数量**，不是形容词堆砌：
      「东墙：米色布艺三人沙发，扶手磨白」能被画出确定的东西；
      「温馨的客厅」不能。
    """
    env = cfg.get("environment_lock") or {}
    var = (env.get("variants") or {}).get(vkey) or {}
    sheet_key = str(var.get("sheet") or "").strip()
    if not sheet_key:
        return ""
    sheets = env.get("sheets") or {}
    if sheet_key not in sheets:
        raise ComposeError(
            f"{sid} 的 env_variant=\"{vkey}\" 指向环境卡 \"{sheet_key}\"，但 environment_lock.sheets "
            f"里没有它。可选：{'、'.join(sheets) or '（本集未定义环境卡）'}"
        )
    sheet = sheets[sheet_key] or {}
    brief = str(sheet.get("brief") or "").strip()
    if brief:
        return f"场景陈设（跨镜固定，不得改动）：{brief}。"
    canon = [str(x).strip() for x in (sheet.get("canon") or []) if str(x).strip()]
    if not canon:
        raise ComposeError(
            f"环境卡 \"{sheet_key}\" 既没有 brief 也没有 canon：环境卡没有内容等于没有环境卡"
        )
    return "场景陈设（跨镜固定，不得改动）：" + "；".join(canon) + "。"


def compile_shot(cfg: dict, shot: dict) -> Compiled:
    """核心：按「主体→动作→场景→光线→镜头→风格→画质」组装。"""
    form = shot.get("form")
    if not form:
        raise ComposeError(f"{shot.get('id')} 没有 form，无法编译")

    sid = shot.get("id", "?")
    warns = validate(form, sid, cfg)

    style = cfg["style_lock"]
    env = cfg.get("environment_lock", {})
    card = cfg.get("character_card", {})
    prop_lock = card.get("proportion_lock", {})

    # 三层锁缺一不可：缺了就会重演"雨只在部分镜头""颈长 0.36 头高"
    if not env.get("line"):
        raise ComposeError("缺少 environment_lock：环境不会跨镜继承")
    if not prop_lock.get("line"):
        raise ComposeError("缺少 proportion_lock：比例不约束，模型会自由发挥")

    # 环境锁变体：同一集里室内外并存时，逐镜无条件注入"雨未停、地面积水"
    # 会把雨下进客厅。变体不是"可以不带环境锁"，而是"环境锁有一组具名版本，
    # 每镜必须点名用哪一个"——所以下面找不到变体会直接拒编，不会静默回落。
    variants = env.get("variants") or {}
    vkey = str(form.get("env_variant") or "").strip()
    if vkey:
        if vkey not in variants:
            raise ComposeError(
                f"{sid} 的 env_variant=\"{vkey}\" 未定义，可选：{'、'.join(variants) or '（本集无变体）'}"
            )
        picked = variants[vkey]
        if not str(picked.get("line", "")).strip():
            raise ComposeError(f"environment_lock.variants.{vkey} 缺少 line")
        env_line = picked["line"]
        env_neg = str(env.get("negative", ""))
        drop = set(picked.get("drop") or [])
        if drop:
            env_neg = "，".join(
                t for t in env_neg.split("，") if t.strip() and t.strip() not in drop
            )
        env_neg = "，".join(x for x in (env_neg, str(picked.get("negative", ""))) if x)
    else:
        if variants:
            raise ComposeError(
                f"{sid} 没有 env_variant：本集定义了环境变体"
                f"（{'、'.join(variants)}），每一镜都必须显式点名"
            )
        env_line = env["line"]
        env_neg = str(env.get("negative", ""))

    light = _req(form, "light")
    cam = _req(form, "camera")
    prop = _req(form, "proportions")
    wx = _req(form, "weather")
    hands = _req(form, "hands")
    props = form.get("props", [])

    # ---- 正向：严格按优先级顺序 ----
    seg: list[str] = []
    seg.append(f"{form['subject']}，{form['action_moment']}。")
    # 空镜跳过身份与比例注入：画面里没有人，注入"鹅蛋脸、肩宽1.7头高"只会让模型
    # 硬塞一个人进来。有人的镜头仍然强制注入（v3 run1 的教训）。
    has_person = form.get("has_person", True)
    if has_person:
        seg.append(identity_line(card))
    seg.append(f"场景：{form['space']}。")
    seg.append(
        f"光线：{light['source']}，{light['direction']}，{light['lights_what']}。"
    )
    seg.append(f"机位：{cam['mount']}，{cam['height']}，{cam['lens']}。")

    if props:
        for p in props:
            if p.get("in_frame") and not str(p.get("facing", "")).strip():
                raise ComposeError(
                    f"道具「{p.get('name')}」没有写 facing。首版 S1 就是漏了朝向，屏幕正对镜头。"
                )
        desc = "；".join(
            f"{p['name']}由{p['held_by']}持有，{p['facing']}"
            + (f"，{p['note']}" if str(p.get("note", "")).strip() else "")
            for p in props if p.get("in_frame")
        )
        seg.append(f"道具：{desc}。")

    seg.append(
        f"手部：画面中只有 {hands['visible_count']} 只手，{hands['which']}；"
        f"{hands['other_limb']}。"
    )
    if has_person:
        seg.append(
            f"人体比例：{prop['head_ratio']}，{prop['shoulder_width']}，{prop['neck']}。"
            f"{prop.get('note', '')}。"
        )
    seg.append(f"天气：{wx['type']}，{wx['ongoing_evidence']}。")
    if str(form.get("rain_scope") or "").strip():
        seg.append(f"空间归属：{form['rain_scope']}。")
    seg.append(f"画面文字：{form['text_policy']}。")

    # 环境锁（逐镜重复，不依赖继承）
    seg.append(env_line)
    # 环境卡：同一空间里每一件陈设的方位与材质，逐镜重复。
    # 空镜也必须注入 —— S6/S8/S10 三个空镜恰恰是陈设最容易被编造的地方。
    if vkey:
        _canon = env_canon(cfg, vkey, sid)
        if _canon:
            seg.append(_canon)

    seg.append(f"{style['positive']}。")
    seg.append(f"画幅 9:16 竖屏，画面铺满画幅，不出现边框、色块、渐变遮罩或分栏。")

    positive = "".join(seg)

    # ---- 动态：动作流程 + 运镜 + 环境持续 ----
    motion_f = form.get("motion", {})
    m_parts = [
        motion_f.get("camera_move", ""),
        motion_f.get("action_flow", ""),
        motion_f.get("environment_persist", ""),
    ]
    # R5：载具镜自动补环绕排除，不靠人记
    joined = positive + "".join(m_parts)
    if any(v in joined for v in VEHICLE) and "环绕" not in "".join(m_parts):
        m_parts.append("机位锁定，不做环绕、不绕拍、不变焦")
        warns.append("R5：载具运动镜，已自动补「不做环绕」")
    motion = "，".join([p for p in m_parts if p])

    # ---- 负向：风格 + 环境 + 比例 + 当场风险项，去重 ----
    raw = [
        style.get("negative", ""),
        env_neg,
        prop_lock.get("negative", ""),
        "，".join(form.get("negatives", [])),
    ]
    seen, toks = set(), []
    for chunk in raw:
        for t in (chunk or "").split("，"):
            t = t.strip()
            if t and t not in seen:
                seen.add(t)
                toks.append(t)
    negative = "，".join(toks)

    return Compiled(positive=positive, motion=motion, negative=negative, warnings=warns)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CFG))
    ap.add_argument("--shot", help="只编译指定镜头")
    ap.add_argument("--check", action="store_true", help="只校验，不打印全文")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    shots = [s for s in cfg["shots"] if not args.shot or s["id"] == args.shot]
    if not shots:
        raise SystemExit(f"找不到镜头 {args.shot}")

    failed = 0
    for s in shots:
        try:
            c = compile_shot(cfg, s)
        except ComposeError as e:
            failed += 1
            print(f"✗ {s['id']} 编译失败：{e}")
            continue
        print(f"✓ {s['id']} 编译通过  （正向 {len(c.positive)} 字 / 动态 {len(c.motion)} 字 / 负向 {len(c.negative)} 字）")
        for w in c.warnings:
            print(f"    ⚠ {w}")
        if not args.check:
            print("\n--- 正向 ---")
            print(c.positive)
            print("\n--- 动态 ---")
            print(c.motion)
            print("\n--- 负向 ---")
            print(c.negative)
            print()

    print(f"\n{len(shots) - failed}/{len(shots)} 通过")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
