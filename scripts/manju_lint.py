# -*- coding: utf-8 -*-
"""
漫剧提示词静态检查（prompt lint）。

为什么需要它：
  到目前为止，EP00 首版的四个缺陷全部是**用户看片之后才发现的**：
    1. 雨只在 S1/S2 出现
    2. S1 头身比怪（颈长 0.36 头高、肩宽 1.64 头高）
    3. S4 被读成流泪
    4. S1 手机屏幕朝向镜头，人物根本看不到屏幕
  每一条都是"提示词没写清楚"，不是模型能力不足。而每一条都可以在**生成之前**
  用一条规则判出来 —— 零成本，不用等出片，也不用等人看。

  这正是把人从"挑错"里解放出来的地方：人只负责审美判断，机器负责查遗漏。

规则分级：
  ERROR  必然导致返工，必须改
  WARN   高风险，建议改

2026-11-18 扩到音频（VOICE_* 三条）：
  配音是整套流程里唯一"生成前就能算准"的环节 —— TTS 按字出音、语速可测，
  于是台词长度在写剧本时就能判死，不必等合成完才发现念不完。
  这是全项目唯一能用"字数"这种确定性单位做门禁的地方，不用白不用。

用法：
  python scripts/manju_lint.py
  python scripts/manju_lint.py --config config/manju_ch01.json
退出码：有 ERROR 返回 1，否则 0（可直接接进 CI / 生成前门禁）
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CFG = ROOT / "config" / "manju_ch01.json"

sys.path.insert(0, str(ROOT / "scripts"))
from compose_prompt import ComposeError, requires_spatial_scope  # noqa: E402


def space_canon(cfg: dict, vkey: str) -> str:
    """本镜所属空间的环境卡清单（brief 优先，否则拼 canon）。"""
    env = cfg.get("environment_lock") or {}
    var = (env.get("variants") or {}).get(vkey) or {}
    key = str(var.get("sheet") or "").strip()
    sheet = (env.get("sheets") or {}).get(key) or {}
    brief = str(sheet.get("brief") or "").strip()
    if brief:
        return brief
    return "；".join(str(x) for x in (sheet.get("canon") or []))

# wan2.6-i2v-flash 支持的时长区间
DURATION_RANGE = (2, 15)

CONTINUOUS = ("持续", "不断", "正在", "一直", "始终")
PAST_STATE = ("已被", "被雨打湿", "已经湿")

# 用户实测的反效果写法：提示词里点名禁止某个"可画的实物"，模型反而把它画得更清楚。
# 证据（2026-11-17）：写"不得出现泪水"→ 泪痕变成两道粗泪柱；
#                   写"没有车牌"   → 车牌上出现可读拉丁字符 "TS 05 6"。
# 正确做法是改结构（换景别/换机位/后期遮挡），不是继续加否定词。
NEGATION_TRAPS = ("泪水", "泪痕", "泪珠", "车牌", "牌照", "血迹", "纹身", "伤疤")

CAMERA_LOCK = ("机位固定", "镜头固定", "固定机位", "镜头锁定", "相机固定")
CAMERA_MOVE = ("推近", "拉远", "上摇", "下摇", "横移", "跟拍", "环绕", "甩镜", "摇镜")
VEHICLE = ("骑车", "骑行", "开车", "驾驶", "电动车", "摩托车")

# ---- 常识四问的判据词表（2026-11-19） ----
# 用户看完 EP02 v3 的原话：「室内下雨和门框出现在一个莫名其妙的空旷地面上，
# 这两个问题不符合人类常识，这类问题更加扎眼。」
# 六条常识清单里，天气归属已有 RAIN_SCOPE 拦；剩下四条在这里代码化。
ENCL_TOP = ("天花板", "吊顶", "屋顶", "楼板", "顶上是", "顶上悬")
ENCL_WALL = ("墙", "壁", "门框", "窗框")
ENCL_GROUND = ("地面", "地板", "地砖", "水泥地", "门槛", "路面")
# 必须"长在结构上"的物件：门、窗、柜。缺宿主就会悬空立在空地上。
SURFACE_OBJ = ("门", "窗", "柜")
SURFACE_ANCHOR = ("墙", "门框", "墙体", "窗框", "贴墙", "靠墙", "嵌", "接严", "门洞", "墙洞")
# 室内空间必须有尺度感，否则模型会给出无限进深（EP02 S11 门内一大片空木地板）。
DEPTH_WORDS = ("进深", "米", "一步", "两步", "靠墙", "贴墙", "紧挨", "狭小", "不到", "横跨")
# 有朝向语义的实物：不写进 props 就拿不到 facing 校验。
# EP02 S3 的「鞋尖朝向屋内」被画成朝门口，就是因为鞋当时只写在 subject 里。
ORIENTED_OBJ = ("鞋", "餐盒", "手机", "杯", "碗", "书", "照片", "伞", "头盔", "刀", "椅")
# 含同一字但语义无关的复合词，必须先剔除，否则「矮鞋柜台面」会被判成一只鞋。
ORIENTED_FALSE = ("鞋柜", "鞋架", "书柜", "书架", "餐椅", "桌椅")

# 同一个物件在两处被写成不同材质 —— 环境卡上线当天就抓到一次：
# S6 的 space 写「玻璃茶几」，新加的环境卡写「深红木矮茶几」，
# 模型两处都照做，于是在同一个画面里画了**两张茶几**。
# 只查这三个最容易撞的物件：词表越窄越准，宁可漏也别误报。
OBJ_MATERIAL = {
    "茶几": ("玻璃", "木"),
    "沙发": ("布艺", "皮"),
    "地板": ("木", "瓷砖", "水磨石"),
}


def material_near(text: str, obj: str, mats: tuple[str, ...], win: int = 9) -> str | None:
    """在物件名前后 win 个字里找材质词。取最近的，找不到返回 None。"""
    best: tuple[int, str] | None = None
    start = 0
    while True:
        i = text.find(obj, start)
        if i < 0:
            break
        lo, hi = max(0, i - win), min(len(text), i + len(obj) + win)
        for m in mats:
            j = text.find(m, lo, hi)
            if j >= 0:
                d = abs(j - i)
                if best is None or d < best[0]:
                    best = (d, m)
        start = i + 1
    return best[1] if best else None


@dataclass
class Issue:
    level: str      # ERROR / WARN
    shot: str
    rule: str
    msg: str


def has_any(text: str, words: tuple[str, ...] | list[str]) -> bool:
    return any(w in text for w in words)


def lint(cfg: dict) -> list[Issue]:
    """检查**实际发给模型的那一份提示词**。

    关键帧提示词是 style_lock + 人物 + 比例锁 + 环境锁 + 镜头 拼装出来的，
    所以必须用 manju_run.build_keyframe_prompt 复现拼装结果再查，
    否则会拿半成品去查、产生一堆假警报。
    动态提示词是原样透传给视频模型的，不受拼装影响，单独查。
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    from manju_run import build_keyframe_prompt, build_motion_prompt, negative_of

    out: list[Issue] = []
    env = cfg.get("environment_lock") or {}
    card = cfg.get("character_card") or {}
    env_is_rainy = "雨" in (env.get("line") or "")

    # ---- 全局规则 -------------------------------------------------------
    if not card.get("proportion_lock"):
        # 首版正是缺这一条，导致肩宽 1.64 头高、颈长 0.36 头高的成片
        out.append(Issue("ERROR", "-", "PROP_LOCK",
                         "形象卡缺少 proportion_lock：比例不显式约束，模型会自由发挥"))

    if not env.get("line"):
        out.append(Issue("ERROR", "-", "ENV_LOCK",
                         "缺少 environment_lock：环境不会跨镜继承，会出现'前两镜下雨、后两镜晴天'"))

    if not card.get("identity_features"):
        out.append(Issue("ERROR", "-", "IDENTITY", "形象卡缺少 identity_features"))

    variants = (cfg.get("environment_lock") or {}).get("variants") or {}

    for shot in cfg.get("shots", []):
        sid = shot.get("id", "?")

        # 0) 环境变体必须被点名（前置：必须在任何编译调用之前判，
        #    否则编译器的拒编会先抛，lint 报不出这条可读提示）
        #    室内外并存的集里，环境锁不能再"一条通吃"——否则"雨未停、地面积水"
        #    会被原样注入客厅，成片里就是屋里下雨。
        # 0b) 天气必须有空间归属（前置：要在任何编译调用之前判）
        #     EP02 S2 没写归属，环境锁说"雨幕清晰"，模型把雨丝画进了门内的书架前与床前，
        #     成片里就是"屋里在下雨"。这类错误观众一眼判死，比场景不一致严重得多。
        _ev = str((shot.get("form") or {}).get("env_variant") or "").strip()
        if requires_spatial_scope(cfg, _ev):
            _sc = str((shot.get("form") or {}).get("rain_scope") or "").strip()
            if not _sc:
                out.append(Issue("ERROR", sid, "RAIN_SCOPE",
                                 f"「{_ev}」类镜头缺 rain_scope：没写雨在画面哪一侧、哪一侧是干的，"
                                 f"雨会被涂满整个画幅（会出现屋里下雨）"))
                continue

        if variants and not str((shot.get("form") or {}).get("env_variant") or "").strip():
            out.append(Issue("ERROR", sid, "ENV_VARIANT",
                             f"本集定义了环境变体（{'、'.join(variants)}），本镜没有点名："
                             f"环境锁无法确定该用哪一条"))
            continue

        neg_tokens = set(t.strip() for t in negative_of(cfg, shot).split("，") if t.strip())
        kp = build_keyframe_prompt(cfg, shot)      # 实际发给图片模型的
        mp = build_motion_prompt(cfg, shot)        # 实际发给视频模型的
        size = shot.get("shot_size") or ""
        both = kp + mp

        # 1) 天气：关键帧靠环境锁覆盖；动态提示词是原样透传，必须自己写
        if env_is_rainy:
            if "雨" not in kp:
                out.append(Issue("ERROR", sid, "ENV_RAIN",
                                 "关键帧提示词里没有'雨'：环境锁可能失效或被覆盖"))
            if "雨" not in mp or not has_any(mp, CONTINUOUS):
                out.append(Issue("ERROR", sid, "ENV_RAIN_MOTION",
                                 "动态提示词缺'持续/不断/正在 + 雨'：模型会理解成雨已停"))

        # 2) 朝向歧义：屏幕类必须写明朝向；镜面类只要写明映出什么即可
        for obj in ("手机", "屏幕", "显示屏"):
            if obj in kp and not has_any(kp, ("朝向", "朝镜头", "背朝", "面向", "正对",
                                              "朝下", "朝上", "朝内", "朝外", "背对", "正面朝")):
                out.append(Issue("ERROR", sid, "FACE_DIR",
                                 f"出现「{obj}」但没写朝向：首版 S1 屏幕正对镜头，人物看不到屏幕"))
                break
        if "后视镜" in kp and not has_any(kp, ("映出", "映着", "映在")):
            out.append(Issue("WARN", sid, "MIRROR_DIR",
                             "出现「后视镜」但没说映出什么，模型可能画成空白镜面"))

        # 3) 面部特写 + 水 = 高风险景别（不能靠写"没有泪水"解决，那违反 R4）
        # 必须看本镜主体，不能看 kp：编译器会强制注入身份特征，kp 里永远有"脸/眼"
        # 注意：不能用「面」这个单字 —— 12 镜实测把「积水路面」「后视镜面」都判成了
        # 面部特写，两处都是误报。改用明确的部位词。
        _FACE_WORDS = ("脸", "眼", "瞳", "眉", "唇", "五官", "面部", "面孔", "脸颊")
        _subj = str((shot.get("form") or {}).get("subject", "") or shot.get("subject", ""))
        _face_in_shot = any(k in _subj for k in _FACE_WORDS)
        if "特写" in size and _face_in_shot and has_any(kp, ("水珠", "水痕", "水滴", "湿")):
            out.append(Issue("WARN", sid, "TEAR_AMBIG",
                             "面部特写含水，实测易被读成哭泣且一致性 0.700；建议改手部/局部特写"))

        # 4) 文字约束（拼装层已加，这里查是否真的带上了）
        if "文字" not in kp:
            out.append(Issue("ERROR", sid, "NO_TEXT", "拼装后的提示词缺少文字约束"))

        # 5) 动态描述必须存在
        if not mp.strip():
            out.append(Issue("ERROR", sid, "MOTION_REQ", "缺少 motion_prompt"))

        # 6) 用完成态描述天气
        if env_is_rainy and has_any(mp, PAST_STATE):
            out.append(Issue("WARN", sid, "TENSE",
                             "动态描述用了完成态（被雨打湿/已湿），容易被理解成雨已停"))

        # 7) 时长
        d = shot.get("duration")
        if not isinstance(d, int) or not (DURATION_RANGE[0] <= d <= DURATION_RANGE[1]):
            out.append(Issue("ERROR", sid, "DURATION",
                             f"duration={d} 超出模型支持区间 {DURATION_RANGE}"))

        # 8) 单镜多主体
        if has_any(both, ("另一个", "两人", "对峙", "面对另一个")):
            out.append(Issue("WARN", sid, "MULTI_SUBJECT",
                             "单镜出现第二个主体，一致性与成功率都会下降"))

        # 8b) 机位没声明 → 模型会自由运镜（首版 S3 就是没锁机位，2 秒内绕到车正后方）
        if mp and not has_any(mp, CAMERA_LOCK + CAMERA_MOVE):
            out.append(Issue("WARN", sid, "CAMERA_LOCK",
                             "动态提示词没有声明机位或运镜，模型会自由环绕/变焦"))

        # 8c) 载具运动镜必须显式排除绕拍，否则"镜头固定"也拦不住（实测）
        if has_any(both, VEHICLE) and "环绕" not in mp and "绕拍" not in mp:
            out.append(Issue("WARN", sid, "NO_ORBIT",
                             "载具运动镜没排除环绕：实测即使写了'镜头固定'，模型仍会绕到车后"))

        # 8d) 否定式写法反效果
        for trap in NEGATION_TRAPS:
            for clause in re.split(r"[。；;，,]", both):
                if trap in clause and has_any(clause, ("不得", "没有", "无", "禁止", "不要")):
                    out.append(Issue("WARN", sid, "NEG_TRAP",
                                     f"用否定词点名「{trap}」会反向强化它，应改结构（换景别/换机位/后期遮挡）"))
                    break

        # 9) 负向词与正向描述冲突（先剔除否定从句，否则"不得出现文字、字母、数字"会自己触发自己）
        positive_clauses = [
            c for c in re.split(r"[。；;，,]", kp)
            if c.strip() and not has_any(c, ("不得", "没有", "禁止", "避免", "不能", "不要"))
        ]
        for tok in neg_tokens & set(positive_clauses):
            out.append(Issue("WARN", sid, "NEG_CONFLICT", f"正向描述里出现了负向词「{tok}」"))

        # 10) 台词时长预算 —— 唯一能在"生成前"拦住音频返工的规则
        #     TTS 按字出音、视频按整秒出片，两者天生对不上。等到合成时才发现念不完，
        #     要么变速毁语气，要么重跑视频（¥0.6-0.75/镜，是 TTS 的 100 倍）。
        #     实测语音语速 4.6 字/秒（Cherry），这里按 4.0 字/秒算预算，留停顿余量。
        from manju_tts import resolve_line, speech_budget
        vtext, from_sub = resolve_line(shot)
        if vtext:
            avail, _ = speech_budget(shot, cfg)
            rate = float((cfg.get("format") or {}).get("speech_rate_chars_per_s", 4.0))
            est = len(vtext) / rate
            if est > avail:
                out.append(Issue("ERROR", sid, "VOICE_BUDGET",
                                 f"台词 {len(vtext)} 字 估 {est:.2f}s，超本镜可用 {avail:.2f}s："
                                 f"删字或加长镜头（加长要重跑视频）"))
            elif est > avail * 0.9:
                out.append(Issue("WARN", sid, "VOICE_BUDGET",
                                 f"台词 {len(vtext)} 字 估 {est:.2f}s，接近上限 {avail:.2f}s，"
                                 f"实际语速偏慢时会被截断"))
            if from_sub:
                out.append(Issue("WARN", sid, "VOICE_FALLBACK",
                                 f"未写 voice_line，回退朗读字幕「{vtext}」："
                                 f"主副标题连读可能重复（如'第47天，辞职做游戏的第47天'）"))
            if re.search(r"[0-9A-Za-z]", vtext):
                out.append(Issue("WARN", sid, "VOICE_NUMERAL",
                                 "台词含阿拉伯数字/拉丁字母，TTS 读法不确定"
                                 "（「第 7 单」可能读'第七单'也可能逐字念），建议改写汉字"))

        # 10) 画外音镜头不许有口部动作
        #     实测（7.23）：S2 是旁白在响，但唇形 55% 的帧在变化、且与音量正相关 r=+0.51，
        #     观众读成"她在说话可口型对不上"。同管线的 S12 唇形 0% 变化——
        #     两者唯一差别是 S2 写了"嘴唇微张"，S12 没写。所以这是写法问题，不是模型能力。
        #     判 WARN 而不是 ERROR：它不必然导致返工，可以接受为"她在低声念"的表演选择，
        #     但必须是有意选的——所以要在生成前就被看见，而不是等观众看出来。
        speaker = str((shot.get("voice_line") or {}).get("as") or "").strip().lower()
        if speaker in ("", "narrator", "unknown", "offscreen", "voiceover", "旁白", "画外音"):
            form = shot.get("form") or {}
            body = str(form.get("action_moment") or "")
            body += "。" + str((form.get("motion") or {}).get("action_flow") or "")
            # 按小句判，必须在同一小句内既有口部词、又没有否定词。
            # 否则「嘴唇没有张开」会被「嘴唇」二字触发——和 NEG_CONFLICT 是同一个坑：
            # 光写字面不看句意，规则就会自己打自己。
            MOUTH = ("嘴唇", "开口", "说话", "出声", "喃喃", "低语", "翕动")
            NEG = ("没有", "未", "不", "别", "禁止", "避免", "不要", "闭合", "闭上", "抿")
            hits = []
            for cl in re.split(r"[。；;，,]", body):
                if not cl.strip():
                    continue
                if any(w in cl for w in MOUTH) and not any(n in cl for n in NEG):
                    hits.append(cl.strip()[:16])
            if hits:
                out.append(Issue("WARN", sid, "VOICE_LIP_MISMATCH",
                                 f"说话人不在镜内（{speaker or '未标注'}）但写了口部动作「{hits[0]}」："
                                 f"唇会跟着动，观众读成'在说话可对不上'。"
                                 f"改法：把口部改成静止（如'嘴唇闭合、嘴角松弛'）"))

        # 12) 抽象物理现象会被画成图形符号
        #     实测：写"热气上升"，模型画出浓白卡通团块，盒底还补一片橙红火光，读起来是"着火"。
        #     和"水花溅开 → 白色同心圆线稿"同源：这类词在训练数据里对应的是符号，不是物理。
        #     要表达热/冷，就写静态的、可画的物理证据（水珠、油光、霜花）。
        body_m = str((form or {}).get("action_moment") or "")
        body_m += str(((form or {}).get("motion") or {}).get("action_flow") or "")
        sym = [w for w in ("热气", "蒸汽", "白烟", "烟雾", "烟柱", "火花") if w in body_m]
        # 带理由的豁免：实测可接受的写法要能留下痕迹，否则门禁会变成长期噪音，
        # 噪音一多就会被整体忽略——那等于没有门禁。
        if sym and not str((form or {}).get("symbol_accepted") or "").strip():
            pass
        elif sym:
            sym = []
        if sym:
            out.append(Issue("WARN", sid, "SYMBOL_MOTION",
                             f"写了抽象物理现象「{'/'.join(sym)}」：模型会画成卡通图形符号"
                             f"（实测「热气」变浓白团块加橙红光，读作着火）。"
                             f"改写成静态证据，如「盒盖内侧的水珠、菜的油光」"))

        # 13) 常识四问 —— 全部来自 EP02 v3 的用户反馈，不是我的推测
        #     「室内下雨」由 RAIN_SCOPE 管；剩下四问在这里。
        #     关键设计：查的是**人对这个空间的描述**（space + 环境卡），不是模型输出。
        #     所以它能在生成前跑、零成本，而不是等出片后靠眼睛发现。
        #     环境卡配齐之后这四条通常全过——它们真正的价值是：
        #     当一个新剧本没配环境卡、又忘了写围合/进深时，立刻报出来。
        _sp = str((form or {}).get("space") or "")
        # 围合信息可能写在三处：镜头自己的 space、环境锁变体的 line、环境卡。
        # 只查 space 会漏掉「作者把围合写在了环境锁里」这种正确写法。
        _vline = str((((cfg.get("environment_lock") or {}).get("variants") or {})
                      .get(_ev) or {}).get("line") or "")
        _whole = _sp + "。" + _vline + "。" + space_canon(cfg, _ev)

        if requires_spatial_scope(cfg, _ev):
            _miss = []
            if not has_any(_whole, ENCL_WALL):
                _miss.append("墙")
            if not has_any(_whole, ENCL_TOP):
                if not has_any(_whole, ENCL_GROUND):
                    _miss.append("顶或地")
            if _miss:
                out.append(Issue("WARN", sid, "SPACE_ENCLOSURE",
                                 f"「{_ev}」的空间描述缺少「{'/'.join(_miss)}」，围合不成立。"
                                 f"EP02 S1 因此出现「门框立在空旷湿地上」——"
                                 f"模型不会替你补围合，它只会把主体放在一片空地上"))

        _surf = [w for w in SURFACE_OBJ if w in _whole]
        if _surf and not has_any(_whole, SURFACE_ANCHOR):
            out.append(Issue("WARN", sid, "SURFACE_ANCHOR",
                             f"描述里有「{'/'.join(_surf)}」却没有任何宿主"
                             f"（墙/门框/窗框/嵌/靠墙）：模型会让它悬空立在空地上。"
                             f"写明「门框嵌在墙体里、上下接严」这类接缝关系"))

        if requires_spatial_scope(cfg, _ev) and not has_any(_whole, DEPTH_WORDS):
            out.append(Issue("WARN", sid, "DEPTH_MARK",
                             "室内空间没有任何尺度词（进深/米/靠墙/紧挨/狭小）："
                             "模型会画出无限进深，EP02 S11 门内因此出现一大片空木地板"))

        # 有朝向语义的实物必须进 props —— 否则拿不到 facing 强制校验。
        # EP02 S3「鞋尖朝向屋内」被画成朝门口，当时鞋只写在 subject 里，
        # 编译器也就没法要求它写 facing，最后只能零成本改字幕兜过去。
        _props = (form or {}).get("props") or []
        _pnames = [str(p.get("name") or "") for p in _props if p.get("name")]
        _subj_txt = str((form or {}).get("subject") or "")
        for _fw in ORIENTED_FALSE:
            _subj_txt = _subj_txt.replace(_fw, "")
        _loose = [
            w for w in ORIENTED_OBJ
            if w in _subj_txt and not any(w in n for n in _pnames)
        ]
        if _loose:
            out.append(Issue("WARN", sid, "PROP_UNSTRUCTURED",
                             f"「{'/'.join(_loose)}」写在了 subject 里但没进 props："
                             f"拿不到 facing 强制校验，朝向只能靠运气。"
                             f"补成 props 项并写明 facing（如「鞋尖朝向门内」）"))

        # 14) 环境卡与镜头描述自相矛盾
        #     环境卡上线当天实测：S6 的 space 写「玻璃茶几」，canon 写「深红木矮茶几」，
        #     模型两处都照做 —— 同一个画面里出现两张茶几。
        #     这类矛盾比任何单点缺陷都糟：提示词越详细，矛盾被画出来的概率越高。
        _cn = space_canon(cfg, _ev)
        if _cn:
            for _obj, _mats in OBJ_MATERIAL.items():
                _a = material_near(_sp, _obj, _mats)
                _b = material_near(_cn, _obj, _mats)
                if _a and _b and _a != _b:
                    out.append(Issue("WARN", sid, "CANON_MATERIAL_CONFLICT",
                                     f"「{_obj}」在本镜空间里是「{_a}」，环境卡里是「{_b}」："
                                     f"模型会两个都画（实测 S6 出现两张茶几）。"
                                     f"统一成环境卡的写法"))
    return out


def selftest(cfg: dict) -> int:
    """回归自检：把 EP00 历史上每一个真实事故，逐条注回配置，看能否拦住。

    规则库没有自检就会慢慢退化成摆设——改着改着就不报了，还没人发现。
    现在主力防线已从 lint 前移到编译器（因为字段缺失比措辞不当更致命），
    所以自检同时覆盖两层：编译器拒编 + 编译期自动补丁。
    """
    import copy
    sys.path.insert(0, str(ROOT / "scripts"))
    from compose_prompt import compile_shot, ComposeError

    def reject(mutate):
        c = copy.deepcopy(cfg)
        mutate(c)
        try:
            compile_shot(c, c["shots"][0] if "shots" in c else c)
            return False, "竟然通过"
        except ComposeError as e:
            return True, str(e)[:46]
        except Exception as e:  # noqa: BLE001
            return False, f"{type(e).__name__}: {e}"[:46]

    def accept_with(mutate, needle):
        c = copy.deepcopy(cfg)
        mutate(c)
        try:
            r = compile_shot(c, c["shots"][2] if "shots" in c else c)
        except Exception as e:  # noqa: BLE001
            return False, f"意外拒编: {e}"[:46]
        hit = (needle in r.positive) or (needle in r.motion) or any(needle in w for w in r.warnings)
        return hit, "已自动补" if hit else "未补"

    def lint_hit(mutate, rule):
        """lint 层规则（编译层拦不到的，比如台词字数）。"""
        c = copy.deepcopy(cfg)
        mutate(c)
        try:
            iss = lint(c)
        except Exception as e:  # noqa: BLE001
            return False, f"{type(e).__name__}: {e}"[:46]
        hits = [i for i in iss if i.rule == rule]
        return bool(hits), (hits[0].msg[:40] if hits else "未报")

    cases = []

    def c1(c): c.pop("environment_lock", None)
    cases.append(("事故①雨只在部分镜头", "删 environment_lock", lambda: reject(c1)))

    def c2(c): c["character_card"].pop("proportion_lock", None)
    cases.append(("事故②头身比怪", "删 proportion_lock", lambda: reject(c2)))

    def c3(c): c["shots"][0]["form"]["props"][0]["facing"] = ""
    cases.append(("事故③屏幕朝镜头", "清空 props.facing", lambda: reject(c3)))

    def c4(c):
        # 事故④的真正病因是「双手各持一物」，不是「画面里有两只手」。
        # R1 放宽到 n>=3 之后，这里必须构造出那句真实的病因才能验证防线还在。
        f = c["shots"][0]["form"]
        f["hands"]["visible_count"] = 2
        f["props"].append({"name": "头盔", "held_by": "左手", "facing": "抱在身前", "in_frame": True})
    cases.append(("事故④两只右手", "双手各持一物", lambda: reject(c4)))

    def c5(c): c["shots"][0]["form"]["negatives"].append("不得出现泪水")
    cases.append(("事故⑤否定词反效果", "negatives 点名泪水", lambda: reject(c5)))

    def c6(c): pass
    cases.append(("事故⑥运镜绕到车后", "载具镜自动补锁定", lambda: accept_with(c6, "不做环绕")))

    def c7(c):
        c["character_card"]["identity_features"] = [
            f for f in c["character_card"]["identity_features"] if f.get("part") != "眼"
        ]
    cases.append(("事故⑦眼部特征丢失", "删形象卡「眼」项", lambda: reject(c7)))

    def c8(c): pass
    cases.append(("事故⑦b 身份未逐镜注入", "检查产物含眼部描述", lambda: accept_with(c8, "单眼皮")))

    # 音频类：这两条只能靠 lint 拦，编译器管不到台词多少
    def c9(c):
        c["shots"][0]["voice_line"]["text"] = (
            "她低头看着手机屏幕上的订单信息，拇指悬在屏幕上方迟迟没有按下去，"
            "雨水顺着发梢一路滴在屏幕上又滑到手腕上。"
        )
    cases.append(("事故⑧台词念不完", "S1 塞 52 字台词", lambda: lint_hit(c9, "VOICE_BUDGET")))

    def c10(c): c["shots"][0].pop("voice_line", None)
    cases.append(("事故⑨字幕连读重复", "删 S1 voice_line", lambda: lint_hit(c10, "VOICE_FALLBACK")))

    def c11(c):
        c["shots"][0]["voice_line"]["as"] = "narrator"
        c["shots"][0]["form"]["action_moment"] = "她低头盯着手机，嘴唇微张，目光发直"
    cases.append(("事故⑩旁白镜嘴在动", "画外音镜写「嘴唇微张」",
                  lambda: lint_hit(c11, "VOICE_LIP_MISMATCH")))

    def c12(c):
        c["environment_lock"]["variants"] = {"interior": {"line": "室内，雨在门外"}}
    cases.append(("事故⑪室内外环境串味", "加变体但镜头不点名", lambda: lint_hit(c12, "ENV_VARIANT")))

    def c13(c):
        c["shots"][0]["form"]["action_moment"] = "盒盖开着，热气一缕缕往上飘"
    cases.append(("事故⑫热气变着火", "写「热气一缕缕往上飘」",
                  lambda: lint_hit(c13, "SYMBOL_MOTION")))

    def c14(c):
        c["environment_lock"]["variants"] = {"interior": {"line": "室内"}}
        for sh in c["shots"]:
            sh["form"]["env_variant"] = "interior"
    cases.append(("事故⑬屋里下雨", "室内镜不写 rain_scope",
                  lambda: lint_hit(c14, "RAIN_SCOPE")))

    # ---- 常识四问（EP02 v3 用户反馈） ----
    def mk_enclosed(c):
        """造一个声明了 enclosed 的室内变体，用来测围合/接缝/进深。"""
        c["environment_lock"]["variants"] = {
            "interior": {"line": "室内，雨被隔在门外",
                         "enclosed": True, "rain_scope_required": True}
        }
        for sh in c["shots"]:
            sh["form"]["env_variant"] = "interior"
            sh["form"]["rain_scope"] = "雨只在窗外，室内干燥"

    def c15(c):
        mk_enclosed(c)
        c["shots"][0]["form"]["space"] = "一间空屋子，什么都没有"
    cases.append(("事故⑭门框立空地", "室内景不写墙/顶/地",
                  lambda: lint_hit(c15, "SPACE_ENCLOSURE")))

    def c16(c):
        mk_enclosed(c)
        c["shots"][0]["form"]["space"] = "一扇防盗门立在画面中央"
    cases.append(("事故⑮门悬空", "写门不写宿主墙",
                  lambda: lint_hit(c16, "SURFACE_ANCHOR")))

    def c17(c):
        mk_enclosed(c)
        c["shots"][0]["form"]["space"] = "一间客厅，摆着沙发和茶几"
    cases.append(("事故⑯无限进深", "室内景无尺度词",
                  lambda: lint_hit(c17, "DEPTH_MARK")))

    def c18(c):
        c["shots"][0]["form"]["subject"] = "地砖上并排放着一双深棕色男式皮鞋"
        c["shots"][0]["form"]["props"] = []
    cases.append(("事故⑰鞋尖朝反", "鞋只写在 subject 没进 props",
                  lambda: lint_hit(c18, "PROP_UNSTRUCTURED")))

    # 环境卡类：同一物件在两处被写成不同材质
    def c19(c):
        mk_enclosed(c)
        c["environment_lock"]["variants"]["interior"]["sheet"] = "lr"
        c["environment_lock"]["sheets"] = {"lr": {"brief": "深红木矮茶几正对布艺沙发"}}
        c["shots"][0]["form"]["space"] = "一间客厅，玻璃茶几"
    cases.append(("事故⑱两张茶几", "space 与环境卡材质冲突",
                  lambda: lint_hit(c19, "CANON_MATERIAL_CONFLICT")))

    print(f"{'历史事故':<24}{'注入方式':<22}{'结果'}")
    print("-" * 78)
    ok = 0
    for name, how, run in cases:
        hit, detail = run()
        ok += hit
        print(f"{name:<24}{how:<22}{'✓ 拦住' if hit else '✗ 漏'}  {detail}")
    print("-" * 78)
    print(f"{ok}/{len(cases)} 通过")
    return 0 if ok == len(cases) else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CFG))
    ap.add_argument("--selftest", action="store_true", help="注入历史缺陷验证规则有效性")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))

    if args.selftest:
        # 自检必须用**固定基线配置**：它测的是"规则库还认不认得历史事故"，
        # 与当前在改哪一集无关。
        # 曾经的写法是直接用 --config 传进来的那份，结果 `--config ch04 --selftest`
        # 拿 ch01 写的事故用例去注 ch04，第一例就因为 S1 没有 voice_line 而 KeyError ——
        # 自检崩了，等于这一整层防线在最需要它的时候不存在。
        base = json.loads(DEFAULT_CFG.read_text(encoding="utf-8"))
        if Path(args.config).resolve() != DEFAULT_CFG.resolve():
            print(f"（自检固定使用基线配置 {DEFAULT_CFG.name}，与 --config 无关）\n")
        sys.exit(selftest(base))

    try:
        issues = lint(cfg)
    except ComposeError as e:
        # 迁移测试暴露：新剧本缺字段时，lint 会把编译异常直接抛成堆栈，
        # 用户看不到"缺什么"，只看到 traceback。
        print(f"✗ 无法检查：该剧本在编译阶段就被拒绝\n  {e}")
        print("  先跑 python scripts/compose_prompt.py --config <剧本> --check")
        sys.exit(2)

    if not issues:
        print("✓ 全部规则通过，可以进入生成阶段")
        return

    n_err = sum(1 for i in issues if i.level == "ERROR")
    n_warn = len(issues) - n_err
    print(f"{'级别':<7}{'镜头':<6}{'规则':<16}说明")
    print("-" * 104)
    for i in sorted(issues, key=lambda x: (x.level != "ERROR", x.shot, x.rule)):
        print(f"{i.level:<7}{i.shot:<6}{i.rule:<16}{i.msg}")
    print("-" * 104)
    print(f"ERROR {n_err} 条，WARN {n_warn} 条")
    # 结构化落盘：让"哪条规则最常报"变成可查询的事实，而不是靠回忆。
    # 只在主流程写（--selftest 不写），否则自检会污染运行日志。
    try:
        from runlog import log as _rlog
        run = f"lint_{Path(args.config).stem}_{time.strftime('%H%M%S')}"
        for i in issues:
            _rlog("lint", i.shot, "fail" if i.level == "ERROR" else "warn",
                  rule=i.rule, reason=i.msg, run=run)
    except Exception:      # noqa: BLE001
        pass
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
