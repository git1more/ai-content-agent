# -*- coding: utf-8 -*-
"""
漫剧最小闭环编排器。

五个阶段：
  character  角色定妆三视图（形象卡）
  keyframe   逐镜关键帧（以形象卡为参考，锁 seed）
  video      关键帧 → 视频片段（wan2.6-i2v-flash）
  voice      剧本台词 → 逐镜配音（qwen3-tts-flash，见 manju_tts.py）
  assemble   拼接 + 字幕 + 音轨 → 成片（见 manju_assemble.py，独立执行）

设计取舍（写清楚，避免后面误以为是偷懒）：
  1. 关键帧阶段承担全部"画面质量"，视频阶段只负责"让它动"。
     这是漫剧的成本结构决定的：视频按秒计费且废片率高，图片按张且便宜。
  2. 画面里不要求模型渲染任何文字。字幕一律后期确定性合成。
     模型渲染中文的失败率极高，这是唯一"一眼废"的硬伤，不赌。
  3. 角色一致性靠"形象卡参考图 + 结构化特征清单"双约束，不靠形容词堆砌。
  4. 人声和画面分开生成。画面模型不出声：wan2.6 的 audio=True 生成的是环境音/音效，
     不能念指定台词，还把 720P 从 ¥0.15/s 抬到 ¥0.30/s。配音走独立 TTS，
     好处是三重的 —— 台词能改、能按角色分轨、成本低两个数量级（一集台词 ≈ ¥0.004）。
  5. video 阶段默认并发 4。理由：视频接口限流 RPM=300、单条要 80-190 秒，
     瓶颈是单条延迟而不是配额；12 镜串行 33 分钟，并发 4 约 9 分钟，并发 12 约 3-4 分钟。
     成本完全不变（按秒计费与并发无关），所以这里没有任何理由串行。

用法：
  python scripts/manju_run.py --stage character
  python scripts/manju_run.py --stage keyframe
  python scripts/manju_run.py --stage video --shots S1
  python scripts/manju_run.py --stage video --concurrency 12
  python scripts/manju_run.py --stage voice
  python scripts/manju_run.py --stage all
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from qwen_image_client import generate as gen_image  # noqa: E402
from compose_prompt import compile_shot, ComposeError  # noqa: E402
from qwen_video_client import generate_video  # noqa: E402

CFG_PATH = ROOT / "config" / "manju_ch01.json"
OUT = ROOT / "assets" / "manju" / "ch01"
MANIFEST = OUT / "manifest.json"
COST = OUT / "cost.csv"

SEED_BASE = 20261117


def set_episode(cfg_path: str | Path) -> None:
    """按配置文件名切换输出目录，实现单剧集脚本复用（迁移测试暴露的硬编码）。"""
    global CFG_PATH, OUT, MANIFEST, COST
    CFG_PATH = Path(cfg_path)
    stem = CFG_PATH.stem.replace("manju_", "")
    OUT = ROOT / "assets" / "manju" / stem
    MANIFEST = OUT / "manifest.json"
    COST = OUT / "cost.csv"


def load_cfg() -> dict[str, Any]:
    return json.loads(CFG_PATH.read_text(encoding="utf-8"))


def load_manifest() -> dict[str, Any]:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {"schema_version": "1.0", "character": {}, "environments": {},
            "keyframes": {}, "clips": {}}


def save_manifest(m: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")


def log_cost(stage: str, item: str, model: str, unit: str, qty: float, cny: float) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    new = not COST.exists()
    with COST.open("a", encoding="utf-8") as f:
        if new:
            f.write("ts,stage,item,model,unit,qty,cost_cny\n")
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')},{stage},{item},{model},{unit},{qty},{cny:.4f}\n")
    # 成本事件同时也是"这次调用成功了"的事件，顺手落结构化日志。
    # 失败走 stage_video 的 except 分支单独记 fail——这样"成了什么/为什么没成"在同一个流里。
    try:
        from runlog import log as _rlog
        _rlog(stage, item, "ok", cost=cny)
    except Exception:      # noqa: BLE001
        pass


def feature_brief(card: dict[str, Any]) -> str:
    """把结构化特征压成一句可读的描述，喂给图片模型。"""
    crit = [f["desc"] for f in card["identity_features"] if f["level"] == "critical"]
    major = [f["desc"] for f in card["identity_features"] if f["level"] == "major"]
    parts = [f"{card['age']} 岁中国女性", "；".join(crit)]
    if major:
        parts.append("；".join(major))
    return "，".join(parts)


def build_keyframe_prompt(cfg: dict[str, Any], shot: dict[str, Any]) -> str:
    """关键帧提示词：有 form 走编译器，没有则回退到旧字段。

    编译器保证三件事：排序（主体→动作→场景→光线→镜头→风格→画质）、
    三层锁（风格/环境/比例）逐镜注入、构图减法强制（visible_count>=2 直接拒编）。
    """
    if shot.get("form"):
        return compile_shot(cfg, shot).positive
    return shot.get("keyframe_prompt", "")


def build_motion_prompt(cfg: dict[str, Any], shot: dict[str, Any]) -> str:
    if shot.get("form"):
        return compile_shot(cfg, shot).motion
    return shot.get("motion_prompt", "")


def negative_of(cfg: dict[str, Any], shot: dict[str, Any] | None = None) -> str:
    """负向：有 form 用编译器合并结果（含当场风险项），否则回退三层锁拼接。"""
    if shot is not None and shot.get("form"):
        return compile_shot(cfg, shot).negative
    parts = [
        cfg["style_lock"]["negative"],
        cfg.get("environment_lock", {}).get("negative", ""),
        cfg["character_card"].get("proportion_lock", {}).get("negative", ""),
    ]
    seen, out = set(), []
    for p in parts:
        for token in (p or "").split("，"):
            t = token.strip()
            if t and t not in seen:
                seen.add(t)
                out.append(t)
    return "，".join(out)


# --------------------------------------------------------------------------- #
# 阶段 1：角色定妆三视图
# --------------------------------------------------------------------------- #
def stage_character(cfg: dict[str, Any], force: bool = False) -> str:
    m = load_manifest()
    dst = OUT / "character_sheet.png"
    if dst.exists() and not force:
        print(f"[skip] 定妆图已存在 {dst.name}")
        return str(dst)

    print("[character] 生成角色三视图设定稿 ...")
    r = gen_image(
        cfg["character_card"]["sheet_prompt"],
        model="qwen-image-3.0",
        size=cfg["format"]["sheet_size"],
        out_dir=OUT,
        tag="character_sheet",
        negative_prompt=negative_of(cfg),
        prompt_extend=False,
        seed=SEED_BASE,
    )
    src = Path(r.images[0])
    src.replace(dst)
    m["character"] = {
        "id": cfg["character_card"]["id"],
        "name": cfg["character_card"]["name"],
        "sheet": str(dst.relative_to(ROOT)),
        "sheet_url": _first_url(r.raw),
        "seed": SEED_BASE,
        "cost_cny": 0.18,
        "elapsed_s": r.elapsed_s,
    }
    save_manifest(m)
    log_cost("character", "LX_sheet", "qwen-image-3.0", "image", 1, 0.18)
    print(f"  -> {dst.relative_to(ROOT)}  ({dst.stat().st_size/1024:.0f} KB, {r.elapsed_s}s)")
    return str(dst)


def _first_url(raw: dict[str, Any]) -> str | None:
    for ch in (raw.get("output") or {}).get("choices", []):
        for it in ch.get("message", {}).get("content", []):
            if isinstance(it, dict) and it.get("image"):
                return it["image"]
    return None


# --------------------------------------------------------------------------- #
# 阶段 1b：环境定妆稿（环境卡）
# --------------------------------------------------------------------------- #
def _env_sheets(cfg: dict[str, Any]) -> dict[str, Any]:
    return (cfg.get("environment_lock") or {}).get("sheets") or {}


def _env_sheet_path(cfg: dict[str, Any], key: str) -> Path:
    """环境卡文件位置。支持跨集复用：同一间屋子在下一集重画会漂，必须指回原图。"""
    sheet = _env_sheets(cfg).get(key) or {}
    reuse = str(sheet.get("reuse_from") or "").strip()
    if reuse:
        return ROOT / reuse
    return OUT / f"env_{key}.png"


def resolve_env_sheet(cfg: dict[str, Any], shot: dict[str, Any]) -> Path | None:
    """本镜落在哪个空间 → 对应的环境卡文件。没定义环境卡就返回 None。

    只对「有室内结构」的镜头有意义：室外的巷子每一镜都是同一个空镜，
    围合关系不成立，也没有需要跨镜锁定的陈设。
    """
    vkey = str((shot.get("form") or {}).get("env_variant") or "").strip()
    if not vkey:
        return None
    var = ((cfg.get("environment_lock") or {}).get("variants") or {}).get(vkey) or {}
    key = str(var.get("sheet") or "").strip()
    if not key:
        return None
    p = _env_sheet_path(cfg, key)
    return p if p.exists() else None


def stage_environment(cfg: dict[str, Any], force: bool = False,
                      only: list[str] | None = None) -> dict[str, str]:
    """为每个空间生成一张环境定妆稿。

    与角色卡同构：一张图 + 一段结构化清单。图负责**质感与配色**，
    文字负责**方位与数量**——后者比图更精确，所以永远是主力，图只是基调。

    刻意不做四机位拼图：拼图喂给参考图模型，模型会试图画成一张拼图。
    其他角度靠 canon 里的方位词（"南墙/东墙/沙发正前方"）表达。
    """
    m = load_manifest()
    sheets = _env_sheets(cfg)
    if not sheets:
        print("[environment] 本集未定义环境卡（environment_lock.sheets），跳过")
        return {}

    out: dict[str, str] = {}
    for key, sheet in sheets.items():
        if only and key not in only:
            continue
        dst = OUT / f"env_{key}.png"
        reuse = str(sheet.get("reuse_from") or "").strip()
        if reuse:
            src_reuse = ROOT / reuse
            if src_reuse.exists():
                print(f"[skip] 环境卡 {key} 复用 {reuse}")
                out[key] = str(src_reuse)
                continue
            raise SystemExit(f"环境卡 {key} 声明复用 {reuse}，但文件不存在")
        if dst.exists() and not force:
            print(f"[skip] 环境卡 {key} 已存在")
            out[key] = str(dst)
            continue
        prompt = str(sheet.get("prompt") or "").strip()
        if not prompt:
            raise SystemExit(f"环境卡 {key} 缺少 prompt")
        print(f"[environment] 生成环境卡 {key}（{sheet.get('name','')}）...")
        r = gen_image(
            prompt,
            model="qwen-image-3.0",
            size=sheet.get("size") or cfg["format"]["sheet_size"],
            out_dir=OUT,
            tag=f"env_{key}",
            negative_prompt=negative_of(cfg),
            prompt_extend=False,
            seed=SEED_BASE,
        )
        src = Path(r.images[0])
        src.replace(dst)
        m.setdefault("environments", {})[key] = {
            "name": sheet.get("name", key),
            "sheet": str(dst.relative_to(ROOT)),
            "sheet_url": _first_url(r.raw),
            "seed": SEED_BASE,
            "cost_cny": 0.18,
            "elapsed_s": r.elapsed_s,
        }
        save_manifest(m)
        log_cost("environment", key, "qwen-image-3.0", "image", 1, 0.18)
        print(f"  -> {dst.relative_to(ROOT)}  ({dst.stat().st_size/1024:.0f} KB, {r.elapsed_s}s)")
        out[key] = str(dst)
    return out


# --------------------------------------------------------------------------- #
# 阶段 2：逐镜关键帧
# --------------------------------------------------------------------------- #
def stage_keyframe(cfg: dict[str, Any], only: list[str] | None = None,
                   force: bool = False, candidates: int = 1) -> dict[str, str]:
    m = load_manifest()
    sheet = OUT / "character_sheet.png"
    if not sheet.exists():
        sys.exit("先跑 --stage character")

    out: dict[str, str] = {}
    for i, shot in enumerate(cfg["shots"]):
        sid = shot["id"]
        if only and sid not in only:
            continue
        dst = OUT / f"{sid}_keyframe.png"
        if dst.exists() and not force:
            print(f"[skip] {sid} 关键帧已存在")
            out[sid] = str(dst)
            continue

        prompt = build_keyframe_prompt(cfg, shot)
        base_seed = SEED_BASE + (i + 1) * 101
        # 参考图：形象卡永远给；环境卡按开关给。
        # 双参考图有实测风险——编辑模型可能把两张图的语义混起来（人长在沙发上）。
        # 所以默认关闭，靠 environment_lock.inject_ref_image 显式打开，
        # 且打开后必须核对角色一致性有没有掉。文本 canon 才是主力。
        refs: list[Path] = [sheet]
        env_sheet = resolve_env_sheet(cfg, shot)
        if env_sheet is not None and (cfg.get("environment_lock") or {}).get("inject_ref_image"):
            refs.append(env_sheet)
        print(f"[keyframe] {sid} {shot['shot_size']} "
              f"{f'（{candidates} 选 1）' if candidates > 1 else ''}"
              f"{'  [图+环境卡]' if len(refs) > 1 else ''} ...")

        if candidates <= 1:
            r = gen_image(
                prompt, ref_images=refs, model="qwen-image-3.0",
                size=cfg["format"]["keyframe_size"], out_dir=OUT,
                tag=f"{sid}_keyframe", negative_prompt=negative_of(cfg, shot),
                prompt_extend=False, seed=base_seed,
            )
            src = Path(r.images[0])
            src.replace(dst)
            source_url = _first_url(r.raw)
            picked = {"candidate": 0, "seed": base_seed, "auto_check": None}
        else:
            # 多候选：每张不同 seed，VLM 逐张判，取第一张全过的；
            # 都不过则取问题最少的，并标记 needs_review
            from manju_vlm_check import judge as vlm_judge
            best_path, best_fails, best_seed, best_url, best_idx = None, None, base_seed, None, -1
            fail_log: list[list[str]] = []
            for c in range(candidates):
                seed = base_seed + c * 7919
                r = gen_image(
                    prompt, ref_images=refs, model="qwen-image-3.0",
                    size=cfg["format"]["keyframe_size"], out_dir=OUT,
                    tag=f"{sid}_keyframe_c{c}", negative_prompt=negative_of(cfg, shot),
                    prompt_extend=False, seed=seed,
                )
                cand = Path(r.images[0])
                res = vlm_judge(cand, cfg, shot)
                print(f"    候选{c}：{'✓ 通过' if res['ok'] else '✗ ' + '；'.join(res['fails'])[:60]}")
                if res["ok"]:
                    best_path, best_fails, best_seed, best_url, best_idx = cand, [], seed, _first_url(r.raw), c
                    break
                fail_log.append(res["fails"])
                if best_fails is None or len(res["fails"]) < len(best_fails):
                    best_path, best_fails, best_seed, best_url, best_idx = cand, res["fails"], seed, _first_url(r.raw), c
            # 未中选的候选清理掉，只留被选中的
            for c in range(candidates):
                leftover = OUT / f"{sid}_keyframe_c{c}.png"
                if leftover.exists() and leftover != best_path:
                    leftover.unlink()
            best_path.replace(dst)
            source_url = best_url
            picked = {"candidate": best_idx, "seed": best_seed,
                      "auto_check": {"passed": not best_fails, "fails": best_fails}}
            # 区分「随机缺陷」与「系统性缺陷」：
            # 若 N 张候选都在同一项失败，说明不是运气问题，重试无意义，应改配置或提示词。
            from collections import Counter
            if len(fail_log) >= 2:
                cnt = Counter(f.split("：")[0] for lst in fail_log for f in lst)
                sys_fails = [k for k, v in cnt.items() if v == len(fail_log)]
                if sys_fails:
                    print(f"    ⚠ 系统性缺陷（每张候选都出现，重试无效）：{', '.join(sys_fails)}")
                    picked["systematic_fails"] = sys_fails
        m.setdefault("keyframes", {})[sid] = {
            "path": str(dst.relative_to(ROOT)),
            "remote_url": source_url,
            "seed": picked["seed"],
            "pick": picked,
            "prompt": prompt,
            "cost_cny": 0.18,
            "elapsed_s": r.elapsed_s,
        }
        save_manifest(m)
        log_cost("keyframe", sid, "qwen-image-3.0", "image", 1, 0.18)
        print(f"  -> {dst.name} ({r.elapsed_s}s)")
        out[sid] = str(dst)
    return out


# --------------------------------------------------------------------------- #
# 阶段 3：图生视频
# --------------------------------------------------------------------------- #
def stage_video(cfg: dict[str, Any], only: list[str] | None = None,
                force: bool = False, concurrency: int = 4) -> dict[str, str]:
    """视频阶段。每个镜头是一次「提交 + 轮询」，彼此独立，可以并发。

    视频接口限流是 RPM=300（官方文档，wan2.6-i2v-flash），单条生成要 80-190 秒，
    所以瓶颈从来不是配额而是单条延迟——并发是这个阶段唯一有效的加速手段。
    12 镜串行约 33 分钟，并发 4 约 9 分钟，并发 12 约 3-4 分钟。
    """
    m = load_manifest()
    vdir = OUT / "clips"
    vdir.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}

    jobs: list[tuple[int, dict, str, Path, Path]] = []
    for i, shot in enumerate(cfg["shots"]):
        sid = shot["id"]
        if only and sid not in only:
            continue
        kf = OUT / f"{sid}_keyframe.png"
        if not kf.exists():
            sys.exit(f"{sid} 关键帧缺失，先跑 --stage keyframe")
        dst = vdir / f"{sid}.mp4"
        if dst.exists() and not force:
            print(f"[skip] {sid} 视频已存在")
            out[sid] = str(dst)
            continue
        jobs.append((i, shot, sid, kf, dst))

    if not jobs:
        return out

    # 共享状态：manifest 是整文件读改写、cost.csv 是追加写，两者都必须串行化
    lock = threading.Lock()
    failed: list[tuple[str, str]] = []
    each_s: list[float] = []          # 每镜实际耗时，用来算真实加速比（不用估算值）

    def run_one(job: tuple[int, dict, str, Path, Path]) -> tuple[str, Any]:
        i, shot, sid, kf, dst = job
        r = generate_video(
            kf,
            build_motion_prompt(cfg, shot),
            dst,
            model=cfg["cost_notes"]["video_model"],
            resolution=cfg["format"]["video_resolution"],
            duration=shot["duration"],
            audio=False,
            negative_prompt=negative_of(cfg, shot),
            seed=SEED_BASE + (i + 1) * 101,
        )
        with lock:
            m.setdefault("clips", {})[sid] = {
                "path": str(dst.relative_to(ROOT)),
                "task_id": r.task_id,
                "duration_s": r.duration_s,
                "elapsed_s": r.elapsed_s,
                "cost_cny": r.cost_cny,
            }
            save_manifest(m)
            log_cost("video", sid, cfg["cost_notes"]["video_model"],
                     "second", r.duration_s, r.cost_cny)
            each_s.append(r.elapsed_s)
            print(f"  ✓ {sid}  {r.duration_s:.0f}s  ¥{r.cost_cny:.2f}  ({r.elapsed_s}s)")
        return sid, r

    t0 = time.time()
    print(f"[video] {len(jobs)} 镜待生成，并发 {max(1, concurrency)}")
    if concurrency <= 1:
        for job in jobs:
            sid = job[2]
            print(f"[video] {sid} {job[1]['shot_size']} {job[1]['duration']}s ...")
            try:
                run_one(job)
                out[sid] = str(job[4])
            except Exception as e:  # noqa: BLE001
                failed.append((sid, str(e)[:200]))
                try:
                    from runlog import log as _rlog
                    _rlog("video", sid, "fail", reason=str(e)[:200])
                except Exception:  # noqa: BLE001
                    pass
                print(f"  ✗ {sid} 失败：{str(e)[:160]}")
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = {ex.submit(run_one, job): job for job in jobs}
            for fut in as_completed(futs):
                sid = futs[fut][2]
                try:
                    fut.result()
                    out[sid] = str(futs[fut][4])
                except Exception as e:  # noqa: BLE001
                    failed.append((sid, str(e)[:200]))
                    try:
                        from runlog import log as _rlog
                        _rlog("video", sid, "fail", reason=str(e)[:200])
                    except Exception:  # noqa: BLE001
                        pass
                    print(f"  ✗ {sid} 失败：{str(e)[:160]}")

    wall = time.time() - t0
    if jobs:
        serial = sum(each_s)
        speed = f"，加速 {serial / wall:.2f}×" if wall > 0 and serial > 0 else ""
        print(f"[video] 完成 {len(out)}/{len(jobs)} 镜，墙钟 {wall:.1f}s"
              f"（各镜耗时之和 {serial:.1f}s{speed}）")
    if failed:
        print(f"\n⚠ {len(failed)} 镜失败：")
        for sid, msg in failed:
            print(f"  {sid}: {msg}")
        print("  重跑未完成镜头：python scripts/manju_run.py --stage video --shots "
              + " ".join(sid for sid, _ in failed))
    return out


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["character", "environment", "keyframe", "video", "voice", "all"])
    ap.add_argument("--shots", nargs="*", help="只跑指定镜头，如 S1 S2")
    ap.add_argument("--envs", nargs="*", help="只跑指定环境卡，如 living_room")
    ap.add_argument("--force", action="store_true", help="覆盖已存在产物")
    ap.add_argument("--config", default=None, help="剧本配置路径，如 config/manju_ch02.json")
    ap.add_argument("--candidates", type=int, default=1,
                    help="每镜生成几张备选，用 VLM 自动挑选（默认 1，建议 3）")
    ap.add_argument("--concurrency", type=int, default=4,
                    help="视频阶段并发数（默认 4）。接口限流 RPM=300，12 镜可开到 12；"
                         "串行调试用 1")
    args = ap.parse_args()

    if args.config:
        set_episode(args.config)
    cfg = load_cfg()
    if args.stage in ("character", "all"):
        stage_character(cfg, force=args.force)
    if args.stage in ("environment", "all"):
        stage_environment(cfg, force=args.force, only=args.envs)
    if args.stage in ("keyframe", "all"):
        stage_keyframe(cfg, only=args.shots, force=args.force, candidates=args.candidates)
    if args.stage in ("video", "all"):
        stage_video(cfg, only=args.shots, force=args.force, concurrency=args.concurrency)
    if args.stage in ("voice", "all"):
        import manju_tts
        manju_tts.set_episode(CFG_PATH)
        manju_tts.run(cfg, only=args.shots, force=args.force)


if __name__ == "__main__":
    main()
