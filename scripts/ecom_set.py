# -*- coding: utf-8 -*-
"""电商套图产能：一个 SKU → 12 张成套图（主图 5 + 场景 3 + 详情 4）。

与 P0 单图验证的区别：
  P0 验的是"一张图能不能复刻准"，这里验的是"一整套能不能稳定产出"。
  指标换成三个：一套一次通过率、单套成本、单套耗时。

链路：
  identity_sheet.json（身份特征）+ prompt_locks.json（身份锁定段）
    → 12 个槽位模板（config/ecom_set.json）
    → qwen-image-3.0 生成（母图作参考）
    → 主图过 postprocess_canvas 合成白底画布（统一尺寸/主体占比）
    → VLM 质检（完整性 / 乱码文字 / 主体占比）
    → 汇总报告

用法：
  python scripts/ecom_set.py --sku TS-WHITE-01
  python scripts/ecom_set.py --sku TS-WHITE-01 --slots M1 S1 D1
  python scripts/ecom_set.py --sku TS-WHITE-01 --dry-run
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from fit_white import measure as measure_subject, fit as fit_white
from qwen_image_client import generate  # noqa: E402
from manju_vlm_check import ask as vlm_ask, _parse_json  # noqa: E402

AD = ROOT / "assets" / "p0_testset"
SET_CFG = ROOT / "config" / "ecom_set.json"
LOCKS_CFG = ROOT / "config" / "prompt_locks.json"
IDENT = AD / "identity_sheet.json"
OUT_ROOT = ROOT / "assets" / "ecom_set"

MODEL = "qwen-image-3.0"
CNY_PER_IMAGE = 0.18
GAP_S = 12.5  # RPM=5 的保守间隔

# 文字口径不能写死：T 恤上不该有字，精华瓶的标签文字却是商品本体。
# 2026-11-18 实测：玻璃瓶整套 12 张里 11 张被判「可辨认文字」，实际标签是清晰的
# "VITAMIN C 15%"，没有一张是乱码——规则错，不是图错。
CHECK_PROMPT = """你在做电商商品图质检。只依据实际看到的回答，输出 JSON。
1. 商品主体是否完整（没有被裁掉关键部位、没有断裂、没有变形、没有出现两个或多个同类商品）？填 true/false。
2. {text_q}
只输出 JSON，键名依次是 "intact"、"text"，不要解释，不要 markdown。"""

DEFAULT_TEXT_Q = ("除了商品本身自带的设计图案（例如印花、织标、五金压印），"
                  "画面里还有没有其他可辨认的文字、字母、数字、水印？有就填 true。")

# 曾用「面积占比」判主体大小，已废弃：提示词约束的是「高度占比」，两者不是同一单位。
# 细长的瓶子按 65% 高度摆放时面积只有 0.20 左右，会被面积口径误判成「主体偏小」。
# 现在改为按槽位在 config 里写 height_range，并直接量包围盒。


def _load() -> tuple[dict, dict, dict]:
    setcfg = json.loads(SET_CFG.read_text(encoding="utf-8"))
    locks = json.loads(LOCKS_CFG.read_text(encoding="utf-8"))
    ident = json.loads(IDENT.read_text(encoding="utf-8"))
    return setcfg, locks, ident


def _sku_meta(ident: dict, sku: str) -> dict:
    for s in ident["skus"]:
        if s["sku"] == sku:
            return s
    raise SystemExit(f"identity_sheet 里没有 SKU：{sku}")


def safe_name(label: str) -> str:
    """Windows 文件名不接受斜杠、冒号、问号等字符，槽位标签由人填写，必须清洗。"""
    return re.sub(r'[\\/:*?"<>|]', '_', label).strip()


def locks_for(lock_seg: str | dict, local_only: bool) -> str:
    """身份锁定段按槽位类型取用。

    一条 locks 里其实混了两种东西：
      · **属性**——「砂红膏体、斜切面、1.5cm 香槟金拉丝环」。任何景别都成立，必须逐格注入。
      · **构图**——「两件平行并排、底端对齐、间距约 1.5 倍管径」。只在全貌镜头成立。

    微距槽位（M4/D1/D2，标了 `skip_intact`）要求「完整轮廓不入画」，
    与构图段直接冲突。实测 LS-BLACK-06：M4/D1/D2 三张全部退化成正面整体图，
    因为模型收到的信号是「两个并列部件、底端对齐」——**两处都很清楚，它就两处都照做**
    （与漫剧 S6「两张茶几」同一条规律）。

    字符串写法 = 全量注入，向后兼容。
    """
    if isinstance(lock_seg, str):
        return lock_seg
    ident = str(lock_seg.get("identity") or "")
    comp = str(lock_seg.get("composition") or "")
    return ident if local_only else (ident + ("；" + comp if comp else ""))


def build_prompt(tpl: str, product: str, locks: str, profile: dict | None = None,
                 local_only: bool = False) -> str:
    """填槽位模板。品类词由 profile 提供，缺键直接报错。

    v1 的模板把品类词汇写死在槽位里（M3「衣服的背面」、D2「缝线密度」），
    套到玻璃瓶上必然跑偏。现在槽位只写「拍什么意图」，品类词走 profile。
    这里刻意做成「缺键就报错」而不是「留空」——留空会让模型自由发挥，
    报错才会在花钱之前停下来。

    `local_only`（该槽位标了 skip_intact）时，locks 只注入属性段，不注入构图段。
    """
    s = tpl.replace("{product}", product).replace("{locks}", locks_for(locks, local_only))
    for k, v in (profile or {}).items():
        if not isinstance(v, str):
            continue          # excluded_slots 之类的配置字段不是提示词片段
        s = s.replace("{" + k + "}", v)
    left = sorted(set(re.findall(r"\{([a-z_]{2,})\}", s)))
    if left:
        raise SystemExit(f"提示词里有未填充的品类占位符：{left}；检查 category_profiles 是否缺键")
    return s


def profile_of(setcfg: dict, locks: dict, sku: str, product: str) -> tuple[str, dict]:
    """取该 SKU 的品类分型。没指定就报错，不猜。"""
    name = (locks["skus"].get(sku) or {}).get("set_profile")
    if not name:
        raise SystemExit(f"{sku} 未在 prompt_locks.json 指定 set_profile（可选："
                         f"{list(setcfg.get('category_profiles', {}))}）")
    prof = setcfg.get("category_profiles", {}).get(name)
    if prof is None:
        raise SystemExit(f"category_profiles 里没有分型：{name}")
    return name, {k: (v.replace("{product}", product) if isinstance(v, str) else v)
                  for k, v in prof.items() if not k.startswith("_")}


def _ask_once(path: Path, prompt: str) -> dict:
    return _parse_json(vlm_ask(path, prompt, max_tokens=240)) or {}


def _majority(path: Path, prompt: str, keys: tuple[str, ...]) -> tuple[dict, str]:
    """两票一致则采信，不一致补第三票取多数。

    为什么需要：同一张图两次判定会不一样（JK 外套首判 12/12，重判 10/12）。
    temperature=0 并不保证服务端完全确定。既然「自动通过率」要当作产能指标用，
    这个数字就必须可复现——否则它只是在报告噪声。
    成本：每次质检约 ¥0.003，多数票后约 ¥0.007，可以忽略。
    """
    a = _ask_once(path, prompt)
    b = _ask_once(path, prompt)
    merged = {}
    need3 = False
    for k in keys:
        if a.get(k) == b.get(k):
            merged[k] = a.get(k)
        else:
            need3 = True
    if need3:
        c = _ask_once(path, prompt)
        for k in keys:
            if k in merged:
                continue
            cnt = Counter([a.get(k), b.get(k), c.get(k)])
            merged[k] = cnt.most_common(1)[0][0]
    return merged, ""


def check_image(path: Path, group: str = "主图",
                height_range: list | None = None, verify: str | None = None,
                text_q: str | None = None, skip_intact: bool = False) -> dict:
    try:
        prompt = CHECK_PROMPT.replace("{text_q}", text_q or DEFAULT_TEXT_Q)
        keys = ["intact", "text"]
        if verify:
            prompt = prompt.replace("只输出 JSON", f'4. {verify} 填 true/false。\n只输出 JSON')
            prompt = prompt.replace('"intact"、"text"', '"intact"、"text"、"extra"')
            keys.append("extra")
        obj, _ = _majority(path, prompt, tuple(keys))
        intact = obj.get("intact")
        text = obj.get("text")
        fails, warns = [], []
        if intact is not True and not skip_intact:
            fails.append(f"主体完整：{intact}")
        # 文字项只作提示，不判失败。
        # 实测它不可靠：同一张图在 v2.3 与 v2.7 两次判定相反；同一枚抽象印花
        # 让 T 恤 6 张图全部被判「有文字」。把它当硬指标，通过率就只是在报噪声。
        # 真正该做的是与母图比对或 OCR，那是后续能力，不能用一句是非题替代。
        if text is not False:
            warns.append(f"疑似有文字，需人工复核：{text}")
        if verify:
            extra = obj.get("extra")
            if extra is not True:
                fails.append(f"槽位专属检查未通过：{extra}")

        # 主体尺度不交给 VLM：它答的是「面积占比」，而提示词约束的是「占多大」。
        # 细长的瓶子按 65% 高度摆好时面积只有 0.20 左右，用面积口径会误判成「偏小」。
        # 这里直接量包围盒，取最长边，量出来的单位和提示词一致。
        m = measure_subject(path)
        sc = m.get("scale_ratio") if m.get("ok") else None
        if height_range and sc is not None:
            lo, hi = height_range
            if not (lo <= sc <= hi):
                fails.append(f"主体尺度 {sc:.2f} 超出 {lo}–{hi}")
        return {"ok": not fails, "intact": intact, "text": text,
                "scale_ratio": sc, "height_ratio": m.get("height_ratio"),
                "area_ratio": m.get("area_ratio"),
                "fails": fails, "warns": warns}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "fails": [f"质检失败 {type(e).__name__}: {e}"], "raw": None}


def compose_white(src: Path, spec: str, out: Path) -> tuple[bool, str]:
    """主图走画布合成：统一尺寸、主体占比、白底纯度校验。

    两级降级：cv2 版（能处理复杂背景）→ fit_white（纯 PIL，前提是源图白底）。
    之前 cv2 装不上就原图照搬，结果主体占比完全不受控——服饰靠模型自觉能做到 85%，
    面霜罐只有 40%。占比是规格问题，不能靠模型碰运气。
    """
    cmd = [sys.executable, str(ROOT / "scripts" / "postprocess_canvas.py"),
           "--input", str(src), "--out", str(out), "--spec", spec]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if r.returncode == 0:
        return True, "postprocess_canvas(cv2)"
    try:
        res = fit_white(src, out, (1200, 1200), 0.70)
        if res.get("ok"):
            return True, f"fit_white(高度{res['height_ratio']:.0%})"
        return False, f"两级合成均失败：{res.get('why')}"
    except Exception as e:  # noqa: BLE001
        tail = (r.stdout or r.stderr).strip().splitlines()
        return False, f"cv2 版失败（{tail[-1][:50] if tail else '无输出'}），fit_white 异常 {e}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sku", required=True)
    ap.add_argument("--slots", nargs="*", help="只跑指定槽位，如 M1 S1 D1")
    ap.add_argument("--dry-run", action="store_true", help="只拼提示词，不调 API")
    ap.add_argument("--force", action="store_true", help="覆盖已存在产物")
    ap.add_argument("--recheck", action="store_true", help="只重判已有图，不重新生成")
    ap.add_argument("--restate", action="store_true",
                    help="只按当前配置重算报告的排除项与分母（不调 API、不重新判定）")
    args = ap.parse_args()

    setcfg, locks, ident = _load()
    meta = _sku_meta(ident, args.sku)
    if args.sku not in locks["skus"]:
        raise SystemExit(f"prompt_locks 里没有 SKU：{args.sku}")
    product = locks["skus"][args.sku]["product"]
    lock_seg = locks["skus"][args.sku]["locks"]
    prof_name, profile = profile_of(setcfg, locks, args.sku, product)

    ref = AD / args.sku / "ref_front.png"
    if not ref.exists():
        raise SystemExit(f"缺母图：{ref}")

    outdir = OUT_ROOT / args.sku
    outdir.mkdir(parents=True, exist_ok=True)

    all_slots = setcfg["slots"]
    # 排除分两种，都必须记进报告，否则通过率会变成"数着玩的"：
    #   ① slot.enabled=false      —— 全局排除（当前没有）
    #   ② profile.excluded_slots  —— 按分型排除
    # ② 才是正确的粒度：M3「背面与反侧」在服饰上 2/2 全过（平面前后差异明确），
    # 在瓶罐/罐体上 2/2 全挂（圆柱体模型没有前后概念，画出来仍是正面）。全局砍会白丢服饰的有效产出。
    prof_excl = (profile.get("excluded_slots") or {})
    excluded: list[dict] = []
    slots: list[dict] = []
    for s in all_slots:
        if not s.get("enabled", True):
            excluded.append({"id": s["id"], "label": s["label"],
                             "reason": s.get("excluded_reason", "")})
        elif s["id"] in prof_excl:
            excluded.append({"id": s["id"], "label": s["label"],
                             "reason": prof_excl[s["id"]]})
        else:
            slots.append(s)

    # --restate：只按新配置重算报告的分母，不调 API、不重新判定。
    # 存在的理由：排除规则一改，旧报告的分母就过期了；而 --recheck 会重新问 VLM，
    # 既花钱又会把判定抖动混进历史数据。改口径和重测是两件事，命令也应该分开。
    if args.restate:
        rp = outdir / "set_report.json"
        if not rp.exists():
            raise SystemExit(f"没有报告可重算：{rp}")
        rep = json.loads(rp.read_text(encoding="utf-8"))
        exc_ids = {e["id"] for e in excluded}
        scored = [r for r in rep.get("rows", []) if r["id"] not in exc_ids]
        rep["excluded"] = excluded
        rep["slots"] = len(scored)
        rep["generated"] = len(scored)
        rep["passed"] = sum(1 for r in scored if (r.get("check") or {}).get("ok"))
        rep["pass_rate"] = round(rep["passed"] / len(scored), 3) if scored else None
        rp.write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{args.sku}: {rep['passed']}/{rep['slots']} = {rep['pass_rate']:.0%}"
              f"（排除 {', '.join(sorted(exc_ids)) or '无'}）")
        return

    if args.slots:
        # 显式点名要跑，就允许跑排除项（用于复核"是不是真的做不到"）。
        # 只把「被点名的排除项」从排除列表里摘掉，其余照旧——
        # 否则跑一次 M4 就会让 M3 重新计入分母，报告前后不自洽。
        slots = [s for s in all_slots if s["id"] in args.slots]
        named = {s["id"] for s in slots}
        excluded = [e for e in excluded if e["id"] not in named]
    if not slots:
        raise SystemExit("没有匹配的槽位")

    print(f"SKU {args.sku} · {meta['name']}")
    print(f"母图 {ref.relative_to(ROOT)}　分型 {prof_name}")
    print(f"槽位 {len(slots)} 个：{', '.join(s['id'] for s in slots)}")
    if excluded and not args.slots:
        for s in excluded:
            print(f"  已排除 {s['id']} {s['label']}：{s.get('reason', '')}")
    print()

    if args.dry_run:
        for s in slots:
            p = build_prompt(s["prompt"], product, lock_seg, profile,
                              local_only=bool(s.get("skip_intact")))
            print(f"[{s['id']}] {s['label']} · {s['size']} · spec={s['spec']}")
            print(f"      {p[:150]}...\n")
        return

    t0 = time.time()
    rows = []
    for i, s in enumerate(slots):
        sid = s["id"]
        dst = outdir / f"{sid}_{safe_name(s['label'])}.png"
        if args.recheck:
            if not dst.exists():
                print(f"[skip] {sid} 不存在，跳过")
                continue
            chk = check_image(dst, s["group"], s.get("scale_range"), s.get("verify"),
                             profile.get("text_question"), s.get("skip_intact", False))
            print(f"[{sid}] {s['group']} · {s['label']} -> "
                  f"{'✓' if chk['ok'] else '✗ ' + '；'.join(chk['fails'])}")
            rows.append({"id": sid, "group": s["group"], "label": s["label"],
                         "path": str(dst.relative_to(ROOT)),
                         "check": {k: chk.get(k) for k in ("ok", "intact", "text", "scale_ratio", "height_ratio", "area_ratio", "fails", "warns")}})
            continue
        if dst.exists() and not args.force:
            # 已存在也要质检，否则整份报告会有空档
            chk = check_image(dst, s["group"], s.get("scale_range"), s.get("verify"),
                             profile.get("text_question"), s.get("skip_intact", False))
            print(f"[skip] {sid} 已存在 -> {'✓' if chk['ok'] else '✗ ' + '；'.join(chk['fails'])}")
            rows.append({"id": sid, "group": s["group"], "label": s["label"],
                         "path": str(dst.relative_to(ROOT)), "reused": True,
                         "check": {k: chk.get(k) for k in ("ok", "intact", "text", "scale_ratio", "height_ratio", "area_ratio", "fails", "warns")}})
            continue

        if i:
            time.sleep(GAP_S)

        prompt = build_prompt(s["prompt"], product, lock_seg, profile,
                                local_only=bool(s.get("skip_intact")))
        print(f"[{sid}] {s['group']} · {s['label']} ...", flush=True)
        try:
            r = generate(prompt, ref_images=[ref], model=MODEL, size=s["size"],
                         out_dir=outdir, tag=sid, negative_prompt=setcfg["negative"],
                         prompt_extend=False)
        except Exception as e:  # noqa: BLE001
            print(f"    生成失败 {type(e).__name__}: {e}")
            rows.append({"id": sid, "group": s["group"], "error": str(e)[:80]})
            continue

        src = Path(r.images[0])
        note = ""
        if s["spec"]:
            ok, msg = compose_white(src, s["spec"], dst)
            if not ok:
                src.replace(dst)
                note = f"画布合成失败，保留原图（{msg}）"
            else:
                if src.exists() and src != dst:
                    src.unlink()
        else:
            fs = s.get("fit_spec")
            if fs:
                try:
                    res = fit_white(src, dst, tuple(fs["size"]), fs["ratio"])
                    if not res.get("ok"):
                        src.replace(dst)
                        note = f"归一化失败，保留原图（{res.get('why')}）"
                    elif src.exists():
                        src.unlink()
                except Exception as e:  # noqa: BLE001
                    src.replace(dst)
                    note = f"归一化异常，保留原图（{e}）"
            else:
                src.replace(dst)

        chk = check_image(dst, s["group"], s.get("scale_range"), s.get("verify"),
                             profile.get("text_question"), s.get("skip_intact", False))
        print(f"    {'✓' if chk['ok'] else '✗'} "
              f"{'；'.join(chk['fails']) if chk['fails'] else '通过'}"
              f"{' | ' + note if note else ''}")
        rows.append({
            "id": sid, "group": s["group"], "label": s["label"],
            "path": str(dst.relative_to(ROOT)),
            "check": {k: chk.get(k) for k in ("ok", "intact", "text", "scale_ratio", "height_ratio", "area_ratio", "fails", "warns")},
            "note": note or None,
        })

    elapsed = time.time() - t0
    ok_n = sum(1 for r in rows if r.get("check", {}).get("ok"))
    done = sum(1 for r in rows if r.get("check"))
    cost = done * CNY_PER_IMAGE
    report = {
        "sku": args.sku, "name": meta["name"],
        "profile": prof_name, "category": meta.get("category"),
        "slots": len(slots), "generated": done, "passed": ok_n,
        "pass_rate": round(ok_n / done, 3) if done else None,
        "cost_cny": round(cost, 2),
        "elapsed_s": round(elapsed, 1),
        "seconds_per_image": round(elapsed / done, 1) if done else None,
        "excluded": excluded,
        "rows": rows,
    }
    # 按槽位合并，不整体覆盖。
    # 之前用 --slots 跑单格会把整套 12 条的记录冲成 1 条，
    # 汇总表因此失真，而且看不出是什么时候丢的。
    rp = outdir / "set_report.json"
    if rp.exists():
        try:
            prev = json.loads(rp.read_text(encoding="utf-8"))
            merged = {r["id"]: r for r in prev.get("rows", [])}
            merged.update({r["id"]: r for r in rows})
            rows = [merged[k] for k in sorted(merged)]
        except Exception:  # noqa: BLE001
            pass

    # 分母始终按「当前配置的排除列表」重算，不沿用旧快照。
    # 被排除的槽位数据仍留在 rows 里（可追溯），但不计入通过率——
    # 否则改一次配置就要重跑一次全量才能让报告自洽。
    exc_ids = {e["id"] for e in excluded}
    scored = [r for r in rows if r["id"] not in exc_ids]
    report["rows"] = rows
    report["slots"] = len(scored)
    report["generated"] = len(scored)
    report["passed"] = sum(1 for r in scored if (r.get("check") or {}).get("ok"))
    report["pass_rate"] = round(report["passed"] / len(scored), 3) if scored else None
    report["excluded"] = excluded
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n" + "-" * 60)
    print(f"一套 {len(scored)} 张计分（另有 {len(exc_ids)} 张按分型排除）· "
          f"一次通过 {report['passed']} 张（{report['pass_rate']:.0%}）"
          if scored else "无产出")
    print(f"成本 ¥{cost:.2f} · 耗时 {elapsed:.0f}s（{report['seconds_per_image']}s/张）")
    print(f"报告 {outdir.relative_to(ROOT)}/set_report.json")
    sys.exit(0 if done and ok_n == done else 1)


if __name__ == "__main__":
    main()
