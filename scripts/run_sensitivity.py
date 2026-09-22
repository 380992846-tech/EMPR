"""Week 2 · 敏感性分析（区分"记忆" vs "计算"）。

核心思路：对同一道题生成三类变体，看模型输出对哪类改动敏感。
  - numeric   ：改数值（如质量 2 kg -> 5 kg），物理结构不变
  - swap      ：互换两个物理量的数值（如质量<->劲度系数）
  - wording   ：改表层措辞，物理不变
若模型是"记忆/检索"，对数值扰动应不敏感；若真做计算，输出应按物理规律成比例变化。

⚠ 真实结果该是什么就是什么，不要为凑结论调阈值。

本版本做了三项硬化（LaTeX 感知，适配 SciBench 的 $...$/\\mathrm{...} 题面）：
  1) numeric_perturb 不再误伤下标/上标/科学计数法指数（H_2、x^2、10^{3} 不再被改变、也不会把
     m/s^2 的指数当数值）。
  2) quantity_swap 能识别 \\mathrm{kg}、\\mathrm{~cm}、km / hr、kJ / mol 这类 LaTeX 单位，
     并映射到物理量。
  3) wording_change 用"零语义改动"的改写规则集（换问法/换同义短语，绝不触碰物理量），
     并报告覆盖率——覆盖率低就如实写，不硬凑。

用法：
  python scripts/run_sensitivity.py --model Qwen/Qwen2.5-7B-Instruct \
      --benchmark scibench --max-problems 40 --backend vllm
  python scripts/run_sensitivity.py --coverage --benchmark scibench   # 干跑：只测覆盖率，不调模型

依赖：numpy（本脚本绘图/统计用）。
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src import data as data_mod
from src import llm as llm_mod
from src import meta as meta_mod
from src import prompts as prompts_mod
from src.grading import parse_number, extract_number, judge
from src.meta import write_json

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


# ============================================================
# LaTeX 感知的数值扫描（CORE 硬化）
# ============================================================
def _scan_numbers(text: str):
    """返回可扰动的数值列表 `(start, end, num_str)`。

    排除：
      * `^{...}` / `_{...}` 组、裸 `^n` / `_n`（指数不是物理量）；
      * **幂的底数**：紧挨在 `^` 前面的数（`10^{3}` 的 10）——把 10 乘 2 会得到
        `20^{3}`=8000，物理上并不等于原来数值的 2 倍，属于"改坏了"而不是"改数值"；
      * 紧贴字母/数字/点的数字（`x2`、`m/s^2` 的指数 2、化学下标）；
      * 撇号年份（`'88`）。
    保留科学计数法的尾数：`1.0 \\times 10^{3}` 里 `1.0` 可扰动（乘 2 得到 `2.0 x 10^3`，
    物理上正确），而 `10` 作为底数被保护。
    """
    n = len(text)
    excluded = [False] * n

    def mark(a, b):
        for k in range(a, min(b, n)):
            if k >= 0:
                excluded[k] = True

    def mark_base_of_power(caret_idx):
        """把 `^` 前面的数字底数标记为不可扰动的。"""
        k = caret_idx - 1
        while k >= 0 and text[k] in " \t":
            k -= 1
        end = k + 1
        while k >= 0 and (text[k].isdigit() or text[k] == "."):
            k -= 1
        if end > k + 1:                       # 确实是一串数字
            mark(k + 1, end)

    i = 0
    while i < n:
        c = text[i]
        if c in "^_" and i + 1 < n:
            j = i + 1
            while j < n and text[j] == " ":
                j += 1
            if j < n and text[j] == "{":
                # 配平花括号的 ^{...} / _{...}
                depth = 0
                k = j
                while k < n:
                    if text[k] == "{":
                        depth += 1
                    elif text[k] == "}":
                        depth -= 1
                        if depth == 0:
                            break
                    k += 1
                mark(i, min(k + 1, n))
                mark_base_of_power(i)
                i = min(k + 1, n)
                continue
            elif j < n and (text[j].isdigit() or text[j] in "+-"):
                # 裸指数 ^2 / _2 / ^-3
                k = j
                while k < n and (text[k].isdigit() or text[k] in ".+-"):
                    k += 1
                mark(i, k)
                mark_base_of_power(i)
                i = k
                continue
        i += 1

    res = []
    # 支持千分位逗号（10,000）、小数、科学计数法；值统一去逗号（供 float() 用）
    for m in re.finditer(r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?", text):
        s, e = m.start(), m.end()
        if any(excluded[s:e]):
            continue
        # 紧贴标识符的数字（x2 的 2、.5 之类）不作为独立量
        if s > 0 and (text[s - 1].isalnum() or text[s - 1] in "._"):
            continue
        # 撇号年份（' 88、'19）不作为物理量扰动（允许撇号与数字间有空格）
        k = s - 1
        while k >= 0 and text[k] in " \t":
            k -= 1
        if k >= 0 and text[k] == "'":
            continue
        res.append((s, e, text[s:e].replace(",", "")))
    return res


def numeric_perturb(question: str, factor: float = 2.0, seed: int = 42, max_repl: int = 2):
    """把题干里若干可扰动数值乘以 factor（物理结构不变）。返回 (新题, 被改的原数值列表)。"""
    rng = random.Random(seed)
    nums = _scan_numbers(question)
    if not nums:
        return question, []
    if len(nums) == 1:
        ids = [0]
    else:
        ids = sorted(rng.sample(range(len(nums)), min(max_repl, len(nums))))
    edits = {}
    changed = []
    for idx in ids:
        s, e, val = nums[idx]
        try:
            new = float(val) * factor
        except ValueError:
            continue
        edits[s] = (e, f"{new:.6g}")
        changed.append(val)
    if not edits:
        return question, changed
    # 按原字符串位置重排，避免偏移
    out = []
    prev = 0
    for s in sorted(edits):
        e, new = edits[s]
        out.append(question[prev:s])
        out.append(new)
        prev = e
    out.append(question[prev:])
    return "".join(out), changed


# ============================================================
# LaTeX 感知的单位->物理量 映射（swap 用，CORE 硬化）
# ============================================================
_UNIT_QTY = {
    # 质量
    "kg": "mass", "g": "mass", "mg": "mass", "lb": "mass", "ton": "mass",
    # 长度/尺寸
    "m": "length", "cm": "length", "mm": "length", "km": "length", "pm": "length",
    "nm": "length", "um": "length", "in": "length", "ft": "length", "mile": "length",
    "au": "length",
    # 体积
    "l": "volume", "ml": "volume", "gal": "volume", "m^3": "volume", "cm^3": "volume",
    # 时间
    "s": "time", "hr": "time", "h": "time", "min": "time", "day": "time",
    "ms": "time", "yr": "time", "year": "time",
    # 速度 / 加速度
    "km/hr": "speed", "km/h": "speed", "m/s": "speed", "mph": "speed",
    "m/s^2": "acceleration", "m/s2": "acceleration",
    # 力 / 能 / 功率 / 压强
    "n": "force", "j": "energy", "kj": "energy", "cal": "energy", "kcal": "energy",
    "ev": "energy", "kev": "energy", "mev": "energy", "kwh": "energy",
    "w": "power",
    "pa": "pressure", "kpa": "pressure", "atm": "pressure", "bar": "pressure",
    "torr": "pressure", "mmhg": "pressure",
    # 温度（K 明确；°C 保留，因 'c' 有歧义跳过）
    "k": "temperature", "kelvin": "temperature",
    # 物质的量 / 浓度
    "mol": "amount", "mmol": "amount", "mol/l": "concentration", "molar": "concentration",
    # 其它
    "hz": "frequency", "cpm": "rate",
    "v": "voltage", "ohm": "resistance",
    "rad": "angle", "degree": "angle", "deg": "angle",
}
# 有歧义、不参与 swap 的单位（避免把不同物理量错配）
_AMBIGUOUS_UNITS = {"c", "g", "m", "a", "k", "f", "q", "r", "u", "t"}


def _read_unit(text: str, i: int):
    """从数字后位置 i 读取一个单位（含 LaTeX 宏，支持 `$4000$ m/s` 这种单位在 `$` 外的情况）。

    返回 (unit_str, end)。
    """
    parts = []
    j = i
    n = len(text)
    guard = 0
    while j < n and guard < 60:
        guard += 1
        c = text[j]
        # LaTeX 空格、空白
        if c in " \t~":
            j += 1
            continue
        # $ 定界符 / 逗号：若已读过单位 token，说明单位在 $...$ 里面 -> 停；
        #   否则（还没读到单位）说明单位在 $ 外（如 $4000$ m/s）-> 跳过继续找。
        if c in "$,":
            if parts:
                break
            j += 1
            continue
        m = re.match(r"\\mathrm\s*\{\s*([^}]*?)\s*\}", text[j:])
        if m:
            parts.append(m.group(1).strip())
            j += len(m.group(0))
            continue
        m = re.match(r"\\operatorname\s*\{\s*([^}]*?)\s*\}", text[j:])
        if m:
            parts.append(m.group(1).strip())
            j += len(m.group(0))
            continue
        m = re.match(r"\\([A-Za-z]+)", text[j:])
        if m:
            parts.append(m.group(1))
            j += len(m.group(0))
            continue
        if c in "/^·×":
            parts.append(c)
            j += 1
            continue
        if c.isalpha():
            k = j
            while k < n and (text[k].isalpha() or text[k] in "^~·"):
                k += 1
            parts.append(text[j:k])
            j = k
            continue
        # 数字/{ 表示下一段开始（新数值或 LaTeX 组），停止
        if c in "0123456789{" or c in ".:+-":
            break
        break
    return "".join(parts), j


def _normalize_unit(unit: str) -> str:
    """清洗单位：去 LaTeX 空格 `~`/空白、统一小写。例：'~kg'->'kg'；'km / hr'->'km/hr'。"""
    return re.sub(r"[\s~]+", "", unit.lower())


def _extract_quantity_items(question: str):
    """返回 `[{s,e,val,qty,unit}]`：能明确归到物理量的 '数+单位' 对（带位置，供位置级 swap）。"""
    items = []
    for s, e, val in _scan_numbers(question):
        unit, _end = _read_unit(question, e)
        if not unit:
            continue
        u = _normalize_unit(unit)
        # 复合单位（能量/物质的量 等）：kJ/mol、kcal/mol
        qty = _UNIT_QTY.get(u)
        if qty is None:
            mo = re.match(r"^(kj|kcal|j|ev|kev|mev)\s*/\s*(mol|g|kg|cm\^?3|l)$", u)
            if mo:
                qty = _UNIT_QTY.get(mo.group(1))
        if qty is None:
            continue
        if u in _AMBIGUOUS_UNITS:
            continue
        items.append({"s": s, "e": e, "val": val, "qty": qty, "unit": u})
    return items


def quantity_swap(question: str, seed: int = 42):
    """互换两个不同物理量的数值（保留其余题面）。识别不出两个可换物理量则原样返回。

    用"位置级"交换（而非字符串替换），避免两个数值互为子串时错位。
    """
    rng = random.Random(seed)
    items = _extract_quantity_items(question)
    by_qty = {}
    for it in items:
        by_qty.setdefault(it["qty"], []).append(it)
    qtys = [q for q, occ in by_qty.items() if occ]
    if len(qtys) < 2:
        return question, 0
    a, b = rng.sample(qtys, 2)
    ia, ib = by_qty[a][0], by_qty[b][0]
    edits = {ia["s"]: (ia["e"], ib["val"]), ib["s"]: (ib["e"], ia["val"])}
    out = []
    prev = 0
    for s in sorted(edits):
        e, new = edits[s]
        out.append(question[prev:s])
        out.append(new)
        prev = e
    out.append(question[prev:])
    newq = "".join(out)
    return newq, (2 if newq != question else 0)


# ============================================================
# wording：零语义改动的改写规则集（CORE 硬化）
# ============================================================
_WORDING_RULES = [
    # 问法（换框架词，绝不碰物理量）
    (["what is", "what's"], ["find", "determine", "compute", "calculate"]),
    (["find", "determine", "calculate", "compute", "evaluate"], ["what is", "find", "determine", "calculate"]),
    (["compute", "calculate"], ["determine", "evaluate", "find"]),
    # 常见动词短语
    (["slides down", "slid down", "slides along"], ["slides along", "moves down", "goes down", "slides down"]),
    (["is released from rest", "released from rest", "starts from rest"],
     ["starts from rest", "begins at rest", "is let go from rest", "is released from rest"]),
    (["ignoring air resistance", "neglect air resistance"],
     ["neglecting air resistance", "in the absence of air resistance", "ignoring air resistance"]),
    (["coefficient of kinetic friction"], ["kinetic friction coefficient", "kinetic friction constant"]),
    (["is traveling", "is moving", "travels", "moves"], ["moves", "travels", "is moving", "is traveling"]),
    (["is dropped", "is thrown", "is released"], ["is released", "is dropped", "is projected"]),
    (["at a speed of", "at a velocity of", "with a speed of"], ["at a speed of", "with speed", "moving at"]),
    (["a distance of", "a length of", "a height of"], ["a distance of", "a height of", "a length of"]),
]


def _match_case(original: str, replacement: str) -> str:
    """替换时保留原短语的首字母大小写（否则 "What is ..." 会被改成 "what is ..."）。"""
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def wording_change(question: str, seed: int = 42, max_rules: int = 1):
    """表层措辞改写（物理、数据完全不变）。返回 (新题, changed_bool)。

    只做"框架词/同义短语"替换，绝不触碰数字、单位、物理公式。命中不了就原样返回。
    正式论文建议用更强的一致改写模型，但本规则已保证语义不变。
    """
    rng = random.Random(seed)
    q = question
    changed = False
    # 收集可命中（不区分大小写）的规则，随机选一条执行
    hits = []
    for i, (phrases, alts) in enumerate(_WORDING_RULES):
        for ph in phrases:
            if re.search(re.escape(ph), q, re.I):
                hits.append((i, ph, alts))
                break
    for i, ph, alts in rng.sample(hits, min(max_rules, len(hits))):
        rep = rng.choice(alts)
        newq, cnt = re.subn(re.escape(ph),
                            lambda m: _match_case(m.group(0), rep),
                            q, count=1, flags=re.I)
        if cnt and newq != q:
            # 把疑问句换成祈使句时，句末问号一并改掉（仍属表层改动，不涉及物理）
            if ph.lower().startswith("what") and not rep.lower().startswith("what"):
                newq = re.sub(r"\?\s*$", ".", newq)
            q = newq
            changed = True
        if changed:
            break
    return q, changed


# ============================================================
# 可重用：三路变体构造器 + 覆盖率统计（干跑，无需模型）
# ============================================================
def build_variant(rows, vname, seed=42):
    """对一批题生成某个变体的 (变体题列表, 覆盖数)。"""
    out = []
    for r in rows:
        q = r["question"]
        if vname == "numeric":
            q2, _ = numeric_perturb(q, seed=seed)
        elif vname == "wording":
            q2, _ = wording_change(q, seed=seed)
        elif vname == "swap":
            q2, _ = quantity_swap(q, seed=seed)
        else:
            raise ValueError(vname)
        if q2 != q:
            out.append(dict(r, question=q2, id=r["id"] + f"-{vname}"))
    return out


def report_coverage(rows, seed=42):
    """干跑：报告每个变体改动了多少题（覆盖率），并打印 2 个样例。返回 {variant: 覆盖率}。"""
    total = len(rows)
    print(f"基准共 {total} 题。三个变体的覆盖率（seed={seed}，数值扰动上限=2、factor=2.0）：\n")
    out = {}
    for vname in ("numeric", "swap", "wording"):
        v = build_variant(rows, vname, seed=seed)
        cov = len(v) / total if total else 0.0
        out[vname] = {"n_changed": len(v), "coverage": cov}
        print(f"  {vname:<8} 改动 {len(v):>4}/{total} 题  (覆盖率 {cov:5.1%})")
        for ex in v[:2]:
            print(f"      Q : {ex['question'][:96].replace(chr(10), ' ')}")
            print(f"      id: {ex['id']}")
        print()
    return out


# ============================================================
# 敏感性度量 + 全量跑（有模型）
# ============================================================
def sensitivity_of_output(base_ans, pert_ans):
    """输出相对变化，带保护项防除零。返回 float 或 None（任一无法解析）。"""
    b, p = parse_number(base_ans), parse_number(pert_ans)
    eps = 1e-6
    if b is None or p is None:
        return None
    denom = abs(b) + eps
    return abs(p - b) / denom


# 「变体是否改变了模型的答案」的阈值：相对变化小于它就算"没变"
UNCHANGED_EPS = 1e-6


def _variant_report(vname, items, arr, base_flags, var_flags):
    """汇总一个变体的所有指标（含配对 McNemar）。items 为逐题审计记录。"""
    import numpy as np
    a = np.array(arr) if arr else np.array([float("nan")])
    n_parsed = len(arr)
    unchanged = float((a < UNCHANGED_EPS).mean()) if n_parsed else float("nan")

    # 配对准确率（只有物理答案不变的变体才可算：wording）
    acc_pair = None
    if base_flags and var_flags and len(base_flags) == len(var_flags):
        base_acc = sum(base_flags) / len(base_flags)
        var_acc = sum(var_flags) / len(var_flags)
        from src.stats import paired_significance
        acc_pair = {
            "n_paired": len(base_flags),
            "base_acc": round(base_acc, 4),
            "variant_acc": round(var_acc, 4),
            "delta": round(var_acc - base_acc, 4),
            "mcnemar_p": paired_significance(base_flags, var_flags),
        }
    return {
        "n_changed": len(items),
        "n_parsed": n_parsed,
        "n_unparsed": len(items) - n_parsed,
        "mean_sensitivity": float(a.mean()) if n_parsed else None,
        "median_sensitivity": float(np.median(a)) if n_parsed else None,
        "unchanged_share": unchanged,
        "response_rate": (1.0 - unchanged) if n_parsed else None,
        "accuracy_paired": acc_pair,
    }


def _run_with_model(model, rows, variants, gen, mode, seed, save, meta_extra=None):
    base_prompts = prompts_mod.build_prompts(rows, mode=mode)
    print(f"[sens] base 推理 {len(rows)} 题（{mode}）...")
    base_out = gen.gen_batch(base_prompts)
    if len(base_out) != len(rows):
        raise RuntimeError(f"base 输出条数({len(base_out)}) != 题数({len(rows)})")
    base_by_id = {r["id"]: o for r, o in zip(rows, base_out)}

    cfg = _load_cfg()
    tol, atol = cfg["eval"]["answer_tolerance"], cfg["eval"]["abs_tolerance"]

    summary = {"model": model, "mode": mode, "total_questions": len(rows),
               "seed": seed, "variants": {}}
    if meta_extra:
        summary.update(meta_extra)

    for vname, fn in variants.items():
        v_rows, pairs = [], []
        for r in rows:
            q2, _ = fn(r["question"])
            if q2 != r["question"]:
                vr = dict(r, question=q2, id=r["id"] + f"-{vname}")
                v_rows.append(vr)
                pairs.append(r["id"])
        if not v_rows:
            print(f"[warn] 变体 {vname}: 覆盖率 0，跳过")
            continue
        print(f"[sens] 变体 {vname}: {len(v_rows)} 题需要推理...")
        try:
            v_out = gen.gen_batch(prompts_mod.build_prompts(v_rows, mode=mode))
        except Exception as e:
            # 一个变体挂掉不该让另外两个变体的结果一起丢（长跑无人值守时尤其重要：
            # 云上跑了两小时，回来发现 OOM 把整份汇总都带走了）。记录错误后继续。
            print(f"[error] 变体 {vname} 推理失败: {type(e).__name__}: {e}")
            summary["variants"][vname] = {"error": f"{type(e).__name__}: {e}",
                                          "n_changed": len(v_rows)}
            continue
        if len(v_out) != len(v_rows):
            summary["variants"][vname] = {
                "error": f"输出条数({len(v_out)}) != 题数({len(v_rows)})",
                "n_changed": len(v_rows)}
            print(f"[error] 变体 {vname}: 输出条数({len(v_out)}) != 题数({len(v_rows)})")
            continue

        sens, items = [], []
        base_flags, var_flags = [], []
        n_unparsed = 0
        for vr, po, base_id in zip(v_rows, v_out, pairs):
            bo = base_by_id.get(base_id, "")
            bval, pval = extract_number(bo), extract_number(po)
            s = None
            if bval is None or pval is None:
                n_unparsed += 1
            else:
                denom = abs(bval) + 1e-6
                s = abs(pval - bval) / denom
                sens.append(s)
            items.append({
                "id": vr["id"], "base_id": base_id, "domain": vr["domain"],
                "question_variant": vr["question"],
                "base_pred": bval, "variant_pred": pval, "sensitivity": s,
            })
            # 配对准确率：base 题的参考答案对变体题是否仍然成立
            #   只有 wording（物理/数据完全没变）成立；numeric/swap 的正确答案
            #   已经变了，拿原参考答案判分没有意义，这里不算。
            if vname == "wording":
                ref = next((x["answer"] for x in rows if x["id"] == base_id), None)
                if ref is not None:
                    bc, _ = judge(bo, ref, tol=tol, abs_tol=atol)
                    vc, _ = judge(po, ref, tol=tol, abs_tol=atol)
                    base_flags.append(bool(bc))
                    var_flags.append(bool(vc))

        rep = _variant_report(vname, items, sens,
                              base_flags if vname == "wording" else None,
                              var_flags if vname == "wording" else None)
        summary["variants"][vname] = rep

        print(f"\n===== {model} | {vname} variant =====")
        print(f"  n changed            : {rep['n_changed']} / {len(rows)}"
              f"  (覆盖率 {rep['n_changed'] / max(len(rows), 1):.1%})")
        print(f"  n parsed / unparsed   : {rep['n_parsed']} / {rep['n_unparsed']}")
        print(f"  mean sensitivity      : {rep['mean_sensitivity']:.4f}"
              if rep["mean_sensitivity"] is not None else "  mean sensitivity      : n/a")
        print(f"  median sensitivity    : {rep['median_sensitivity']:.4f}"
              if rep["median_sensitivity"] is not None else "  median sensitivity    : n/a")
        print(f"  share unchanged       : {rep['unchanged_share']:.3f}"
              if rep["unchanged_share"] is not None else "  share unchanged       : n/a")
        if rep["accuracy_paired"]:
            ap = rep["accuracy_paired"]
            print(f"  配对准确率            : base {ap['base_acc']:.3f} -> "
                  f"{vname} {ap['variant_acc']:.3f} (Δ={ap['delta']:+.3f}, "
                  f"McNemar p={ap['mcnemar_p']:.4f})")

        if save:
            f = RESULTS / f"sensitivity_{vname}_{model.replace('/', '__')}.json"
            write_json(f, {"model": model, "variant": vname, "mode": mode,
                           "n": len(v_rows), **rep, "items": items})
            print(f"[save] {f}")

    if save:
        f = RESULTS / f"sensitivity_summary_{model.replace('/', '__')}.json"
        write_json(f, summary)
        print(f"[save] {f}")


def _load_cfg():
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None)
    p.add_argument("--benchmark", default=None)
    p.add_argument("--max-problems", type=int, default=40)
    p.add_argument("--backend", default="auto", choices=["auto", "vllm", "transformers"])
    p.add_argument("--device", default="auto")
    p.add_argument("--mode", default=None, choices=["cot", "direct"])
    p.add_argument("--cot", action="store_true", help="等价 --mode cot")
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--variants", default="numeric,wording,swap",
                   help="逗号分隔，默认三个全跑")
    p.add_argument("--save", action="store_true")
    p.add_argument("--coverage", action="store_true",
                   help="干跑：只测三个变体的覆盖率，不调用模型")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    cfg = _load_cfg()
    eval_cfg = cfg.get("eval", {})
    seed = args.seed if args.seed is not None else eval_cfg.get("seed", 42)
    benchmark = args.benchmark or cfg.get("defaults", {}).get("benchmark", "scibench")
    mode = args.mode or ("cot" if args.cot else cfg.get("defaults", {}).get("mode", "cot"))
    max_new_tokens = args.max_new_tokens or eval_cfg.get("max_new_tokens", 1024)

    rows = data_mod.load_benchmark(benchmark, max_problems=args.max_problems)
    print(f"[data] {benchmark}: {len(rows)} 题 | {data_mod.format_composition(rows)}")
    if args.coverage:
        rep = report_coverage(rows, seed=seed)
        if args.save:
            f = RESULTS / f"sensitivity_coverage_{benchmark}.json"
            write_json(f, {"meta": meta_mod.result_meta(
                "run_sensitivity.py --coverage", benchmark=benchmark, seed=seed,
                n_problems=len(rows)), "benchmark": benchmark, "n_problems": len(rows),
                "seed": seed, "coverage": rep})
            print(f"[save] {f}")
        return

    model = args.model or cfg.get("defaults", {}).get("model") or cfg["models"]["core"][0]
    gen = llm_mod.make_generator(model, backend=args.backend, seed=seed,
                                 max_new_tokens=max_new_tokens, device=args.device,
                                 batch_size=args.batch_size)

    # 传入只接收 question 的包装函数（保持与覆盖率干跑一致）
    all_variants = {
        "numeric": lambda q: numeric_perturb(q, seed=seed),
        "wording": lambda q: wording_change(q, seed=seed),
        "swap": lambda q: quantity_swap(q, seed=seed),
    }
    want = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in want if v not in all_variants]
    if unknown:
        raise SystemExit(f"未知变体 {unknown}，可选 {list(all_variants)}")
    variants = {k: all_variants[k] for k in want}

    _run_with_model(model, rows, variants, gen, mode, seed, args.save, meta_extra={
        "meta": meta_mod.result_meta("run_sensitivity.py", model=model,
                                     benchmark=benchmark, mode=mode,
                                     backend=getattr(gen, "backend", args.backend),
                                     max_new_tokens=max_new_tokens,
                                     n_problems=len(rows)),
        "benchmark": benchmark,
        "max_new_tokens": max_new_tokens,
        "data_composition": data_mod.composition(rows),
    })


if __name__ == "__main__":
    main()
