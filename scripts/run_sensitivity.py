"""Week 2 · 敏感性分析（区分"记忆" vs "计算"）。

核心思路：对同一道题生成三类变体，看模型输出对哪类改动敏感。
  - numeric   ：改数值（如质量 2 kg -> 5 kg），物理结构不变
  - swap      ：互换两个物理量（如质量<->劲度系数）[Week2 中段实现]
  - wording   ：改表层措辞，物理不变
若模型是"记忆/检索"，对数值扰动应不敏感；若真做计算，输出应按物理规律成比例变化。

注意：真实结果该是什么就是什么，不要为凑结论调阈值。

用法：
  python scripts/run_sensitivity.py --model Qwen/Qwen2.5-7B-Instruct \
      --benchmark scibench --max-problems 40 --backend vllm
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src import data as data_mod
from src import llm as llm_mod
from src import prompts as prompts_mod
from src.grading import parse_number, extract_number

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# 常见物理量（用于 swap 变体；Week2 中段实现）
_QUANTITIES = ["mass", "kg", "m", "s", "N", "m/s", "m/s^2", "friction",
               "coefficient", "velocity", "force", "spring", "temperature", "charge"]

_NUM_RE = re.compile(r"(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")


def numeric_perturb(question: str, factor: float = 2.0, seed: int = 42, max_repl: int = 2):
    """把题干里的若干数值乘以 factor（物理结构不变）。返回 (新题, 原始数列表)。"""
    import random
    rng = random.Random(seed)
    nums = _NUM_RE.findall(question)
    if not nums:
        return question, []
    # 随机挑最多 max_repl 个数值进行扰动
    idxs = set(rng.sample(range(len(nums)), min(max_repl, len(nums))))
    seen = 0
    def fn(m):
        nonlocal seen
        cur = seen
        seen += 1
        if cur in idxs:
            return f"{float(m.group()) * factor:.6g}"
        return m.group()
    new_q = _NUM_RE.sub(fn, question)
    return new_q, nums


# 常见"数量单位->物理量"映射（用于 quantity_swap 识别哪些数是哪个物理量）
_UNIT_QUANTITY = {
    "kg": "mass", "g": "mass", "m": "length", "cm": "length",
    "m/s": "velocity", "m s^-1": "velocity", "ms^-1": "velocity",
    "m/s^2": "acceleration", "m/s2": "acceleration",
    "n": "force", "n/m": "spring_constant", "n/m^2": "pressure",
    "k": "spring_constant",
    "degree": "angle", "deg": "angle", "rad": "angle",
    "j": "energy", "w": "power", "t": "time", "s": "time",
    "c": "charge", "v": "voltage", "a": "current",
    "hz": "frequency", "celsius": "temp_celsius",
}


def wording_change(question: str):
    """表层措辞改写（物理完全不变）。

    ⚠ 这是确定性规则版，用于快速验证"措辞改动应几乎不敏感"。
      正式论文建议用同义改写模型（或模板化题面），确保改后物理含义与数据完全一致。
    """
    import random
    rng = random.Random(42)
    q = question
    rules = [
        ("slides down", ["slides along", "moves down", "travels down"]),
        ("is released from rest", ["starts from rest", "begins at rest", "is let go from rest"]),
        ("What is its velocity", ["Find its velocity", "Compute the velocity", "Determine its velocity"]),
        ("coefficient of kinetic friction", ["kinetic friction coefficient", "kinetic friction constant"]),
        ("ignoring air resistance", ["neglecting air resistance", "in the absence of air resistance"]),
        ("Calculate", ["Compute", "Find", "Determine"]),
    ]
    for phrase, alts in rules:
        idx = q.lower().find(phrase.lower())
        if idx >= 0:
            rep = rng.choice(alts)
            q = q[:idx] + rep + q[idx + len(phrase):]
    return q


def quantity_swap(question: str, seed: int = 42):
    """互换两个物理量的数值（如质量<->速度），保留题面其余部分。

    启发式：把 "数 单位" 按单位归类到物理量，挑两个不同物理量交换其数值。
    ⚠ 对自由文本题面较脆弱，正确率有限——Week2 需按实际题库调优，
      或改用"模板化题面"以保证能可靠识别物理量。
    """
    import random
    import re
    rng = random.Random(seed)
    pat = re.compile(r"(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*([A-Za-z][A-Za-z0-9_^/]*)")
    groups = {}
    for m in pat.finditer(question):
        val, unit = m.group(1), m.group(2).lower()
        qty = _UNIT_QUANTITY.get(unit)
        if qty is None:
            continue
        groups.setdefault(qty, []).append((m.start(), m.end(), val))
    qtys = [q for q, occ in groups.items() if occ]
    if len(qtys) < 2:
        return question  # 识别不出两个可换物理量，原样返回
    a, b = rng.sample(qtys, 2)
    (pa, _, va), (pb, _, vb) = groups[a][0], groups[b][0]
    # 先换位置靠后的，避免偏移
    if pa < pb:
        q = question[:pa] + vb + question[pa + len(va):]
        q = q[:pb + (len(vb) - len(va))] + va + q[pb + (len(vb) - len(va)) + len(vb):]
    else:
        q = question[:pb] + va + question[pb + len(vb):]
        q = q[:pa + (len(va) - len(vb))] + vb + q[pa + (len(va) - len(vb)) + len(va):]
    return q


def sensitivity_of_output(base_out, pert_out, base_ans, pert_ans):
    """输出相对变化，带保护项防除零。返回 float。"""
    b, p = parse_number(base_ans), parse_number(pert_ans)
    eps = 1e-6
    if b is None or p is None:
        return None
    denom = abs(b) + eps
    return abs(p - b) / denom


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None)
    p.add_argument("--benchmark", default="scibench")
    p.add_argument("--max-problems", type=int, default=40)
    p.add_argument("--backend", default="auto")
    p.add_argument("--cot", action="store_true")
    p.add_argument("--save", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    model = args.model or cfg.get("defaults", {}).get("model") or cfg["models"]["core"][0]
    rows = data_mod.load_benchmark(args.benchmark, max_problems=args.max_problems)
    gen = llm_mod.make_generator(model, backend=args.backend,
                                 seed=cfg["eval"]["seed"])
    mode = "cot" if args.cot else "direct"

    # 三路变体：数值扰动 / 表层措辞 / 物理量互换
    variants = {
        "numeric": lambda q: numeric_perturb(q, factor=2.0)[0],
        "wording": wording_change,
        "swap": quantity_swap,
    }
    print(f"[data] {len(rows)} problems; generating base outputs...")
    base_prompts = prompts_mod.build_prompts(rows, mode=mode)
    base_out = gen.gen_batch(base_prompts)

    import numpy as np
    summary = {"model": model, "variants": {}}
    for vname, fn in variants.items():
        v_rows = []
        for r in rows:
            q2 = fn(r["question"])
            if q2 != r["question"]:
                v_rows.append(dict(r, question=q2, id=r["id"] + f"-{vname}"))
        if not v_rows:
            print(f"[warn] 变体 {vname}: 没有生成任何改动（题面不适配），跳过")
            continue
        v_out = gen.gen_batch(prompts_mod.build_prompts(v_rows, mode=mode))
        sens = []
        for r, bo, po in zip(v_rows, base_out, v_out):
            bval, pval = extract_number(bo), extract_number(po)
            s = sensitivity_of_output(bo, po,
                                      str(bval) if bval else "",
                                      str(pval) if pval else "")
            sens.append(s if s is not None else 0.0)
        arr = np.array(sens)
        mean_sens = float(arr.mean())
        unchanged = float((arr < 1e-6).mean())
        print(f"\n===== {model} | {vname} variant =====")
        print(f"  n changed           : {len(v_rows)}")
        print(f"  mean sensitivity    : {mean_sens:.4f}")
        print(f"  median sensitivity  : {float(np.median(arr)):.4f}")
        print(f"  share of unchanged  : {unchanged:.3f}")
        summary["variants"][vname] = {
            "n_changed": len(v_rows),
            "mean_sensitivity": mean_sens,
            "median_sensitivity": float(np.median(arr)),
            "unchanged_share": unchanged,
        }
        if args.save:
            f = RESULTS / f"sensitivity_{vname}_{model.replace('/', '__')}.json"
            f.write_text(json.dumps({"model": model, "variant": vname,
                                     "n": len(v_rows),
                                     "mean_sensitivity": mean_sens,
                                     "unchanged_share": unchanged},
                                    ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[save] {f}")

    if args.save:
        f = RESULTS / f"sensitivity_summary_{model.replace('/', '__')}.json"
        f.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[save] {f}")


if __name__ == "__main__":
    main()
