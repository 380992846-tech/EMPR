"""反事实稳定性探针：结构性提示能否让模型对"纯表层改写"更稳健？

背景：Week 2 发现 wording 变体（物理与数据一字不改、只换措辞）会改变 24%-34% 的答案，
而整体正确率几乎不变 —— 说明这些翻转是对称的错误抖动，而非"改写让题变难"。
自然的工程追问是：**推理时的结构性脚手架能不能压住这种不稳定？**

设计（有统计功效的版本）：
  对全部 wording 覆盖题（~240）跑两种提示词，比较同一个变体题面下的表现：
    (a) plain      —— 与 Week 2 相同的 direct 提示
    (b) structured —— 加一句 "Before solving, first write down the knowns and
                       unknowns explicitly."
  指标：
    * 变体正确率（两种提示），配对 McNemar
    * 相对原题答案的**稳定性**：变体答案是否等于原题答案（对/错都算稳定）
    * broke 子集（原题对、plain 下变体错）在 structured 下的救回数
      —— 该子集天生很小（因为错误是抖动而非集中在固定题目），因此只作描述性报告

⚠ 若 structured 不能提升变体正确率、也不能提升稳定性，则结论是：
   这类不稳定不是"缺少解题脚手架"，提示工程补不回来 —— 这是能力层面的限制。

用法：
  python scripts/run_repair_probe.py --model Qwen/Qwen2.5-7B-Instruct \
      --wording results/sensitivity_wording_Qwen2.5-7B-Instruct.json \
      --eval results/eval_scibench_direct_Qwen2.5-7B-Instruct.json \
      --backend transformers --device cuda --save
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml

from src import llm as llm_mod
from src import meta as meta_mod
from src import prompts as prompts_mod
from src.grading import extract_number, parse_number
from src.meta import write_json
from src.stats import paired_significance

RESULTS = ROOT / "results"


def _load(path):
    p = Path(path)
    if not p.exists():
        p = RESULTS / Path(path).name
    return json.loads(p.read_text(encoding="utf-8"))


def _correct(pred, ref, tol, atol):
    """已提取的数值 -> 是否判对（口径与 src.grading.judge 一致）。"""
    rv = parse_number(ref)
    if rv is None or pred is None:
        return False
    if abs(rv) <= atol:
        return abs(pred - rv) <= atol
    return abs(pred - rv) / abs(rv) <= tol


def _same_answer(a, b, tol=0.05, atol=0.01):
    """两个已提取答案是否"等价"（用于衡量答案稳定性，而不是对错）。"""
    if a is None or b is None:
        return a is None and b is None
    if abs(a) <= atol and abs(b) <= atol:
        return True
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom <= tol


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None)
    p.add_argument("--wording", required=True, help="sensitivity_wording_*.json")
    p.add_argument("--eval", required=True, help="同模型 eval json（取参考答案）")
    p.add_argument("--backend", default="auto", choices=["auto", "vllm", "transformers"])
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--max-items", type=int, default=400)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--save", action="store_true")
    p.add_argument("--out-dir", default=str(RESULTS))
    args = p.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    ec = cfg.get("eval", {})
    seed = args.seed if args.seed is not None else ec.get("seed", 42)
    tol, atol = ec.get("answer_tolerance", 0.05), ec.get("abs_tolerance", 0.01)
    model = args.model or cfg.get("defaults", {}).get("model")

    word = _load(args.wording)
    ev = _load(args.eval)
    refs = {d["id"]: d["ref"] for d in ev["detail"]}

    items = []
    for it in word.get("items", []):
        ref = refs.get(it.get("base_id"))
        if ref is None:
            continue
        items.append({"base_id": it["base_id"], "ref": ref,
                      "question_variant": it["question_variant"],
                      "base_pred": it.get("base_pred"),
                      "domain": it.get("domain")})
    items = items[: args.max_items]
    n = len(items)
    print(f"[probe] wording 覆盖 {len(word.get('items', []))} 题，可用 {n} 题")
    if n == 0:
        print("[probe] 无可用样本")
        return 2

    gen = llm_mod.make_generator(model, backend=args.backend, seed=seed,
                                 max_new_tokens=args.max_new_tokens,
                                 device=args.device, batch_size=args.batch_size)
    plain_prompts = [prompts_mod.build_prompt(i["question_variant"], mode="direct") for i in items]
    struct_prompts = [prompts_mod.build_prompt(i["question_variant"], mode="structured") for i in items]
    print(f"[probe] plain 推理 {n} 条 ...")
    plain_out = gen.gen_batch(plain_prompts, batch_size=args.batch_size)
    print(f"[probe] structured 推理 {n} 条 ...")
    struct_out = gen.gen_batch(struct_prompts, batch_size=args.batch_size)
    if len(plain_out) != n or len(struct_out) != n:
        raise RuntimeError("输出条数与样本数不一致，拒绝继续")

    plain_flags, struct_flags = [], []
    plain_stable, struct_stable = [], []
    n_broke = n_fixed = n_worsened = 0
    detail = []
    for it, po, so in zip(items, plain_out, struct_out):
        pv, sv = extract_number(po), extract_number(so)
        pok, sok = _correct(pv, it["ref"], tol, atol), _correct(sv, it["ref"], tol, atol)
        pstable = _same_answer(pv, it["base_pred"], tol, atol)
        sstable = _same_answer(sv, it["base_pred"], tol, atol)
        plain_flags.append(pok)
        struct_flags.append(sok)
        plain_stable.append(pstable)
        struct_stable.append(sstable)
        if not pok:
            n_broke += 1
        if not pok and sok:
            n_fixed += 1
        if pok and not sok:
            n_worsened += 1
        detail.append({"id": it["base_id"], "ref": it["ref"],
                       "base_pred": it["base_pred"],
                       "plain_pred": pv, "plain_correct": pok, "plain_stable": pstable,
                       "struct_pred": sv, "struct_correct": sok, "struct_stable": sstable})

    def rate(flags):
        return sum(flags) / n

    p_mcnemar = paired_significance(plain_flags, struct_flags)
    p_stable = paired_significance(plain_stable, struct_stable)
    res = {
        "n_items": n,
        "plain_acc": round(rate(plain_flags), 4),
        "struct_acc": round(rate(struct_flags), 4),
        "delta_acc": round(rate(struct_flags) - rate(plain_flags), 4),
        "mcnemar_p_acc": None if p_mcnemar != p_mcnemar else p_mcnemar,
        "plain_stable_rate": round(rate(plain_stable), 4),
        "struct_stable_rate": round(rate(struct_stable), 4),
        "mcnemar_p_stable": None if p_stable != p_stable else p_stable,
        "n_plain_wrong": n_broke, "n_fixed_by_structure": n_fixed,
        "n_worsened_by_structure": n_worsened,
    }
    print(f"\n===== 反事实稳定性探针（wording 变体，n={n}）=====")
    print(f"  plain      (direct)        变体正确率: {res['plain_acc']:.3f}")
    print(f"  structured (已知/未知)      变体正确率: {res['struct_acc']:.3f}  "
          f"(Δ={res['delta_acc']:+.3f}, McNemar p={res['mcnemar_p_acc']})")
    print(f"  答案稳定性 vs 原题答案      : plain {res['plain_stable_rate']:.3f} -> "
          f"structured {res['struct_stable_rate']:.3f}  (McNemar p={res['mcnemar_p_stable']})")
    print(f"  救回 / 弄坏                 : {n_fixed} / {n_worsened}")

    f = Path(args.out_dir) / f"repair_probe_{meta_mod.model_slug(model)}.json"
    if args.save:
        write_json(f, {
            **res,
            "model": model,
            "meta": meta_mod.result_meta(
                "run_repair_probe.py", model=model,
                backend=getattr(gen, "backend", args.backend),
                max_new_tokens=args.max_new_tokens, n_items=n,
                prompt_fingerprint_structured=meta_mod.fingerprint(
                    prompts_mod.build_prompt("x", mode="structured"))),
            "items": detail,
        })
        print(f"[save] {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
