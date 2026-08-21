"""Week 1 · 评估管道主入口。

跑某个基准上的若干开放模型，输出每模型×每子域的正确率表（含 bootstrap 95% CI）。

用法示例：
  python scripts/run_eval.py --model Qwen/Qwen2.5-14B-Instruct \
      --benchmark scibench --cot --max-problems 100 --backend vllm
  python scripts/run_eval.py --models-all --benchmark scibench     # 跑 config 里 core+large 全部
  python scripts/run_eval.py --benchmark scibench                   # 用 config defaults.model

输出：results/eval_<benchmark>_<model>.json  +  终端表格。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src import data as data_mod
from src import llm as llm_mod
from src import prompts as prompts_mod
from src.grading import judge
from src.stats import summarize_accuracy

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def load_config():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    return cfg


def run_model(model_name, rows, cfg, backend, mode, max_new_tokens, batch_size):
    """对一个模型跑完整评估，返回每行 (question, pred, ref, correct)。"""
    gen = llm_mod.make_generator(
        model_name, backend=backend, seed=cfg["eval"]["seed"],
        max_new_tokens=max_new_tokens)
    prompts = prompts_mod.build_prompts(rows, mode=mode)
    t0 = time.time()
    outputs = gen.gen_batch(prompts, batch_size=batch_size)
    dt = time.time() - t0

    tol = cfg["eval"]["answer_tolerance"]
    atol = cfg["eval"]["abs_tolerance"]
    detail = []
    for r, pred in zip(rows, outputs):
        correct, pred_val = judge(pred, r["answer"], tol=tol, abs_tol=atol)
        detail.append({
            "id": r["id"],
            "question": r["question"],
            "domain": r["domain"],
            "subset": r["subset"],
            "ref": r["answer"],
            "pred_raw": pred,
            "pred_val": pred_val,
            "correct": bool(correct),
        })
    return detail, dt


def report_table(detail, rows, cfg, model_name, benchmark, mode):
    """按子域汇总正确率 + CI，并打印表格。"""
    boot_iter = cfg["eval"]["bootstrap_iterations"]
    seed = cfg["eval"]["seed"]
    print(f"\n===== {model_name} | {benchmark} | {mode} | n={len(detail)} =====")
    domains = {}
    for d in detail:
        domains.setdefault(d["domain"], []).append(d)
    overall = [d["correct"] for d in detail]
    if not overall:
        print("  (无样本)")
        return
    acc, (lo, hi, _) = summarize_accuracy(overall, boot_iter, seed)
    print(f"  OVERALL    {acc}")
    for dom, dd in sorted(domains.items()):
        acc, _ = summarize_accuracy([d["correct"] for d in dd], boot_iter, seed)
        print(f"  {dom:<18} n={len(dd):<4} {acc}")
    return {
        "model": model_name, "benchmark": benchmark, "mode": mode,
        "n": len(detail), "overall_acc": acc, "detail": detail,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None, help="模型名；默认取 config defaults.model")
    p.add_argument("--models", default=None, help="逗号分隔的多个模型")
    p.add_argument("--models-all", action="store_true", help="跑 config 里 core+large 全部模型")
    p.add_argument("--benchmark", default=None,
                   choices=["scibench", "olympiadbench", "mmlu_physics"])
    p.add_argument("--max-problems", type=int, default=None)
    p.add_argument("--cot", action="store_true", help="用 zero-shot CoT（等价 --mode cot）")
    p.add_argument("--mode", default=None, choices=["cot", "direct"], help="提示词模式")
    p.add_argument("--backend", default="auto", choices=["auto", "vllm", "transformers"])
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--save", action="store_true", help="保存 json")
    args = p.parse_args()

    cfg = load_config()
    defaults = cfg.get("defaults", {})
    # 模式：显式 --mode 优先；否则 --cot 别名；否则 config defaults.mode
    mode = args.mode or ("cot" if args.cot else defaults.get("mode", "direct"))
    # 基准：--benchmark 优先；否则 defaults.benchmark
    benchmark = args.benchmark or defaults.get("benchmark", "scibench")

    # 模型列表（config.models 现为 {core:[...], large:[...]}）
    mdict = cfg.get("models", {})
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    elif args.models_all:
        models = list(mdict.get("core", [])) + list(mdict.get("large", []))
    elif args.model:
        models = [args.model]
    else:
        models = [args.model or defaults.get("model") or
                  (mdict.get("core", [""])[0])]

    # 基准
    bench_cfg = cfg["benchmarks"].get(benchmark)
    subsets = (bench_cfg.get("subsets") if bench_cfg else None) or ("*",)
    rows = data_mod.load_benchmark(benchmark, subsets=subsets,
                                   max_problems=args.max_problems)
    print(f"[data] {benchmark}: {len(rows)} problems loaded")

    for m in models:
        if not m:
            continue
        try:
            detail, dt = run_model(m, rows, cfg, args.backend, mode,
                                   args.max_new_tokens, args.batch_size)
            rep = report_table(detail, rows, cfg, m, benchmark, mode)
            rep["wall_time_s"] = dt
            if args.save:
                fname = f"eval_{benchmark}_{mode}_{m.replace('/', '__')}.json"
                (RESULTS / fname).write_text(
                    json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[save] results/{fname}")
        except Exception as e:
            print(f"[error] {m}: {type(e).__name__}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
