"""Week 1 · 评估管道主入口。

跑某个基准上的若干开放模型，输出 每模型 × 每子域 的正确率表（含 bootstrap 95% CI）。

用法示例：
  # 单模型（云上 GPU）
  python scripts/run_eval.py --model Qwen/Qwen2.5-14B-Instruct \
      --benchmark scibench --cot --max-problems 100 --backend vllm --save

  # 本地 CPU 冒烟（不需要 GPU，验证链路；结果不可作为论文数字）
  python scripts/run_eval.py --model Qwen/Qwen2.5-0.5B-Instruct \
      --benchmark scibench --cot --max-problems 20 --backend transformers --save

  # 判分自检：用参考答案当预测跑全流程，正确率必须 100%（否则是判分 bug）
  python scripts/run_eval.py --self-check --benchmark scibench

  # 跑 config 里 core+large 全部 / 用 config defaults.model
  python scripts/run_eval.py --models-all --benchmark scibench
  python scripts/run_eval.py --benchmark scibench

输出：results/eval_<benchmark>_<mode>_<model>.json（含 detail + 元数据 + CI）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src import data as data_mod
from src import llm as llm_mod
from src import meta as meta_mod
from src import prompts as prompts_mod
from src.grading import judge_ex
from src.stats import summarize_accuracy

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def load_config() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def run_model(model_name, rows, cfg, backend, mode, max_new_tokens, batch_size,
              gpus=1, device="auto", chat="auto"):
    """对一个模型跑完整评估，返回 (detail, wall_time_s, gen_info)。"""
    gen = llm_mod.make_generator(
        model_name, backend=backend, seed=cfg["eval"]["seed"],
        max_new_tokens=max_new_tokens, gpus=gpus, device=device, batch_size=batch_size,
        chat=chat)
    prompts = prompts_mod.build_prompts(rows, mode=mode)
    import time
    t0 = time.time()
    outputs = gen.gen_batch(prompts, batch_size=batch_size)
    dt = time.time() - t0
    if len(outputs) != len(rows):
        # 静默错位会把 A 题输出判到 B 题答案上，宁可报错
        raise RuntimeError(f"输出条数({len(outputs)}) != 题数({len(rows)})，拒绝继续")

    tol = cfg["eval"]["answer_tolerance"]
    atol = cfg["eval"]["abs_tolerance"]
    detail = []
    for r, pred in zip(rows, outputs):
        res = judge_ex(pred, r["answer"], tol=tol, abs_tol=atol)
        detail.append({
            "id": r["id"],
            "question": r["question"],
            "domain": r["domain"],
            "subset": r["subset"],
            "ref": r["answer"],
            "pred_raw": pred,
            "pred_val": res["pred"],
            "correct": bool(res["correct"]),
            "status": res["status"],
            "scorable": bool(res["scorable"]),
            "n_pred_chars": len(pred or ""),
        })
    info = {"backend": getattr(gen, "backend", backend), "device": getattr(gen, "device", None),
            "chat_template": getattr(gen, "chat", None)}
    return detail, dt, info


def summarize(detail, rows_unused=None):
    """汇总正确率：分母只算 scorable 样本，参考答案解析不了的单独报 n_unscored。"""
    scorable = [d for d in detail if d["scorable"]]
    unscorable = [d for d in detail if not d["scorable"]]
    unparsed = [d for d in scorable if d["status"] == "unparsed_pred"]
    return {
        "n": len(detail),
        "n_scorable": len(scorable),
        "n_unscored": len(unscorable),
        "n_unparsed_pred": len(unparsed),
        "n_correct": sum(1 for d in scorable if d["correct"]),
        "flags": [d["correct"] for d in scorable],
    }


def report_table(detail, cfg, model_name, benchmark, mode):
    """按子域汇总正确率 + CI，打印表格，返回可落盘的结果 dict。"""
    boot_iter = cfg["eval"]["bootstrap_iterations"]
    seed = cfg["eval"]["seed"]
    agg = summarize(detail)

    print(f"\n===== {model_name} | {benchmark} | {mode} | n={agg['n']} =====")
    if agg["n_scorable"] == 0:
        print("  (没有可判分样本)")
        return None

    overall_acc, _ = summarize_accuracy(agg["flags"], boot_iter, seed)
    print(f"  OVERALL            n={agg['n_scorable']:<4} {overall_acc}")
    if agg["n_unscored"]:
        print(f"  [warn] 参考答案不可解析 {agg['n_unscored']} 条，已剔除出分母（不是模型错）")
    if agg["n_unparsed_pred"]:
        pct = agg["n_unparsed_pred"] / max(agg["n_scorable"], 1)
        print(f"  [warn] 模型输出抠不出数值 {agg['n_unparsed_pred']} 条 ({pct:.1%})，计为错——"
              f"若比例偏高，先查判分/提示词，别全量跑")

    domains = {}
    for d in detail:
        if d["scorable"]:
            domains.setdefault(d["domain"], []).append(d)
    per_domain = {}
    for dom, dd in sorted(domains.items()):
        acc, _ = summarize_accuracy([d["correct"] for d in dd], boot_iter, seed)
        per_domain[dom] = acc
        print(f"  {dom:<18} n={len(dd):<4} {acc}")

    return {
        "model": model_name, "benchmark": benchmark, "mode": mode,
        "n": agg["n"], "n_scorable": agg["n_scorable"], "n_unscored": agg["n_unscored"],
        "n_unparsed_pred": agg["n_unparsed_pred"],
        "overall_acc": overall_acc, "per_domain_acc": per_domain, "detail": detail,
    }


# ---------------------------------------------------------------- 判分自检
def self_check(rows, cfg, benchmark, mode):
    """用参考答案当预测跑一遍判分：正确率必须 1.000，否则是判分/提示词的 bug。

    这是零成本的有效性检查（不调模型）：能抓到符号、单位、千分位、科学计数法、
    选择题字母、负号这类会让"答对的题被算错"的系统性偏差。
    """
    tol, atol = cfg["eval"]["answer_tolerance"], cfg["eval"]["abs_tolerance"]
    detail = []
    for r in rows:
        pred = f"ANSWER: {r['answer']}"
        res = judge_ex(pred, r["answer"], tol=tol, abs_tol=atol)
        detail.append({
            "id": r["id"], "question": r["question"], "domain": r["domain"],
            "subset": r["subset"], "ref": r["answer"], "pred_raw": pred,
            "pred_val": res["pred"], "correct": bool(res["correct"]),
            "status": res["status"], "scorable": bool(res["scorable"]),
        })
    agg = summarize(detail)
    bad = [d for d in detail if not d["correct"]]
    print(f"\n===== 判分自检 | {benchmark} | n={agg['n']} =====")
    print(f"  参考答不可解析 : {agg['n_unscored']}")
    print(f"  自判一致       : {agg['n_correct']}/{agg['n']} "
          f"({agg['n_correct'] / max(agg['n'], 1):.4f})")
    for d in bad[:20]:
        print(f"  [不一致] {d['id']}: ref={d['ref']!r} -> pred_val={d['pred_val']!r} "
              f"status={d['status']}")
    if bad:
        print(f"  !! {len(bad)} 条自判不一致：判定逻辑不成立，请先修 src/grading.py")
    return detail, len(bad)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None, help="模型名；默认取 config defaults.model")
    p.add_argument("--models", default=None, help="逗号分隔的多个模型")
    p.add_argument("--models-all", action="store_true", help="跑 config 里 core+large 全部模型")
    p.add_argument("--benchmark", default=None,
                   choices=["scibench", "olympiadbench", "mmlu_physics"])
    p.add_argument("--max-problems", type=int, default=None)
    p.add_argument("--cot", action="store_true", help="零样本 CoT（等价 --mode cot）")
    p.add_argument("--mode", default=None, choices=["cot", "direct"], help="提示词模式")
    p.add_argument("--backend", default="auto", choices=["auto", "vllm", "transformers"])
    p.add_argument("--device", default="auto", help="auto|cpu|cuda（transformers 后端）")
    p.add_argument("--no-chat", action="store_true",
                   help="不走 tokenizer 的 chat template（默认 auto：有模板就用）")
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--save", action="store_true", help="保存 json")
    p.add_argument("--out-dir", default=str(RESULTS),
                   help="结果输出目录；管道验证建议用 results/smoke 以区别于正式实验")
    p.add_argument("--self-check", action="store_true",
                   help="判分自检：用参考答案当预测，正确率必须 100%%（不调模型）")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    cfg = load_config()
    defaults = cfg.get("defaults", {})
    eval_cfg = cfg.get("eval", {})
    mode = args.mode or ("cot" if args.cot else defaults.get("mode", "direct"))
    benchmark = args.benchmark or defaults.get("benchmark", "scibench")
    max_new_tokens = args.max_new_tokens or eval_cfg.get("max_new_tokens", 1024)

    bench_cfg = cfg["benchmarks"].get(benchmark)
    subsets = (bench_cfg.get("subsets") if bench_cfg else None) or ("*",)
    rows = data_mod.load_benchmark(benchmark, subsets=subsets,
                                  max_problems=args.max_problems)
    print(f"[data] {benchmark}: {len(rows)} problems | {data_mod.format_composition(rows)}")
    if not rows:
        print("[data] 没加载到题目：先跑 scripts/download_data.py（见 README）")
        return 2

    # ---------------- 判分自检（零成本，先跑这个）
    if args.self_check:
        detail, n_bad = self_check(rows, cfg, benchmark, mode)
        if args.save:
            f = Path(args.out_dir) / f"selfcheck_{benchmark}.json"
            meta_mod.write_json(f, {
                "meta": meta_mod.result_meta("run_eval.py --self-check", benchmark=benchmark,
                                             mode=mode, n=len(rows)),
                "benchmark": benchmark, "n": len(rows), "n_inconsistent": n_bad,
                "judge_self_consistency": (len(rows) - n_bad) / max(len(rows), 1),
                "detail": detail,
            })
            print(f"[save] {f}")
        return 1 if n_bad else 0

    # ---------------- 模型列表（config.models = {smoke:[...], core:[...], large:[...]}）
    # --models 的每一项既可以是模型名，也可以是池名（smoke/core/large），
    # 例如 --models "core,Qwen/Qwen2.5-0.5B-Instruct"。
    mdict = cfg.get("models", {})
    if args.models:
        models = []
        for tok in (m.strip() for m in args.models.split(",")):
            if not tok:
                continue
            if tok in mdict:
                print(f"[models] 展开池 '{tok}': {len(mdict[tok])} 个模型")
                models.extend(mdict[tok])
            else:
                models.append(tok)
    elif args.models_all:
        # 不含 smoke：该池仅用于管道验证
        models = list(mdict.get("core", [])) + list(mdict.get("large", []))
    elif args.model:
        models = [args.model]
    else:
        models = [defaults.get("model") or (mdict.get("core") or [""])[0]]

    for m in models:
        if not m:
            continue
        try:
            detail, dt, info = run_model(m, rows, cfg, args.backend, mode,
                                         max_new_tokens, args.batch_size, args.gpus,
                                         args.device,
                                         chat=(False if args.no_chat else "auto"))
            rep = report_table(detail, cfg, m, benchmark, mode)
            if rep is None:
                continue
            rep["wall_time_s"] = round(dt, 1)
            rep["max_new_tokens"] = max_new_tokens
            rep["data_composition"] = data_mod.composition(rows)
            fp = meta_mod.fingerprint(prompts_mod.build_prompt(rows[0]["question"], mode=mode))
            rep["prompt_fingerprint"] = fp
            rep["meta"] = meta_mod.result_meta(
                "run_eval.py", model=m, benchmark=benchmark, mode=mode,
                backend=info.get("backend"), device=info.get("device"),
                chat_template=info.get("chat_template"),
                n_problems=len(rows), max_new_tokens=max_new_tokens,
                prompt_fingerprint=fp, self_check=args.self_check)
            if args.save:
                fname = f"eval_{benchmark}_{mode}_{m.replace('/', '__')}.json"
                out_dir = Path(args.out_dir)
                meta_mod.write_json(out_dir / fname, rep)
                print(f"[save] {out_dir / fname}")
                print(f"       单题均耗时 {dt / max(len(rows), 1):.2f}s | 提示词指纹 "
                      f"{rep['prompt_fingerprint']} | commit {rep['meta']['git_commit']}")
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"[error] {m}: {type(e).__name__}: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
