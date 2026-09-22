"""一条命令的"上云前体检"：把所有零成本检查跑完，输出一份可归档的体检报告。

用途：在执行 GPU 全量实验之前（以及 CI 流水线中）确认
  1. 依赖与数据是否齐；
  2. 参考答案是否都能解析（严格解析器口径）；
  3. 判分回归用例是否全过；
  4. 拿参考答案自判是否 100% 一致（判分管道自洽）；
  5. 三个变体探针的覆盖率；
并把这些数字写进 `results/preflight_<benchmark>.json`，任何一项不过就以非零码退出。

用法：
  python scripts/preflight.py                          # 默认 scibench 全量
  python scripts/preflight.py --benchmark scibench --max-problems 60
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import data as data_mod          # noqa: E402
from src import meta as meta_mod          # noqa: E402
from src.grading import _self_test, parse_reference  # noqa: E402
from src.meta import write_json           # noqa: E402

RESULTS = ROOT / "results"


def _load_script(name: str):
    """把 scripts/<name>.py 当模块加载（scripts 不是包，不能直接 import）。"""
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_empr_preflight_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _check_deps():
    """返回 (结果 dict, 是否致命)。torch/transformers 缺失不算致命（不跑模型也能体检）。"""
    optional = {}
    for name in ("yaml", "sympy", "numpy", "scipy", "torch", "transformers", "vllm", "pytest"):
        try:
            mod = __import__(name)
            optional[name] = getattr(mod, "__version__", "ok")
        except Exception:
            optional[name] = None
    required = ("yaml", "sympy")
    missing = [k for k in required if optional.get(k) is None]
    print(f"[1/5] 依赖: " + ", ".join(
        f"{k}={v or '缺失'}" for k, v in optional.items()))
    if missing:
        print(f"      [FAIL] 缺少必需依赖: {missing}（pip install -r requirements.txt）")
    elif not optional.get("torch"):
        print("      [note] 没有 torch/transformers -> 只能做体检，不能跑模型（云上再装）")
    return {"versions": optional, "missing_required": missing}, bool(missing)


def _check_data(benchmark, max_problems):
    print(f"[2/5] 数据: 加载 {benchmark} ...")
    rows = data_mod.load_benchmark(benchmark, max_problems=max_problems)
    if not rows:
        print("      [FAIL] 没加载到题目：先跑 python scripts/download_data.py")
        return {"n": 0}, rows, True
    comp = data_mod.composition(rows)
    bad = [r for r in rows if parse_reference(r["answer"]) is None]
    print(f"      n={len(rows)} | {data_mod.format_composition(rows)}")
    print(f"      参考答案可解析: {len(rows) - len(bad)}/{len(rows)}"
          f"（{len(bad)} 条不可判分会剔除出分母）")
    for r in bad[:5]:
        print(f"        - {r['id']}: {r['answer']!r}")
    return {"n": len(rows), "composition": comp,
            "n_unscorable_ref": len(bad),
            "scorable_ratio": round((len(rows) - len(bad)) / len(rows), 4)}, rows, False


def _check_grading():
    print("[3/5] 判分回归用例 ...")
    fails = _self_test(verbose=False)
    print(f"      {'[FAIL]' if fails else '[OK]'} 通过 {23 - fails}/23"
          f"（python -m src.grading 可看逐条明细）")
    return {"n_cases": 23, "n_fail": fails}, bool(fails)


def _check_judge_self_consistency(rows, benchmark):
    print("[4/5] 判分自洽（参考答案当预测）...")
    from src.grading import judge_ex
    bad, n_scorable = [], 0
    for r in rows:
        res = judge_ex(f"ANSWER: {r['answer']}", r["answer"])
        if res["status"] == "unscorable_ref":
            continue
        n_scorable += 1
        if not res["correct"]:
            bad.append({"id": r["id"], "ref": r["answer"], "pred_val": res["pred"]})
    ratio = (n_scorable - len(bad)) / max(n_scorable, 1)
    n_unscorable = len(rows) - n_scorable
    print(f"      {'[FAIL]' if bad else '[OK]'} 可判分 {n_scorable}/{len(rows)} 题，"
          f"其中自洽 {n_scorable - len(bad)}/{n_scorable} ({ratio:.4f})")
    if n_unscorable:
        print(f"      另有 {n_unscorable} 条参考答案不可判分（符号/多变量），"
              f"已剔除出分母——论文里必须报这个覆盖率")
    for b in bad[:5]:
        print(f"        - {b['id']}: ref={b['ref']!r} -> {b['pred_val']!r}")
    return {"n": len(rows), "n_scorable": n_scorable, "n_unscorable": n_unscorable,
            "n_inconsistent": len(bad), "judge_self_consistency": round(ratio, 4)}, bool(bad)


def _check_coverage(rows, seed):
    print("[5/5] 变体覆盖率（干跑，不调模型）...")
    sens = _load_script("run_sensitivity")
    cov = sens.report_coverage(rows, seed=seed) if len(rows) <= 60 else {}
    if not cov:
        # 题多时 report_coverage 的样例打印太吵，这里只算数
        total = len(rows) or 1
        for vname in ("numeric", "swap", "wording"):
            n = len(sens.build_variant(rows, vname, seed=seed))
            cov[vname] = {"n_changed": n, "coverage": n / total}
        for k, v in cov.items():
            print(f"      {k:<8} {v['n_changed']:>4}/{len(rows)}  ({v['coverage']:.1%})")
    return cov, False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default=None)
    p.add_argument("--max-problems", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--save", action="store_true", default=True)
    args = p.parse_args()

    cfg_path = ROOT / "config.yaml"
    cfg = {}
    if cfg_path.exists():
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    benchmark = args.benchmark or cfg.get("defaults", {}).get("benchmark", "scibench")
    seed = args.seed if args.seed is not None else cfg.get("eval", {}).get("seed", 42)

    t0 = time.time()
    report = {"benchmark": benchmark, "max_problems": args.max_problems, "seed": seed}
    fatal = False

    deps, bad_deps = _check_deps()
    report["deps"] = deps
    fatal = fatal or bad_deps

    data_rep, rows, bad_data = _check_data(benchmark, args.max_problems)
    report["data"] = data_rep
    if bad_data:
        fatal = True
    if rows:
        grad, bad_grad = _check_grading()
        report["grading_self_test"] = grad
        judge, bad_judge = _check_judge_self_consistency(rows, benchmark)
        report["judge_self_consistency"] = judge
        cov, _ = _check_coverage(rows, seed)
        report["variant_coverage"] = cov
        fatal = fatal or bad_grad or bad_judge

    report["elapsed_s"] = round(time.time() - t0, 1)
    report["passed"] = not fatal
    report["meta"] = meta_mod.result_meta("preflight.py", benchmark=benchmark, seed=seed)

    print(f"\n===== 体检结论：{'PASS' if not fatal else 'FAIL'} "
          f"（{report['elapsed_s']}s）=====")
    if fatal:
        print("请先修复上方 FAIL 项，再执行 GPU 全量实验（判分不成立时，全量结果无效）。")
    else:
        print("可以上云跑全量了。建议顺序见 DEPLOY_CHECKLIST.md 第 5 节。")
    if args.save:
        f = RESULTS / f"preflight_{benchmark}.json"
        write_json(f, report)
        print(f"[save] {f}")
    return 1 if fatal else 0


if __name__ == "__main__":
    sys.exit(main())
