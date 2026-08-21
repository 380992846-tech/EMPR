"""从 results/eval_*.json 自动生成论文主表（模型 × 子域 × 正确率+CI）。

用法：
  python scripts/make_table.py --dir results                # 全部，按 domain 分组，打印表格
  python scripts/make_table.py --group subset              # 按 SciBench 科目分组
  python scripts/make_table.py --benchmark scibench --mode cot --format md
  python scripts/make_table.py --format md --out results/table.md
  python scripts/make_table.py --format tex --out results/table.tex

格式：
  table | md | csv | tex。默认打印控制台表格。
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stats import bootstrap_ci

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def _fmt(acc, lo, hi):
    if acc is None:
        return "-"
    return f"{acc:.3f} [{lo:.3f}, {hi:.3f}]"


def _load_files(directory, benchmark=None, mode=None):
    files = sorted(glob.glob(str(Path(directory) / "eval_*.json")))
    out = []
    for fp in files:
        try:
            d = json.loads(Path(fp).read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[skip] {fp}: {e}", file=sys.stderr)
            continue
        if benchmark and d.get("benchmark") != benchmark:
            continue
        if mode and d.get("mode") != mode:
            continue
        out.append(d)
    return out


def _acc_matrix(data_list, group_key, ci=0.95, seed=42, n_iter=10000):
    """返回 (models_sorted, groups_sorted, matrix, meta)。
    matrix[(model, group)] = (acc, lo, hi, n)。meta[(model,group)]=(benchmark, mode)。

    组名加 benchmark 前缀，避免多基准在 results/ 混放时同名 domain（如 mechanics）
    被错误合并进同一列。
    """
    rows = {}
    meta = {}
    for d in data_list:
        m = d["model"]
        bm = d.get("benchmark", "?")
        for r in d["detail"]:
            g = r.get(group_key) or "unknown"
            key = f"{bm}/{g}"  # 带基准前缀，避免跨基准混表
            rows.setdefault(m, {}).setdefault(key, []).append(1.0 if r["correct"] else 0.0)
            meta[(m, key)] = (bm, d.get("mode"))
    models = sorted(rows.keys())
    groups = sorted({g for mm in rows.values() for g in mm})
    matrix = {}
    for m in models:
        for g in groups:
            vals = rows.get(m, {}).get(g, [])
            if vals:
                lo, hi, mean = bootstrap_ci(vals, n_iter=n_iter, ci=ci, seed=seed)
                matrix[(m, g)] = (mean, lo, hi, len(vals))
            else:
                matrix[(m, g)] = None
    return models, groups, matrix, meta


def _print_table(models, groups, matrix, meta, fmt, out):
    lines = []
    if fmt == "tex":
        lines.append("\\begin{tabular}{l" + "c" * len(groups) + "}")
        lines.append(" & ".join(["Model"] + groups) + " \\\\ \\hline")
    else:
        lines.append("| Model | " + " | ".join(groups) + " |")
        lines.append("|" + "---|" * (len(groups) + 1))

    for m in models:
        cells = []
        for g in groups:
            v = matrix.get((m, g))
            if v is None:
                cells.append("-")
            else:
                cells.append(_fmt(*v[:3]))
        if fmt == "tex":
            lines.append(" & ".join([m.split("/")[-1]] + cells) + " \\\\")
        else:
            lines.append("| " + m.split("/")[-1] + " | " + " | ".join(cells) + " |")

    if fmt == "tex":
        lines.append("\\end{tabular}")

    text = "\n".join(lines)
    if out:
        Path(out).write_text(text + "\n", encoding="utf-8")
        print(f"[save] {out}")
    print(text)

    # 附带：每个模型-组的样本量提示
    print("\n[n] 样本量（若某格 <30，CI 会很宽，标注一下）:")
    for m in models:
        for g in groups:
            v = matrix.get((m, g))
            if v and v[3] < 30:
                print(f"  {m.split('/')[-1]} × {g}: n={v[3]}  <- 样本少，CI 参考价值低")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default=str(RESULTS))
    p.add_argument("--group", default="domain", choices=["domain", "subset"])
    p.add_argument("--benchmark", default=None)
    p.add_argument("--mode", default=None)
    p.add_argument("--format", default="table", choices=["table", "md", "csv", "tex"])
    p.add_argument("--out", default=None)
    args = p.parse_args()

    data_list = _load_files(args.dir, args.benchmark, args.mode)
    if not data_list:
        print(f"没找到 results/eval_*.json（在 {args.dir}）。先跑 run_eval.py --save 生成。")
        return

    # 从 config.yaml 读 bootstrap 迭代数与 seed，避免与 run_eval 的 CI 不一致
    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    eval_cfg = cfg.get("eval", {})
    n_iter = eval_cfg.get("bootstrap_iterations", 10000)
    seed = eval_cfg.get("seed", 42)

    print(f"加载 {len(data_list)} 个评估结果:",
          ", ".join(f"{d['model'].split('/')[-1]}({d.get('mode')})" for d in data_list))

    # 用与 run_eval 一致的 n_iter 重算 CI（跨 benchmark 分列后 group 已有前缀）
    models, groups, matrix, meta = _acc_matrix(data_list, args.group, ci=0.95, seed=seed, n_iter=n_iter)

    if args.format == "csv":
        import csv
        import io
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["model"] + groups)
        for m in models:
            w.writerow([m.split("/")[-1]] + [
                _fmt(*matrix[(m, g)][:3]) if matrix.get((m, g)) else "-" for g in groups])
        text = buf.getvalue().strip()
        if args.out:
            Path(args.out).write_text(text + "\n", encoding="utf-8")
            print(f"[save] {args.out}")
        print(text)
        return

    _print_table(models, groups, matrix, meta, args.format, args.out)


if __name__ == "__main__":
    main()
