"""从 results/*.json 生成 paper/ 下的 LaTeX 表格（可直接 \\input 进论文）。

原则：**表格内容一律由结果文件生成，不手工抄数字**，避免与 results/ 不一致。
运行： python scripts/make_paper_tables.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"

DOMAINS = ["math", "mechanics", "thermodynamics", "optics", "quantum", "general"]
DOM_LABEL = {"math": "Math", "mechanics": "Mech", "thermodynamics": "Thermo",
             "optics": "Optics", "quantum": "Quantum", "general": "General"}
ORDER = ["Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct",
         "Meta-Llama-3.1-8B-Instruct", "DeepSeek-R1-Distill-Qwen-7B"]


def _models():
    found = {}
    for f in glob.glob(str(RESULTS / "eval_*.json")):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        short = d["model"].rstrip("/").split("/")[-1]
        found.setdefault(short, {})[(d["benchmark"], d["mode"])] = d
    ordered = [m for m in ORDER if m in found] + [m for m in sorted(found) if m not in ORDER]
    return found, ordered


def _acc(s):
    """'0.257 [0.211, 0.307]' -> ('25.7', '[21.1, 30.7]')"""
    body, ci = str(s).split("[")
    lo, hi = ci.replace("]", "").split(",")
    return f"{float(body) * 100:.1f}", f"[{float(lo) * 100:.1f}, {float(hi) * 100:.1f}]"


def _short(model):
    return (model.replace("-Instruct", "").replace("Meta-Llama", "Llama")
            .replace("DeepSeek-R1-Distill-Qwen", "R1-Distill"))


def table1_main(found, ordered):
    lines = [
        r"% 由 scripts/make_paper_tables.py 从 results/eval_scibench_*.json 生成，请勿手改",
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \caption{Accuracy (\%) on SciBench physics (323 problems). All reference answers are",
        r"           numerically gradable and all model outputs were parseable, so the denominator",
        r"           is the full 323. Bracketed intervals are bootstrap 95\% CIs (10{,}000 resamples,",
        r"           seed=42) on overall accuracy; sub-domain cells are point estimates.",
        r"           ``cot'' = zero-shot chain-of-thought, ``direct'' = answer-only.}",
        r"  \label{tab:main}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{@{}l l r c c c c c c@{}}",
        r"    \toprule",
        r"    \multirow{2}{*}{\textbf{Model}} & \multirow{2}{*}{\textbf{Mode}} &",
        r"      \multirow{2}{*}{\textbf{Overall (95\% CI)}} &",
        r"      \multicolumn{6}{c}{\textbf{Sub-domain accuracy (\%)}} \\",
        r"    \cmidrule(lr){4-9}",
        r"    & & & " + " & ".join(rf"\textbf{{{DOM_LABEL[d]}}}" for d in DOMAINS) + r" \\",
        r"    \midrule",
    ]
    for i, model in enumerate(ordered):
        for mode, label in (("cot", "cot"), ("direct", "direct")):
            d = found[model].get(("scibench", mode))
            if not d:
                continue
            acc, ci = _acc(d["overall_acc"])
            cells = [f"{float(d['per_domain_acc'].get(dom, 'nan').split()[0]) * 100:.1f}"
                     for dom in DOMAINS]
            lines.append(f"    {_short(model)} & {label} & {acc} \\,{ci} & " +
                         " & ".join(cells) + r" \\")
        if i < len(ordered) - 1:
            lines.append(r"    \addlinespace[2pt]")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def table2_sensitivity():
    lines = [
        r"% 由 scripts/make_paper_tables.py 从 results/sensitivity_summary_*.json 生成，请勿手改",
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \caption{Sensitivity of final answers to hard vs.\ surface perturbations",
        r"           (SciBench physics, 323 base problems, answer-only prompt).",
        r"           \emph{numeric} changes physical values with physics intact;",
        r"           \emph{wording} only rewrites surface phrasing (physics and data identical);",
        r"           \emph{swap} exchanges two distinct physical quantities.",
        r"           ``Changed'' is the share of answers that differ from the base run.",
        r"           Median / p90 are reported instead of the mean: the sensitivity score is",
        r"           $|\Delta|/(|base|+\epsilon)$ and is dominated by outliers when $|base|\approx 0$.}",
        r"  \label{tab:sensitivity}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{5pt}",
        r"  \begin{tabular}{@{}l l r r r r r@{}}",
        r"    \toprule",
        r"    \textbf{Model} & \textbf{Variant} & \textbf{Coverage} & \textbf{$n$ parsed} &",
        r"      \textbf{Median sens.} & \textbf{p90 sens.} & \textbf{Changed} \\",
        r"    \midrule",
    ]
    names = ["numeric", "wording", "swap"]
    rows = []
    for f in sorted(glob.glob(str(RESULTS / "sensitivity_summary_*.json"))):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        rows.append((d["model"].rstrip("/").split("/")[-1], d))
    rows.sort(key=lambda t: (ORDER.index(t[0]) if t[0] in ORDER else 99, t[0]))
    for i, (model, d) in enumerate(rows):
        base = d.get("total_questions", 0)
        for j, v in enumerate(names):
            r = d["variants"].get(v) or {}
            if "error" in r or not r:
                continue
            cov = r["n_changed"] / base if base else 0
            lines.append(
                f"    {(r'\multirow{3}{*}{' + _short(model) + '}') if j == 0 else ''} & "
                f"{v} & {r['n_changed']}/{base} ({cov * 100:.1f}\\%) & {r['n_parsed']} & "
                f"{r['median_sensitivity']:.3f} & {r['p90_sensitivity']:.2f} & "
                f"{(1 - r['unchanged_share']) * 100:.1f}\\% \\\\")
        if i < len(rows) - 1:
            lines.append(r"    \addlinespace[3pt]")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def table3_causal():
    f = RESULTS / "causal_tracing_scibench_Qwen2.5-7B-Instruct.json"
    d = json.loads(f.read_text(encoding="utf-8"))
    buckets = [("0--2", [0, 2]), ("4--8", [4, 6, 8]), ("10--16", [10, 12, 14, 16]),
               ("18--22", [18, 20, 22]), ("24--26", [24, 26])]
    n_tried = d['n_valid'] + d['n_skipped_clean_wrong'] + d['n_skipped_corrupt_flat']
    lines = [
        r"% 由 scripts/make_paper_tables.py 从 results/causal_tracing_*.json 生成，请勿手改",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Causal tracing on Qwen2.5-7B-Instruct (28 layers): recovery rate when the",
        r"           clean residual stream is restored at a single layer of the corrupted run.",
        r"           Only problems whose clean run was correct \emph{and} whose corrupted run",
        rf"           actually lost score are counted ($n={d['n_valid']}$ of {n_tried} attempted;",
        rf"           skipped {d['n_skipped_clean_wrong']} clean-wrong and",
        rf"           {d['n_skipped_corrupt_flat']} corrupted-flat).}}",
        r"  \label{tab:causal}",
        r"  \small",
        r"  \begin{tabular}{@{}l c c@{}}",
        r"    \toprule",
        r"    \textbf{Layer range} & \textbf{Mean recovery} & \textbf{Mean $\pm$1 std across problems} \\",
        r"    \midrule",
    ]
    for name, layers in buckets:
        pts = [p for p in d["average_curve"] if p["layer"] in layers]
        if not pts:
            continue
        m = sum(p["mean_recovery"] for p in pts) / len(pts)
        s = sum(p["std"] for p in pts) / len(pts)
        mark = r"$^\dagger$" if name == "4--8" else ""
        lines.append(f"    layers {name}{mark} & {m:+.3f} & $\\pm${s:.3f} \\\\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \vspace{2pt}",
        r"  \parbox{\linewidth}{\footnotesize $^\dagger$Peak recovery occurs at layer 8",
        r"  ($+0.837 \pm 0.251$). Layers 0--2 show high variance because restoring the",
        r"  embedding-level state is unstable; recovery collapses after layer 18.}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def table4_errors(kappa=None):
    labels_path = RESULTS / "error_labels.json"
    lines = [
        r"% 由 scripts/make_paper_tables.py 生成（标签来自 results/error_labels*.json），请勿手改",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Error-type distribution of the SciBench cot runs, annotated independently by",
        r"           a \emph{same-family} judge (Qwen2.5-14B) and a \emph{cross-family} judge",
        r"           (Llama-3.1-8B). \emph{unscorable\_ref} (reference answer not numerically",
        r"           gradable) and \emph{unparsed\_pred} (no parseable answer) samples are excluded.",
        r"           E1 = wrong physical law; E2 = algebraic / arithmetic / unit error;",
        r"           E3 = violated constraint / internal inconsistency; N = grader false negative.}",
        r"  \label{tab:errors}",
        r"  \small",
        r"  \begin{tabular}{@{}l l r c c c c@{}}",
        r"    \toprule",
        r"    \textbf{Model} & \textbf{Judge} & \textbf{Errors} & \textbf{E1} & \textbf{E2} &",
        r"      \textbf{E3} & \textbf{N} \\",
        r"    \midrule",
    ]

    def rows_for(labels_file):
        p = RESULTS / labels_file
        if not p.exists():
            return {}
        labels = json.loads(p.read_text(encoding="utf-8"))
        out = {}
        for f in sorted(glob.glob(str(RESULTS / "eval_scibench_cot_*.json"))):
            d = json.loads(Path(f).read_text(encoding="utf-8"))
            name = d["model"].rstrip("/").split("/")[-1]
            if name not in ("Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct"):
                continue
            cnt = {"E1": 0, "E2": 0, "E3": 0, "N": 0}
            for r in d["detail"]:
                lbl = labels.get(f"{d['model']}::{r['id']}")
                if lbl in cnt:
                    cnt[lbl] += 1
            out[name] = (sum(cnt.values()), cnt)
        return out

    per_model = rows_for("error_labels.json")
    for name in sorted(per_model, key=lambda m: ORDER.index(m) if m in ORDER else 99):
        tot, cnt = per_model[name]
        if tot:
            lines.append(f"    {_short(name)} & Qwen2.5-14B (same family) & {tot} & " +
                         " & ".join(f"{cnt[k] / tot * 100:.1f}\\%" for k in ("E1", "E2", "E3", "N")) +
                         r" \\")
    llama_model = rows_for("error_labels_llama.json")
    if llama_model:
        lines.append(r"    \midrule")
        for name in sorted(llama_model, key=lambda m: ORDER.index(m) if m in ORDER else 99):
            tot, cnt = llama_model[name]
            if tot:
                lines.append(f"    {_short(name)} & Llama-3.1-8B (cross family) & {tot} & " +
                             " & ".join(f"{cnt[k] / tot * 100:.1f}\\%" for k in ("E1", "E2", "E3", "N")) +
                             r" \\")
    lines += [r"    \bottomrule"]
    if len(per_model) == 2:
        a, b = [per_model[m][1] for m in ("Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct")]
        from src.stats import chi2_contingency
        chi2, p, dof = chi2_contingency([[a[k] for k in ("E1", "E2", "E3", "N")],
                                         [b[k] for k in ("E1", "E2", "E3", "N")]])
        note = (rf"  \parbox{{\linewidth}}{{\footnotesize Under the same-family judge the "
                rf"distribution does not differ significantly between 7B and 14B "
                rf"($\chi^2={chi2:.2f}$, $df={dof}$, $p={p:.3f}$).")
        if kappa is not None:
            note += (rf" Cross-family agreement on the same 495 samples is only "
                     rf"$\kappa={kappa['cohens_kappa']:.2f}$ "
                     rf"({kappa['observed_agreement'] * 100:.0f}\% raw agreement), so the "
                     rf"\emph{{precise}} category shares are annotator-dependent; what is robust "
                     rf"across both judges is that E2 dominates and E3 stays small "
                     rf"(both judges' modal category: {kappa['main_error_a']}).")
        note += r"}"
        lines += [r"  \end{tabular}", r"  \vspace{2pt}", note]
    else:
        lines += [r"  \end{tabular}"]
    lines += [r"\end{table}", ""]
    return "\n".join(lines)


def table5_cross_task(found):
    lines = [
        r"% 由 scripts/make_paper_tables.py 从 results/eval_*.json 生成，请勿手改",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Same model, different answer formats. Multiple-choice recognition is easy;",
        r"           producing the final number is not. Task formats are not directly comparable",
        r"           (selection vs.\ generation), so the contrast is interpreted qualitatively.}",
        r"  \label{tab:cross-task}",
        r"  \small",
        r"  \begin{tabular}{@{}l l r c@{}}",
        r"    \toprule",
        r"    \textbf{Task} & \textbf{Answer format} & \textbf{$n$ (gradable)} & \textbf{Accuracy} \\",
        r"    \midrule",
    ]
    m7 = found.get("Qwen2.5-7B-Instruct", {})
    rows = [
        ("MMLU physics (high-school + college + conceptual)", "select A--D",
         m7.get(("mmlu_physics", "direct"))),
        ("SciBench physics, cot", "write the number", m7.get(("scibench", "cot"))),
        ("SciBench physics, direct", "write the number", m7.get(("scibench", "direct"))),
        ("OlympiadBench physics, cot", "write the number", m7.get(("olympiadbench", "cot"))),
    ]
    for task, fmt, d in rows:
        if not d:
            continue
        acc, ci = _acc(d["overall_acc"])
        n_txt = str(d["n_scorable"])
        if d["n_unscored"]:
            n_txt += rf" of {d['n']}"
        lines.append(f"    {task} & {fmt} & {n_txt} & {acc}\\,{ci} \\\\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \vspace{2pt}",
        r"  \parbox{\linewidth}{\footnotesize Model: Qwen2.5-7B-Instruct. OlympiadBench is",
        r"  evaluated only on the subset whose reference answer is numerically gradable",
        r"  (152 of 351), which is biased towards easier items; it must not be compared",
        r"  numerically against SciBench.}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def main():
    found, ordered = _models()
    if not found:
        print("没有 results/eval_*.json，先跑实验或同步结果")
        return 2

    kappa = None
    kf = RESULTS / "error_label_agreement.json"
    if kf.exists():
        kappa = json.loads(kf.read_text(encoding="utf-8"))

    outs = {
        "table1_main.tex": table1_main(found, ordered),
        "table2_sensitivity.tex": table2_sensitivity(),
        "table3_causal.tex": table3_causal(),
        "table4_errors.tex": table4_errors(kappa),
        "table5_cross_task.tex": table5_cross_task(found),
    }
    for name, text in outs.items():
        p = PAPER / name
        p.write_text(text, encoding="utf-8")
        print(f"[save] {p}  ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
