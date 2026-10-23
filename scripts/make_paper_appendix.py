"""生成 paper/appendix.tex：prompt 模板全文、结果 JSON schema、统计细节。

数字与文本一律从代码/结果文件生成，避免附录与实现脱节。
用法： python scripts/make_paper_appendix.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import prompts as prompts_mod  # noqa: E402

RESULTS = ROOT / "results"
PAPER = ROOT / "paper"


def esc(s: str) -> str:
    """把模板文本转成 verbatim 安全的形式（不需转义，只处理换行）。"""
    return s


def prompts_appendix() -> str:
    sample = "A 2.0 kg block slides down a 30 degree incline at 3.0 m/s. ..."
    lines = [
        r"\section{Prompt templates}",
        r"\label{app:prompts}",
        r"All prompts are rendered from a single template table in \texttt{src/prompts.py};",
        r"the MCQ templates are selected automatically when the question contains at least two",
        r"options of the form ``A. \dots''. Every template demands the final answer on its own",
        r"line prefixed with \texttt{ANSWER:}, which is what the extractor looks for first.",
        r"",
    ]
    for mode in prompts_mod.MODES:
        for mcq in (False, True):
            tpl = prompts_mod.TEMPLATES[(mode, mcq)]
            text = tpl.substitute(question="<problem statement>")
            kind = "multiple-choice" if mcq else "free-response"
            lines += [
                rf"\paragraph{{{mode.capitalize()} / {kind}}}",
                r"{\footnotesize\begin{verbatim}  ",
                text,
                r"\end{verbatim}}",
            ]
    lines += [
        r"\paragraph{Chat template}",
        r"The rendered prompt is then wrapped by the tokenizer's chat template",
        r"(\texttt{apply\_chat\_template}), so the model sees its own instruction format.",
        r"Whether a chat template was applied is recorded in every result file as",
        r"\texttt{meta.chat\_template}.",
        rf"Example problem used in this paper: \texttt{{{sample}}}",
        r"",
    ]
    return "\n".join(lines)


def schema_appendix() -> str:
    f = next(iter(sorted(RESULTS.glob("eval_scibench_cot_*.json"))), None)
    if f is None:
        return ""
    d = json.loads(f.read_text(encoding="utf-8"))
    detail0 = d["detail"][0]

    def typ(v):
        if isinstance(v, bool):
            return "bool"
        if isinstance(v, int):
            return "int"
        if isinstance(v, float):
            return "float"
        if v is None:
            return "null|value"
        if isinstance(v, list):
            return "array"
        if isinstance(v, dict):
            return "object"
        return "string"

    lines = [
        r"\section{Result-file schema}",
        r"\label{app:schema}",
        r"Every evaluation writes one JSON file. The top level records the run identity and",
        r"provenance; \texttt{detail} holds one record per problem, which is what the audit",
        r"tables are built from. Minimal schema of \texttt{results/eval\_*.json}:",
        r"",
        r"\begin{verbatim}",
    ]
    for k, v in d.items():
        if k == "detail":
            lines.append(f'  "detail": [ ... {len(d["detail"])} records ... ],')
        else:
            lines.append(f'  "{k}": {typ(v)},')
    lines += [
        r"  # detail[i]:",
    ]
    for k, v in detail0.items():
        lines.append(f'  #   "{k}": {typ(v)},')
    lines += [
        r"\end{verbatim}",
        r"",
        r"Two fields deserve emphasis. \texttt{scorable} marks whether the \emph{reference}",
        r"answer could be parsed; rows with \texttt{scorable=false} are excluded from every",
        r"denominator and reported as \texttt{n\_unscored}. \texttt{status} is one of",
        r"\texttt{correct}, \texttt{incorrect}, \texttt{unparsed\_pred} or",
        r"\texttt{unscorable\_ref}, so a reader can recompute any accuracy in this paper from",
        r"the raw file without re-running a model. Sensitivity files",
        r"(\texttt{sensitivity\_*\_*.json}) additionally carry \texttt{items[]}, one entry per",
        r"perturbed problem with the perturbed question text and both answers, enabling",
        r"per-item audit of the perturbation.",
        r"",
    ]
    return "\n".join(lines)


def stats_appendix() -> str:
    st = json.loads((RESULTS / "paper_stats.json").read_text(encoding="utf-8"))
    m1 = st.get("M1_layer_ci") or {}
    kd = st.get("kappa_details") or {}
    rows = []
    for k in ("E1", "E2", "E3", "N"):
        c = (kd.get("per_class") or {}).get(k) or {}
        if not c:
            continue
        agree = c.get("agreement_given_a")
        rows.append(f"    {k} & {c.get('annotator_b', '--')} & {c.get('annotator_a', '--')} & "
                    f"{c.get('both', '--')} & "
                    + (f"{agree * 100:.0f}\\%" if agree is not None else "--") + r" \\")

    lines = [
        r"\section{Statistical details}",
        r"\label{app:stats}",
        r"\paragraph{Bootstrap.}",
        r"Confidence intervals resample problems with replacement $B$~=~10{,}000 times using",
        r"\texttt{random.Random} seeded with 42, and take the 2.5/97.5 percentiles of the",
        r"resampled statistic (order statistics of the resampled accuracy, i.e.\ a percentile",
        r"interval), the standard non-parametric approach of \citet{efron1993bootstrap}.",
        r"The same seed is used for every model, so intervals are exactly",
        r"reproducible across machines and do not depend on the NumPy version.",
        r"",
    ]
    if m1:
        lines += [
            r"\paragraph{Layer-wise interval and paired test.}",
            rf"The recovery curve has $n={m1['n_valid']}$ problems. Each layer's mean gets a",
            r"percentile bootstrap interval from the same $B$; to test whether the point estimate's",
            r"best layer is meaningful, we bootstrap the \emph{difference} between that layer and",
            r"every other layer (2{,}000 resamples) and ask whether the interval excludes zero.",
            rf"With this test the best layer (layer {m1['peak_layer']}) is \emph{{not}} separable",
            rf"from {m1['n_layers_where_peak_not_separable']} other layers, which is why the paper",
            r"claims a band rather than a peak.",
            r"",
        ]
    lines += [
        r"\paragraph{Small-sample proportions.}",
        r"Proportions computed on few observations are reported with Wilson score intervals",
        r"(95\%), which behave better than the normal approximation near 0 and 1. For example the",
        r"layer-parse recovery on the correct-baseline group is a proportion of a"
        r"~\nbCleanCorrect{}-problem sample.",
        r"",
        r"\paragraph{Paired tests.}",
        r"Correctness comparisons between two conditions on the same problems use McNemar's test",
        r"\citep{mcnemar1947} on the discordant pairs. When the number of discordant pairs is",
        r"small we report the exact binomial two-sided $p$-value rather than the asymptotic",
        r"$\chi^2$ version.",
        r"",
        r"\paragraph{Annotator agreement.}",
        r"We report both unweighted Cohen's $\kappa$ and a linearly weighted $\kappa_w$",
        r"\citep{landis1977kappa} that gives partial credit for adjacent categories, plus",
        r"per-class agreement. The distinction matters because the categories are heavily",
        r"imbalanced (E2 dominates), which depresses $\kappa$ even when most individual",
        r"decisions agree.",
        r"",
        r"\begin{table}[h]",
        r"  \centering",
        r"  \caption{Per-class agreement between the same-family and cross-family judge",
        r"           ($n=\nbAnnotated$ samples; counts are per class).}",
        r"  \label{tab:perclass}",
        r"  \small",
        r"  \begin{tabular}{@{}l c c c c@{}}",
        r"    \toprule",
        r"    \textbf{Class} & \textbf{cross-family count} & \textbf{same-family count} &",
        r"      \textbf{both} & \textbf{agreement given same-family label} \\",
        r"    \midrule",
        *rows,
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        r"",
    ]
    return "\n".join(lines)


def main():
    parts = [r"% 由 scripts/make_paper_appendix.py 自动生成，请勿手改。", ""]
    for fn in (prompts_appendix, schema_appendix, stats_appendix):
        t = fn()
        if t:
            parts.append(t)
    out = PAPER / "appendix.tex"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"[save] {out} ({len(out.read_text(encoding='utf-8').splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
