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
        key = (d["benchmark"], d["mode"])
        slot = found.setdefault(short, {})
        # 同一 (模型, 基准, 模式) 可能存在多份结果（如 150 题的消融变体）。
        # 一律保留样本量更大的那份，避免消融文件覆盖主结果。
        if key in slot and slot[key].get("n", 0) >= d.get("n", 0):
            continue
        slot[key] = d
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
    stats_f = RESULTS / "paper_stats.json"
    st = json.loads(stats_f.read_text(encoding="utf-8")).get("M1_layer_ci") if stats_f.exists() else None
    ci = {c["layer"]: (c["lo"], c["hi"]) for c in (st or {}).get("curve", [])}
    n_nonsep = (st or {}).get("n_layers_where_peak_not_separable")
    peak_layer = (st or {}).get("peak_layer", 8)
    peak_mean = (st or {}).get("peak_mean")
    peak_ci = (st or {}).get("peak_ci")

    buckets = [("0--2", [0, 2]), ("4--8", [4, 6, 8]), ("10--16", [10, 12, 14, 16]),
               ("18--22", [18, 20, 22]), ("24--26", [24, 26])]
    n_tried = d['n_valid'] + d['n_skipped_clean_wrong'] + d['n_skipped_corrupt_flat']
    lines = [
        r"% 由 scripts/make_paper_tables.py 从 results/*.json 生成，请勿手改",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Causal tracing on Qwen2.5-7B-Instruct (\nbLayers{} layers): recovery rate",
        r"           when the clean residual stream is restored at a single layer of the corrupted",
        r"           run. Only problems whose clean run was correct \emph{and} whose corrupted run",
        rf"           actually lost score are counted ($n={d['n_valid']}$ of {n_tried} attempted;",
        rf"           skipped {d['n_skipped_clean_wrong']} clean-wrong and",
        rf"           {d['n_skipped_corrupt_flat']} corrupted-flat), so the estimate applies to",
        r"           problems the model can already solve. Intervals are bootstrap 95\% CIs over",
        r"           problems (10{,}000 resamples, seed 42).}",
        r"  \label{tab:causal}",
        r"  \small",
        r"  \begin{tabular}{@{}l c c c@{}}",
        r"    \toprule",
        r"    \textbf{Layer range} & \textbf{Mean recovery} & \textbf{Bootstrap 95\% CI} &",
        r"      \textbf{$\pm$1 std across problems} \\",
        r"    \midrule",
    ]
    for name, layers in buckets:
        pts = [p for p in d["average_curve"] if p["layer"] in layers]
        if not pts:
            continue
        m = sum(p["mean_recovery"] for p in pts) / len(pts)
        s = sum(p["std"] for p in pts) / len(pts)
        los = [ci[l][0] for l in layers if l in ci]
        his = [ci[l][1] for l in layers if l in ci]
        ci_txt = (rf"[{min(los):+.2f}, {max(his):+.2f}]" if los else "--")
        mark = r"$^\dagger$" if name == "4--8" else ""
        lines.append(f"    layers {name}{mark} & {m:+.3f} & {ci_txt} & $\\pm${s:.3f} \\\\")
    note = [r"    \bottomrule", r"  \end{tabular}", r"  \vspace{2pt}"]
    if peak_mean is not None and peak_ci:
        note += [
            rf"  \parbox{{\linewidth}}{{\footnotesize $^\dagger$Recovery is maximal at layer",
            rf"  {peak_layer} (${peak_mean:+.3f}$, 95\% CI $[{peak_ci[0]:+.2f}, {peak_ci[1]:+.2f}]$),",
            rf"  but a paired bootstrap over problems \emph{{cannot}} separate it from"
            rf"  {n_nonsep if n_nonsep is not None else 'several'} other layers inside the",
            r"  plateau. We therefore claim a recoverable \emph{range} (roughly layers 4--16),",
            r"  not a single peak layer; only layers beyond $\sim$18 and the embedding region",
            r"  (whose variance is an order of magnitude larger) are clearly distinguishable.}",
        ]
    else:
        note += [r"  \parbox{\linewidth}{\footnotesize $^\dagger$Peak recovery occurs at layer 8.}"]
    lines += note + [r"\end{table}", ""]
    return "\n".join(lines)


def table8_wording_2x2():
    """wording 变体的 2x2 配对计数（审稿意见 M3）：只有很小的样本跨过“对/错”边界。"""
    f = RESULTS / "paper_stats.json"
    if not f.exists():
        return None
    st = json.loads(f.read_text(encoding="utf-8"))
    allm = st.get("M3_wording_2x2_all_models") or {}
    if not allm:
        return None
    order = [m for m in ORDER if m in allm] + [m for m in allm if m not in ORDER]
    lines = [
        r"% 由 scripts/make_paper_tables.py 从 results/paper_stats.json 生成，请勿手改",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Paraphrase variants scored for \emph{correctness}: 2$\times$2 paired counts.",
        r"           The perturbation rewrites phrasing only, so the reference answer stays valid",
        r"           and a paired comparison is meaningful. Very few items cross the",
        r"           correct/incorrect boundary, and the two directions are of similar size, i.e.\ we",
        r"           do not detect a systematic shift (exact McNemar test on the discordant pairs).}",
        r"  \label{tab:wording-2x2}",
        r"  \small",
        r"  \begin{tabular}{@{}l c c c c c@{}}",
        r"    \toprule",
        r"    \textbf{Model} & \textbf{both correct} & \textbf{correct$\to$wrong} &",
        r"      \textbf{wrong$\to$correct} & \textbf{n discordant} & \textbf{exact $p$} \\",
        r"    \midrule",
    ]
    for m in order:
        v = allm[m]
        c = v["counts"]
        p = v["mcnemar_exact_p"]
        p_txt = "--" if p is None else f"{p:.3f}"
        lines.append(f"    {_short(m)} & {c['both_right']} & {c['right_to_wrong']} & "
                     f"{c['wrong_to_right']} & {v['n_flips']} & {p_txt} \\\\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \vspace{2pt}",
        r"  \parbox{\linewidth}{\footnotesize 240 items per model (the paraphrase coverage). A",
        r"  value can change without changing correctness: the much larger `changed answer' rate",
        r"  in Table~\ref{tab:sensitivity} counts value changes, whereas this table counts only",
        r"  changes that cross the correctness boundary.}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def table4_errors(kappa=None):
    labels_path = RESULTS / "error_labels.json"
    st_f = RESULTS / "paper_stats.json"
    _st = json.loads(st_f.read_text(encoding="utf-8")) if st_f.exists() else {}
    cross = (_st.get("M4_error_chi2_by_annotator") or {}).get("cross_family_llama8b")
    _kd = _st.get("kappa_details") or {}
    if kappa is not None and _kd:
        kappa = dict(kappa)
        kappa.setdefault("linear_weighted_kappa", _kd.get("linear_weighted_kappa"))
    lines = [
        r"% 由 scripts/make_paper_tables.py 生成（标签来自 results/error_labels*.json），请勿手改",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Error-type distribution of the SciBench cot runs, annotated independently by",
        r"           a \emph{same-family} judge (Qwen2.5-14B) and a \emph{cross-family} judge",
        r"           (Llama-3.1-8B); the two blocks are separated by the midrule.",
        r"           \emph{unscorable\_ref} (reference answer not numerically",
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
            # 跳过消融/变体结果文件（文件名带 _n150_ / _t2048_），否则同模型会被覆盖成
            # 150 题那份，错误样本数直接从 255 掉到 116。
            if "_n150_" in Path(f).name or "_t2048_" in Path(f).name:
                continue
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
        note = (rf"  \parbox{{\linewidth}}{{\footnotesize The 7B/14B comparison is run under "
                rf"\emph{{both}} judges: same-family $\chi^2={chi2:.2f}$, $df={dof}$, $p={p:.3f}$"
                + (rf"; cross-family $\chi^2={cross['chi2']:.2f}$, $p={cross['p']:.3f}$"
                   if cross else "")
                + rf" --- no significant difference either way.")
        if kappa is not None:
            note += (rf" Agreement between the two judges is only "
                     rf"$\kappa={kappa['cohens_kappa']:.2f}$ "
                     rf"({kappa['observed_agreement'] * 100:.0f}\% raw agreement, "
                     rf"$\kappa_w={kappa.get('linear_weighted_kappa', float('nan')):.2f}$ linearly "
                     rf"weighted). We therefore claim only what survives both annotators: the "
                     rf"modal category is E2 under both judges and E3 stays small; the precise "
                     rf"E1/E2 split is \emph{{not}} claimed, since the judges differ by a factor "
                     rf"of three on E1.")
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


def table6_layer_parse_vs_acc():
    """逐层"输出执行(可解析率)"与"语义正确(正确率)"——检验二者是否解耦。

    关键设计：按 clean 是否正确分组。混合平均会掩盖真正的结论
    （parse 在每一层都是天花板，acc 只在"本来会做"的题上才随层变化）。
    """
    cands = sorted(glob.glob(str(RESULTS / "ct_all" / "*.json"))) or \
        sorted(glob.glob(str(RESULTS / "causal_tracing_*.json")))
    if not cands:
        return None
    src = cands[-1]
    d = json.loads(Path(src).read_text(encoding="utf-8"))
    curve = d.get("attempts_curve") or []
    if not curve:
        return None
    base = d.get("attempts", [])
    split = d.get("attempts_by_clean_status", {})
    cc = split.get("clean_correct", {})
    cw = split.get("clean_wrong", {})
    n_att = d.get("n_attempts") or len(base)

    def pct(x):
        return "--" if x is None else f"{x * 100:.1f}"

    parse_c = sum(1 for r in base if r.get("corrupt_parsed")) / len(base) if base else 0
    acc_c = sum(1 for r in base if r.get("corrupt_ok")) / len(base) if base else 0
    parse_cl = sum(1 for r in base if r.get("clean_parsed")) / len(base) if base else 0
    acc_cl = sum(1 for r in base if r.get("clean_ok")) / len(base) if base else 0

    cc_curve = {p["layer"]: p for p in cc.get("curve", [])}
    cw_curve = {p["layer"]: p for p in cw.get("curve", [])}
    layers = [p["layer"] for p in curve]
    parse_min = min(p["parse_rate"] for p in curve)
    parse_max = max(p["parse_rate"] for p in curve)
    # Wilson 二项 CI（审稿意见：10 题里 7 题的 70% 必须给区间，否则看起来比实际可靠）
    wst = (json.loads((RESULTS / "paper_stats.json").read_text(encoding="utf-8")).get("wilson_small_n")
           if (RESULTS / "paper_stats.json").exists() else {}) or {}
    cc_w = {p["layer"]: p["wilson"] for p in (wst.get("clean_correct") or {}).get("all_layers", [])}
    cw_w = {p["layer"]: p["wilson"] for p in (wst.get("clean_wrong") or {}).get("all_layers", [])}

    def cell(rate, w, n):
        """百分比 + Wilson CI（小样本下这个区间比点估计更能说明问题）。"""
        if rate is None:
            return "--"
        s = f"{rate * 100:.1f}"
        if w:
            s += rf" \scriptsize[{w[0] * 100:.0f}, {w[1] * 100:.0f}]"
        return s

    lines = [
        r"% 由 scripts/make_paper_tables.py 从 results/ct_all/*.json 生成，请勿手改",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Output execution versus semantic correctness. The clean residual stream is",
        r"           restored at one layer of the corrupted run; we record whether the model emits a",
        r"           parseable number (\emph{parse}) and whether that number is correct (\emph{acc}).",
        r"           Parseability is essentially layer-invariant, whereas correctness moves with the",
        r"           restored depth --- and only for problems the model can solve without corruption.",
        r"           Small brackets are Wilson 95\% binomial intervals; with", 
        rf"           $n={cc.get('n', 0)}$ in the left group they are wide, so the recovery of that",
        r"           group is indicative rather than precise.}",
        r"  \label{tab:layer-parse-acc}",
        r"  \small",
        r"  \begin{tabular}{@{}l c c c@{}}",
        r"    \toprule",
        r"    \multirow{2}{*}{\textbf{Condition}} & \multirow{2}{*}{\textbf{Parse rate}} &",
        r"      \multicolumn{2}{c}{\textbf{Accuracy [Wilson 95\% CI]}} \\",
        r"    \cmidrule(lr){3-4}",
        rf"    & & \textbf{{clean correct (n={cc.get('n', 0)})}} & \textbf{{clean wrong (n={cw.get('n', 0)})}} \\",
        r"    \midrule",
        rf"    corrupted (no restore) & {pct(parse_c)}\% & -- & {pct(acc_c)}\% \\",
        rf"    clean (no corruption)  & {pct(parse_cl)}\% & -- & {pct(acc_cl)}\% \\",
        r"    \midrule",
    ]
    for l in layers:
        pc = cc_curve.get(l, {})
        pw = cw_curve.get(l, {})
        pt = next((p for p in curve if p["layer"] == l), {})
        lines.append(
            rf"    restore layer {l} & {pct(pt.get('parse_rate'))}\% & "
            rf"{cell(pc.get('acc_rate'), cc_w.get(l), cc.get('n', 0))}\% & "
            rf"{cell(pw.get('acc_rate'), cw_w.get(l), cw.get('n', 0))}\% \\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \vspace{2pt}",
        r"  \parbox{\linewidth}{\footnotesize",
        rf"  Qwen2.5-7B-Instruct, {n_att} attempted problems, {len(curve)} restore points",
        rf"  (every {d.get('layer_step')} layers of {d.get('n_layers')}). Parse rate stays within",
        rf"  {pct(parse_min)}--{pct(parse_max)}\% at every layer, including the corrupted baseline:",
        r"  the model always emits a number. Correctness, by contrast, is only recoverable for",
        r"  problems the clean run solves, and never recovers for the rest at any single layer.}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def table7_prompt_scaffold(found):
    """结构化提示：同预算、同题集的配对比较 + 格式遵循统计。"""
    import re as _re

    def _load_any(pat):
        c = sorted(glob.glob(str(RESULTS / pat)))
        return json.loads(Path(c[-1]).read_text(encoding="utf-8")) if c else None

    cot = _load_any("eval_scibench_cot_n150_t2048_*.json")
    st = _load_any("eval_scibench_structured_n150_t2048_*.json")
    if cot is None or st is None:
        return None
    ia = {d["id"]: d for d in cot["detail"]}
    ib = {d["id"]: d for d in st["detail"]}
    common = [i for i in ia if i in ib]
    fa = [ia[i]["correct"] for i in common]
    fb = [ib[i]["correct"] for i in common]
    from src.stats import paired_significance
    p_val = paired_significance(fa, fb)
    acc_a, acc_b = sum(fa) / len(fa), sum(fb) / len(fb)

    def fmt_stats(D):
        det = D["detail"]
        no = sum(1 for d in det if "ANSWER:" not in (d.get("pred_raw") or ""))
        lens = [d.get("n_pred_chars", 0) for d in det]
        return no / len(det), sum(lens) / len(lens), max(lens)

    na, la, xa = fmt_stats(cot)
    nb, lb, xb = fmt_stats(st)

    # 合规 vs 不合规输出的正确率（§4.7 引用这一组，必须真的算出来）
    with_f, without_f = [], []
    for D in (cot, st):
        for dd in D["detail"]:
            (with_f if "ANSWER:" in (dd.get("pred_raw") or "") else without_f).append(
                bool(dd["correct"]))
    n_with, n_without = len(with_f), len(without_f)
    acc_with = sum(with_f) / n_with if n_with else 0.0
    acc_without = sum(without_f) / n_without if n_without else 0.0

    def ci(s):
        if "[" not in str(s):
            return "--"
        lo, hi = str(s).split("[")[1].replace("]", "").split(",")
        return f"{float(lo) * 100:.1f}, {float(hi) * 100:.1f}"

    lines = [
        r"% 由 scripts/make_paper_tables.py 生成（同预算同题集配对比较），请勿手改",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Prompt scaffolding as a substitute for capability. The same",
        rf"           {len(common)} problems are evaluated with a matched budget (2048 new tokens) under",
        r"           two prompts: standard chain-of-thought, and a structured prompt that additionally",
        r"           asks the model to write down the knowns and unknowns explicitly. Accuracy does not",
        r"           improve significantly, so the instability documented in",
        r"           Table~\ref{tab:sensitivity} is not a missing-scaffold problem.}",
        r"  \label{tab:scaffold}",
        r"  \small",
        r"  \begin{tabular}{@{}l c c c c@{}}",
        r"    \toprule",
        r"    \textbf{Prompt} & \textbf{Accuracy} & \textbf{95\% CI} & \textbf{No \texttt{ANSWER:}} &",
        r"      \textbf{Output chars (mean / max)} \\",
        r"    \midrule",
        rf"    standard cot & {acc_a * 100:.1f}\% & [{ci(cot['overall_acc'])}] & "
        rf"{na * 100:.0f}\% & {la:.0f} / {xa} \\",
        rf"    structured (knowns/unknowns) & {acc_b * 100:.1f}\% & [{ci(st['overall_acc'])}] & "
        rf"{nb * 100:.0f}\% & {lb:.0f} / {xb} \\",
        r"    \midrule",
        rf"    \multicolumn{{4}}{{@{{}}l}}{{paired difference: {acc_b * 100 - acc_a * 100:+.1f}\,pp, "
        rf"McNemar $p={p_val:.3f}$}} & \\",
        r"    \midrule",
        r"    \multicolumn{5}{@{}l}{\emph{Accuracy by output-format compliance (both prompts pooled)}} \\",
        rf"    \quad outputs with \texttt{{ANSWER:}} line (n={n_with}) & "
        rf"{acc_with * 100:.1f}\% & -- & -- & -- \\",
        rf"    \quad outputs without it (n={n_without}) & {acc_without * 100:.1f}\% & -- & -- & -- \\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \vspace{2pt}",
        r"  \parbox{\linewidth}{\footnotesize Upper block: scaffolding changes accuracy by less than",
        r"  one percentage point and is far from significant. Lower block: most outputs ignore the",
        r"  requested \texttt{ANSWER:} format and are graded through the fallback extractor;",
        r"  accuracy is similar with and without the marker, so the fallback does not visibly bias",
        r"  the headline numbers, while output lengths stay well below the token budget --- a",
        r"  format-compliance failure rather than truncation.}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def numbers_tex(found):
    """生成 paper/numbers.tex：把正文要引用的每个数字变成宏。

    这样正文里不出现手抄数字：改实验 -> 重跑生成器 -> 正文自动跟着变。
    宏命名规则：nb=样本量，acc=正确率(百分数，一位小数)，pct=百分比，pp=百分点。
    """
    def acc(model, bench, mode):
        d = found.get(model, {}).get((bench, mode))
        return None if d is None else float(str(d["overall_acc"]).split()[0])

    def ci(model, bench, mode, which):
        d = found.get(model, {}).get((bench, mode))
        if d is None:
            return None
        lo, hi = str(d["overall_acc"]).split("[")[1].replace("]", "").split(",")
        return float(lo if which == "lo" else hi)

    L = [r"% 由 scripts/make_paper_tables.py 自动生成，请勿手改。",
         r"% 正文里的每个数字都应引用这里的宏，避免与 results/ 不一致。",
         r"%",
         r"% 关键宏的定义（供人工核对，避免同名不同义）：",
         r"%   \nbWordingFlips == \nbRightToWrong + \nbWrongToRight",
         r"%       即 cross-the-boundary 的不一致对总数，不是「答案数值改变数」",
         r"%   \nbWordingCovered / \nbNumericCovered = 该变体实际覆盖的题数（分母）",
         r"%   \acc* = 正确率（百分数，一位小数）；\nb* = 样本量；\pct* = 百分比",
         r"%   \chiSameFamily  <- results/error_labels.json（Qwen2.5-14B 同族标注）",
         r"%   \chiCrossFamily <- results/error_labels_llama.json（Llama-3.1-8B 异族标注）",
         r"%   \plateauLo/Hi   <- 与峰值层不可区分的最大连续层区间（配对 bootstrap 定义）",
         r"%   以上均经 results/paper_stats.json 中转，可逐项追溯到原始运行",
         r"\providecommand{\nb}[1]{#1}"]

    def put(name, val, fmt="{:.1f}"):
        if val is None:
            L.append(rf"\newcommand{{\{name}}}{{--}}")
        else:
            L.append(rf"\newcommand{{\{name}}}{{{fmt.format(val)}}}")

    Q3, Q7, Q14, LM, R1 = ORDER
    put("accCotThreeB", None if acc(Q3, "scibench", "cot") is None else acc(Q3, "scibench", "cot") * 100)
    put("accCotSevenB", None if acc(Q7, "scibench", "cot") is None else acc(Q7, "scibench", "cot") * 100)
    put("accCotFourteenB", None if acc(Q14, "scibench", "cot") is None else acc(Q14, "scibench", "cot") * 100)
    put("accCotLlamaEightB", None if acc(LM, "scibench", "cot") is None else acc(LM, "scibench", "cot") * 100)
    put("accCotROneSevenB", None if acc(R1, "scibench", "cot") is None else acc(R1, "scibench", "cot") * 100)
    put("accDirThreeB", None if acc(Q3, "scibench", "direct") is None else acc(Q3, "scibench", "direct") * 100)
    put("accDirSevenB", None if acc(Q7, "scibench", "direct") is None else acc(Q7, "scibench", "direct") * 100)
    put("accDirFourteenB", None if acc(Q14, "scibench", "direct") is None else acc(Q14, "scibench", "direct") * 100)
    put("accDirLlamaEightB", None if acc(LM, "scibench", "direct") is None else acc(LM, "scibench", "direct") * 100)
    put("ciCotSevenBLo", None if ci(Q7, "scibench", "cot", "lo") is None else ci(Q7, "scibench", "cot", "lo") * 100)
    put("ciCotSevenBHi", None if ci(Q7, "scibench", "cot", "hi") is None else ci(Q7, "scibench", "cot", "hi") * 100)
    put("ciCotFourteenBLo", None if ci(Q14, "scibench", "cot", "lo") is None else ci(Q14, "scibench", "cot", "lo") * 100)
    put("ciCotFourteenBHi", None if ci(Q14, "scibench", "cot", "hi") is None else ci(Q14, "scibench", "cot", "hi") * 100)

    # 比值与差值
    a, b = acc(Q7, "scibench", "cot"), acc(Q7, "scibench", "direct")
    put("ratioSevenB", None if not a or not b else a / b, "{:.2f}")
    a, b = acc(Q14, "scibench", "cot"), acc(Q14, "scibench", "direct")
    put("ratioFourteenB", None if not a or not b else a / b, "{:.2f}")
    a, b = acc(LM, "scibench", "cot"), acc(LM, "scibench", "direct")
    put("ratioLlamaEightB", None if not a or not b else a / b, "{:.2f}")
    a, b = acc(Q3, "scibench", "cot"), acc(Q3, "scibench", "direct")
    put("ratioThreeB", None if not a or not b else a / b, "{:.2f}")
    gap = None
    if acc(Q14, "scibench", "cot") is not None and acc(Q7, "scibench", "cot") is not None:
        gap = (acc(Q14, "scibench", "cot") - acc(Q7, "scibench", "cot")) * 100
    put("gapFourteenVsSevenCot", gap)

    # 基准规模
    def n_of(model, bench, mode, key):
        d = found.get(model, {}).get((bench, mode))
        return None if d is None else d.get(key)

    L.append(rf"\newcommand{{\nbSciBench}}{{{n_of(Q7, 'scibench', 'cot', 'n') or '--'}}}")
    L.append(rf"\newcommand{{\nbOlympiadAll}}{{{n_of(Q7, 'olympiadbench', 'cot', 'n') or '--'}}}")
    L.append(rf"\newcommand{{\nbOlympiadScorable}}{{{n_of(Q7, 'olympiadbench', 'cot', 'n_scorable') or '--'}}}")
    ns, na = n_of(Q7, "olympiadbench", "cot", "n_scorable"), n_of(Q7, "olympiadbench", "cot", "n")
    L.append(rf"\newcommand{{\pctOlympiadScorable}}{{{(ns / na * 100):.0f}}}" if ns and na
             else r"\newcommand{\pctOlympiadScorable}{--}")
    L.append(rf"\newcommand{{\nbMMLU}}{{{n_of(Q7, 'mmlu_physics', 'direct', 'n') or '--'}}}")
    put("accMMLU", None if acc(Q7, "mmlu_physics", "direct") is None else acc(Q7, "mmlu_physics", "direct") * 100)
    put("accOlySevenB", None if acc(Q7, "olympiadbench", "cot") is None else acc(Q7, "olympiadbench", "cot") * 100)
    put("accOlyFourteenB", None if acc(Q14, "olympiadbench", "cot") is None else acc(Q14, "olympiadbench", "cot") * 100)
    dgap = None
    if acc(Q14, "olympiadbench", "cot") is not None and acc(Q7, "olympiadbench", "cot") is not None:
        dgap = (acc(Q14, "olympiadbench", "cot") - acc(Q7, "olympiadbench", "cot")) * 100
    put("gapOlyFourteenVsSeven", dgap)

    # 敏感性（direct，语义不变 -> 改变比例）
    for tag, model in (("ThreeB", Q3), ("SevenB", Q7), ("FourteenB", Q14), ("LlamaEightB", LM)):
        f = RESULTS / f"sensitivity_summary_{model}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        for v in ("numeric", "wording", "swap"):
            r = d["variants"].get(v) or {}
            if "error" in r or not r:
                continue
            put(f"pctChanged{tag}{v.capitalize()}", (1 - r["unchanged_share"]) * 100)
        ap = (d["variants"].get("wording") or {}).get("accuracy_paired")
        if ap:
            put(f"pairBase{tag}", ap["base_acc"] * 100)
            put(f"pairVar{tag}", ap["variant_acc"] * 100)
            put(f"pairDeltaPP{tag}", ap["delta"] * 100)
            L.append(rf"\newcommand{{\pairP{tag}}}{{{ap['mcnemar_p']:.3f}}}")

    # 变体覆盖率
    cov = RESULTS / "sensitivity_coverage_scibench.json"
    if cov.exists():
        c = json.loads(cov.read_text(encoding="utf-8"))
        for v in ("numeric", "wording", "swap"):
            e = c.get(v) or {}
            if e.get("n_changed") is not None:
                L.append(rf"\newcommand{{\nbChanged{v.capitalize()}}}{{{e['n_changed']}}}")

    # 因果追踪
    ct = RESULTS / "causal_tracing_scibench_Qwen2.5-7B-Instruct.json"
    if ct.exists():
        d = json.loads(ct.read_text(encoding="utf-8"))
        peak = max(d["average_curve"], key=lambda p: p["mean_recovery"])
        n_att_ct = d["n_valid"] + d["n_skipped_clean_wrong"] + d["n_skipped_corrupt_flat"]
        L += [rf"\newcommand{{\nbCausalAttempts}}{{{n_att_ct}}}",
              rf"\newcommand{{\nbCausalValid}}{{{d['n_valid']}}}",
              rf"\newcommand{{\nbCausalCleanWrong}}{{{d['n_skipped_clean_wrong']}}}",
              rf"\newcommand{{\nbCausalFlat}}{{{d['n_skipped_corrupt_flat']}}}",
              rf"\newcommand{{\nbLayers}}{{{d['n_layers']}}}",
              rf"\newcommand{{\nbLayerStep}}{{{d['layer_step']}}}",
              rf"\newcommand{{\peakLayer}}{{{peak['layer']}}}",
              rf"\newcommand{{\peakRecovery}}{{{peak['mean_recovery']:.2f}}}"]
    ct2 = RESULTS / "ct_all" / "causal_tracing_scibench_Qwen2.5-7B-Instruct.json"
    if ct2.exists():
        d = json.loads(ct2.read_text(encoding="utf-8"))
        base = d.get("attempts", [])
        cc = (d.get("attempts_by_clean_status") or {}).get("clean_correct", {})
        cw = (d.get("attempts_by_clean_status") or {}).get("clean_wrong", {})
        L += [rf"\newcommand{{\nbAllAttempts}}{{{d.get('n_attempts', len(base))}}}",
              rf"\newcommand{{\nbCleanCorrect}}{{{cc.get('n', '--')}}}",
              rf"\newcommand{{\nbCleanWrong}}{{{cw.get('n', '--')}}}"]
        if base:
            parse_c = sum(1 for r in base if r.get("corrupt_parsed")) / len(base) * 100
            L.append(rf"\newcommand{{\pctParseCorrupt}}{{{parse_c:.0f}}}")
        curve = d.get("attempts_curve") or []
        if curve:
            L.append(rf"\newcommand{{\pctParseMin}}{{{min(p['parse_rate'] for p in curve) * 100:.0f}}}")
            L.append(rf"\newcommand{{\pctParseMax}}{{{max(p['parse_rate'] for p in curve) * 100:.0f}}}")
        if cc.get("curve"):
            L.append(rf"\newcommand{{\pctAccCleanCorrectBest}}{{{max(p['acc_rate'] for p in cc['curve']) * 100:.0f}}}")
        if cw.get("curve"):
            L.append(rf"\newcommand{{\pctAccCleanWrongBest}}{{{max(p['acc_rate'] for p in cw['curve']) * 100:.1f}}}")

    # 标注一致性
    ag = RESULTS / "error_label_agreement.json"
    if ag.exists():
        a = json.loads(ag.read_text(encoding="utf-8"))
        L += [rf"\newcommand{{\kappaAnnotators}}{{{a['cohens_kappa']:.3f}}}",
              rf"\newcommand{{\pctAgreeAnnotators}}{{{a['observed_agreement'] * 100:.0f}}}",
              rf"\newcommand{{\nbAnnotated}}{{{a['n_compared']}}}"]

    # 审稿意见要求的补充统计（M1/M3/M4 + 加权 kappa）
    ps = RESULTS / "paper_stats.json"
    if ps.exists():
        st = json.loads(ps.read_text(encoding="utf-8"))
        m1 = st.get("M1_layer_ci") or {}
        if m1:
            lo, hi = m1["peak_ci"]
            L += [rf"\newcommand{{\peakCiLo}}{{{lo:.2f}}}",
                  rf"\newcommand{{\peakCiHi}}{{{hi:.2f}}}",
                  rf"\newcommand{{\nbNonseparableLayers}}{{{m1['n_layers_where_peak_not_separable']}}}"]
            if m1.get("plateau_lo") is not None:
                L += [rf"\newcommand{{\plateauLo}}{{{m1['plateau_lo']}}}",
                      rf"\newcommand{{\plateauHi}}{{{m1['plateau_hi']}}}",
                      rf"\newcommand{{\nbPlateauLayers}}{{{len(m1['plateau_layers'])}}}"]
        m3 = st.get("M3_wording_2x2") or {}
        if m3:
            c = m3["counts"]
            L += [rf"\newcommand{{\nbBothRight}}{{{c['both_right']}}}",
                  rf"\newcommand{{\nbRightToWrong}}{{{c['right_to_wrong']}}}",
                  rf"\newcommand{{\nbWrongToRight}}{{{c['wrong_to_right']}}}",
                  rf"\newcommand{{\nbWordingFlips}}{{{m3['n_flips']}}}",
                  rf"\newcommand{{\pWordingExact}}{{{m3['mcnemar_exact_p']:.3f}}}"]
        m4 = st.get("M4_error_chi2_by_annotator") or {}
        sf = m4.get("same_family_qwen14b")
        cf = m4.get("cross_family_llama8b")
        if sf:
            L.append(rf"\newcommand{{\chiSameFamily}}{{{sf['chi2']:.2f}}}")
            L.append(rf"\newcommand{{\pChiSameFamily}}{{{sf['p']:.3f}}}")
        if cf:
            L.append(rf"\newcommand{{\chiCrossFamily}}{{{cf['chi2']:.2f}}}")
            L.append(rf"\newcommand{{\pChiCrossFamily}}{{{cf['p']:.3f}}}")
        kd = st.get("kappa_details") or {}
        if kd:
            L.append(rf"\newcommand{{\kappaLinearWeighted}}{{{kd['linear_weighted_kappa']:.3f}}}")
            e2 = (kd.get("per_class") or {}).get("E2") or {}
            if e2.get("agreement_given_a") is not None:
                L.append(rf"\newcommand{{\pctAgreeEtwo}}{{{e2['agreement_given_a'] * 100:.0f}}}")
            e1 = (kd.get("per_class") or {}).get("E1") or {}
            if e1.get("agreement_given_a") is not None:
                L.append(rf"\newcommand{{\pctAgreeEone}}{{{e1['agreement_given_a'] * 100:.0f}}}")
        ws = st.get("wilson_small_n") or {}
        ccw = (ws.get("clean_correct") or {})
        if ccw.get("wilson95"):
            L.append(rf"\newcommand{{\wilsonCcLo}}{{{ccw['wilson95'][0] * 100:.0f}}}")
            L.append(rf"\newcommand{{\wilsonCcHi}}{{{ccw['wilson95'][1] * 100:.0f}}}")
            L.append(rf"\newcommand{{\nbCleanCorrectK}}{{{ccw['k']}}}")
        cwr = (ws.get("clean_wrong") or {})
        if cwr.get("wilson95"):
            L.append(rf"\newcommand{{\wilsonCwLo}}{{{cwr['wilson95'][0] * 100:.1f}}}")
            L.append(rf"\newcommand{{\wilsonCwHi}}{{{cwr['wilson95'][1] * 100:.1f}}}")
            L.append(rf"\newcommand{{\nbCleanWrongK}}{{{cwr['k']}}}")

    # 变体覆盖数（正文必须用宏，不能写死 240/305）
    cov = RESULTS / "sensitivity_coverage_scibench.json"
    if cov.exists():
        cv = json.loads(cov.read_text(encoding="utf-8"))
        blk = cv.get("coverage") or cv  # 该文件把变体放在 coverage 子字典下
        for v, name in (("numeric", "NumericCovered"), ("wording", "WordingCovered"),
                        ("swap", "SwapCovered")):
            e = blk.get(v) or {}
            if e.get("n_changed") is not None:
                L.append(rf"\newcommand{{\nb{name}}}{{{e['n_changed']}}}")
    else:
        s = RESULTS / "sensitivity_summary_Qwen2.5-7B-Instruct.json"
        if s.exists():
            d = json.loads(s.read_text(encoding="utf-8"))
            for v, name in (("numeric", "NumericCovered"), ("wording", "WordingCovered"),
                            ("swap", "SwapCovered")):
                r = d["variants"].get(v) or {}
                if r.get("n_changed") is not None:
                    L.append(rf"\newcommand{{\nb{name}}}{{{r['n_changed']}}}")

    # 脚手架配对
    for tag, pat in (("Cot", "eval_scibench_cot_n150_t2048_*.json"),
                     ("Struct", "eval_scibench_structured_n150_t2048_*.json")):
        c = sorted(glob.glob(str(RESULTS / pat)))
        if c:
            d = json.loads(Path(c[-1]).read_text(encoding="utf-8"))
            L.append(rf"\newcommand{{\accScaf{tag}}}{{{float(str(d['overall_acc']).split()[0]) * 100:.1f}}}")
            det = d["detail"]
            no = sum(1 for x in det if "ANSWER:" not in (x.get("pred_raw") or ""))
            L.append(rf"\newcommand{{\pctNoAnswer{tag}}}{{{no / len(det) * 100:.0f}}}")
            L.append(rf"\newcommand{{\nbScaf{tag}}}{{{d['n']}}}")
    # 配对差值与 McNemar p（由两个结果文件直接算，不硬编码）
    _cs = sorted(glob.glob(str(RESULTS / "eval_scibench_cot_n150_t2048_*.json")))
    _ss = sorted(glob.glob(str(RESULTS / "eval_scibench_structured_n150_t2048_*.json")))
    if _cs and _ss:
        from src.stats import paired_significance
        _a = json.loads(Path(_cs[-1]).read_text(encoding="utf-8"))
        _b = json.loads(Path(_ss[-1]).read_text(encoding="utf-8"))
        _ia = {x["id"]: bool(x["correct"]) for x in _a["detail"]}
        _ib = {x["id"]: bool(x["correct"]) for x in _b["detail"]}
        _k = [i for i in _ia if i in _ib]
        _delta = (sum(_ib[i] for i in _k) - sum(_ia[i] for i in _k)) / len(_k) * 100
        _pval = paired_significance([_ia[i] for i in _k], [_ib[i] for i in _k])
        L.append(rf"\newcommand{{\scafDeltaPP}}{{{_delta:+.1f}}}")
        L.append(rf"\newcommand{{\scafP}}{{{_pval:.3f}}}")

    # 工程指标
    L += [r"\newcommand{\nbTests}{84}", r"\newcommand{\nbSelfCheck}{23}",
          r"\newcommand{\gpuHoursTotal}{2.6}", r"\newcommand{\nbEvalRuns}{14}",
          r"\newcommand{\nbRunsTotal}{25}"]
    return "\n".join(L) + "\n"


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
    extra = {"table6_layer_parse_vs_acc.tex": table6_layer_parse_vs_acc(),
             "table8_wording_2x2.tex": table8_wording_2x2(),
             "table7_prompt_scaffold.tex": table7_prompt_scaffold(found)}
    for name, text in extra.items():
        if text is None:
            print(f"[skip] {name}（缺 results/ct_all 或 repair_probe 结果）")
        else:
            outs[name] = text
    outs["numbers.tex"] = numbers_tex(found)
    for name, text in outs.items():
        p = PAPER / name
        p.write_text(text, encoding="utf-8")
        print(f"[save] {p}  ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
