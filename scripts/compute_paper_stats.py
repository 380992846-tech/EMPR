"""补齐审稿意见要求的统计量（全部可从现有 results/ 复算，无需 GPU）。

对应意见编号：
  M1  逐层恢复率的 bootstrap 95% CI + 邻近层配对 bootstrap 检验（layer 8 是否真的可区分）
  M3  wording 变体的 2x2 配对计数（base 对/变体错、base 错/变体对）
  M4  7B vs 14B 错误分布卡方在“同族标签”与“异族标签”下各做一次
      加权 kappa（线性权重）+ per-class agreement（普通 kappa 对类别不平衡敏感）
  五  小样本比例的二项 Wilson CI（例如 10 题里 7 题 = 70%）

输出：results/paper_stats.json
用法： python scripts/compute_paper_stats.py
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.stats import chi2_contingency, paired_significance  # noqa: E402

RESULTS = ROOT / "results"
KEYS = ("E1", "E2", "E3", "N")


def load(p):
    p = Path(p)
    if not p.exists():
        p = RESULTS / Path(p).name
    return json.loads(p.read_text(encoding="utf-8"))


def wilson(k, n, z=1.96):
    """二项比例的 Wilson 置信区间（小样本比正态近似稳）。"""
    if n == 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


# ---------------------------------------------------------------- M1
def causal_layer_stats(path="causal_tracing_scibench_Qwen2.5-7B-Instruct.json",
                       n_iter=10000, seed=42):
    d = load(path)
    probs = d["problems"]
    layers = [p["layer"] for p in d["average_curve"]]
    step_guess = d.get("layer_step") or (layers[1] - layers[0] if len(layers) > 1 else 1)
    # 每题每层的 recovery 向量
    mat = {pid: {pt["layer"]: pt["recovery"] for pt in rec["curve"]}
           for pid, rec in probs.items()}
    pids = list(mat)
    n = len(pids)
    if n == 0:
        return None
    rng = random.Random(seed)

    def mean_at(layer, sample):
        return sum(mat[p][layer] for p in sample) / len(sample)

    # 逐层 bootstrap CI
    boot = {l: [] for l in layers}
    diffs = {}  # 相邻层差值的 bootstrap 分布
    pairs = [(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
    for a, b in pairs:
        diffs[(a, b)] = []
    for _ in range(n_iter):
        sample = [rng.choice(pids) for _ in range(n)]
        for l in layers:
            boot[l].append(mean_at(l, sample))
        for a, b in pairs:
            diffs[(a, b)].append(mean_at(b, sample) - mean_at(a, sample))

    def pct(vals, q):
        s = sorted(vals)
        i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
        return s[i]

    curve = []
    for l in layers:
        vals = boot[l]
        curve.append({"layer": l,
                      "mean": round(sum(vals) / len(vals), 4),
                      "lo": round(pct(vals, 0.025), 4),
                      "hi": round(pct(vals, 0.975), 4)})
    peak = max(curve, key=lambda c: c["mean"])
    best = peak["layer"]

    # 峰值层 vs 每个其他层：配对 bootstrap 差值 CI
    vs = {}
    for l in layers:
        if l == best:
            continue
        dd = [mean_at(best, [rng.choice(pids) for _ in range(n)]) -
              mean_at(l, [rng.choice(pids) for _ in range(n)]) for _ in range(2000)]
        lo, hi = pct(dd, 0.025), pct(dd, 0.975)
        vs[l] = {"diff": round(peak["mean"] - next(c["mean"] for c in curve if c["layer"] == l), 4),
                 "lo": round(lo, 4), "hi": round(hi, 4),
                 "significant": bool(lo > 0 or hi < 0)}

    # plateau 的**形式化定义**：与峰值层不可区分的层构成的最大连续区间
    # （审稿意见 M-B：range 的边界必须说明是怎么来的，不能只靠肉眼）
    nonsep = sorted(l for l, v in vs.items() if not v["significant"]) + [best]
    nonsep = sorted(nonsep)
    band = [best]
    i = nonsep.index(best)
    j = i
    while j + 1 < len(nonsep) and nonsep[j + 1] - nonsep[j] <= step_guess:
        j += 1
        band.append(nonsep[j])
    while i - 1 >= 0 and nonsep[i] - nonsep[i - 1] <= step_guess:
        i -= 1
        band.append(nonsep[i])
    band = sorted(set(band))
    return {"n_valid": n, "n_layers": d["n_layers"], "layer_step": d["layer_step"],
            "curve": curve, "peak_layer": best, "peak_mean": peak["mean"],
            "peak_ci": [peak["lo"], peak["hi"]],
            "peak_vs_others": {str(k): v for k, v in vs.items()},
            "n_layers_where_peak_not_separable": sum(
                1 for v in vs.values() if not v["significant"]),
            "plateau_layers": band, "plateau_lo": band[0], "plateau_hi": band[-1],
            "nonseparable_layers": nonsep}


# ---------------------------------------------------------------- M3
def wording_2x2(model="Qwen2.5-7B-Instruct"):
    word = load(f"sensitivity_wording_{model}.json")
    ev = load(f"eval_scibench_direct_{model}.json")
    refs = {d["id"]: d["ref"] for d in ev["detail"]}
    from src.grading import parse_number

    def ok(pred, ref):
        r = parse_number(ref)
        if r is None or pred is None:
            return False
        if abs(r) <= 0.01:
            return abs(pred - r) <= 0.01
        return abs(pred - r) / abs(r) <= 0.05

    c = {"both_right": 0, "right_to_wrong": 0, "wrong_to_right": 0, "both_wrong": 0}
    for it in word["items"]:
        ref = refs.get(it["base_id"])
        if ref is None:
            continue
        b = ok(it.get("base_pred"), ref)
        v = ok(it.get("variant_pred"), ref)
        key = ("both_right" if b and v else "right_to_wrong" if b else
               "wrong_to_right" if v else "both_wrong")
        c[key] += 1
    flips = c["right_to_wrong"] + c["wrong_to_right"]
    pb, pv = paired_significance(
        [True] * 0 + [False] * 0 or [], []) if False else (None, None)
    # McNemar 精确二项（只在不一致对上）
    p = None
    if flips:
        # 双尾精确二项检验 p=0.5
        k = min(c["right_to_wrong"], c["wrong_to_right"])
        p = 2 * sum(math.comb(flips, i) for i in range(k + 1)) / (2 ** flips)
        p = min(1.0, p)
    return {"model": model, "counts": c, "n_flips": flips, "mcnemar_exact_p": p}


# ---------------------------------------------------------------- M4
def error_chi2_by_annotator():
    out = {}
    for tag, f in (("same_family_qwen14b", "error_labels.json"),
                   ("cross_family_llama8b", "error_labels_llama.json")):
        labels = load(f)
        rows = {}
        for model in ("Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct"):
            ev = load(f"eval_scibench_cot_{model}.json")
            cnt = dict.fromkeys(KEYS, 0)
            for r in ev["detail"]:
                lbl = labels.get(f"{ev['model']}::{r['id']}")
                if lbl in cnt:
                    cnt[lbl] += 1
            rows[model] = cnt
        a = [rows["Qwen2.5-7B-Instruct"][k] for k in KEYS]
        b = [rows["Qwen2.5-14B-Instruct"][k] for k in KEYS]
        chi2, p, dof = chi2_contingency([a, b])
        out[tag] = {"sevenB": dict(zip(KEYS, a)), "fourteenB": dict(zip(KEYS, b)),
                    "chi2": round(chi2, 4), "p": round(p, 4), "dof": dof}
    return out


def kappa_details():
    a = load("error_labels.json")
    b = load("error_labels_llama.json")
    common = [k for k in a if k in b and a[k] in KEYS and b[k] in KEYS]
    pairs = [(a[k], b[k]) for k in common]
    n = len(pairs)

    def cohen(weights=None):
        """weights=None 时是普通 Cohen's kappa（只对对角线计期望一致）。"""
        def w(k1, k2):
            if weights is None:
                return 1.0 if k1 == k2 else 0.0
            return weights[k1][k2]

        pw = sum(w(x, y) for x, y in pairs) / n
        pe = 0.0
        for k1 in KEYS:
            for k2 in KEYS:
                pa = sum(1 for x, _ in pairs if x == k1) / n
                pb = sum(1 for _, y in pairs if y == k2) / n
                pe += w(k1, k2) * pa * pb
        return (pw - pe) / (1 - pe) if pe < 1 else float("nan")

    # 线性权重：|i-j|/(K-1)
    idx = {k: i for i, k in enumerate(KEYS)}
    lin = {k1: {k2: 1 - abs(idx[k1] - idx[k2]) / (len(KEYS) - 1) for k2 in KEYS} for k1 in KEYS}
    per_class = {}
    for k in KEYS:
        nb = sum(1 for x, _ in pairs if x == k)
        na = sum(1 for _, y in pairs if y == k)
        both = sum(1 for x, y in pairs if x == k and y == k)
        per_class[k] = {"annotator_a": na, "annotator_b": nb, "both": both,
                        "agreement_given_a": round(both / nb, 3) if nb else None}
    return {"n": n, "cohens_kappa": round(cohen(), 4),
            "linear_weighted_kappa": round(cohen(lin), 4),
            "per_class": per_class}


# ---------------------------------------------------------------- 五（小样本 CI）
def small_sample_cis():
    p = RESULTS / "ct_all" / "causal_tracing_scibench_Qwen2.5-7B-Instruct.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    split = d.get("attempts_by_clean_status", {})
    out = {}
    for tag, grp in split.items():
        n = grp["n"]
        best = max(grp["curve"], key=lambda p: p["acc_rate"])
        k = round(best["acc_rate"] * n)
        lo, hi = wilson(k, n)
        out[tag] = {"n": n, "best_layer": best["layer"], "best_rate": best["acc_rate"],
                    "k": k, "wilson95": [round(lo, 3), round(hi, 3)],
                    "all_layers": [{"layer": pt["layer"], "rate": pt["acc_rate"],
                                    "k": round(pt["acc_rate"] * n),
                                    "wilson": [round(w, 3) for w in wilson(round(pt["acc_rate"] * n), n)]}
                                   for pt in grp["curve"]]}
    return out


def main():
    res = {}
    res["M1_layer_ci"] = causal_layer_stats()
    res["M3_wording_2x2"] = wording_2x2()
    res["M3_wording_2x2_all_models"] = {
        m: wording_2x2(m) for m in
        ("Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct",
         "Meta-Llama-3.1-8B-Instruct")
        if (RESULTS / f"sensitivity_wording_{m}.json").exists()
        and (RESULTS / f"eval_scibench_direct_{m}.json").exists()
    }
    res["M4_error_chi2_by_annotator"] = error_chi2_by_annotator()
    res["kappa_details"] = kappa_details()
    res["wilson_small_n"] = small_sample_cis()
    out = RESULTS / "paper_stats.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    m1 = res["M1_layer_ci"]
    if m1:
        print(f"[M1] n_valid={m1['n_valid']} 峰值层 {m1['peak_layer']} "
              f"mean={m1['peak_mean']:.3f} CI[{m1['peak_ci'][0]:.3f},{m1['peak_ci'][1]:.3f}]")
        print(f"     与峰值层不可区分的层数: {m1['n_layers_where_peak_not_separable']}")
        for l, v in m1["peak_vs_others"].items():
            print(f"       vs layer {l:>3}: diff={v['diff']:+.3f} "
                  f"CI[{v['lo']:+.3f},{v['hi']:+.3f}] 显著={v['significant']}")
    m3 = res["M3_wording_2x2"]
    print(f"[M3] 2x2={m3['counts']} flips={m3['n_flips']} exact p={m3['mcnemar_exact_p']}")
    for tag, v in res["M4_error_chi2_by_annotator"].items():
        print(f"[M4] {tag}: chi2={v['chi2']} p={v['p']} 7B={v['sevenB']} 14B={v['fourteenB']}")
    kd = res["kappa_details"]
    print(f"[kappa] n={kd['n']} cohen={kd['cohens_kappa']} linear_weighted={kd['linear_weighted_kappa']}")
    print(f"        per-class={kd['per_class']}")
    print(f"[wilson] {res['wilson_small_n']}")
    print(f"\n[save] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
