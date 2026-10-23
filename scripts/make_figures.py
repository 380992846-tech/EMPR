"""图 2 / 图 3 重绘：加误差棒与 n 标注、加 corrupt 基线参照线（审稿意见 m9）。"""
import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def wilson_halfwidth(p, n, z=1.96):
    """单比例 Wilson 区间的半宽（样本量已知，用于给柱状图加误差棒）。"""
    d = 1 + z * z / n
    return z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d


# ---------------------------------------------------------------- 图 2
models, num, wrd, ns = [], [], [], []
for f in sorted(glob.glob("results/sensitivity_summary_*.json")):
    s = json.load(open(f, encoding="utf-8"))
    name = s["model"].rstrip("/").split("/")[-1].replace("-Instruct", "")
    models.append(name)
    num.append(1 - s["variants"]["numeric"]["unchanged_share"])
    wrd.append(1 - s["variants"]["wording"]["unchanged_share"])
    ns.append((s["variants"]["numeric"]["n_changed"], s["variants"]["wording"]["n_changed"]))

fig, ax = plt.subplots(figsize=(7.8, 4.3))
idx = np.arange(len(models))
w = 0.38
err_num = [wilson_halfwidth(num[i], ns[i][0]) for i in range(len(models))]
err_wrd = [wilson_halfwidth(wrd[i], ns[i][1]) for i in range(len(models))]
ax.bar(idx - w / 2, num, w, yerr=err_num, capsize=3,
       label="numeric (values rescaled)", color="#1d4ed8",
       error_kw={"lw": 1, "ecolor": "#1e3a8a"})
ax.bar(idx + w / 2, wrd, w, yerr=err_wrd, capsize=3,
       label="wording (phrasing only; physics identical)", color="#f59e0b",
       error_kw={"lw": 1, "ecolor": "#92400e"})
ax.set_xticks(idx)
ax.set_xticklabels([f"{m}\n(n={ns[i][0]} / {ns[i][1]})" for i, m in enumerate(models)],
                   fontsize=8.5)
ax.set_ylabel("share of answers that changed")
ax.set_ylim(0, 1.08)
ax.set_title("Output sensitivity: numeric vs surface-form perturbation "
             "(bars: Wilson 95% CI)")
ax.legend(fontsize=8.5, loc="upper right")
ax.grid(axis="y", alpha=0.25)
for i, (a, b) in enumerate(zip(num, wrd)):
    ax.text(i - w / 2, a + err_num[i] + 0.02, f"{a:.0%}", ha="center", fontsize=7.5)
    ax.text(i + w / 2, b + err_wrd[i] + 0.02, f"{b:.0%}", ha="center", fontsize=7.5)
fig.tight_layout()
fig.savefig("results/sensitivity_comparison.png", dpi=160)
print("saved results/sensitivity_comparison.png")

# ---------------------------------------------------------------- 图 3
d = json.load(open("results/causal_tracing_scibench_Qwen2.5-7B-Instruct.json", encoding="utf-8"))
try:
    st = json.load(open("results/paper_stats.json", encoding="utf-8")).get("M1_layer_ci") or {}
except FileNotFoundError:
    st = {}
ci = {c["layer"]: (c["lo"], c["hi"]) for c in st.get("curve", [])}
cur = d["average_curve"]
xs = [p["layer"] for p in cur]
ys = [p["mean_recovery"] for p in cur]
sd = [p["std"] for p in cur]
peak = max(cur, key=lambda p: p["mean_recovery"])
lo = [ci.get(x, (y - s, y + s))[0] for x, y, s in zip(xs, ys, sd)]
hi = [ci.get(x, (y - s, y + s))[1] for x, y, s in zip(xs, ys, sd)]
plo = st.get("plateau_lo", 4)
phi = st.get("plateau_hi", 14)
n_tried = d.get("n_problems") or (d["n_valid"] + d["n_skipped_clean_wrong"]
                                  + d["n_skipped_corrupt_flat"])

fig, ax = plt.subplots(figsize=(7.4, 4.4))
ax.axhline(0, color="black", lw=1.0, ls=":", label="corrupted baseline (no restore)")
ax.axhline(1, color="#16a34a", lw=0.9, ls="--", alpha=0.7,
           label="clean run (= 1 by construction)")
ax.axvspan(plo, phi, color="#dbeafe", alpha=0.6, lw=0,
           label=f"recoverable band (layers {plo}-{phi}, formal definition)")
ax.fill_between(xs, [y - s for y, s in zip(ys, sd)],
                [y + s for y, s in zip(ys, sd)],
                color="#64748b", alpha=0.18, lw=0, label="+/-1 std across problems")
ax.fill_between(xs, lo, hi, color="#1d4ed8", alpha=0.22, lw=0,
                label="bootstrap 95% CI of the mean")
ax.plot(xs, ys, marker="o", ms=4.5, lw=1.8, color="#1d4ed8", label="mean recovery")
ax.axvline(peak["layer"], color="#dc2626", ls="--", lw=1.2,
           label=f"point-estimate max: layer {peak['layer']} "
                 f"(+{peak['mean_recovery']:.2f}); not separable from 6 layers")
ax.set_xlabel("restored layer index")
ax.set_ylabel("recovery rate")
ax.set_title(f"Causal tracing - Qwen2.5-7B, SciBench physics "
             f"(n={d['n_valid']} valid / {n_tried} attempted; {d['n_layers']} layers)")
ax.set_ylim(-0.45, 1.4)
ax.legend(loc="upper right", fontsize=7.0, framealpha=0.92)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("results/causal_tracing_Qwen2.5-7B.png", dpi=160)
print("saved results/causal_tracing_Qwen2.5-7B.png")
