"""用因果追踪结果画图 3：逐层恢复率曲线（均值 ±1 std，标注峰值层与平台区）。"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("results/causal_tracing_scibench_Qwen2.5-7B-Instruct.json", encoding="utf-8"))
cur = d["average_curve"]
xs = [p["layer"] for p in cur]
ys = [p["mean_recovery"] for p in cur]
sd = [p["std"] for p in cur]
peak = max(cur, key=lambda p: p["mean_recovery"])

fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.axvspan(4, 16, color="#dbeafe", alpha=0.7, lw=0, label="plateau (layer 4-16)")
ax.fill_between(xs, [y - s for y, s in zip(ys, sd)], [y + s for y, s in zip(ys, sd)],
                color="#64748b", alpha=0.22, lw=0, label="+/-1 std")
ax.plot(xs, ys, marker="o", ms=4.5, lw=1.8, color="#1d4ed8", label="mean recovery")
ax.axhline(0, color="black", lw=0.8, ls=":")
ax.axvline(peak["layer"], color="#dc2626", ls="--", lw=1.3,
           label=f"peak: layer {peak['layer']} (+{peak['mean_recovery']:.2f})")
ax.set_xlabel("layer index")
ax.set_ylabel("recovery rate")
ax.set_title(f"Causal tracing — Qwen2.5-7B on SciBench physics "
             f"({d['n_valid']} valid problems / {d['n_layers']} layers)")
ax.set_ylim(-0.35, 1.35)
ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
ax.grid(alpha=0.25)

fig.tight_layout()
out = "results/causal_tracing_Qwen2.5-7B.png"
fig.savefig(out, dpi=160)
print("saved", out)

# 同时给敏感性画一张：各模型 numeric/wording 的输出改变比例
import glob
models, num_changed, wrd_changed = [], [], []
for f in sorted(glob.glob("results/sensitivity_summary_*.json")):
    s = json.load(open(f, encoding="utf-8"))
    name = s["model"].rstrip("/").split("/")[-1].replace("-Instruct", "")
    models.append(name)
    num_changed.append(1 - s["variants"]["numeric"]["unchanged_share"])
    wrd_changed.append(1 - s["variants"]["wording"]["unchanged_share"])

fig2, ax2 = plt.subplots(figsize=(7.2, 4.0))
import numpy as np
idx = np.arange(len(models))
w = 0.38
ax2.bar(idx - w / 2, num_changed, w, label="numeric (values changed)", color="#1d4ed8")
ax2.bar(idx + w / 2, wrd_changed, w, label="wording (surface only)", color="#f59e0b")
ax2.set_xticks(idx)
ax2.set_xticklabels(models, fontsize=8.5)
ax2.set_ylabel("share of answers that changed")
ax2.set_ylim(0, 1)
ax2.set_title("Output sensitivity to numeric vs surface-form perturbation")
ax2.legend(fontsize=8.5)
ax2.grid(axis="y", alpha=0.25)
for i, (a, b) in enumerate(zip(num_changed, wrd_changed)):
    ax2.text(i - w / 2, a + 0.02, f"{a:.0%}", ha="center", fontsize=7.5)
    ax2.text(i + w / 2, b + 0.02, f"{b:.0%}", ha="center", fontsize=7.5)
fig2.tight_layout()
out2 = "results/sensitivity_comparison.png"
fig2.savefig(out2, dpi=160)
print("saved", out2)
