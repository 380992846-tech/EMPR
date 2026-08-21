"""Week 3 · 因果追踪：逐层恢复率曲线（开放权重模型）。

方法（Meng et al. 2022，仅限开放权重模型）：
  1. clean run    ：对原题前向，记录每层输出的 hidden state（residual stream）。
  2. corrupt run  ：把物理量换成无语义占位符（C2 法），得到损坏输出（性能低）。
  3. per-layer restore：对每一层 k，在 corrupt 前向时把第 k 层输出替换为 clean[k]
                       （等价于恢复残差流在 k 处的状态），重跑，量性能。
  4. recovery rate(k) = (perf_restore_k - perf_corrupt) / (perf_clean - perf_corrupt)
  5. 画 recovery-rate-vs-layer 曲线，找关键层区间。

性能度量：用"答案相对误差"的连续分（更平滑），也可切 0/1。
⚠ 每层都要一次前向，7B/32 层每道题约 32 次前向——题数请给小（8-20）。

用法：
  python scripts/run_causal_tracing.py --model Qwen/Qwen2.5-7B-Instruct \
      --benchmark scibench --max-problems 8 --save
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src import data as data_mod
from src.grading import parse_number

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

_PLACEHOLDER = "zork blork"


def corrupt_quantities(question: str):
    """C2：把"数字 单位"替换为无语义占位符，保留句法结构。"""
    pat = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\s*[A-Za-z^]*(?:/[A-Za-z^]+)?")
    return pat.sub(_PLACEHOLDER, question)


def load_model(model_name, dtype="auto", device="cuda"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.eval()
    return model, tok, device


def run_and_score(model, tok, prompt, ref, device, max_new=96):
    """一次前向，返回 (文本, 性能分 0..1)。性能分=1-相对误差(截断到0)。"""
    import torch
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(inputs["input_ids"], do_sample=False, max_new_tokens=max_new,
                             pad_token_id=tok.pad_token_id)
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    ref_v, pred_v = parse_number(ref), parse_number(text)
    if ref_v is None:
        return text, 0.0
    if pred_v is None:
        return text, 0.0
    if abs(ref_v) < 1e-9:
        score = 1.0 if abs(pred_v) < 1e-6 else 0.0
    else:
        score = max(0.0, 1.0 - abs(pred_v - ref_v) / abs(ref_v))
    return text, score


def collect_clean_states(model, tok, prompt, device):
    """clean run：记录每层输出 hidden（residual stream）。返回 {layer: tensor}。"""
    import torch
    layers = model.model.layers
    states = {}
    handles = []
    def make_hook(idx):
        def hook(module, args, output):
            h = output[0] if isinstance(output, (tuple, list)) else output
            states[idx] = h.detach().cpu()
        return hook
    for i in range(len(layers)):
        handles.append(layers[i].register_forward_hook(make_hook(i)))
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        model(**inputs)
    for h in handles:
        h.remove()
    return states


def run_restore(model, tok, prompt, restore_layer, clean_states, device, max_new=96):
    """corrupt 前向，但在第 restore_layer 层用 clean 状态覆盖。返回文本。"""
    import torch
    layers = model.model.layers
    inputs = tok(prompt, return_tensors="pt").to(device)
    seq_len = inputs["input_ids"].shape[1]
    handles = []
    def make_hook(idx):
        def hook(module, args, output):
            h = output[0] if isinstance(output, (tuple, list)) else output
            if idx in clean_states:
                c = clean_states[idx].to(h.device)
                # 位置对齐（截到当前 seq 长度）
                h[:, : min(seq_len, c.shape[1])] = c[:, : min(seq_len, h.shape[1])]
            out = (h,) if isinstance(output, (tuple, list)) else h
            return out
        return hook
    for i in range(len(layers)):
        handles.append(layers[i].register_forward_hook(make_hook(i)))
    with torch.no_grad():
        out = model.generate(inputs["input_ids"], do_sample=False, max_new_tokens=max_new,
                             pad_token_id=tok.pad_token_id)
    for h in handles:
        h.remove()
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None)
    p.add_argument("--benchmark", default="scibench")
    p.add_argument("--max-problems", type=int, default=8)
    p.add_argument("--save", action="store_true")
    p.add_argument("--plot", action="store_true", help="画 recovery-vs-layer 图（需 matplotlib）")
    args = p.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    model_name = args.model or cfg.get("defaults", {}).get("model") or cfg["models"]["core"][0]
    rows = data_mod.load_benchmark(args.benchmark, max_problems=args.max_problems)

    print(f"[ct] 加载模型 {model_name} ...")
    model, tok, device = load_model(model_name)
    n_layers = len(model.model.layers)
    print(f"[ct] 层数 = {n_layers}，题数 = {len(rows)}")

    all_curves = {}
    for r in rows:
        clean_prompt, corrupt_prompt = r["question"], corrupt_quantities(r["question"])
        # clean / corrupt 性能
        _, perf_clean = run_and_score(model, tok, clean_prompt, r["answer"], device)
        _, perf_corrupt = run_and_score(model, tok, corrupt_prompt, r["answer"], device)
        denom = perf_clean - perf_corrupt
        if denom <= 1e-9:
            print(f"  [skip] {r['id']}: clean~corrupt（无法恢复，跳过）")
            continue
        # 记录 clean states（用同一 clean run 的 hidden）
        clean_states = collect_clean_states(model, tok, clean_prompt, device)

        # 逐层恢复（取一个粗网格省时，全量可设 step=1）
        step = max(1, n_layers // 16)
        curve = []
        for k in range(0, n_layers, step):
            restored = run_restore(model, tok, corrupt_prompt, k, clean_states, device)
            # 用 restored 文本算性能（相对 ref）
            ref_v, pred_v = parse_number(r["answer"]), parse_number(restored)
            if ref_v is None or pred_v is None:
                perf_k = 0.0
            elif abs(ref_v) < 1e-9:
                perf_k = 1.0 if abs(pred_v) < 1e-6 else 0.0
            else:
                perf_k = max(0.0, 1.0 - abs(pred_v - ref_v) / abs(ref_v))
            recovery = (perf_k - perf_corrupt) / denom
            curve.append({"layer": k, "perf": perf_k, "recovery": float(recovery)})
        all_curves[r["id"]] = {
            "ref": r["answer"], "perf_clean": perf_clean, "perf_corrupt": perf_corrupt,
            "domain": r["domain"], "curve": curve,
        }
        peak = max(curve, key=lambda c: c["recovery"])
        print(f"  {r['id']:<20} clean={perf_clean:.2f} corrupt={perf_corrupt:.2f} "
              f"peak_recovery={peak['recovery']:.2f}@{peak['layer']}")

    # 汇总：跨题目平均恢复率曲线
    if all_curves:
        by_layer = {}
        for r in all_curves.values():
            for pt in r["curve"]:
                by_layer.setdefault(pt["layer"], []).append(pt["recovery"])
        avg = sorted((l, sum(v) / len(v)) for l, v in by_layer.items())
        print("\n===== 平均 recovery-vs-layer =====")
        for l, v in avg:
            bar = "#" * int(v * 40)
            print(f"  layer {l:>3}: {v:+.2f} {bar}")

    if args.save:
        f = RESULTS / f"causal_tracing_{model_name.replace('/', '__')}.json"
        f.write_text(json.dumps({"model": model_name, "n_layers": n_layers,
                                 "problems": all_curves},
                                ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[save] {f}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            for rid, r in all_curves.items():
                xs = [p["layer"] for p in r["curve"]]
                ys = [p["recovery"] for p in r["curve"]]
                ax.plot(xs, ys, alpha=0.4, label=rid[:20])
            ax.set_xlabel("layer"); ax.set_ylabel("recovery rate")
            ax.set_title(f"causal tracing - {model_name.split('/')[-1]}")
            out = RESULTS / f"causal_tracing_{model_name.replace('/', '__')}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"[plot] {out}")
        except Exception as e:
            print(f"[plot] 跳过（matplotlib 不可用）: {e}")


if __name__ == "__main__":
    main()
