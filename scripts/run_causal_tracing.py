"""Week 3 · 因果追踪：逐层恢复率曲线（仅开放权重模型）。

方法（Meng et al. 2022 causal tracing / ROME，这里取"逐层恢复"近似）：
  1. clean run    ：原题前向，记录每层输出的 hidden state（residual stream）。
  2. corrupt run  ：把"数字+单位"换成无语义占位符（C2 法），性能应显著下降。
  3. per-layer restore：corrupt 前向时，只把第 k 层的输出替换回 clean 的值。
  4. recovery(k) = (perf_restore_k - perf_corrupt) / (perf_clean - perf_corrupt)
  5. 画 recovery-vs-layer 曲线，峰值层 = 该题答案"经过"的关键层。

注意：三个必须守住的点（否则曲线没有意义）：
  * **clean 必须本来答对**：clean 都答错时，"恢复多少"无从谈起，直接跳过并计数。
  * **corrupt 必须真的掉分**：如果换掉数字后性能不变，说明这道题模型本来就没算，
    recovery 的分母≈0，会炸成任意大数；这类题跳过并计数。
  * **只恢复目标层**：其余层保持 corrupt 状态，否则等价于全层污染，曲线失去意义。

注意：只对开放权重模型做隐藏态干预（闭源 API 不可复现，审稿人会质疑）。
注意：逐层要一次前向：7B/28 层 × 8 题 ≈ 200+ 次前向，题数给 8-20，`--layer-step` 可再省。

用法：
  python scripts/run_causal_tracing.py --model Qwen/Qwen2.5-7B-Instruct \
      --benchmark scibench --max-problems 8 --mode direct --save --plot
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src import data as data_mod
from src import meta as meta_mod
from src import prompts as prompts_mod
from src.grading import extract_number, parse_number
from src.meta import write_json

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

_PLACEHOLDER = "zork blork"

# 数字 + 可选单位（含 LaTeX \mathrm{...}/\text{...}、a/b 复合单位、^幂）
_UNIT_ATOM = r"(?:\\mathrm\s*\{[^}]*\}|\\text\s*\{[^}]*\}|\\operatorname\s*\{[^}]*\}|[A-Za-z\u03bc\u00b0]+)"
_UNIT_TAIL = (r"(?:\s*" + _UNIT_ATOM + r")?"
              r"(?:\s*/\s*" + _UNIT_ATOM + r")?"
              r"(?:\s*\^\s*\{?-?\d+\}?)?")
_QUANTITY_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?" + _UNIT_TAIL)


def corrupt_quantities(question: str):
    """C2：把"数字 + 单位"整体替换为无语义占位符，保留句法结构。

    注意：必须吃下 LaTeX 单位。SciBench 题面几乎全是
    ``$1000 \\mathrm{~kg}$`` / ``$9.8 \\mathrm{m/s^2}$`` 这种写法，正则必须能识别
    LaTeX 单位：只认 ``[A-Za-z]`` 会漏掉数值本身，只把 ``m/s^2`` 的指数换成占位符，
    结果是既没有替换物理量、又留下畸形单位，整条曲线建立在无效的 corrupt 之上。
    """
    return _QUANTITY_RE.sub(_PLACEHOLDER, question)


def score_answer(text: str, ref: str, tol: float = 0.05, abs_tol: float = 0.01):
    """返回 (correct: bool, perf: float in [0,1])。

    **必须用 extract_number**（抠出"最终答案"）而不是 parse_number(整段输出)：
    对 CoT 输出直接 parse_number 会取到推理过程里的第一个数字，把"答对"算成"答错"，
    进而让 clean/corrupt 的性能差变成噪声。
    perf 用连续分 1-相对误差，比 0/1 平滑，适合做逐层曲线；correct 用容差判定。
    """
    ref_v = parse_number(ref)
    pred_v = extract_number(text)
    if ref_v is None or pred_v is None:
        return False, 0.0
    if abs(ref_v) <= abs_tol:
        return abs(pred_v - ref_v) <= abs_tol, 1.0 if abs(pred_v - ref_v) <= abs_tol else 0.0
    rel = abs(pred_v - ref_v) / abs(ref_v)
    return rel <= tol, max(0.0, 1.0 - rel)


def load_model(model_name, dtype="auto", device="auto", verbose=True):
    """加载 HF 模型 + tokenizer（因果追踪需要 hidden state，只能用 transformers）。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.llm import _as_chat, _dtype_kwarg, resolve_device

    dev = resolve_device(device)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    if verbose:
        print(f"[ct] 加载 {model_name} -> {dev}（{_dtype_kwarg()}={dtype}）")
    kwargs = {_dtype_kwarg(): dtype}
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).to(dev)
    model.eval()
    return model, tok, dev, _as_chat


def _layers(model):
    """取 transformer 层列表；不同架构路径不同，这里做一次防御式解析。"""
    for path in ("model.layers", "model.decoder.layers", "transformer.h",
                 "model.model.layers", "gpt_neox.layers"):
        obj = model
        ok = True
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                ok = False
                break
        if ok and len(obj) > 0:
            return list(obj)
    raise RuntimeError("找不到 transformer 层列表：请为本架构补 src/run_causal_tracing 的 _layers")


def _unpack_hidden(output):
    """从层输出中安全取出 residual hidden state（tuple 或 dataclass 两种形态）。"""
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def _align_right(tensor_c, target_len):
    """把 clean 的 hidden 右对齐到 corrupt 的长度（末尾 token 对齐）。

    简化处理：C2 占位符与数字的 token 数不同，长度必然有差。ROPN 原文按"主语 token 位置"
    对齐；这里按**右端对齐**（题干末尾 = 问题位置，对答案影响最大），并在论文里写明。
    """
    import torch
    L = tensor_c.shape[1]
    if L == target_len:
        return tensor_c
    if L > target_len:
        return tensor_c[:, L - target_len:, :]
    pad = torch.zeros((tensor_c.shape[0], target_len - L, tensor_c.shape[2]),
                      dtype=tensor_c.dtype)
    return torch.cat([pad, tensor_c], dim=1)


def _encode(tok, prompt, device):
    """编码单条 prompt，并**显式给出 attention_mask**。

    为什么必须显式：这些模型的 pad_token == eos_token，transformers 无法从 input_ids
    推断 mask（会打 warning，且对含 eos 的输入会误屏蔽真实 token）。这里是单条前向，
    不涉及批量，因此不需要左填充。
    """
    import torch
    inputs = tok(prompt, return_tensors="pt").to(device)
    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    return inputs


def collect_clean_states(model, tok, prompt, device):
    """clean run：记录每层输出 hidden。返回 {layer_idx: tensor(cpu)}。"""
    import torch
    layers = _layers(model)
    states, handles = {}, []

    def make_hook(idx):
        def hook(module, args, output):
            h = _unpack_hidden(output)
            if h.shape[1] > 1:          # 只记 prefill，忽略逐 token 解码步
                states[idx] = h.detach().cpu()
        return hook

    for i in range(len(layers)):
        handles.append(layers[i].register_forward_hook(make_hook(i)))
    inputs = _encode(tok, prompt, device)
    with torch.no_grad():
        model(**inputs)
    for h in handles:
        h.remove()
    return states


def run_once(model, tok, prompt, device, max_new=128):
    """单次贪心生成，返回解码文本。"""
    import torch
    inputs = _encode(tok, prompt, device)
    with torch.no_grad():
        out = model.generate(**inputs, do_sample=False,
                             max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def make_restore_hook(idx, restore_layer, clean_states):
    """构造"只在目标层恢复 clean hidden"的 forward hook（独立出来便于单测）。

    两条守卫：
      * ``idx == restore_layer``：只碰目标层，其余层保持 corrupt（逐层恢复的核心语义）；
      * ``h.shape[1] > 1``：只在 prefill 时替换。解码步每步序列长度是 1，
        否则会把 clean 的位置 0 写进新 token 的位置，生成结果失真且曲线上看不出异常。
    """
    def hook(module, args, output):
        h = _unpack_hidden(output)
        if idx == restore_layer and idx in clean_states and h.shape[1] > 1:
            return _align_right(clean_states[idx], h.shape[1]).to(h.device, h.dtype).clone()
        return h
    return hook


def run_restore(model, tok, prompt, restore_layer, clean_states, device, max_new=128):
    """corrupt 前向，但只把第 restore_layer 层的输出换成 clean 的对应值。"""
    import torch
    layers = _layers(model)
    inputs = _encode(tok, prompt, device)
    handles = []
    for i in range(len(layers)):
        handles.append(layers[i].register_forward_hook(
            make_restore_hook(i, restore_layer, clean_states)))
    with torch.no_grad():
        out = model.generate(**inputs, do_sample=False,
                             max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
    for h in handles:
        h.remove()
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None)
    p.add_argument("--benchmark", default=None)
    p.add_argument("--max-problems", type=int, default=8)
    p.add_argument("--mode", default="direct", choices=["cot", "direct"],
                   help="因果追踪用 direct 更省时（逐层要多次前向）")
    p.add_argument("--layer-step", type=int, default=0, help="层采样间隔；0=自动(~16 个点)")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--device", default="auto")
    p.add_argument("--save", action="store_true")
    p.add_argument("--out-dir", default=str(RESULTS),
                   help="结果输出目录；管道验证建议用 results/smoke 以区别于正式实验")
    p.add_argument("--plot", action="store_true", help="画平均恢复率曲线（需 matplotlib）")
    args = p.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    eval_cfg = cfg.get("eval", {})
    tol, atol = eval_cfg.get("answer_tolerance", 0.05), eval_cfg.get("abs_tolerance", 0.01)
    benchmark = args.benchmark or cfg.get("defaults", {}).get("benchmark", "scibench")
    model_name = args.model or cfg.get("defaults", {}).get("model") or cfg["models"]["core"][0]
    rows = data_mod.load_benchmark(benchmark, max_problems=args.max_problems)
    print(f"[ct] {benchmark}: {len(rows)} 题 | {data_mod.format_composition(rows)}")

    model, tok, device, as_chat = load_model(model_name, device=args.device)
    layers = _layers(model)
    n_layers = len(layers)
    step = args.layer_step or max(1, n_layers // 16)
    print(f"[ct] 层数={n_layers}，采样间隔={step}，模式={args.mode}，"
          f"max_new_tokens={args.max_new_tokens}")

    def prep(text):
        """题干 -> 实际喂给模型的 prompt（与 run_eval 保持一致：同样的模板 + chat 模板）。"""
        p_ = prompts_mod.build_prompt(text, mode=args.mode)
        return as_chat(tok, [p_])[0] if getattr(tok, "chat_template", None) else p_

    all_curves, n_skip_clean, n_skip_corrupt = {}, 0, 0
    for r in rows:
        clean_prompt = prep(r["question"])
        corrupt_prompt = prep(corrupt_quantities(r["question"]))
        clean_text = run_once(model, tok, clean_prompt, device, args.max_new_tokens)
        corrupt_text = run_once(model, tok, corrupt_prompt, device, args.max_new_tokens)
        clean_ok, perf_clean = score_answer(clean_text, r["answer"], tol, atol)
        _, perf_corrupt = score_answer(corrupt_text, r["answer"], tol, atol)

        if not clean_ok:
            # clean 都没答对：恢复率没有意义（分子分母都是噪声）
            n_skip_clean += 1
            print(f"  [skip] {r['id']}: clean 未答对（perf={perf_clean:.2f}）")
            continue
        denom = perf_clean - perf_corrupt
        if denom <= 1e-9:
            n_skip_corrupt += 1
            print(f"  [skip] {r['id']}: corrupt 没掉分（clean={perf_clean:.2f}, "
                  f"corrupt={perf_corrupt:.2f}）— 这道题模型本来就没在算")
            continue

        clean_states = collect_clean_states(model, tok, clean_prompt, device)
        curve = []
        for k in range(0, n_layers, step):
            text_k = run_restore(model, tok, corrupt_prompt, k, clean_states, device,
                                 args.max_new_tokens)
            ok_k, perf_k = score_answer(text_k, r["answer"], tol, atol)
            curve.append({"layer": k, "perf": round(perf_k, 4),
                          "recovery": round((perf_k - perf_corrupt) / denom, 4),
                          "correct": bool(ok_k)})
        peak = max(curve, key=lambda c: c["recovery"])
        all_curves[r["id"]] = {
            "ref": r["answer"], "domain": r["domain"],
            "perf_clean": round(perf_clean, 4), "perf_corrupt": round(perf_corrupt, 4),
            "peak_layer": peak["layer"], "peak_recovery": peak["recovery"],
            "corrupt_prompt": corrupt_quantities(r["question"]),
            "curve": curve,
        }
        print(f"  {r['id']:<22} clean={perf_clean:.2f} corrupt={perf_corrupt:.2f} "
              f"peak={peak['recovery']:+.2f}@{peak['layer']}")

    # ---------------- 汇总：跨题平均曲线（含 std）
    avg = []
    if all_curves:
        by_layer = {}
        for rec in all_curves.values():
            for pt in rec["curve"]:
                by_layer.setdefault(pt["layer"], []).append(pt["recovery"])
        for l in sorted(by_layer):
            vals = by_layer[l]
            m = sum(vals) / len(vals)
            var = sum((v - m) ** 2 for v in vals) / len(vals) if len(vals) > 1 else 0.0
            avg.append({"layer": l, "mean_recovery": round(m, 4),
                        "std": round(var ** 0.5, 4), "n": len(vals)})
        print(f"\n===== 平均 recovery-vs-layer（{len(all_curves)} 题）=====")
        for pt in avg:
            bar = "#" * int(max(0.0, pt["mean_recovery"]) * 40)
            print(f"  layer {pt['layer']:>3}: {pt['mean_recovery']:+.2f} ±{pt['std']:.2f} {bar}")

    print(f"\n[ct] 有效题目 {len(all_curves)} 题；跳过：clean 答错 {n_skip_clean} 题、"
          f"corrupt 未掉分 {n_skip_corrupt} 题")
    if not all_curves:
        print("[ct] 没有任何有效题目：换更大的模型/更多题数，或先看 clean 是否答得对。")

    if args.save:
        f = Path(args.out_dir) / f"causal_tracing_{benchmark}_{meta_mod.model_slug(model_name)}.json"
        write_json(f, {
            "meta": meta_mod.result_meta(
                "run_causal_tracing.py", model=model_name, benchmark=benchmark,
                mode=args.mode, n_layers=n_layers, layer_step=step,
                max_new_tokens=args.max_new_tokens, device=device,
                n_problems=len(rows),
                chat_template=bool(getattr(tok, "chat_template", None)),
                prompt_fingerprint=meta_mod.fingerprint(
                    prompts_mod.build_prompt(rows[0]["question"], mode=args.mode))
                if rows else None),
            "model": model_name, "benchmark": benchmark, "mode": args.mode,
            "n_layers": n_layers, "layer_step": step,
            "n_valid": len(all_curves), "n_skipped_clean_wrong": n_skip_clean,
            "n_skipped_corrupt_flat": n_skip_corrupt,
            "average_curve": avg, "problems": all_curves,
        })
        print(f"[save] {f}")

    if args.plot and avg:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            xs = [pt["layer"] for pt in avg]
            ys = [pt["mean_recovery"] for pt in avg]
            sd = [pt["std"] for pt in avg]
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(xs, ys, marker="o", label="mean recovery")
            ax.fill_between(xs, [y - s for y, s in zip(ys, sd)],
                            [y + s for y, s in zip(ys, sd)], alpha=0.2, label="±1 std")
            best = max(avg, key=lambda pt: pt["mean_recovery"])
            ax.axvline(best["layer"], ls="--", color="crimson",
                       label=f"peak layer {best['layer']}")
            ax.set_xlabel("layer index")
            ax.set_ylabel("recovery rate")
            ax.set_title(f"causal tracing - {model_name.split('/')[-1]} "
                         f"({len(all_curves)} problems)")
            ax.legend()
            out = Path(args.out_dir) / f"causal_tracing_{benchmark}_{meta_mod.model_slug(model_name)}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"[plot] {out}")
        except Exception as e:
            print(f"[plot] 跳过（matplotlib 不可用）: {e}")


if __name__ == "__main__":
    main()
