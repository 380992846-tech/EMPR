"""GPU 时间预算估算器（基于实测数据，可复算）。

思路：把每个实验折算成 **(prefill tokens, decode tokens, 启动次数)**，再按各模型在
单张 80GB 卡上的 vLLM 吞吐折算成 GPU 时间。prompt 长度是**实测的**（见
`results/prompt_token_stats.json`，由真实题面分词得到），生成长度与吞吐是**假设**，
写在下面的常量里——你跑完第一次后可以用结果 json 里的 `wall_time_s` 反推并替换。

用法：
  python scripts/gpu_budget.py                 # 打印预算表（默认 A800-80G）
  python scripts/gpu_budget.py --json results/gpu_budget.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.meta import write_json  # noqa: E402

# ============================================================
# 假设（改这里就能换成你的实测值）
# ============================================================
# 生成长度（tokens/题）。cot 是"要求写全过程"，实测 7B~14B 一般在 300~700；
# direct 只要一个数字，但 instruct 模型常带一两句解释，按 30 估。
OUTPUT_TOKENS = {"direct": 30, "cot": 450, "cot_reasoning": 1400}
MAX_NEW_TOKENS = {"direct": 96, "cot": 1024, "cot_reasoning": 2048}

# vLLM bf16 在**单张 A800-80G（≈A100-80G）**上的聚合输出吞吐（tokens/s，高并发）。
# 这是保守中值；社区实测 7B/8B 可到 3~4k，14B 1.5~2k。用 --throughput-scale 可整体缩放。
THROUGHPUT = {
    "7B": 2500, "8B": 2500, "14B": 1500, "32B": 800, "70B": 400, "72B": 380,
}
# 每次启动的固定开销（加载权重 + CUDA graph 捕获 + vLLM 预热），单位分钟。
# 权重已在本地盘缓存的前提下；首次还得从网上下（7B≈15GB，14B≈30GB，72B≈145GB）。
STARTUP_MIN = {"7B": 2.0, "8B": 2.0, "14B": 3.0, "32B": 4.0, "70B": 8.0, "72B": 8.0}
# prefill 折算：把 prompt tokens 按"预填充比解码快 N 倍"计入，避免忽略它
PREFILL_SPEEDUP = 10.0
# 变体覆盖率（实测，seed=42；见 results/sensitivity_coverage_scibench.json）
COVERAGE = {"numeric": 0.9443, "wording": 0.7430, "swap": 0.1115}


def _tokens(prompt_stats, bench, mode, n):
    """从实测的 prompt 统计里取平均 prompt 长度，换算 n 条的总 prefill tokens。"""
    key = f"{bench}/{mode}"
    mean = prompt_stats[key]["mean"]
    return mean * n


def run_estimate(label, model_size, bench, mode, n, prompt_stats, extra_generations=1.0,
                 tp=1, hf_path=False, note=""):
    """一个 run 的估算。extra_generations 用于敏感性（同一题要跑多个变体）。"""
    prefill = _tokens(prompt_stats, bench, mode, n) * extra_generations
    out_per_q = OUTPUT_TOKENS[mode]
    decode = out_per_q * n * extra_generations
    thr = THROUGHPUT[model_size] / (2.0 if tp == 2 else 1.0)  # TP=2 每卡吞吐减半
    if hf_path:
        # 因果追踪要 hook 隐藏态，不能用 vLLM：HF generate 单序列吞吐约 1/25
        thr = max(20.0, THROUGHPUT[model_size] / 25.0)
    decode_s = decode / thr
    prefill_s = prefill / (thr * PREFILL_SPEEDUP)
    startup_s = STARTUP_MIN[model_size] * 60
    gpu_s = (decode_s + prefill_s + startup_s) * tp
    return {
        "label": label, "model_size": model_size, "bench": bench, "mode": mode,
        "n": n, "generations": round(n * extra_generations),
        "prefill_tokens": round(prefill), "decode_tokens": round(decode),
        "gpus": tp,
        "gpu_seconds": round(gpu_s, 1), "gpu_minutes": round(gpu_s / 60, 1),
        "note": note,
    }


def build_plan(prompt_stats, throughput_scale=1.0):
    """返回 (runs, 汇总)。这就是"跑完这篇论文要多少卡时"的清单。"""
    global THROUGHPUT
    THROUGHPUT = {k: v * throughput_scale for k, v in THROUGHPUT.items()}

    runs = []
    N = 323            # SciBench 物理全量
    N_OLY = 351        # OlympiadBench
    N_MMLU = 488       # MMLU 物理三 config

    # ---- Week 1：主表（core 池 × cot + direct）
    for size, name in (("7B", "Qwen2.5-7B"), ("7B", "Mistral-7B"),
                       ("8B", "Llama-3.1-8B"), ("14B", "Qwen2.5-14B")):
        runs.append(run_estimate(f"W1 {name} cot", size, "scibench", "cot", N, prompt_stats))
        runs.append(run_estimate(f"W1 {name} direct", size, "scibench", "direct", N, prompt_stats))
    # ---- Week 1：large 池（单卡 14B；70B/72B 需 TP=2）
    runs.append(run_estimate("W1 R1-Distill-14B cot", "14B", "scibench", "cot", N,
                             prompt_stats, note="推理模型会写满 1024+，按 cot_reasoning 更准"))
    runs.append(run_estimate("W1 Llama-3.1-70B cot", "70B", "scibench", "cot", N,
                             prompt_stats, tp=2, note="bf16 需 2×80G；或单卡 4bit"))
    runs.append(run_estimate("W1 Qwen2.5-72B cot", "72B", "scibench", "cot", N,
                             prompt_stats, tp=2, note="同上"))
    # ---- 跨基准（可选，用于外部效度）
    runs.append(run_estimate("W1 7B cot @OlympiadBench(152 可判分)", "7B", "olympiadbench",
                             "cot", 152, prompt_stats))
    runs.append(run_estimate("W1 7B cot @MMLU 物理", "7B", "mmlu_physics", "cot", N_MMLU, prompt_stats))

    # ---- Week 2：敏感性（base + 三变体 = 1+0.944+0.743+0.112 = 2.80 倍生成量）
    mult = 1.0 + sum(COVERAGE.values())
    for size, name in (("7B", "Qwen2.5-7B"), ("14B", "Qwen2.5-14B")):
        runs.append(run_estimate(f"W2 {name} 敏感性(cot, 全量)", size, "scibench", "cot", N,
                                 prompt_stats, extra_generations=mult,
                                 note=f"base+3 变体，生成量 ×{mult:.2f}"))

    # ---- Week 3：因果追踪（HF 逐层前向，不能用 vLLM）
    # 每题 1 clean + 1 corrupt + K 个层采样；K≈16（层步长自动）
    gens = 2 + 16
    for size, name in (("7B", "Qwen2.5-7B"), ("14B", "Qwen2.5-14B")):
        runs.append(run_estimate(f"W3 {name} 因果追踪 12 题", size, "scibench", "direct", 12,
                                 prompt_stats, extra_generations=1.0, hf_path=True,
                                 note=f"HF 路径，每题 {gens} 次前向（1 clean+1 corrupt+16 层）")
                                 )

    # ---- Week 4：错误分类（只需对错样本，judge 用 14B）
    runs.append(run_estimate("W4 错误分类 judge(14B, 240 条)", "14B", "scibench", "direct", 240,
                             prompt_stats, note="judge prompt 比原题长，实际再加 ~40%", ))

    total_gpu_s = sum(r["gpu_seconds"] for r in runs)
    summary = {
        "n_runs": len(runs),
        "gpu_seconds": round(total_gpu_s, 1),
        "gpu_hours": round(total_gpu_s / 3600, 2),
        "a800_single_card_hours": round(total_gpu_s / 3600, 2),
        "with_safety_1_5x_gpu_hours": round(total_gpu_s / 3600 * 1.5, 2),
        "with_safety_2x_gpu_hours": round(total_gpu_s / 3600 * 2.0, 2),
    }
    return runs, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt-stats", default=str(ROOT / "results" / "prompt_token_stats.json"))
    p.add_argument("--throughput-scale", type=float, default=1.0,
                   help="吞吐缩放：你实测更快就 >1，更慢就 <1")
    p.add_argument("--json", default=str(ROOT / "results" / "gpu_budget_estimate.json"))
    args = p.parse_args()

    stats_path = Path(args.prompt_stats)
    if not stats_path.exists():
        print(f"缺少实测 prompt 统计 {stats_path}\n先跑： python scripts/prompt_stats.py")
        return 2
    blob = json.loads(stats_path.read_text(encoding="utf-8"))
    prompt_stats = blob["prompt_tokens"]

    runs, summary = build_plan(prompt_stats, throughput_scale=args.throughput_scale)

    print("假设（可改脚本顶部常量，或 --throughput-scale 整体缩放）：")
    print(f"  生成长度: direct {OUTPUT_TOKENS['direct']} tok/题, cot {OUTPUT_TOKENS['cot']} tok/题")
    print(f"  vLLM 聚合吞吐(A800-80G, bf16): " +
          ", ".join(f"{k}={int(v)}tok/s" for k, v in THROUGHPUT.items()))
    print(f"  启动开销: " + ", ".join(f"{k}={v}min" for k, v in STARTUP_MIN.items()))
    print(f"  变体覆盖率(实测): " + ", ".join(f"{k}={v:.1%}" for k, v in COVERAGE.items()))
    print()
    print(f"{'run':<42}{'卡':>3}{'生成数':>7}{'decode tok':>11}{'prefill tok':>12}{'卡时(分)':>10}")
    for r in runs:
        print(f"{r['label']:<42}{r['gpus']:>3}{r['generations']:>7}"
              f"{r['decode_tokens']:>11,}{r['prefill_tokens']:>12,}{r['gpu_minutes']:>10.1f}")
    print("-" * 88)
    print(f"合计：{summary['n_runs']} 个 run，"
          f"{summary['gpu_hours']} 卡时（含 1.5× 安全余量 {summary['with_safety_1_5x_gpu_hours']} 卡时，"
          f"2× 余量 {summary['with_safety_2x_gpu_hours']} 卡时）")

    write_json(args.json, {"assumptions": {
        "output_tokens": OUTPUT_TOKENS, "max_new_tokens": MAX_NEW_TOKENS,
        "throughput_tok_s": {k: round(v, 1) for k, v in THROUGHPUT.items()},
        "startup_min": STARTUP_MIN, "coverage": COVERAGE,
        "throughput_scale": args.throughput_scale,
    }, "summary": summary, "runs": runs})
    print(f"[save] {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
