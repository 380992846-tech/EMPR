"""实测各基准/模式的 prompt token 数（用 Qwen2.5 分词器），供 gpu_budget.py 使用。

prompt 长度是**实测**的（真实题面分词），不是估的；生成长度才是假设。
Qwen2.5 系列与 R1-Distill-Qwen 同分词器；Llama/Mistral 的英文 BPE 通常更碎，
估算时按 +10% 处理（gpu_budget.py 的吞吐常量里已隐含这层保守性）。

用法： python scripts/prompt_stats.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import data as dm            # noqa: E402
from src import meta as meta_mod      # noqa: E402
from src import prompts as pm         # noqa: E402

BENCHES = ("scibench", "olympiadbench", "mmlu_physics")
MODES = ("cot", "direct")
# 用最小的已缓存模型分词（同系列分词器一致，不必下 7B）
TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    out = {}
    for bench in BENCHES:
        try:
            rows = dm.load_benchmark(bench)
        except Exception as e:
            print(f"[skip] {bench}: {type(e).__name__}: {e}")
            continue
        for mode in MODES:
            lens = sorted(len(tok(pm.build_prompt(r["question"], mode=mode))["input_ids"])
                          for r in rows)
            out[f"{bench}/{mode}"] = {
                "n": len(lens),
                "mean": round(statistics.mean(lens), 1),
                "p50": lens[len(lens) // 2],
                "p90": lens[int(len(lens) * 0.9)],
                "max": lens[-1],
                "total": sum(lens),
            }
    chat_overhead = len(tok.apply_chat_template(
        [{"role": "user", "content": "x"}], tokenize=True, add_generation_prompt=True)) - 1

    print(f"chat 模板额外开销: {chat_overhead} tokens/请求")
    print(f"{'baseline/mode':<26}{'n':>5}{'mean':>8}{'p50':>7}{'p90':>7}{'max':>7}{'total':>10}")
    for k, v in out.items():
        print(f"{k:<26}{v['n']:>5}{v['mean']:>8}{v['p50']:>7}{v['p90']:>7}{v['max']:>7}{v['total']:>10}")

    f = ROOT / "results" / "prompt_token_stats.json"
    meta_mod.write_json(f, {
        "meta": meta_mod.result_meta("prompt_stats.py", tokenizer=TOKENIZER),
        "tokenizer": TOKENIZER, "chat_overhead": chat_overhead, "prompt_tokens": out,
    })
    print(f"[save] {f}")


if __name__ == "__main__":
    main()
