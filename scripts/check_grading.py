"""判定质量抽查：检查对错判定（grading）是否可靠。

两种用法：
  1) 检查已保存的评估结果：python scripts/check_grading.py --file results/eval_*.json
  2) 先跑一小段再检查：python scripts/run_eval.py --model ... --max-problems 20 --save
     然后 --file 指向刚生成的 json。

它会：
  - 统计 pred_val=null（答案没抠出来）的比例 —— 这是判定最大的风险源
  - 统计 correct 比例
  - 打印一批样例（优先打印 null 和判定为错但答案其实接近的），让你人工核对
  - 提示常见问题（ref 是符号/单位、pred 含多余文本等）

判定的目标：null 比例尽量低、被判定 wrong 的确实是错的。若 null 高，先调 src/grading.py 再全量跑。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.grading import parse_number

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def _summarize(detail):
    n = len(detail)
    nulls = [d for d in detail if d.get("pred_val") is None]
    correct = [d for d in detail if d.get("correct")]
    n_wrong = n - len(correct)
    print(f"总样本         : {n}")
    print(f"pred_val=null  : {len(nulls)}  ({len(nulls)/max(n,1):.1%})   <- 答案没抠出来")
    print(f"判定正确(correct): {len(correct)}  ({len(correct)/max(n,1):.1%})")
    print(f"判定错误(wrong) : {n_wrong}  ({n_wrong/max(n,1):.1%})")
    return nulls


def _print_samples(detail, nulls, k=8):
    # 优先打印 null 的样例
    shown = 0
    for d in nulls[:k]:
        shown += 1
        print("\n----- [pred_val=null] -----")
        print(f"问题: {d.get('question','')[:110]}")
        print(f"ref : {d.get('ref','')!r}")
        print(f"pred: {d.get('pred_raw','')[:220]!r}")
    # 再从判定为错的里面抽几个打印
    wrong = [d for d in detail if not d.get("correct")]
    import random
    random.Random(0).shuffle(wrong)
    for d in wrong[: max(0, k - shown)]:
        print("\n----- [wrong] -----")
        print(f"问题: {d.get('question','')[:110]}")
        print(f"ref : {d.get('ref','')!r}")
        print(f"pred: {d.get('pred_raw','')[:220]!r}")

    # 检查 ref 是否可解析（若 ref 本身是符号/表达式，数值判定会天然失效）
    bad_ref = [d for d in detail if parse_number(d.get("ref")) is None]
    if bad_ref:
        print(f"\n[注意] 有 {len(bad_ref)} 条的参考答案 ref 无法解析为数值（符号/表达式/含单位）")
        for d in bad_ref[:5]:
            print(f"  - ref={d.get('ref','')!r}   (id={d.get('id')})")
        print("  -> 这些题数值判定会失效，建议：要么改 grading 支持符号等价，要么在评估里剔除/单列")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True, help="run_eval --save 生成的 results/eval_*.json")
    p.add_argument("--samples", type=int, default=8)
    args = p.parse_args()

    path = Path(args.file)
    if not path.exists():
        path = RESULTS / args.file
    data = json.loads(path.read_text(encoding="utf-8"))
    detail = data["detail"]
    print(f"加载: {data.get('model')} | {data.get('benchmark')} | {data.get('mode')}")
    nulls = _summarize(detail)
    _print_samples(detail, nulls, k=args.samples)


if __name__ == "__main__":
    main()
