"""判定质量抽查：检查对错判定（grading）是否可靠。

用法：
  python scripts/check_grading.py --file results/eval_*.json

它会报出四件在花钱全量跑之前必须确认的事：
  1) ``status`` 分布：correct / incorrect / unparsed_pred / unscorable_ref
     - ``unparsed_pred``（模型没给出可解析答案）比例高 -> 判分或提示词有问题
     - ``unscorable_ref``（参考答案本身解析不了）-> 这些题已剔除出分母，需在论文里写明
  2) 按子域看 unparsed_pred 比例：某个子域特别高通常意味着题面里的答案形式特殊
  3) 参考答案可解析性（严格解析器 parse_reference）
  4) 抽样打印（优先 null / 判错样例），供人工核对

判定目标：unparsed_pred 尽量低、被判 wrong 的确实是错的。
若 unparsed_pred 偏高，先调 src/grading.py 或 src/prompts.py，别急着全量跑。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.grading import parse_reference

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

_STATUS_DESC = {
    "correct": "判定正确",
    "incorrect": "判定错误",
    "unparsed_pred": "模型输出抠不出数值（计为错）",
    "unscorable_ref": "参考答案不可解析（已剔除分母）",
}


def _summarize(detail):
    n = len(detail)
    counts = {}
    for d in detail:
        st = d.get("status") or ("correct" if d.get("correct") else "incorrect")
        counts[st] = counts.get(st, 0) + 1
    scorable = [d for d in detail if d.get("scorable", True)]
    n_correct = sum(1 for d in scorable if d.get("correct"))

    print(f"总样本            : {n}")
    for st, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {st:<16}: {c:>5}  ({c / max(n, 1):.1%})  {_STATUS_DESC.get(st, '')}")
    print(f"\n可判分样本        : {len(scorable)}")
    print(f"其中判定正确      : {n_correct}  ({n_correct / max(len(scorable), 1):.1%})")

    # 按子域看 unparsed_pred：某个子域异常高 = 该子域答案形式特殊
    by_dom = {}
    for d in detail:
        dom = d.get("domain", "?")
        by_dom.setdefault(dom, [0, 0])
        by_dom[dom][0] += 1
        if (d.get("status") == "unparsed_pred"):
            by_dom[dom][1] += 1
    bad = {k: v for k, v in by_dom.items() if v[1]}
    if bad:
        print("\n按子域的 unparsed_pred 比例（定位题面/答案形式问题）：")
        for dom, (tot, bad_n) in sorted(bad.items(), key=lambda kv: -kv[1][1] / max(kv[1][0], 1)):
            print(f"  {dom:<18} {bad_n}/{tot}  ({bad_n / tot:.0%})")

    returns = [d for d in detail if d.get("pred_val") is None
               and d.get("status") != "unscorable_ref"]
    return returns


def _print_samples(detail, nulls, k=8):
    for d in nulls[:k]:
        print("\n----- [抠不出答案] -----")
        print(f"问题: {d.get('question', '')[:110]}")
        print(f"ref : {d.get('ref', '')!r}")
        print(f"pred: {d.get('pred_raw', '')[:220]!r}")
    wrong = [d for d in detail if not d.get("correct") and d.get("scorable", True)]
    random.Random(0).shuffle(wrong)
    for d in wrong[:max(0, k - min(len(nulls), k))]:
        print("\n----- [判错] -----")
        print(f"问题: {d.get('question', '')[:110]}")
        print(f"ref : {d.get('ref', '')!r} -> {d.get('pred_val')!r}")
        print(f"pred: {d.get('pred_raw', '')[:220]!r}")

    # 严格解析器视角下不可判分的参考答案
    bad_ref = [d for d in detail if parse_reference(d.get("ref")) is None]
    if bad_ref:
        print(f"\n[注意] {len(bad_ref)} 条参考答案无法解析为数值（符号/表达式/含多数字）：")
        for d in bad_ref[:5]:
            print(f"  - ref={d.get('ref', '')!r}  (id={d.get('id')})")
        print("  -> 这些题不计入分母。若某模型这类题很多，单独在论文里说明覆盖率。")


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
    print(f"加载: {data.get('model')} | {data.get('benchmark')} | {data.get('mode')} "
          f"| n={data.get('n')} | commit={data.get('meta', {}).get('git_commit', '?')}")
    nulls = _summarize(detail)
    _print_samples(detail, nulls, k=args.samples)


if __name__ == "__main__":
    main()
