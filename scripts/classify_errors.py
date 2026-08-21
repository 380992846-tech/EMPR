"""Week 4 · 错误类型分类（E1/E2/E3）。

错误类型（与论文定义一致）：
  E1 定律误用   : 用了错误的物理定律（该用动量守恒却用能量守恒）
  E2 代数/计算错: 定律选对但算错（符号、单位换算、数值）
  E3 约束违背   : 未同时满足多个约束/一致性检查（独立算力却互不检验）
  N            : 判为错但模型其实接近/应视为正确（判定误报）
  S            : 跳过（看不清楚）

两种标注方式：
  1) 手动标注（默认，推荐给"只有自己"）：逐个打印错误样例，你输入 E1/E2/E3/N/S。
     进度存 results/error_labels.json，可随时中断、下次继续（已标的不重复）。
  2) LLM 自动标注：--judge-model <模型>，批量让模型打标（注意：判官与被评模型同族可能有偏）。

用法：
  # 手动标注（从某次评估的错误样本开始）
  python scripts/classify_errors.py --file results/eval_*.json

  # LLM 自动标注
  python scripts/classify_errors.py --file results/eval_*.json --judge-model Qwen/Qwen2.5-14B-Instruct

  # 只看统计报告（基于已存 labels）
  python scripts/classify_errors.py --report --file results/eval_*.json

  # 两模型错误分布做卡方对比
  python scripts/classify_errors.py --report --file A.json --compare-file B.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import llm as llm_mod
from src.stats import chi2_contingency

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
LABELS_FILE = RESULTS / "error_labels.json"

_JUDGE_PROMPT = (
    "You are grading a physics solution. Below is a physics problem, its correct reference answer, "
    "and a model's answer that is WRONG. Classify the model's error into exactly one category.\n"
    "E1 = wrong physical law applied (used wrong law entirely).\n"
    "E2 = right approach but algebraic/arithmetic/unit error.\n"
    "E3 = failed to satisfy multiple constraints / internal inconsistency / missing consistency check.\n"
    "N = the answer is actually correct or the judged value is close enough (false negative by grader).\n"
    "Reply with a single token: E1, E2, E3, or N.\n\n"
    "PROBLEM: {question}\n"
    "REFERENCE ANSWER: {ref}\n"
    "MODEL OUTPUT: {pred}\n"
    "CATEGORY:"
)


# ---------------------------------------------------------------- 加载
def _load_eval(file_or_path):
    path = Path(file_or_path)
    if not path.exists():
        path = RESULTS / file_or_path
    data = json.loads(path.read_text(encoding="utf-8"))
    wrong = [d for d in data["detail"] if not d.get("correct")]
    return data, wrong


def _load_labels():
    if LABELS_FILE.exists():
        return json.loads(LABELS_FILE.read_text(encoding="utf-8"))
    return {}


def _save_labels(labels):
    LABELS_FILE.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- 手动标注
def _manual_label(data, wrong):
    labels = _load_labels()
    todo = [d for d in wrong if d["id"] not in labels]
    print(f"共 {len(wrong)} 个错误样本，未标注 {len(todo)} 个（可 Ctrl-C 中断，进度会保存）")
    if not todo:
        print("已全部标注。")
        return labels
    for i, d in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] id={d['id']}  domain={d['domain']}")
        print(f"  问题: {d['question'][:180]}")
        print(f"  ref : {d['ref']!r}")
        print(f"  pred: {d['pred_raw'][:260]!r}")
        while True:
            ans = input("  类别 (E1/E2/E3/N/S，q=退出): ").strip().upper()
            if ans == "Q":
                _save_labels(labels)
                print("已保存，退出。")
                return labels
            if ans in ("E1", "E2", "E3", "N", "S"):
                labels[d["id"]] = ans
                _save_labels(labels)
                break
            print("  输入无效，请输入 E1/E2/E3/N/S")
    return labels


# ---------------------------------------------------------------- LLM 标注
def _judge_label(data, wrong, judge_model, backend, cfg):
    labels = _load_labels()
    todo = [d for d in wrong if d["id"] not in labels]
    if not todo:
        return labels
    gen = llm_mod.make_generator(judge_model, backend=backend, seed=cfg["eval"]["seed"])
    prompts = [_JUDGE_PROMPT.format(question=d["question"], ref=d["ref"], pred=d["pred_raw"])
               for d in todo]
    outs = gen.gen_batch(prompts)
    for d, out in zip(todo, outs):
        tag = (out.strip() or "?")[:2].upper()
        if tag.startswith("E1"):
            labels[d["id"]] = "E1"
        elif tag.startswith("E2"):
            labels[d["id"]] = "E2"
        elif tag.startswith("E3"):
            labels[d["id"]] = "E3"
        elif tag.startswith("N"):
            labels[d["id"]] = "N"
        else:
            labels[d["id"]] = "?"
    _save_labels(labels)
    return labels


# ---------------------------------------------------------------- 报告
def _report(data, compare=None, label_keys=("E1", "E2", "E3", "N")):
    labels = _load_labels()
    count = {k: sum(1 for d in data["detail"] if labels.get(d["id"]) == k) for k in label_keys}
    total_lab = sum(count.values())
    print(f"模型: {data.get('model')}  | 已标注错误样本: {total_lab}")
    print(f"{'类别':<6}{'数量':<6}{'占比':<8} 说明")
    desc = {"E1": "定律误用", "E2": "代数/计算错", "E3": "约束违背", "N": "判定误报"}
    for k in label_keys:
        pct = count[k] / total_lab if total_lab else 0
        print(f"{k:<6}{count[k]:<6}{pct:>6.1%}  {desc.get(k, '')}")
    # 按 domain 的错误类型
    print("\n按 domain：")
    doms = {}
    for d in data["detail"]:
        lbl = labels.get(d["id"])
        if lbl in label_keys:
            doms.setdefault(d["domain"], {}).setdefault(lbl, 0)
            doms[d["domain"]][lbl] += 1
    for dom, cc in sorted(doms.items()):
        tot = sum(cc.values())
        print(f"  {dom:<14} " + "  ".join(f"{k}={cc.get(k,0)}" for k in label_keys) +
              f"  (n={tot})")
    # 对比另一个分布（卡方）
    if compare is not None:
        clabels = _load_labels_for(compare)
        a = [count[k] for k in label_keys]
        b = [sum(1 for d in compare["detail"] if clabels.get(d["id"]) == k) for k in label_keys]
        if sum(a) > 0 and sum(b) > 0:
            chi2, p, dof = chi2_contingency([a, b])
            print(f"\n两模型错误分布卡方: chi2={chi2:.2f}, df={dof}, p={p:.4f}")


def _load_labels_for(data):
    # 对比文件用同一 labels 文件（按 id 匹配）
    return _load_labels()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True, help="run_eval --save 生成的 results/eval_*.json")
    p.add_argument("--judge-model", default=None, help="LLM 自动标注（默认手动）")
    p.add_argument("--backend", default="auto")
    p.add_argument("--report", action="store_true", help="只出统计报告")
    p.add_argument("--compare-file", default=None, help="另一个 eval json，做错误分布卡方对比")
    args = p.parse_args()

    data, wrong = _load_eval(args.file)
    cfg = {"eval": {"seed": 42}}

    if args.report:
        cmp = _load_eval(args.compare_file)[0] if args.compare_file else None
        _report(data, cmp)
        return

    if args.judge_model:
        _judge_label(data, wrong, args.judge_model, args.backend, cfg)
    else:
        _manual_label(data, wrong)

    print("\n标注完成，分布：")
    _report(data)


if __name__ == "__main__":
    main()
