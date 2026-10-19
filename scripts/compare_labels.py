"""对比两位标注器（同族 vs 异族）的错误类型标签一致性。

为什么需要它：Week 4 的错误分类如果用与被评模型同族的模型当标注器，可能存在系统性偏好。
用异族标注器（如 Llama）重标一遍，报告两者的一致率与 Cohen's kappa，
才能判断"E2 是主要错误类型"这一结论是否稳健。

用法：
  python scripts/compare_labels.py --a results/error_labels.json \
      --b results/error_labels_llama.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

KEYS = ("E1", "E2", "E3", "N")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cohen_kappa(pairs, labels=KEYS):
    """Cohen's kappa（多类别）。pairs 为 [(a_label, b_label)]。"""
    n = len(pairs)
    if n == 0:
        return float("nan")
    po = sum(1 for a, b in pairs if a == b) / n
    pe = 0.0
    for k in labels:
        pa = sum(1 for a, _ in pairs if a == k) / n
        pb = sum(1 for _, b in pairs if b == k) / n
        pe += pa * pb
    if pe >= 1.0:
        return float("nan")
    return (po - pe) / (1 - pe)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="标注器 A 的标签文件（同族）")
    p.add_argument("--b", required=True, help="标注器 B 的标签文件（异族）")
    p.add_argument("--json", default=None, help="把对比结果写入 json")
    args = p.parse_args()

    a, b = load(args.a), load(args.b)
    common = [k for k in a if k in b]
    print(f"标注器 A: {len(a)} 条 | 标注器 B: {len(b)} 条 | 共同标注: {len(common)} 条")

    pairs = [(a[k], b[k]) for k in common if a[k] in KEYS and b[k] in KEYS]
    skipped = len(common) - len(pairs)
    if skipped:
        print(f"[note] 有 {skipped} 条其中一方给出 '?' 或 'S'，未纳入一致性统计")

    if not pairs:
        print("没有可比对的标签")
        return 2

    print(f"\n{'':<8}" + "".join(f"{'B:' + k:<10}" for k in KEYS))
    for ka in KEYS:
        row = [sum(1 for x, y in pairs if x == ka and y == kb) for kb in KEYS]
        print(f"A:{ka:<6}" + "".join(f"{v:<10}" for v in row) +
              f"  (A 侧合计 {sum(row)})")

    agree = [1 for x, y in pairs if x == y]
    po = len(agree) / len(pairs)
    kappa = cohen_kappa(pairs)
    print(f"\n一致率 (observed agreement): {po:.3f}  ({len(agree)}/{len(pairs)})")
    print(f"Cohen's kappa             : {kappa:.3f}")

    # 各自的主类别占比
    print(f"\n{'类别':<6}{'A 占比':<10}{'B 占比':<10}")
    dist = {}
    for k in KEYS:
        pa = sum(1 for x, _ in pairs if x == k) / len(pairs)
        pb = sum(1 for _, y in pairs if y == k) / len(pairs)
        dist[k] = {"a": round(pa, 4), "b": round(pb, 4)}
        print(f"{k:<6}{pa:>7.1%}   {pb:>7.1%}")

    main_a = max(dist, key=lambda k: dist[k]["a"])
    main_b = max(dist, key=lambda k: dist[k]["b"])
    print(f"\n主错误类型：A = {main_a}，B = {main_b}"
          f"  -> {'一致' if main_a == main_b else '不一致（结论需谨慎）'}")

    kappa_txt = ("几乎一致" if kappa >= 0.8 else "高度一致" if kappa >= 0.6 else
                 "中等一致" if kappa >= 0.4 else "一致性偏低")
    print(f"kappa 解读：{kappa_txt}（Landis & Koch 分级）")

    if args.json:
        out = {"n_a": len(a), "n_b": len(b), "n_common": len(common),
               "n_compared": len(pairs), "n_skipped": skipped,
               "observed_agreement": round(po, 4), "cohens_kappa": round(kappa, 4),
               "distribution": dist, "main_error_a": main_a, "main_error_b": main_b,
               "main_error_agree": main_a == main_b}
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
        print(f"[save] {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
