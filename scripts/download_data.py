"""数据下载器。

- SciBench     ：从 GitHub 直接拉官方 JSON（urllib，无额外依赖）。
                  官方源 github.com/mandyyyyii/scibench 的 dataset/original/<subject>.json
- OlympiadBench：从 HuggingFace 拉（需 datasets 包，已在 requirements）。
                  含官方 final_answer + solution（解题步骤）。

落地：data/scibench/*.jsonl、data/olympiadbench/*.jsonl，src/data.py 会自动扫描。

用法：
  python scripts/download_data.py --benchmark scibench          # 默认，物理 6 科 ~323 题
  python scripts/download_data.py --benchmark scibench --all    # 全部科目
  python scripts/download_data.py --benchmark olympiadbench     # 物理（英+中）
  python scripts/download_data.py --benchmark all               # 两者都下
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import SCIBENCH_SUBJECT_DOMAIN, _domain_of  # 子域映射的唯一真源

DATA = ROOT / "data"

SCIBENCH_BRANCH = "main"
SCIBENCH_RAW = f"https://raw.githubusercontent.com/mandyyyyii/scibench/{SCIBENCH_BRANCH}/dataset/original"

# SciBench 科目 -> 显示域。**不要在这里另写一套映射**：早期版本把 quan/diff 都写成
# "general"，落盘后再加载时子域字段就固定成 general，154/323 题挤进同一个桶，
# 于是"模型 × 子域"主表直接失真。现在统一从 src.data 取。
SCIBENCH_PHYSICS = {s: _domain_of(s) for s in
                    ("matter", "class", "fund", "diff", "quan", "thermo")}
SCIBENCH_OTHER = {s: SCIBENCH_SUBJECT_DOMAIN[s] for s in
                  ("atkins", "chemmc", "stat", "calculus")}

# OlympiadBench 物理配置（HF: Hothan/OlympiadBench）
OLYMPIAD_PHYSICS = {
    "OE_TO_physics_en_COMP": "physics",     # 英语，开放作答，竞赛级
    "OE_TO_physics_zh_CEE": "physics",      # 中文，高考级
}


# ---------------------------------------------------------------- 通用
def _fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "dsh"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8")


def _fetch_json(url: str):
    return json.loads(_fetch(url))


def _strip_tex(s: str) -> str:
    """轻量清洗答案：只去掉 $ 与 \\mathrm{}/\\text{} 外壳，**保留数学结构**。

    早期实现是 `re.sub(r"\\$|\\\\mathrm|\\\\text|{|}", "", s)` —— 它把花括号一起删掉，
    于是 ``\\frac{\\sqrt{2}}{2}`` 变成 ``\\frac\\sqrt22``（不可解析的噪声）。
    LaTeX 转换交给 src/grading.py 的 normalize_text 统一处理，这里不要抢着破坏结构。
    """
    s = str(s or "").strip()
    s = s.replace("$", "")
    s = re.sub(r"\\(?:mathrm|text|operatorname)\s*\{([^{}]*)\}", r"\1", s)
    return s.strip()


def _first_scalar(x):
    """HF 字段常是单元素列表（如 final_answer=['2022.2']），取出标量。

    早期直接把列表 str() 进答案字段，落盘成 "['2022.2']" —— 351 条 OlympiadBench
    参考答案全军覆没（100% 不可判分），而且宽松解析器还会"碰巧"从里面抠出 2022.2，
    把 bug 藏起来。
    """
    if isinstance(x, (list, tuple)):
        for v in x:
            if v not in (None, ""):
                return v
        return ""
    return x


def _first(*vals):
    for v in vals:
        if v not in (None, "", [], [""]):
            return v
    return ""


# ---------------------------------------------------------------- SciBench（GitHub）
def _norm_problem_id(raw) -> str:
    """把 problemid 归一化为裸编号（用于匹配 _sol.json）。

    problem 的 id 形如 ``"Problem 3.40"``，_sol.json 里形如 ``"3.40"``；
    去掉 ``Problem`` 前缀、空白与其它非编号字符后对齐。
    """
    s = str(raw).strip()
    if s.lower() in ("none", "null") or not s:
        return ""
    s = re.sub(r"(?i)problem\s*", "", s)   # 去 "Problem " 前缀
    return s.strip()


def download_scibench(out_dir: Path, subjects: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for subject, domain in subjects.items():
        try:
            probs = _fetch_json(f"{SCIBENCH_RAW}/{subject}.json")
        except Exception as e:
            print(f"  [skip] scibench/{subject}: {type(e).__name__}: {e}")
            continue
        sol_by_id = {}
        try:
            for s in _fetch_json(f"{SCIBENCH_RAW}/{subject}_sol.json"):
                sid = _norm_problem_id(s.get("problemid"))
                if sid and s.get("solution"):
                    sol_by_id[sid] = s["solution"]
        except Exception as e:
            print(f"  [warn] scibench/{subject} solution 抓取失败: {e}")
        n = 0
        with open(out_dir / f"{subject}.jsonl", "w", encoding="utf-8") as f:
            for p in probs:
                q = _first(p.get("problem_text"))
                if not q:
                    continue
                pid = _norm_problem_id(p.get("problemid"))
                rec = {
                    "id": f"scibench-{subject}-{pid.replace(' ', '_')}",
                    "question": str(q),
                    "answer": _first(p.get("answer_number"), p.get("answer_latex")),
                    "solution": sol_by_id.get(pid, ""),
                    "domain": domain,
                    "subset": subject,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
        with_sol = sum(1 for p in probs if _norm_problem_id(p.get("problemid")) in sol_by_id)
        print(f"  scibench/{subject:<7} {domain:<12} {n:>4} 题 (含解 {with_sol})")
        total += n
    return total


# ---------------------------------------------------------------- OlympiadBench（HF）
def download_olympiadbench(out_dir: Path, configs: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from datasets import load_dataset
    except Exception as e:
        print(f"  [skip] OlympiadBench 需 datasets 包: {e}")
        return 0
    total = 0
    for config, domain in configs.items():
        try:
            ds = load_dataset("Hothan/OlympiadBench", config, split="train")
        except Exception as e:
            print(f"  [skip] olympiadbench/{config}: {type(e).__name__}: {e}")
            continue
        n = 0
        with open(out_dir / f"{config}.jsonl", "w", encoding="utf-8") as f:
            for i, r in enumerate(ds):
                q = r.get("question")
                if not q:
                    continue
                ans = _first_scalar(_first(r.get("final_answer")))
                sol = _first(r.get("solution"))
                rec = {
                    "id": f"olympiadbench-{config}-{i}",
                    "question": str(q),
                    "answer": _strip_tex(ans) if ans else "",
                    "solution": _join_list(sol),
                    "domain": r.get("subfield") or domain,
                    "subset": config,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
        print(f"  olympiadbench/{config:<24} {n:>4} 题 (含解 {n})")
        total += n
    return total


def _join_list(x):
    if isinstance(x, list):
        return "\n".join(str(i) for i in x)
    return str(x or "")


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="scibench",
                   choices=["scibench", "olympiadbench", "all"])
    p.add_argument("--scibench-subjects", default=None,
                   help="逗号分隔的 SciBench 科目；默认物理 6 科；'all'=全部")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    if args.out_dir:
        out_root = Path(args.out_dir)
    else:
        out_root = DATA

    total = 0
    if args.benchmark in ("scibench", "all"):
        if args.scibench_subjects == "all":
            subs = {**SCIBENCH_PHYSICS, **SCIBENCH_OTHER}
        elif args.scibench_subjects:
            subs = {s.strip(): SCIBENCH_PHYSICS.get(s.strip(), "general")
                    for s in args.scibench_subjects.split(",") if s.strip()}
        else:
            subs = dict(SCIBENCH_PHYSICS)
        total += download_scibench(out_root / "scibench", subs)

    if args.benchmark in ("olympiadbench", "all"):
        total += download_olympiadbench(out_root / "olympiadbench", OLYMPIAD_PHYSICS)

    print(f"\n完成：共 {total} 题")


if __name__ == "__main__":
    main()
