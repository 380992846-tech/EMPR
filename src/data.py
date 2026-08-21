"""基准数据加载：SciBench / OlympiadBench / MMLU-physics。

设计：
  1. 优先尝试多个已知 HF 数据集名（字段可能不同，做防御式提取）。
  2. 失败/无 HF 时回退到本地目录 data/<name>/ 或 data/<name>.jsonl。

SciBench 官方源：github.com/mandyyyyii/scibench（按 subject 的 JSON）。
下载：git clone 或手动放到 data/scibench/，脚本会扫描目录里所有 *.json / *.jsonl，
对每条记录用常见键（problem/question、solution、answer/final_answer）防御式提取。

每个样本归一化为 dict:
    {"id", "question", "answer", "solution", "domain", "subset"}
"""
from __future__ import annotations

import json
import os
from pathlib import Path

try:
    from datasets import load_dataset
    _HAS_HF = True
except Exception:  # pragma: no cover
    _HAS_HF = False


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# 各基准可能的 HF 数据集名（按优先级尝试）
HF_CANDIDATES = {
    "scibench": ["IACS/SciBench", "lmms-lab-encoder/SciBench_Math", "xw27/scibench"],
    "olympiadbench": ["Hothan/OlympiadBench"],
    "mmlu_physics": ["cais/mmlu"],
}


# ---------------------------------------------------------------- 归一化工具
def _norm(rows, name, subset, question_keys, answer_keys, solution_keys,
          domain_keys=(), extra_filter=None):
    """从任意行的 dict 里按候选键抽字段。"""
    out = []
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            continue
        q = _first(r, question_keys)
        if not q:
            continue
        if extra_filter and not extra_filter(r):
            continue
        ans = _first(r, answer_keys)
        sol = _first(r, solution_keys)
        dom = _first(r, domain_keys) or subset
        out.append({
            "id": f"{name}-{subset}-{i}",
            "question": str(q),
            "answer": str(ans or ""),
            "solution": str(sol or ""),
            "domain": _domain_of(dom),
            "subset": subset,
        })
    return out


def _first(d, keys):
    for k in keys:
        if k in d and d[k] not in (None, "", []):
            return d[k]
    return ""


def _is_physics(r, keys=("subject", "subset", "domain", "field", "category")):
    s = " ".join(str(r.get(k, "")) for k in keys).lower()
    return any(t in s for t in ("phys", "mech", "therm", "electr", "optic", "wave"))


# ---------------------------------------------------------------- SciBench
def load_scibench(subsets=("phys",), max_n=None):
    if _HAS_HF:
        for ds_name in HF_CANDIDATES["scibench"]:
            try:
                ds = load_dataset(ds_name, split="test")
                return _norm_scibench_hf(ds, subsets, max_n)
            except Exception as e:
                print(f"[data] HF '{ds_name}' 加载失败: {type(e).__name__}")
    # 本地目录回退：目录由 download_data.py 精挑（默认纯物理科目），不再做域关键词过滤
    return _load_local_dir("scibench", ("*",), max_n)


def _norm_scibench_hf(ds, subsets, max_n):
    # HF 上不同镜像字段不一；统一抽取
    rows = list(ds)
    if subsets != ("*",):
        rows = [r for r in rows if _is_physics(r)]
    out = _norm(rows, "scibench", "-".join(subsets) if subsets != ("*",) else "all",
                question_keys=["problem", "question", "text", "prompt"],
                answer_keys=["answer", "final_answer", "result", "output"],
                solution_keys=["solution", "sol", "reasoning", "steps", "full_solution"],
                domain_keys=["subject", "domain", "category"])
    return out[:max_n] if max_n else out


# ---------------------------------------------------------------- OlympiadBench
def load_olympiadbench(subsets=("physics",), max_n=None):
    # 优先用 download_data.py 拉好的本地目录（含官方 final_answer + solution）
    try:
        rows = _load_local_dir("olympiadbench", subsets, max_n)
        if rows:
            return rows
    except FileNotFoundError:
        pass
    # 否则尝试 HF（config 为 OE_TO_physics_en_COMP 等，见 download_data.py）
    if _HAS_HF:
        for ds_name in HF_CANDIDATES["olympiadbench"]:
            try:
                configs = ["OE_TO_physics_en_COMP", "OE_TO_physics_zh_CEE"]
                rows = []
                for cfg in configs:
                    ds = load_dataset(ds_name, cfg, split="train")
                    for i, r in enumerate(ds):
                        q = r.get("question")
                        if not q:
                            continue
                        rows.append({
                            "id": f"olympiadbench-{cfg}-{i}",
                            "question": str(q),
                            "answer": str((r.get("final_answer") or [""])[0]),
                            "solution": _join_solution(r.get("solution")),
                            "domain": r.get("subfield") or "physics",
                            "subset": cfg,
                        })
                if rows:
                    return rows[:max_n] if max_n else rows
            except Exception as e:
                print(f"[data] HF '{ds_name}' 加载失败: {type(e).__name__}")
    return _load_local_dir("olympiadbench", subsets, max_n)


def _join_solution(x):
    if isinstance(x, list):
        return "\n".join(str(i) for i in x)
    return str(x or "")


# ---------------------------------------------------------------- MMLU
def load_mmlu_physics(max_n=None):
    if not _HAS_HF:
        return _load_local_dir("mmlu_physics", ("*",), max_n)
    try:
        ds = load_dataset("cais/mmlu", "physics", split="test")
        rows = list(ds)
        out = []
        for i, r in enumerate(rows):
            letters = ["A", "B", "C", "D", "E"][: len(r["choices"])]
            opts = "".join(f"\n{letters[j]}. {r['choices'][j]}" for j in range(len(r["choices"])))
            # MMLU 的 answer 是选项下标(0-4)，转成字母以便与 MCQ 判定/模型输出对齐
            ans_idx = int(r["answer"]) if str(r["answer"]).isdigit() else -1
            ans = letters[ans_idx] if 0 <= ans_idx < len(letters) else str(r["answer"])
            out.append({
                "id": f"mmlu-physics-{i}",
                "question": f"{r['question']}{opts}",
                "answer": ans,
                "solution": "",
                "domain": "mcq",
                "subset": "mmlu_physics",
            })
        return out[:max_n] if max_n else out
    except Exception as e:
        print(f"[data] HF MMLU 加载失败: {e}")
    return _load_local_dir("mmlu_physics", ("*",), max_n)


# ---------------------------------------------------------------- 本地目录/文件回退
def _load_local_dir(name, subsets, max_n):
    """扫描 data/<name>/ 下所有 .json/.jsonl，防御式提取。"""
    d = DATA_DIR / name
    if not d.exists():
        return _load_local_jsonl(name, subsets, max_n)
    out = []
    files = sorted(d.rglob("*.json")) + sorted(d.rglob("*.jsonl"))
    for fp in files:
        try:
            if fp.suffix == ".jsonl":
                rows = [json.loads(l) for l in fp.read_text(encoding="utf-8").splitlines() if l.strip()]
            else:
                data = json.loads(fp.read_text(encoding="utf-8"))
                rows = _flatten_json(data)
            batch = _norm(rows, name, fp.stem,
                          question_keys=["problem", "question", "text", "prompt"],
                          answer_keys=["answer", "final_answer", "result", "output"],
                          solution_keys=["solution", "sol", "reasoning", "steps", "full_solution"],
                          domain_keys=["subject", "domain", "category", "field"],
                          extra_filter=(None if subsets == ("*",) else _is_physics))
            out.extend(batch)
            if max_n and len(out) >= max_n:
                return out[:max_n]
        except Exception as e:
            print(f"[data] 解析 {fp.name} 失败: {e}")
    return out[:max_n] if max_n else out


def _load_local_jsonl(name, subsets, max_n):
    path = DATA_DIR / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"未找到 {name} 数据：HF 加载失败，本地 {path} 也不存在。\n"
            f"请下载后放到 data/{name}/ 或 data/{name}.jsonl（见 README）。")
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            r = json.loads(line)
            s = " ".join(str(r.get(k, "")) for k in ("subject", "domain", "subset")).lower()
            if subsets != ("*",) and not any(t in s for t in subsets):
                continue
            r.setdefault("id", f"{name}-{i}")
            out.append(r)
            if max_n and len(out) >= max_n:
                break
    return out


def _flatten_json(data):
    """把 JSON 结构摊平成记录列表：dict->[dict 本身或其值]，list->列表。"""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "problem" in data or "question" in data:
            return [data]
        # 可能按 id 索引的 dict: {"p1": {...}, ...}
        vals = list(data.values())
        if vals and all(isinstance(v, dict) for v in vals):
            return vals
        # 可能 {"problems": [...]}
        for k in ("problems", "data", "examples", "questions", "items"):
            if k in data:
                return _flatten_json(data[k])
    return [data]


def _domain_of(s):
    s = str(s).lower()
    # SciBench 科目名（显式优先，避免量子/数学被混入 "general"）
    if s in ("class", "classical") or "mech" in s:
        return "mechanics"
    if s in ("thermo",) or "therm" in s:
        return "thermodynamics"
    if s in ("matter",) or "wave" in s or "optic" in s:
        return "optics"
    if s in ("quan", "quantum") or "quant" in s:
        return "quantum"
    if s in ("calculus", "diff") or "calculus" in s or "diff" in s:
        return "math"
    if s in ("fund", "stat", "atkins", "chemmc", "general"):
        return "general"
    # 通用关键词
    if any(k in s for k in ("mech", "dynam", "kinemat", "force")):
        return "mechanics"
    if any(k in s for k in ("therm", "heat")):
        return "thermodynamics"
    if any(k in s for k in ("electromagnet", "elec", "magnet")):
        return "electromagnetism"
    if any(k in s for k in ("optic", "light", "wave")):
        return "optics"
    if any(k in s for k in ("quant", "quantum")):
        return "quantum"
    return s or "unknown"


# ---------------------------------------------------------------- 统一入口
def load_benchmark(name, subsets=None, max_problems=None):
    name = name.lower()
    if name == "scibench":
        rows = load_scibench(subsets or ("phys",), max_problems)
    elif name == "olympiadbench":
        rows = load_olympiadbench(subsets or ("physics",), max_problems)
    elif name == "mmlu_physics":
        rows = load_mmlu_physics(max_problems)
    else:
        raise ValueError(f"未知基准: {name}")
    if max_problems:
        rows = rows[:max_problems]
    return rows
