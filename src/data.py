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
import re
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


# ---------------------------------------------------------------- 子域映射（唯一真源）
# SciBench 科目 -> 物理子域。**这里是唯一的真源**：src/data.py 的加载路径与
# scripts/download_data.py 的落盘路径都从这里取，避免"落盘写 general、加载时又
# 想标成 quantum"这类两处不一致（早期版本正是如此，导致 154/323 题被并进 general）。
SCIBENCH_SUBJECT_DOMAIN = {
    "class": "mechanics",        # 经典力学
    "classical": "mechanics",    # 兼容 "Classical Mechanics" 写法
    "mechanics": "mechanics",
    "matter": "optics",          # 波动与近代物理（X 射线衍射、光电效应、能级）
    "thermo": "thermodynamics",  # 热力学/统计物理
    "thermodynamics": "thermodynamics",
    "quan": "quantum",           # 量子化学/量子力学
    "quantum": "quantum",
    "diff": "math",              # 微分方程、级数等数学工具题
    "fund": "general",           # 综合基础题（跨子域混合）
    "stat": "statistics",
    "atkins": "chemistry",
    "chemmc": "chemistry",
    "calculus": "math",
}


# ---------------------------------------------------------------- 归一化工具
def _norm(rows, name, subset, question_keys, answer_keys, solution_keys,
          domain_keys=(), extra_filter=None, domain_from_subset=False):
    """从任意行的 dict 里按候选键抽字段。

    domain_from_subset=True 时忽略行内自带的 domain 字段，改用文件名/子集名推导子域
    （SciBench 本地 jsonl 的 domain 字段是早期版本写下的，已在部分科目上过时）。
    """
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
        dom = _domain_of(subset if domain_from_subset else (_first(r, domain_keys) or subset))
        out.append({
            "id": f"{name}-{subset}-{i}",
            "question": str(q),
            "answer": str(ans or ""),
            "solution": str(sol or ""),
            "domain": dom,
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
def _prefer_hf() -> bool:
    """EMPR_PREFER_HF=1 时强制走 HF（默认本地优先，见 load_scibench）。"""
    return os.environ.get("EMPR_PREFER_HF", "").strip().lower() not in ("", "0", "false", "no")


def _has_local(name: str) -> bool:
    d = DATA_DIR / name
    return d.exists() and any(d.glob("*.json*"))


def load_scibench(subsets=("phys",), max_n=None):
    """加载 SciBench。

    **本地优先**：`data/scibench/*.jsonl`（download_data.py 落盘）存在时直接用它，
    不再先试 HF。理由：HF 镜像的名字/字段/切分随时会变，同一份代码在不同机器上
    可能悄悄加载到不同数据集（实测镜像 lmms-lab-encoder/SciBench_Math 与
    xw27/scibench 会抛 ValueError，而一旦某个镜像哪天可用，题目数、子域分布、
    答案格式都会变，结果不可比）。要强制走 HF：设 EMPR_PREFER_HF=1。
    """
    if _has_local("scibench") and not _prefer_hf():
        rows = _load_local_dir("scibench", ("*",), max_n)
        if rows:
            print(f"[data] scibench: 用本地 data/scibench/ 共 {len(rows)} 题"
                  f"（本地优先保证可复现；强制 HF 请设 EMPR_PREFER_HF=1）")
            return rows
        print("[data] 本地 data/scibench/ 为空，改试 HF")
    if _HAS_HF:
        for ds_name in HF_CANDIDATES["scibench"]:
            try:
                ds = load_dataset(ds_name, split="test")
                print(f"[data] scibench: 用 HF 数据集 '{ds_name}'")
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
    """扫描 data/<name>/ 下所有 .json/.jsonl，防御式提取。

    `domain_from_subset=True`（SciBench）：子域由文件名（科目）推导，而不是用行内
    自带的 domain 字段——早期 download_data.py 把 quan/diff 都写成了 "general"，
    直接用行内字段会让 154/323 题挤进同一个桶，子域表失去意义。
    """
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
                          extra_filter=(None if subsets == ("*",) else _is_physics),
                          domain_from_subset=(name == "scibench"))
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
    s = str(s).strip()
    low = s.lower()
    # 0) "physics.class" / "physics/class" 这类带前缀的写法，取最后一段
    if any(sep in low for sep in (".", "/", ":")):
        tail = re.split(r"[./:]", low)[-1].strip()
        if tail and tail != low:
            mapped = _domain_of(tail)
            if mapped and mapped != tail:
                return mapped
    # 1) 显式科目名（唯一真源，避免 quan/diff 被并进 general）
    if low in SCIBENCH_SUBJECT_DOMAIN:
        return SCIBENCH_SUBJECT_DOMAIN[low]
    # 2) 兼容旧写法（"Classical"、"_sol" 之类）
    for key, dom in SCIBENCH_SUBJECT_DOMAIN.items():
        if low == key or low.startswith(key + "_"):
            return dom
    # 3) 通用关键词
    if any(k in low for k in ("mech", "dynam", "kinemat", "force")):
        return "mechanics"
    if any(k in low for k in ("therm", "heat")):
        return "thermodynamics"
    if any(k in low for k in ("electromagnet", "elec", "magnet")):
        return "electromagnetism"
    if any(k in low for k in ("optic", "light", "wave")):
        return "optics"
    if any(k in low for k in ("quant", "quantum")):
        return "quantum"
    if any(k in low for k in ("calculus", "algebra", "equation")):
        return "math"
    return low or "unknown"


def composition(rows):
    """返回 `{domain: n}`（按题数降序），用于打印/记录数据构成。"""
    counts = {}
    for r in rows:
        counts[r.get("domain", "unknown")] = counts.get(r.get("domain", "unknown"), 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def format_composition(rows):
    """一行式数据构成，便于写进日志/结果 json（复现时不用再猜题数分布）。"""
    comp = composition(rows)
    total = len(rows) or 1
    return " | ".join(f"{d}={n}({n / total:.0%})" for d, n in comp.items())


# ---------------------------------------------------------------- 统一入口
def stratified_take(rows, max_n, key="subset"):
    """按 subset 轮转取前 max_n 题（跨科目均衡），保持各科目内部原有顺序。

    为什么需要它：``rows[:max_n]`` 取到的是文件排序里的前几个科目——
    SciBench 按字母序扫描文件，``--max-problems 40`` 会拿到清一色的 mechanics
    题，然后在论文里写成"模型在 SciBench 上 40 题"（子域表直接失真）。
    轮转采样是确定性的（不用随机数），所以不影响可复现性。
    """
    if not max_n or max_n >= len(rows):
        return list(rows)
    buckets = {}
    for r in rows:
        buckets.setdefault(r.get(key) or "unknown", []).append(r)
    order = sorted(buckets)                      # 确定性顺序
    picked, i = [], 0
    while len(picked) < max_n and any(buckets[k] for k in order):
        k = order[i % len(order)]
        if buckets[k]:
            picked.append(buckets[k].pop(0))
        i += 1
    return picked[:max_n]


def load_benchmark(name, subsets=None, max_problems=None, stratified=True, key="subset"):
    """统一入口。stratified=True 时按科目轮转抽样（默认开，见 stratified_take）。"""
    name = name.lower()
    if name == "scibench":
        rows = load_scibench(subsets or ("phys",), None)
    elif name == "olympiadbench":
        rows = load_olympiadbench(subsets or ("physics",), None)
    elif name == "mmlu_physics":
        rows = load_mmlu_physics(None)
    else:
        raise ValueError(f"未知基准: {name}")
    if max_problems:
        rows = stratified_take(rows, max_problems, key=key) if stratified else rows[:max_problems]
    return rows
