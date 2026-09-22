"""提示词模板（唯一真源）。

模板**不在** config.yaml 中：判分与推理都只依赖本文件，避免两处定义不一致。
调整问法只需修改本文件。

与判分的一致性（重要）：``src/grading.py`` 的提取顺序是
``ANSWER:`` 行 -> ``\\boxed{}`` -> "answer is X" -> 末行数 -> 全文最后一个数。
因此**每种模式都必须显式要求 "ANSWER:" 行**；否则判定只能依赖"全文最后一个数"兜底，
而 Week 2 的敏感性实验默认使用 direct 模式，精度会整体下降。

另：选择题（MMLU / 带 A. B. C. D. 选项的题）不使用数值模板，会自动切换为要求字母的模板。
"""
from __future__ import annotations

import re
from string import Template

# ---------------------------------------------------------------- 模板
_ZERO_COT = Template(
    "Solve the following physics problem step by step. Show all work.\n"
    "At the end, give the final numeric answer on its own line, prefixed with \"ANSWER:\".\n"
    "Problem: $question"
)
_DIRECT = Template(
    "Answer the following physics problem directly with a single number (no explanation).\n"
    "Give the final answer on its own line, prefixed with \"ANSWER:\".\n"
    "Problem: $question"
)
_MCQ_COT = Template(
    "Solve the following multiple-choice physics problem step by step. Show all work.\n"
    "At the end, give the chosen option letter on its own line, prefixed with \"ANSWER:\".\n"
    "Problem: $question"
)
_MCQ_DIRECT = Template(
    "Answer the following multiple-choice physics problem.\n"
    "Give the chosen option letter on its own line, prefixed with \"ANSWER:\".\n"
    "Problem: $question"
)

TEMPLATES = {
    ("cot", False): _ZERO_COT,
    ("direct", False): _DIRECT,
    ("cot", True): _MCQ_COT,
    ("direct", True): _MCQ_DIRECT,
}

MODES = ("cot", "direct")

_OPTION_RE = re.compile(r"(?:^|\n)\s*[A-F][\.\)]\s+\S")


def is_mcq(question: str) -> bool:
    """粗判选择题：题干里出现 >=2 个 "A. xxx" / "B) xxx" 形式的选项。"""
    return len(_OPTION_RE.findall(question or "")) >= 2


def build_prompt(question: str, mode: str = "cot", template=None) -> str:
    """单条 prompt。mode ∈ {cot, direct}；选择题自动换模板。"""
    if template is not None:
        return template.substitute(question=question)
    if mode not in MODES:
        raise ValueError(f"未知模式: {mode!r}（可选 {MODES}）")
    tpl = TEMPLATES[(mode, is_mcq(question))]
    return tpl.substitute(question=question)


def build_prompts(rows, mode: str = "cot", template=None):
    """rows: data.load_benchmark 的输出。返回与 rows 等长的 prompt 列表。"""
    return [build_prompt(r["question"], mode=mode, template=template) for r in rows]


if __name__ == "__main__":  # 快速看一眼 prompt 长什么样
    from src.data import load_benchmark
    rows = load_benchmark("scibench", max_problems=1)
    for m in MODES:
        print(f"----- {m} -----")
        print(build_prompt(rows[0]["question"], mode=m)[:400])
    print("----- mcq 检测 -----")
    print(is_mcq("Which is true?\nA. one\nB. two\nC. three"), is_mcq("A car moves at 3 m/s"))
