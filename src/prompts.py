"""提示词模板（来自 config.yaml，也可编程覆盖）。"""
from __future__ import annotations

from string import Template

_ZERO_COT = Template(
    "Solve the following physics problem step by step. Show all work.\n"
    "At the end, give the final numeric answer on its own line, prefixed with \"ANSWER:\".\n"
    "Problem: $question"
)
_DIRECT = Template(
    "Answer the following physics problem directly. Give only the final numeric answer.\n"
    "Problem: $question"
)


def build_prompts(rows, mode="cot", template=None):
    """rows: data.load_benchmark 的输出。返回与 rows 等长的 prompt 列表。"""
    tpl = template or (_ZERO_COT if mode == "cot" else _DIRECT)
    return [tpl.substitute(question=r["question"]) for r in rows]
