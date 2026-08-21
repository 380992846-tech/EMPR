"""答案提取与对错判定。

- extract_number: 从模型输出里抠出最终数值（优先找 "ANSWER:" 行，其次取末尾数字）。
- parse_number: 把字符串解析成 float（处理百分号、分数、科学计数）。
- judge: 数值比对（sympy），相对误差 <= tol 判对，近零用绝对差；
         可选 LLM-judge 作为辅助（默认关闭，避免循环论证）。

对错判定是评估的核心，务必先跑 `python -m src.grading --self-test` 看样例行为。
"""
from __future__ import annotations

import math
import re

# sympy 可选（装了就支持分数/符号表达式求值；没装也能用纯 Python 解析）
try:
    import sympy as sp
except Exception:  # pragma: no cover
    sp = None


# ---------------------------------------------------------------- number parsing
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
# 匹配 "x = 12.34"、"= 1/2"、"2.5%"、"-3e-5"
_FRAC_RE = re.compile(r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*/\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")


def _try_float(s: str):
    try:
        return float(s)
    except Exception:
        return None


def parse_number(s: str):
    """把各种文本解析成一个 float；失败返回 None。"""
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    # 1) sympy（可选）：支持 "sqrt(2)"、"1/2"、"3*pi" 等
    if sp is not None:
        try:
            return float(sp.sympify(s).evalf())
        except Exception:
            pass
    # 2) 分数 "a/b"
    m = _FRAC_RE.search(s)
    if m:
        a, b = _try_float(m.group(1)), _try_float(m.group(2))
        if a is not None and b not in (None, 0):
            return a / b
    # 3) 纯数字
    v = _try_float(s)
    if v is not None:
        return v
    # 4) 从字符串里抠第一个数
    m = _NUM_RE.search(s)
    if m:
        return _try_float(m.group())
    return None


def extract_number(text: str):
    """从模型输出提取最终答案数值。优先找 ANSWER: 行；末尾数值时跳过 ^2 这类单位指数。"""
    if not text:
        return None
    # 1) 显式 ANSWER: 行
    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("ANSWER:"):
            return parse_number(line[len("ANSWER:"):])
    # 2) "answer is X" / "result = X"
    m = re.search(r"(?:answer|result)\s*(?:is|:|=)\s*(.+)", text, re.I)
    if m:
        v = parse_number(m.group(1))
        if v is not None:
            return v
    # 3) 末尾最后一个数（启发式）。跳过 "^" 后的指数（如 m/s^2 的 2）。
    _plain = re.compile(r"(?<!\^)-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
    candidates = []
    for g in _plain.findall(text):
        v = parse_number(g)
        if v is not None:
            candidates.append(v)
    return candidates[-1] if candidates else None


# ---------------------------------------------------------------- judging
def judge(pred_text: str, reference: str, tol: float = 0.05, abs_tol: float = 0.01):
    """返回 (bool correct, float pred_value or None)。

    reference 支持纯数值字符串或带单位的（会先 parse_number）。
    """
    ref = parse_number(reference)
    if ref is None:
        return False, None
    pred = extract_number(pred_text)
    if pred is None:
        return False, None
    if abs(ref) <= abs_tol:
        correct = abs(pred - ref) <= abs_tol
    else:
        correct = abs(pred - ref) / abs(ref) <= tol
    return correct, pred


# ---------------------------------------------------------------- self-test
def _self_test():
    cases = [
        ("ANSWER: 19.8", "19.8", True),
        ("The velocity is 20 m/s.", "19.8", True),          # 5% tol
        ("final answer = 1.00 J", "1", True),
        ("ANSWER: 1/2", "0.5", True),
        ("The velocity is 2.5 m/s^2", "2.5", True),
        ("", "19.8", False),
        ("Nope, can't solve.", "19.8", False),
    ]
    for text, ref, want in cases:
        got, pred = judge(text, ref)
        status = "OK" if got == want else "FAIL"
        print(f"[{status}] pred={pred} ref={ref} got={got} want={want}")
    # 打印 extract_number 行为供人工检查
    for t in ["ANSWER: 42", "The acceleration is a = 2.5 m/s^2", "19.8", "no answer"]:
        print(f"extract({t!r}) -> {extract_number(t)}")


if __name__ == "__main__":
    _self_test()
