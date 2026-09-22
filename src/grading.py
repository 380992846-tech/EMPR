"""答案提取与对错判定（评估的有效性核心）。

设计原则：**任何"模型答对了却被判错"的系统性偏差都会直接毁掉主结论**，所以这里
优先保证提取稳健，并显式区分三类结果，而不是把它们都压成 False：

  correct        —— 判定为对
  incorrect      —— 判定为错（含模型根本没给出可解析答案：这是模型的失败，计入分母）
  unscorable_ref —— **参考答案本身无法解析**（符号表达式/带复杂单位）。这类样本不计入
                    分母，而是单独报 n_unscored，避免"数据集标注问题"被算成"模型不会"。

提取顺序（越靠前越可信）：
  1) 最后一行 ``ANSWER:`` / ``答案:`` 前缀
  2) LaTeX ``\\boxed{...}`` / ``\\fbox{...}``
  3) ``the answer is X`` / ``final answer = X`` / ``result: X``
  4) 末行里最后一个数（跳过 ``^2`` 这类指数）
  5) 全文最后一个数（仅在前四步都失败时兜底）

关键行为（每条都有对应回归用例，见 tests/test_grading.py）：
  * 千分位逗号先规范化，``"10,000 kg"`` 解析为 10000 而非 0
  * 科学计数法整体解析：``"2.5 x 10^3"`` 解析为 2500 而非尾数 2.5
  * 多段 ``ANSWER:`` 时取**最后一段**（CoT 模型的最终答案位于末尾，取第一段会命中中间结论）
  * Unicode 减号 ``−``、上标 ``²``、``×`` 以及 LaTeX ``\\times`` 统一规范化后再解析
  * MCQ 仅在前缀行或 "answer is X" 附近识别选项字母，
    不对全文做 ``\\b[A-E]\\b`` 搜索（避免把 "A car starts..." 的 A 当作选项）

用法自检：``python -m src.grading``（打印每个用例的判定过程）。
"""
from __future__ import annotations

import re

# sympy 可选（装了就支持分数/sqrt/pi 等表达式；没装也能用纯 Python 解析）
try:
    import sympy as sp
except Exception:  # pragma: no cover
    sp = None


# ---------------------------------------------------------------- 文本规范化
_UNICODE_FIXES = {
    "\u2212": "-",   # − 数学减号
    "\u2013": "-",   # – en dash
    "\u2014": "-",   # — em dash
    "\u00d7": "x",   # × 乘号
    "\u00b2": "^2",  # ² 上标
    "\u00b3": "^3",  # ³
    "\u2070": "^0",
    "\u00b7": "*",   # · 中点乘
    "\u2044": "/",   # ⁄ 分数斜杠
    "\uff1a": ":",   # ：全角冒号
    "\uff0c": ",",   # ，全角逗号
    "\u3002": ".",   # 。全角句号
}
_COMMA_IN_NUMBER = re.compile(r"(?<=\d),(?=\d{3}\b)")
# LaTeX 里的乘除/数学宏先转成普通符号，避免 "2.5 \times 10^{3}" 解析失败
_LATEX_FIXES = (
    (re.compile(r"\\times\b"), "x"),
    (re.compile(r"\\cdot\b"), "*"),
    (re.compile(r"\\div\b"), "/"),
    (re.compile(r"\\left|\\right|\\,|\\;|\\!|\\ "), " "),
    (re.compile(r"\\mathrm\s*\{([^{}]*)\}"), r"\1"),
    (re.compile(r"\\text\s*\{([^{}]*)\}"), r"\1"),
    (re.compile(r"\\%"), "%"),
    # \frac{a}{b} -> ((a)/(b))：**必须保留括号**，否则 "\frac{1}{2}mv^2" 这类
    # 符号表达式会被降级成一串裸数字（"12mv^2"），参考答案就判错了。
    (re.compile(r"\\dfrac|\\tfrac"), r"\\frac"),
)
_FRAC_LOOP = (re.compile(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}"), r"((\1)/(\2))")
_SQRT_LOOP = (re.compile(r"\\sqrt\s*\{([^{}]*)\}"), r"sqrt(\1)")
# LaTeX 数学函数 -> sympy 函数名。很多"看起来是符号答案"其实有唯一数值
# （\cos^{10}(\frac{\pi}{10})、\cot\frac{\pi}{8}），映射后就能判分。
_MATH_FUNCS = {
    "pi": "pi", "sin": "sin", "cos": "cos", "tan": "tan", "cot": "cot",
    "sec": "sec", "csc": "csc", "ln": "log", "log": "log", "exp": "exp",
    "arcsin": "asin", "arccos": "acos", "arctan": "atan",
    "sinh": "sinh", "cosh": "cosh", "tanh": "tanh",
}
# 隐式乘法："2sqrt(5)" / "2 pi" -> "2*sqrt(5)" / "2*pi"
_IMPLICIT_MUL = re.compile(
    r"(?<=[\d)])\s*(?=(?:sqrt|pi|sin|cos|tan|cot|sec|csc|log|exp|asin|acos|atan)\b)")

ANSWER_LABEL = re.compile(r"^\s*(?:\*\*)?\s*(?:final\s+)?(?:ANSWER|答案)\s*(?:\*\*)?\s*[:：]", re.I)
BOXED = re.compile(r"\\(?:boxed|fbox)\s*\{([^{}]*)\}")
INLINE_ANSWER = re.compile(
    r"(?:final\s+answer|the\s+answer|answer|result|答案)\s*(?:is|are|=|:|：)\s*(.+)", re.I)


def normalize_text(text: str) -> str:
    """统一 Unicode 符号、LaTeX 宏与千分位逗号，让后续正则只处理一种写法。"""
    if not text:
        return ""
    out = str(text)
    for k, v in _UNICODE_FIXES.items():
        out = out.replace(k, v)
    for pat, repl in _LATEX_FIXES:
        out = pat.sub(repl, out)
    # LaTeX 数学函数（\pi、\cos …）先变成可求值形式
    for name, fn in _MATH_FUNCS.items():
        out = re.sub(r"\\" + name + r"\b", fn, out)
    # \frac/\sqrt 可能嵌套（\frac{\sqrt{2}}{2}），多跑几轮直到稳定
    for _ in range(5):
        new = _FRAC_LOOP[0].sub(_FRAC_LOOP[1], out)
        new = _SQRT_LOOP[0].sub(_SQRT_LOOP[1], new)
        if new == out:
            break
        out = new
    out = _IMPLICIT_MUL.sub("*", out)
    # "10,000" -> "10000"（但保留 "a, b" 这种普通逗号）
    out = _COMMA_IN_NUMBER.sub("", out)
    return out


# ---------------------------------------------------------------- 数值解析
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_FRAC_RE = re.compile(
    r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*/\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
_SCI_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:[xX*]\s*)?10\s*\^\s*\{?\s*([-+]?\d+)\s*\}?")
_SUPER_SCI_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*[xX*]\s*10([-+]\d+)")  # 10-3 这种被压平的写法
_E_RE = re.compile(r"(-?\d+(?:\.\d+)?)[eE]([-+]?\d+)")


def _try_float(s: str):
    try:
        return float(s)
    except Exception:
        return None


def _sympy_value(s: str):
    """用 sympy 求值（仅对短字符串，避免病态表达式把评估卡死）。"""
    if sp is None or len(s) > 60:
        return None
    try:
        v = sp.sympify(s.replace("^", "**"), evaluate=True)
        if v.is_number:
            return float(v.evalf())
    except Exception:
        return None
    return None


def parse_number(s: str):
    """把各种文本解析成一个 float；失败返回 None。"""
    if s is None:
        return None
    s = normalize_text(s).strip()
    if not s:
        return None
    # 0) 科学计数法（"2.5 x 10^3" / "2.5e3" / "2.5 x 103"）——必须整体解析，
    #    否则会被下面的裸数字规则截成尾数 2.5
    for pat in (_SCI_RE, _SUPER_SCI_RE, _E_RE):
        m = pat.search(s)
        if m:
            mant, exp = _try_float(m.group(1)), _try_float(m.group(2))
            if mant is not None and exp is not None:
                return mant * (10 ** int(exp))
    # 1) 纯数字
    v = _try_float(s)
    if v is not None:
        return v
    # 2) 分数 "a/b"
    m = _FRAC_RE.search(s)
    if m:
        a, b = _try_float(m.group(1)), _try_float(m.group(2))
        if a is not None and b not in (None, 0):
            return a / b
    # 3) sympy（sqrt(2)、3*pi 等）
    v = _sympy_value(s)
    if v is not None:
        return v
    # 4) 从字符串里抠第一个数
    m = _NUM_RE.search(s)
    if m:
        return _try_float(m.group())
    return None


def parse_reference(s: str):
    """**严格**解析参考答案；无法确定是数值时返回 None（= 该题不可判分）。

    为什么不直接用宽松的 parse_number：它的兜底规则是"抠出第一个数字"，于是
    ``x^2 + 2x`` 会被读成 ``2``、``\\frac{1}{2}mv^2`` 会被读成 ``1``，
    然后拿模型答案跟这个假数字比 —— 样本被当成"可判分"，分母被污染，
    而错误完全看不出来。参考答案必须严进宽出。

    接受：``19.8`` / ``-0.16`` / ``7.25%`` / ``1/2`` / ``2.5e3`` / ``2.5 x 10^3``
    / ``1.0 J`` / ``0.5 kg/mol`` / ``sqrt(2)``（纯数值表达式）。
    拒绝：含符号的表达式、含多个数字的句子、空串。
    """
    if s is None:
        return None
    t = normalize_text(str(s)).strip()
    if not t:
        return None
    t = re.sub(r"^\s*(?:ANSWER|答案)\s*[:：]?\s*", "", t, flags=re.I).strip()
    t = t.strip("$").strip()
    if not t:
        return None

    # 1) 科学计数法必须**先**判：否则 "2.5 x 10^3" 会被"数值 + 单位尾巴"规则
    #    匹配成 2.5（尾巴 "x 10^3" 看起来像单位）
    for pat in (_SCI_RE, _SUPER_SCI_RE, _E_RE):
        m = pat.fullmatch(t)
        if m:
            mant, exp = _try_float(m.group(1)), _try_float(m.group(2))
            if mant is not None and exp is not None:
                return mant * (10 ** int(exp))
    # 2) 数值 + 可选单位尾巴（单位里允许字母/斜杠/幂/百分号）
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*"
                     r"([A-Za-z%\u00b0][A-Za-z0-9%\u00b0^/*\s.\u00b7-]*)?", t)
    if m:
        return _try_float(m.group(1))
    # 3) 分数 a/b
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)", t)
    if m:
        a, b = _try_float(m.group(1)), _try_float(m.group(2))
        if a is not None and b not in (None, 0):
            return a / b
    # 4) 纯数值表达式（sqrt(2)、pi）——必须没有自由符号
    if sp is not None and len(t) <= 60:
        try:
            expr = sp.sympify(t.replace("^", "**"), evaluate=True)
            if getattr(expr, "is_number", False) and not getattr(expr, "free_symbols", None):
                return float(expr.evalf())
        except Exception:
            pass
    return None


def _parse_with_sci(s: str):
    """先整体科学计数法，再退回 parse_number。"""
    return parse_number(s)


# ---------------------------------------------------------------- 答案提取
def _answer_lines(text: str):
    """返回所有 ``ANSWER:`` 行（已去掉标签）与它们的行号，保持文件顺序。"""
    hits = []
    for i, line in enumerate(text.splitlines()):
        if ANSWER_LABEL.match(line):
            hits.append((i, ANSWER_LABEL.sub("", line).strip()))
    return hits


def extract_answer_span(text: str):
    """返回最可能的"最终答案"文本片段（不做数值化），失败返回 None。"""
    if not text:
        return None
    t = normalize_text(text)
    # 1) 最后一段 ANSWER: 行（CoT 模型的最终答案在末尾，取第一段会命中中间结论）
    lines = _answer_lines(t)
    if lines:
        return lines[-1][1]
    # 2) \boxed{...}
    boxes = BOXED.findall(t)
    if boxes:
        return boxes[-1].strip()
    # 3) "answer is X" / "final answer = X"
    inline = INLINE_ANSWER.findall(t)
    if inline:
        return inline[-1].strip()
    # 4) 末行（最后 3 行里找，模型常把结果单独放一行）
    tail = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if tail:
        return tail[-1]
    return None


def extract_number(text: str):
    """从模型输出提取最终答案数值；失败返回 None。"""
    if not text:
        return None
    span = extract_answer_span(text)
    if span:
        v = _parse_with_sci(span)
        if v is not None:
            return v
    # 兜底：全文最后一个数（跳过 "^" 后的指数，如 m/s^2 的 2）
    t = normalize_text(text)
    plain = re.compile(r"(?<!\^)-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
    candidates = [parse_number(g) for g in plain.findall(t)]
    candidates = [c for c in candidates if c is not None]
    return candidates[-1] if candidates else None


def extract_choice(text: str, letters: str = "ABCDEF"):
    """MCQ：提取选项字母。只认 ANSWER: 行 / "answer is X"，避免全文搜字母误命中普通单词首字母。"""
    if not text:
        return None
    t = normalize_text(text)
    lines = _answer_lines(t)
    pat = re.compile(rf"\b([{letters}])\b")
    for _, payload in reversed(lines):
        m = pat.search(payload)
        if m:
            return m.group(1).upper()
    inline = INLINE_ANSWER.findall(t)
    for payload in reversed(inline):
        m = pat.search(payload)
        if m:
            return m.group(1).upper()
    # 最后兜底：末行"就是"一个字母（可带粗体/标点）。长的自然语言末行不算，
    # 否则 "A car starts from rest..." 里的 A 会被当选项。
    tail = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if tail and len(tail[-1]) <= 12:
        found = set(pat.findall(tail[-1]))
        if len(found) == 1:
            return found.pop().upper()
    return None


# ---------------------------------------------------------------- 判定
def judge_ex(pred_text: str, reference: str, tol: float = 0.05, abs_tol: float = 0.01):
    """完整判定，返回 dict（run_eval 记录用）。

    字段：status / correct / scorable / pred / ref / detail
      status ∈ {correct, incorrect, unscorable_ref, unparsed_pred}
      scorable=False 的样本应从分母里剔除（参考答案本身无法解析）。
    """
    ref_str = normalize_text(str(reference)).strip()
    # MCQ：reference 是单个字母
    if re.fullmatch(r"[A-Fa-f]", ref_str):
        choice = extract_choice(pred_text or "", letters="ABCDEF")
        ok = choice is not None and choice == ref_str.upper()
        return {"status": "correct" if ok else "incorrect", "correct": bool(ok),
                "scorable": True, "pred": choice, "ref": ref_str.upper(),
                "detail": "mcq"}

    ref = parse_reference(ref_str)
    if ref is None:
        return {"status": "unscorable_ref", "correct": False, "scorable": False,
                "pred": None, "ref": ref_str, "detail": "reference not parseable"}

    pred = extract_number(pred_text or "")
    if pred is None:
        return {"status": "unparsed_pred", "correct": False, "scorable": True,
                "pred": None, "ref": ref, "detail": "no numeric answer extracted"}

    if abs(ref) <= abs_tol:
        ok = abs(pred - ref) <= abs_tol
    else:
        ok = abs(pred - ref) / abs(ref) <= tol
    return {"status": "correct" if ok else "incorrect", "correct": bool(ok),
            "scorable": True, "pred": pred, "ref": ref,
            "detail": "" if ok else "numeric mismatch"}


def judge(pred_text: str, reference: str, tol: float = 0.05, abs_tol: float = 0.01):
    """向后兼容的二元判定：返回 (correct, pred_value)。

    注意：参考答案无法解析时返回 (False, None)——**不要**直接用这个函数统计正确率，
    请用 ``judge_ex`` 并把 ``scorable=False`` 的样本剔除（run_eval 已这样做）。
    """
    r = judge_ex(pred_text, reference, tol=tol, abs_tol=abs_tol)
    return r["correct"], r["pred"]


# ---------------------------------------------------------------- self-test
# (模型输出, 参考答案, 期望判定)  —— 回归测试集，防止"改提取逻辑把对的改错"
# 期望判定 want 语义：True=判对 / False=判错 / None=参考答案不可判分（不算模型错）
_CASES = [
    ("ANSWER: 19.8", "19.8", True),
    ("Let me compute...\nANSWER: 19.8\n", "19.8", True),
    ("ANSWER: 20", "19.8", True),                       # 5% 容差内
    ("ANSWER: 30", "19.8", False),
    ("first guess ANSWER: 999\nso ANSWER: 19.8", "19.8", True),   # 取最后一段
    ("The velocity is 20 m/s.", "19.8", True),
    ("final answer = 1.00 J", "1", True),
    ("ANSWER: 1/2", "0.5", True),
    ("ANSWER: 2.14", "2.14", True),
    ("$\\boxed{4.8}$", "4.8", True),
    ("The mass is $10,000 \\mathrm{kg}$.", "10000", True),        # 千分位
    ("ANSWER: $2.5 \\times 10^{3}$", "2500", True),               # 科学计数法
    ("ANSWER: 2.5e3", "2500", True),
    ("ANSWER: -0.16", "-0.16", True),
    ("ANSWER: \u22120.16", "-0.16", True),                        # − 减号
    ("ANSWER: 7.25%", "7.25", True),
    ("ANSWER: 15.425", "15.425", True),
    ("", "19.8", False),
    ("Nope, can't solve.", "19.8", False),
    ("ANSWER: A", "A", True),                                     # MCQ
    ("A car starts from rest and reaches ANSWER: B", "B", True),  # MCQ：不被 "A car" 骗
    ("A car starts from rest and accelerates.", "A", False),      # MCQ：无答案 -> 错
    ("ANSWER: 19.8", "x^2 + 2x", None),                           # 参考答案不可判分
]


def _self_test(verbose: bool = True) -> int:
    """跑回归用例；返回失败数（0 表示全过）。"""
    fails = 0
    for text, ref, want in _CASES:
        r = judge_ex(text, ref)
        if want is None:
            ok = r["status"] == "unscorable_ref"
        else:
            ok = (r["status"] != "unscorable_ref") and (r["correct"] is want)
        if not ok:
            fails += 1
        if verbose:
            flag = "OK  " if ok else "FAIL"
            print(f"[{flag}] want={want!s:<5} got={r['correct']!s:<5} "
                  f"status={r['status']:<14} pred={r['pred']!s:<8} "
                  f"ref={ref!r:<14} text={text[:38]!r}")
    if verbose:
        print(f"\n自检：{len(_CASES) - fails}/{len(_CASES)} 通过")
    return fails


if __name__ == "__main__":
    import sys
    sys.exit(1 if _self_test() else 0)
