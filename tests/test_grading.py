"""判分（src/grading.py）回归测试。

判分是整个项目的有效性底座：一个"答对却判错"的系统性偏差，会直接把论文主结论
写成反的。这里把每个真实踩过的坑都钉成用例。
"""
from __future__ import annotations

import pytest

from src.grading import (judge, judge_ex, extract_number, extract_choice,
                         extract_answer_span, parse_number, parse_reference,
                         normalize_text, _CASES)


@pytest.mark.parametrize("text,ref,want", _CASES)
def test_self_check_cases(text, ref, want):
    """与 `python -m src.grading` 同一套用例，保证两条路径不会各说各话。"""
    r = judge_ex(text, ref)
    if want is None:
        assert r["status"] == "unscorable_ref"
    else:
        assert r["status"] != "unscorable_ref"
        assert r["correct"] is want, r


def test_thousands_separator_not_split():
    # 早期 bug：NUM_RE 把 "10,000" 切成 10 和 000，取到 0 -> 答对被判错
    assert extract_number("The mass is $10,000 \\mathrm{kg}$.") == 10000.0
    assert judge("ANSWER: 10000", "10,000")[0]


def test_scientific_notation_whole_value():
    # 早期 bug：只取到尾数 2.5
    assert extract_number("ANSWER: $2.5 \\times 10^{3}$") == 2500.0
    assert extract_number("ANSWER: 2.5e3") == 2500.0
    assert extract_number("ANSWER: 2.5 x 10^3") == 2500.0


def test_last_answer_line_wins():
    # CoT 模型常先写一个中间结论，再给最终答案；取第一段会错
    text = "First I get ANSWER: 999\nBut correcting the unit:\nANSWER: 19.8"
    assert extract_number(text) == 19.8


def test_unicode_and_latex_normalization():
    assert extract_number("ANSWER: \u22120.16") == -0.16          # U+2212
    assert extract_number("ANSWER: 3\u00b2") == 9.0                # 上标平方
    assert normalize_text("1 \\times 2") == "1 x 2"


def test_boxed_and_inline_forms():
    assert extract_number("$\\boxed{4.8}$") == 4.8
    assert extract_number("So the final answer is 1.00 J") == 1.0


def test_mcq_letter_not_confused_with_prose():
    # 早期 bug：\b[A-E]\b 全文搜索，"A car..." 的 A 被当成选项
    assert extract_choice("A car starts from rest and accelerates.") is None
    assert extract_choice("ANSWER: B") == "B"
    assert judge("A car accelerates, so ANSWER: C", "C")[0]
    assert not judge("A car accelerates.", "A")[0]


def test_unscorable_reference_is_flagged_not_counted_wrong():
    r = judge_ex("ANSWER: 5", "\\frac{1}{2}mv^2 + something")
    assert r["status"] == "unscorable_ref"
    assert r["scorable"] is False


def test_reference_parser_is_strict_but_not_brittle():
    """参考答案必须严进：宽松的 parse_number 会把 x^2+2x 读成 1，污染分母。"""
    # 可解析（真数值）
    for s, want in [("19.8", 19.8), ("-0.16", -0.16), ("7.25%", 7.25), ("1/2", 0.5),
                    ("2.5e3", 2500.0), ("2.5 x 10^3", 2500.0), ("1.0 J", 1.0),
                    ("0.5 kg/mol", 0.5), ("$4.8$", 4.8), ("ANSWER: 3.14", 3.14)]:
        assert parse_reference(s) == pytest.approx(want), s
    assert parse_reference("sqrt(2)") == pytest.approx(2 ** 0.5, rel=1e-6)
    # LaTeX 的 \times 也算科学计数法
    assert parse_reference("2.5 \\times 10^{3}") == 2500.0
    # 不可解析（符号/句子/空）
    for s in ["x^2 + 2x", "\\frac{1}{2}mv^2", "", "see figure 2 and 3", None]:
        assert parse_reference(s) is None, s
    # parse_number 依然宽松（用于模型输出），两者分工明确：
    # 对符号表达式它会"抠出第一个数字"，正是这种宽松让参考答案必须走严格解析
    assert parse_number("x^2 + 2x") == 2.0


def test_model_failure_to_answer_is_scored_wrong_but_flagged():
    r = judge_ex("I cannot solve this.", "19.8")
    assert r["status"] == "unparsed_pred"
    assert r["scorable"] is True and r["correct"] is False


def test_tolerance_behaviour():
    assert judge("ANSWER: 20.5", "19.8")[0]          # 3.5% < 5%
    assert not judge("ANSWER: 21.5", "19.8")[0]      # 8.6% > 5%
    assert judge("ANSWER: 0.005", "0")[0]            # 近零用绝对容差
    assert not judge("ANSWER: 0.5", "0")[0]


def test_parse_number_edge_cases():
    assert parse_number("sqrt(2)") == pytest.approx(2 ** 0.5, rel=1e-6)
    assert parse_number("1/2") == 0.5
    assert parse_number("") is None
    assert parse_number("no digits here") is None


def test_extract_answer_span_prefers_answer_line():
    assert extract_answer_span("blah\nANSWER: 42\nblah") == "42"
    assert extract_answer_span("") is None
