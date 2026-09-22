"""三种变体生成器的语义不变量测试。

论文的 Week 2 结论完全建立在这三个变体上，所以它们的"语义边界"必须被钉死：
  * numeric：只动数值，绝不碰指数（x^2、m/s^2、10^{3} 的指数）、不碰单位
  * swap    ：只交换两个不同物理量的数值，题目其余部分逐字不变
  * wording ：数字、单位一个都不许动（否则就不是"表层措辞"了）
"""
from __future__ import annotations

import re

import pytest


# ---------------------------------------------------------------- numeric
def test_numeric_changes_numbers_only(sensitivity):
    q = "A mass of $2 \\mathrm{kg}$ moves at $3.5 \\mathrm{m/s}$ for 10 s."
    new, changed = sensitivity.numeric_perturb(q, factor=2.0, seed=42)
    assert new != q and changed
    # 单位与文字结构原样保留
    assert "\\mathrm{kg}" in new and "\\mathrm{m/s}" in new and "for" in new
    assert new.count("\\mathrm") == q.count("\\mathrm")


def test_numeric_leaves_exponents_alone(sensitivity):
    q = "The area is $x^2$ and the acceleration is $9.8 \\mathrm{m/s^2}$, $10^{3}$ Pa."
    new, _ = sensitivity.numeric_perturb(q, factor=2.0, seed=1)
    # 指数 2 / 3 与幂底数 10 都不能被扰动（把 10 乘 2 会得到 20^3=8000，语义已变）
    assert "x^2" in new and "10^{3}" in new and "m/s^2" in new


def test_numeric_keeps_sci_notation_mantissa_only(sensitivity):
    q = "The pressure is $1.0 \\times 10^{3} \\mathrm{Pa}$."
    new, changed = sensitivity.numeric_perturb(q, factor=2.0, seed=5)
    assert changed
    assert "10^{3}" in new and "10^{6}" not in new      # 指数必须原样
    assert new != q
    # 尾数 1.0 -> 2.0（6g 格式化后写作 2），即 2000 Pa，语义正确
    assert "2 \\times 10^{3}" in new or "2.0 \\times 10^{3}" in new


def test_numeric_ignores_year_apostrophe(sensitivity):
    q = "In the '88 experiment the speed was 12 m/s."
    new, changed = sensitivity.numeric_perturb(q, factor=2.0, seed=3)
    assert "'88" in new


# ---------------------------------------------------------------- swap
def test_swap_exchanges_two_quantities(sensitivity):
    q = "A $5 \\mathrm{kg}$ block is pushed with $20 \\mathrm{N}$."
    new, n = sensitivity.quantity_swap(q, seed=0)
    assert n == 2
    assert "$20 \\mathrm{kg}$" in new and "$5 \\mathrm{N}$" in new
    # 除两个数值外逐字不变
    assert re.sub(r"\d+", "#", new) == re.sub(r"\d+", "#", q)


def test_swap_needs_two_distinct_quantities(sensitivity):
    q = "A block of $5 \\mathrm{kg}$ and another of $9 \\mathrm{kg}$."
    new, n = sensitivity.quantity_swap(q, seed=0)
    assert n == 0 and new == q


# ---------------------------------------------------------------- wording
def test_wording_never_touches_numbers(sensitivity):
    samples = [
        "What is the acceleration of a $2 \\mathrm{kg}$ block moving at $4 \\mathrm{m/s}$?",
        "Find the total energy of the system with mass 12 kg and speed 3 m/s.",
        "A ball is released from rest at a height of 10 m. Determine the impact speed.",
    ]
    for q in samples:
        new, _ = sensitivity.wording_change(q, seed=7)
        assert [v for _, _, v in sensitivity._scan_numbers(new)] == \
               [v for _, _, v in sensitivity._scan_numbers(q)], q


def test_wording_preserves_capitalization(sensitivity):
    q = "What is the velocity of the car?"
    new, changed = sensitivity.wording_change(q, seed=0)
    assert changed
    assert new[0].isupper(), new


def test_build_variant_reports_subset_only(sensitivity):
    rows = [
        {"id": "a", "question": "What is the speed of 5 kg at 2 m/s?", "answer": "1"},
        {"id": "b", "question": "", "answer": "0"},
    ]
    out = sensitivity.build_variant(rows, "numeric", seed=1)
    assert all(r["id"].endswith("-numeric") for r in out)
    assert all(isinstance(r["question"], str) and r["question"] for r in out)


# ---------------------------------------------------------------- corrupt（因果追踪）
def test_corrupt_replaces_quantities_but_keeps_structure(causal):
    """SciBench 题面全是 LaTeX，corrupt 必须连 $1000 \\mathrm{~kg}$ 一起换掉。

    早期正则只认 [A-Za-z] 单位：数字没被换掉，反而把 `m/s^2` 的指数换成占位符，
    得到 `\\mathrm{m/s^zork blork}` —— corrupt 既没损坏物理量，又引入了畸形单位。
    """
    q = "A $2 \\mathrm{kg}$ block slides $3 \\mathrm{m}$ at $9.8 \\mathrm{m/s^2}$."
    c = causal.corrupt_quantities(q)
    for num in ("2", "3", "9.8"):
        assert num not in c, c
    assert c.count("zork blork") == 3
    assert "block slides" in c
    # 单位/幂结构不应被破坏成畸形写法
    assert "m/s^zork" not in c and "^zork" not in c


def test_corrupt_handles_sci_notation_and_bare_units(causal):
    c = causal.corrupt_quantities("The pressure is $1.0 \\times 10^{3} \\mathrm{Pa}$ "
                                  "and the speed is 20 m/s.")
    assert not any(ch.isdigit() for ch in c), c
    assert "zork blork" in c


def test_score_answer_uses_final_answer_not_first_number(causal):
    # 关键回归：CoT 输出里第一个数字是中间量，早期用 parse_number(整段) 会把答对算成答错
    text = "Step 1: we have 3 known values. Step 2: compute. ANSWER: 19.8"
    ok, perf = causal.score_answer(text, "19.8")
    assert ok and perf == 1.0
    # 判错时 perf 是连续分（1-相对误差），不是 0/1
    ok2, perf2 = causal.score_answer("ANSWER: 30", "19.8")
    assert not ok2
    assert perf2 == pytest.approx(1 - abs(30 - 19.8) / 19.8, abs=1e-9)
