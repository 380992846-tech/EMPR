"""数据加载 / 子域映射 / 抽样 / 提示词 测试。

重点固定两件事：
 1. 子域映射（`_domain_of`）：科目到子域必须一一对应，否则部分科目会被并入同一子域，
    子域统计失真。此处固定正确映射。
 2. 分层抽样：直接切片 `rows[:n]` 只会取到同一个科目，此处固定"按科目轮转"的行为。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src import data as data_mod
from src import prompts as prompts_mod

ROOT = Path(__file__).resolve().parent.parent
DATA_OK = (ROOT / "data" / "scibench").exists() and any(
    (ROOT / "data" / "scibench").glob("*.jsonl"))


# ---------------------------------------------------------------- 子域映射
@pytest.mark.parametrize("raw,want", [
    ("class", "mechanics"), ("Classical", "mechanics"), ("mechanics", "mechanics"),
    ("thermo", "thermodynamics"), ("matter", "optics"),
    ("quan", "quantum"), ("diff", "math"), ("fund", "general"),
    ("physics.class", "mechanics"),       # 带前缀的写法要能剥到科目
    ("astrophysics", "astrophysics"),     # 未知科目原样返回，不静默并进 general
])
def test_domain_mapping(raw, want):
    assert data_mod._domain_of(raw) == want


def test_known_subjects_have_distinct_domains():
    """四个物理科目不能都被映到同一个 domain（旧 bug 的体检项）。"""
    doms = {data_mod._domain_of(s) for s in ("class", "thermo", "matter", "quan")}
    assert len(doms) == 4


def test_composition_and_format():
    rows = [{"domain": "mechanics"}, {"domain": "mechanics"}, {"domain": "optics"}]
    assert data_mod.composition(rows) == {"mechanics": 2, "optics": 1}
    assert "mechanics=2" in data_mod.format_composition(rows)


# ---------------------------------------------------------------- 分层抽样
def _fake_rows():
    rows = []
    for sub, n in (("class", 10), ("quan", 3), ("thermo", 8)):
        rows += [{"id": f"{sub}-{i}", "subset": sub} for i in range(n)]
    return rows


def test_stratified_take_balances_subjects():
    picked = data_mod.stratified_take(_fake_rows(), 6, key="subset")
    counts = {s: sum(1 for r in picked if r["subset"] == s) for s in ("class", "quan", "thermo")}
    assert len(picked) == 6
    assert all(v >= 1 for v in counts.values()), counts
    assert max(counts.values()) - min(counts.values()) <= 2, counts


def test_stratified_take_is_deterministic_and_caps():
    a = data_mod.stratified_take(_fake_rows(), 5, key="subset")
    b = data_mod.stratified_take(_fake_rows(), 5, key="subset")
    assert [r["id"] for r in a] == [r["id"] for r in b]
    assert len(data_mod.stratified_take(_fake_rows(), 100, key="subset")) == 21


def test_naive_slice_would_be_biased():
    """说明为什么需要 stratified_take：直接切片只拿到第一个科目。"""
    rows = sorted(_fake_rows(), key=lambda r: r["subset"])
    naive = rows[:6]
    assert len({r["subset"] for r in naive}) == 1


# ---------------------------------------------------------------- MMLU（MCQ 分支）
def test_mmlu_physics_configs_exist_in_cais_mmlu():
    """回归：cais/mmlu 里没有 "physics" 这个 config，写错会让整条 MCQ sanity check 静默失效。

    这里不联网，只钉住"用的是真实存在的 config 名"这一事实（名字拼错就等于功能死掉）。
    """
    assert "physics" not in data_mod.MMLU_PHYSICS_CONFIGS
    assert set(data_mod.MMLU_PHYSICS_CONFIGS) == {
        "high_school_physics", "college_physics", "conceptual_physics"}


def test_mmlu_row_normalization():
    """把 HF 的 answer 下标转成字母，并在题干里拼出 A./B./C./D. 选项。"""
    fake = [
        {"question": "Q1?", "choices": ["one", "two", "three", "four"], "answer": 2},
        {"question": "Q2?", "choices": ["only"], "answer": 9},          # 越界 -> 保留原值
    ]
    rows = data_mod._norm_mmlu(fake, "conceptual_physics")
    assert rows[0]["answer"] == "C"
    assert "B. two" in rows[0]["question"] and rows[0]["domain"] == "mcq"
    assert rows[0]["id"] == "mmlu-conceptual_physics-0"
    assert rows[1]["answer"] == "9"


# ---------------------------------------------------------------- 本地数据
@pytest.mark.skipif(not DATA_OK, reason="需要 data/scibench/*.jsonl（先跑 download_data.py）")
def test_load_scibench_local():
    rows = data_mod.load_benchmark("scibench")
    assert len(rows) > 200
    assert all(r["question"] and r["answer"] for r in rows)
    assert all(r["domain"] and r["domain"] != "general" for r in rows
               if r["subset"] in ("class", "thermo", "matter", "quan", "diff"))
    # 每个 id 唯一（错位判分的隐患之一就是 id 重复）
    assert len({r["id"] for r in rows}) == len(rows)


@pytest.mark.skipif(not DATA_OK, reason="需要 data/scibench/*.jsonl")
def test_load_benchmark_unknown_raises():
    with pytest.raises(ValueError):
        data_mod.load_benchmark("nope")


# ---------------------------------------------------------------- 提示词
def test_prompts_ask_for_answer_line():
    """每个模板都必须要求 ANSWER: 行，否则判分只能靠"全文最后一个数"兜底。"""
    for mode in prompts_mod.MODES:
        for mcq in (False, True):
            tpl = prompts_mod.TEMPLATES[(mode, mcq)]
            assert "ANSWER:" in tpl.template, (mode, mcq)


def test_prompt_mode_selection_and_mcq_detection():
    q = "A 2 kg block moves at 3 m/s. What is its kinetic energy?"
    assert "step by step" in prompts_mod.build_prompt(q, mode="cot")
    assert "directly" in prompts_mod.build_prompt(q, mode="direct")
    mcq = "Which is true?\nA. one\nB. two\nC. three"
    assert prompts_mod.is_mcq(mcq) and not prompts_mod.is_mcq(q)
    assert "option letter" in prompts_mod.build_prompt(mcq, mode="cot")
    with pytest.raises(ValueError):
        prompts_mod.build_prompt(q, mode="nope")


def test_build_prompts_length_and_question_embedded():
    rows = [{"question": "Q1"}, {"question": "Q2"}]
    ps = prompts_mod.build_prompts(rows, mode="cot")
    assert len(ps) == 2 and all("Q1" in ps[0] and "Q2" in ps[1] for _ in [0])
