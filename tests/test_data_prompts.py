"""数据加载 / 子域映射 / 抽样 / 提示词 测试。

重点钉住两件事：
 1. 子域映射（`_domain_of`）：早期 `download_data.py` 把 quan/diff 都落成 "general"，
    导致 154/323 题挤进同一个桶，子域表失去意义——这里锁住正确映射。
 2. 分层抽样：`rows[:n]` 会拿到清一色同一科目，这里锁住"按科目轮转"。
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
