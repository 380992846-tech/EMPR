"""统计工具（src/stats.py）测试。

CI 的**可复现性**是硬要求：同一 seed 必须在任何机器给出同一个 CI，否则云上/本地
两份结果对不上。若 CI 同时存在 numpy 与纯 Python 两条实现路径，
两种环境会得到不同结果，因此这里统一为纯 Python 实现并固定下来。
"""
from __future__ import annotations

import math

import pytest

from src.stats import (bootstrap_ci, summarize_accuracy, paired_significance,
                       chi2_contingency, chi2_contingency_table)


def test_bootstrap_ci_is_deterministic():
    vals = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1] * 5
    a = bootstrap_ci(vals, n_iter=2000, seed=42)
    b = bootstrap_ci(vals, n_iter=2000, seed=42)
    assert a == b


def test_bootstrap_ci_seed_changes_result_but_brackets_mean():
    vals = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1] * 5
    lo, hi, mean = bootstrap_ci(vals, n_iter=2000, seed=1)
    assert lo <= mean <= hi
    assert math.isclose(mean, sum(vals) / len(vals), rel_tol=1e-12)


def test_bootstrap_ci_empty_and_degenerate():
    lo, hi, mean = bootstrap_ci([], n_iter=100)
    assert math.isnan(lo) and math.isnan(mean)
    lo, hi, mean = bootstrap_ci([1, 1, 1, 1], n_iter=100)
    assert lo == hi == mean == 1.0


def test_summarize_accuracy_format():
    s, (lo, hi, mean) = summarize_accuracy([1, 0, 1, 0], n_iter=500)
    assert s.startswith(f"{mean:.3f} [") and lo <= mean <= hi
    assert summarize_accuracy([])[0] == "N/A"


def test_mcnemar_symmetry_and_direction():
    a = [1] * 10 + [0] * 10
    b = [1] * 10 + [0] * 10
    assert math.isnan(paired_significance(a, b))          # 完全一致 -> 无差异可检验
    # a 全对、b 全错 -> 差异极端显著
    p = paired_significance([1] * 20, [0] * 20)
    assert p < 0.01
    # 交换顺序不改变 p（对称）
    assert abs(paired_significance([1] * 10 + [0] * 10, [0] * 10 + [1] * 10) -
               paired_significance([0] * 10 + [1] * 10, [1] * 10 + [0] * 10)) < 1e-12


def test_chi2_matches_scipy_when_available():
    """对 2x2，scipy 默认带 Yates 连续性校正，我们实现的是未校正 Pearson 卡方，
    所以要和 scipy 的 correction=False 比（差异是有意为之，不是 bug）。"""
    table = [[10, 20], [30, 15]]
    chi2, p, dof = chi2_contingency(table)
    assert dof == 1 and chi2 > 0 and 0 <= p <= 1
    try:
        from scipy.stats import chi2_contingency as sp_chi2
        ref_chi2, ref_p, ref_dof, _ = sp_chi2(table, correction=False)
        assert chi2 == pytest.approx(ref_chi2, rel=1e-6)
        assert p == pytest.approx(ref_p, rel=1e-6)
        assert dof == ref_dof
        # 3x2 表（无 Yates 之说）也必须一致
        big = [[10, 20, 5], [30, 15, 12]]
        c2, p2, dof2 = chi2_contingency(big)
        r2, rp2, rd2, _ = sp_chi2(big)
        assert c2 == pytest.approx(r2, rel=1e-6) and dof2 == rd2
    except ImportError:  # pragma: no cover
        pass


def test_chi2_2x2_wrapper_agrees_with_general():
    t = [[4, 6], [9, 2]]
    assert chi2_contingency_table(t) == chi2_contingency(t)
