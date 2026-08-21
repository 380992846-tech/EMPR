"""统计工具：bootstrap 95% CI、配对显著性。

约定：所有正确率/性能指标都用 bootstrap CI，随机种子统一 42。
numpy 可选：装了用 numpy（快）；没装用纯 Python（本机也能跑）。scipy 可选。
"""
from __future__ import annotations

try:
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

try:
    from scipy import stats as _sp
except Exception:  # pragma: no cover
    _sp = None


def _sample(arr, rng, n):
    if _np is not None:
        return arr[rng.integers(0, n, size=n)]
    return [arr[rng.randrange(n)] for _ in range(n)]


def _mean(arr):
    return (sum(arr) / len(arr)) if arr else float("nan")


def _median(arr):
    if not arr:
        return float("nan")
    s = sorted(arr)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2


def bootstrap_ci(scores, n_iter=10000, ci=0.95, seed=42, statistic="mean"):
    """对一维样本计算 bootstrap 置信区间。返回 (lo, hi, mean)。"""
    arr = list(scores) if _np is None else _np.asarray(scores, dtype=float)
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    if _np is not None:
        rng = _np.random.default_rng(seed)
        fn = _np.mean if statistic == "mean" else _np.median
        boot = [fn(_sample(arr, rng, n)) for _ in range(n_iter)]
        lo, hi = _np.percentile(boot, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
        return float(lo), float(hi), float(fn(arr))
    # 纯 Python 兜底
    import random as _random
    rng = _random.Random(seed)
    fn = _mean if statistic == "mean" else _median
    boot = sorted(fn(_sample(arr, rng, n)) for _ in range(n_iter))
    q = int((1 - ci) / 2 * n_iter), int((1 + ci) / 2 * n_iter) - 1
    return float(boot[q[0]]), float(boot[q[1]]), float(fn(arr))


def summarize_accuracy(correct_flags, n_iter=10000, seed=42):
    """把 0/1 数组转成 "0.695 [0.66, 0.73]" 格式并返回原始 CI。"""
    flags = list(correct_flags)
    if not flags:
        return "N/A", (float("nan"), float("nan"), float("nan"))
    lo, hi, mean = bootstrap_ci(flags, n_iter=n_iter, seed=seed)
    return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]", (lo, hi, mean)


def paired_significance(a, b):
    """对配对 0/1 样本做 McNemar 检验（a 为模型 A 对错，b 为模型 B 对错）。"""
    a = list(a); b = list(b)
    n01 = sum(1 for x, y in zip(a, b) if not x and y)  # A错B对
    n10 = sum(1 for x, y in zip(a, b) if x and not y)  # A对B错
    if n01 + n10 == 0:
        return float("nan")
    z = (abs(n01 - n10) - 1) / (n01 + n10) ** 0.5
    return _normal_sf(abs(z)) * 2


def _normal_sf(x):
    # 标准正态尾概率（纯 Python 近似，或用 scipy）
    if _sp is not None:
        return float(_sp.norm.sf(x))
    # 数值近似（Abramowitz & Stegun 26.2.17）
    import math
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    d = 0.3989422804014327 * math.exp(-x * x / 2.0)
    p = d * t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 +
        t * (-1.821255978 + t * 1.330274429))))
    return max(0.0, min(1.0, p))


def chi2_contingency_table(observed_2x2):
    """2x2 卡方（用于错误类型分布对比）。返回 (chi2, p, dof)。"""
    a = [[float(x) for x in row] for row in observed_2x2]
    n = sum(sum(r) for r in a)
    row_tot = [sum(r) for r in a]
    col_tot = [a[0][i] + a[1][i] for i in range(2)]
    chi2 = 0.0
    for i in range(2):
        for j in range(2):
            exp = row_tot[i] * col_tot[j] / n
            if exp > 0:
                chi2 += (a[i][j] - exp) ** 2 / exp
    p = _chi2_sf(chi2, 1)
    return float(chi2), float(p), 1


def chi2_contingency(matrix):
    """任意 r×c 列联表卡方（如 2 模型 × 4 错误类别）。返回 (chi2, p, dof)。"""
    mat = [[float(x) for x in row] for row in matrix]
    n = sum(sum(r) for r in mat)
    if n == 0:
        return float("nan"), float("nan"), 0
    r, c = len(mat), len(mat[0])
    row_tot = [sum(row) for row in mat]
    col_tot = [sum(mat[i][j] for i in range(r)) for j in range(c)]
    chi2 = 0.0
    for i in range(r):
        for j in range(c):
            exp = row_tot[i] * col_tot[j] / n
            if exp > 0:
                chi2 += (mat[i][j] - exp) ** 2 / exp
    dof = (r - 1) * (c - 1)
    p = _chi2_sf(chi2, dof)
    return float(chi2), float(p), dof


def _chi2_sf(chi2, dof):
    if _sp is not None:
        return float(_sp.chi2.sf(chi2, dof))
    # 1-dof 卡方 ~ 标准正态的平方：P = 2*(1-Phi(sqrt(chi2)))
    return _normal_sf(chi2 ** 0.5) * 2
