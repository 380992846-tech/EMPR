"""pytest 公共夹具：把仓库根与 scripts/ 下的脚本当模块加载。

scripts/ 不是包（没有 __init__.py），直接 `import scripts.run_sensitivity` 会失败，
这里用 importlib 按文件路径加载，既能测脚本里的纯函数，又不用为了测试去改包结构。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_empr_script_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def sensitivity():
    """scripts/run_sensitivity.py 模块（变体生成器所在）。"""
    return _load_script("run_sensitivity")


@pytest.fixture(scope="session")
def causal():
    """scripts/run_causal_tracing.py 模块。"""
    return _load_script("run_causal_tracing")
