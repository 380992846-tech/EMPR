r"""论文一致性守护测试：把 scripts/check_paper.py 接进测试套件。

论文里的数字全部来自 paper/numbers.tex 与 paper/table*.tex（由脚本从 results/ 生成），
因此一旦有人手改了表格、写错宏名、或新增 \ref 却没有 \label，这里就会失败。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_paper_selfcheck_passes():
    # Windows 上 subprocess 默认按系统 locale(GBK) 解码，check_paper.py 输出 UTF-8 中文会炸，
    # 因此显式指定 encoding；errors="replace" 保证即使解码异常也能看到真正的失败原因。
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_paper.py")],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=ROOT, timeout=120)
    assert r.returncode == 0, (
        "论文自检失败（手改数字或交叉引用不一致）：\n" + (r.stdout or "") + (r.stderr or ""))
