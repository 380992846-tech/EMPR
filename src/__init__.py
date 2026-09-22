"""EMPR 评估管道源码包。

副作用（有意为之）：把 stdout/stderr 的错误处理设为 backslashreplace。Windows 上
Python 默认按 GBK 编码控制台输出，遇到 ``−``（U+2212）、``²`` 这类字符会直接
UnicodeEncodeError 崩掉整个脚本——而物理题面/模型输出里这类字符很常见。
这里不改编码（避免中文变乱码），只让不可映射的字符降级成转义，脚本不再中断。
"""
from __future__ import annotations

import sys


def _make_console_robust() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except Exception:  # pragma: no cover - 非 TTY / 老版本 Python
            pass


_make_console_robust()
