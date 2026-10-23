r"""论文自检：交叉引用、宏定义、\input 目标、tex 结构。

检查项：
  1. main.tex 里所有 \input{...} 的文件是否存在
  2. 正文引用的数字宏是否都在 numbers.tex 里定义过（防手抄与拼写错误）
  3. 每个 \ref 是否有对应 \label
  4. 每个 \cite 键是否在 refs.bib 里
  5. 花括号配平、环境嵌套、无 __占位符__

用法： python scripts/check_paper.py
"""
from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
MAIN = PAPER / "main.tex"

# 本项目生成的宏前缀（用于区分 LaTeX 内置命令）
OUR_PREFIXES = ("acc", "nb", "pct", "pair", "ratio", "gap", "ci", "kappa",
                 "scaf", "peak", "gpu")


def strip_comments(s: str) -> str:
    return "\n".join(l for l in s.splitlines() if not l.lstrip().startswith("%"))


def collect(files):
    return "\n".join(strip_comments(Path(f).read_text(encoding="utf-8")) for f in files if Path(f).exists())


def main() -> int:
    problems = []
    if not MAIN.exists():
        print("缺少 paper/main.tex")
        return 2
    main_src = strip_comments(MAIN.read_text(encoding="utf-8"))

    # 1) \input 目标
    inputs = re.findall(r"\\input\{([^}]+)\}", main_src)
    for t in inputs:
        p = PAPER / t
        if not p.exists():
            problems.append(f"[input] 找不到 {t}")
    print(f"[input] {len(inputs)} 个 input 目标：{', '.join(inputs)}")

    # 2) 宏定义 vs 使用
    defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", strip_comments(
        (PAPER / "numbers.tex").read_text(encoding="utf-8"))))
    body = collect([MAIN, PAPER / "discussion.tex",
                    *[p for p in glob.glob(str(PAPER / "table*.tex"))]])
    used = set(re.findall(r"\\([A-Za-z]+)", body))
    # 只认"前缀 + 大写字母/数字"的形式，避免把 \\citep、\\kappa 这类内置命令当成我们的宏
    prefix_re = re.compile(r"^(?:" + "|".join(OUR_PREFIXES) + r")[A-Z0-9]")
    our_used = {u for u in used if prefix_re.match(u)}
    missing = sorted(our_used - defined)
    if missing:
        problems.append(f"[macro] 未定义但被引用: {missing}")
    unused = sorted(defined - used)
    print(f"[macro] 定义 {len(defined)} 个，正文用到 {len(our_used)} 个"
          + (f"，未使用 {len(unused)} 个" if unused else ""))

    # 3) \ref vs \label
    labels = set(re.findall(r"\\label\{([^}]+)\}", body))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", body))
    dangling = sorted(refs - labels)
    if dangling:
        problems.append(f"[ref] 无对应 label: {dangling}")
    print(f"[ref] {len(refs)} 个引用 / {len(labels)} 个标签")

    # 4) \cite vs bib
    bib = (PAPER / "refs.bib").read_text(encoding="utf-8")
    bibkeys = set(re.findall(r"@\w+\{([^,]+),", bib))
    cites = set()
    for m in re.findall(r"\\cite[pt]?\{([^}]+)\}", main_src):
        cites.update(k.strip() for k in m.split(","))
    bad_cites = sorted(cites - bibkeys)
    if bad_cites:
        problems.append(f"[cite] refs.bib 里没有: {bad_cites}")
    unused_keys = sorted(bibkeys - cites)
    print(f"[cite] 引用 {len(cites)} 个 / bib {len(bibkeys)} 条"
          + (f"，未引用 {len(unused_keys)} 条" if unused_keys else ""))

    # 5) 结构
    for f in sorted(glob.glob(str(PAPER / "*.tex"))):
        src = strip_comments(Path(f).read_text(encoding="utf-8"))
        b = src.count("{") - src.count("}")
        stack, bad = [], []
        for m in re.finditer(r"\\(begin|end)\{(\w+\*?)\}", src):
            if m.group(1) == "begin":
                stack.append(m.group(2))
            elif not stack or stack[-1] != m.group(2):
                bad.append(m.group(2))
            else:
                stack.pop()
        ph = re.findall(r"__[A-Za-z0-9_]+__", src)
        if b or stack or bad or ph:
            problems.append(f"[struct] {Path(f).name}: braces {b:+d} unclosed {stack} bad {bad} ph {len(ph)}")

    print()
    if problems:
        for p in problems:
            print("FAIL", p)
        return 1
    print("论文自检全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
