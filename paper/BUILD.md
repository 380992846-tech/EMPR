# 论文编译说明

## 主稿

`paper/main.tex` 是唯一的论文主文件。它**不含任何手抄数字**：

- 表格：`\input{table1_main.tex}` 等 7 张，全部由 `scripts/make_paper_tables.py` 生成；
- 行内数字：引用 `numbers.tex` 里的宏（同脚本生成），例如 `\accCotSevenB`、`\kappaAnnotators`；
- 章节片段：`\input{discussion.tex}`、`\input{reproducibility_statement.tex}`。

因此"论文以哪个为准"的答案是：**以 `main.tex` 为准**，而它的每个数字都可追溯到 `results/*.json`。

## 编译

```bash
# 从仓库根目录
python scripts/make_paper_tables.py     # 重新生成 7 张表 + numbers.tex
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

或一条命令：

```bash
cd paper && latexmk -pdf main.tex
```

已验证环境：MiKTeX（pdflatex + bibtex），输出 12 页，11 条引用与 23 个标签全部解析，
0 个未定义引用。所需宏包：`fontenc, inputenc, amsmath, amssymb, booktabs, multirow, array,
graphicx, xcolor, geometry, hyperref, natbib, caption, titlesec`（MiKTeX/TeX Live 默认齐全）。

## 改数字的正确流程

1. 重跑实验（结果写入 `results/`）；
2. `python scripts/make_paper_tables.py` —— 表格与 numbers.tex 一起更新；
3. 重新编译；`python scripts/check_paper.py` 会校验宏定义、`\input` 目标、`\ref`/`\cite` 是否齐全。

**不要**直接改 `table*.tex` 或 `numbers.tex`——它们每次都会被生成器覆盖（文件头也写了这句）。

## 提交前自检

```bash
python scripts/check_paper.py    # 论文一致性（也是 tests/test_paper_consistency.py 的内容）
python -m pytest tests -q        # 全量单元测试
```

`check_paper.py` 会检查：`\input` 目标存在、正文引用的数字宏都已定义、
`\ref` 有对应 `\label`、`\cite` 键在 `refs.bib` 里、花括号配平、环境嵌套、无 `__占位符__`。

## 仍需人工补写的位置

正文里用 `\TODO{...}` 标出，编译后是红色文字，投稿前必须清零：

- 作者、单位、致谢（`\author{}`、Acknowledgements 段）。

其余章节（Abstract / Intro / Related / Method / Results / Discussion / Limitations /
Reproducibility）均已成稿，可由现有实验数据直接支撑。
