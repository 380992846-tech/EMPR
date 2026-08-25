# EMPR 论文表格骨架（跑完实验后填数字）

**用途**：每一张表对应哪条脚本、列结构是什么、数字从哪儿来、填之前要确认什么。
**原则**：只填 `run_*` 脚本**真实跑出来**的数字；**不要**引用 `论文/物理/物理直觉 ...pdf` 里 AI 稿的旧数字（那是编的）。

> 运行环境：云 GPU（见 `DEPLOY_CHECKLIST.md`）。所有表都要带 **bootstrap 95% CI**（脚本已内置，seed=42，n_iter=10000）。

---

## 表 1 · 主结果：模型 × 子域 正确率 + 95% CI

- **来源脚本**：`scripts/run_eval.py --save`（每模型生成 `results/eval_*.json`）→ `scripts/make_table.py --dir results --benchmark scibench --format tex`。
- **列说明**：Model | n(总题数) | Overall | mechanics | thermodynamics | optics | general | …（子域名以 `per_domain_acc` 实际键为准）。
- **子域从哪来**：`src/data.py` 的 `_domain_of()` 把 SciBench 科目映射成 mechanics/thermodynamics/optics/electromagnetism/general。
- **注意**：`make_table.py` 现在把组名前缀上了 benchmark（`scibench/mechanics`），避免多个基准混表；上表列名用 `scibench/…` 的显示名即可。
- **填空清单**：
  - [ ] 覆盖模型池 core + large（Llama-3.1-8B、Qwen2.5-7B/14B、Mistral-7B、DeepSeek-R1-Distill-14B、Llama-3.1-70B、Qwen2.5-72B 等，**跑了几只填几只**，别装全部跑过）。
  - [ ] 每个数字都要带 CI（`acc [lo, hi]`）；`n`（该子域题数）也列出来。
  - [ ] Overall 与各子域数字来自 `results/eval_*.json` 的 `overall_acc` / `per_domain_acc`。

模板：见 `table1_main.tex`。

---

## 表 2 · 敏感性：记忆 vs 计算（numeric / wording / swap）

- **来源脚本**：`scripts/run_sensitivity.py --save` → `results/sensitivity_summary_*.json`（含 `total_questions`、各变体 `mean/median sensitivity`、`unchanged_share`、`n_changed`、`n_unparsed`）。
- **列说明**：Variant | 覆盖率(=n_changed/total) | n_parsed | mean sensitivity | median | unchanged share。
- **覆盖率计算**：`n_changed / total_questions`。
- **三个变体的语义**（论文里必须写清）：
  - `numeric`：换数值（物理结构不变）——**主信号**。若模型"真算"，输出应成比例变；若靠检索/模式匹配，则可能不变。
  - `wording`：换表层措辞（物理、数据完全不变）——模型输出应几乎不变。
  - `swap`：互换两个不同物理量的数值（制造物理不一致题）——模型应明显变化 / 出错。
- **⚠ 诚实标注（重要）**：
  - `wording` 覆盖率 ~74%，`numeric` ~95%，`swap` ~11%（**swap 覆盖低是数据集固有**：很多题所有数都在同一物理量或没有单位）。在论文里如实报告覆盖率，别硬凑。
  - `n_unparsed`（输出答案提取失败的样本）单独列出，**不要**按 0 敏感处理。
- **填空清单**：见 `table2_sensitivity.tex`。

---

## 表 3 · 因果追踪：逐层恢复率（核心机制定位）

- **来源脚本**：`scripts/run_causal_tracing.py --save --plot` → `results/` 的 json + png（恢复率-层曲线、热力图）。
- **列说明**：Layer index | recovery rate（恢复到该层时正确率回升多少）。或按物理子域分组的层区间对比（如热力学偏后、光学偏前）。
- **⚠ 只用开放权重模型**（8B / 14B，70B 可选）。**不要**做 GPT-3.5 那种闭源 API 的隐藏态干预（不可复现，审稿人会质疑）。
- **注意**：脚本已修"只恢复目标层"，避免全层污染。图上要标关键层区间。
- 模板：见 `table3_causal.tex`。

---

## 表 4 · 错误类型分布（E1 定律误用 / E2 代数 / E3 约束违背）

- **来源脚本**：`scripts/classify_errors.py --file results/eval_*.json --report --compare-file …`（手动/LLM 标注后统计 + 卡方）。
- **列说明**：Error type | 数量 | 占比 | 按 domain 分布。
- **⚠**：标签按 (模型, 题id) 复合键，跨模型对比卡方时用各自模型的键。
- 模板：建议在论文里做成 `E1/E2/E3 × domain` 的小表（可用 `classify_errors.py --report` 输出直接改）。

---

## 其它图
- 图 1：规模-性能曲线（log 尺度）+ 按组合深度退化曲线 —— 由表 1 数据派生（排序即可画，或小脚本）。
- 图 2：敏感性核密度/箱线（numeric vs wording）—— 由 `results/sensitivity_summary_*.json` 派生。

---

## 投稿前统一检查
- [ ] 所有表数字来自 `results/`，带 95% CI。
- [ ] 子域名、模型名、benchmark 前缀（`scibench/…`）全文一致。
- [ ] 删掉任何 AI 稿里的旧数字 / 空表格 / `[Figure about here]` 占位。
- [ ] 若某个变体（如 swap）覆盖率低，在正文 Limitations 里写明。
