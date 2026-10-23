# 论文产物索引与口径说明

**这个文件回答一个问题：论文里每个数字以什么为准。**

## 权威来源（按优先级）

| 内容 | 权威文件 | 说明 |
|---|---|---|
| **所有数字** | `results/*.json` | 每次运行写入的原始结果，含 `meta`（commit / prompt_fingerprint / max_new_tokens / wall_time_s） |
| **论文表格** | `paper/table1..7_*.tex` | **由 `scripts/make_paper_tables.py` 从 `results/` 自动生成**，表头标注"请勿手改"。改数字只能重跑实验，不能改表 |
| **中文叙述** | `README.md` 第 1-9 节 | results-first 结构，每个结论后附对应结果文件 |
| **English 叙述** | `paper/discussion.tex`、`paper/reproducibility_statement.tex` | 可直接 `\input` 进论文 |
| 填表口径 | 本文件 | 只讲"怎么算、分母是什么、哪些不能报" |

> 本仓库**没有**完整的论文草稿（无 abstract / intro / method / experiments 的 .tex）。
> 已产出的是可投稿的**表格 + Discussion + 可复现性声明**，正文需要另行撰写。
> 数字一律不要在正文里手抄 —— 引用表格编号即可。

重新生成全部表格：

```bash
python scripts/make_paper_tables.py      # 7 张 tex 覆盖写
```

## 七个表格对应关系

| 表 | 文件 | 内容 | 数据来源 |
|---|---|---|---|
| 1 | `table1_main.tex` | 5 模型 × cot/direct，overall CI + 6 子域 | `eval_scibench_*.json` |
| 2 | `table2_sensitivity.tex` | 4 模型 × 3 变体（覆盖/median/p90/改变比例） | `sensitivity_summary_*.json` |
| 3 | `table3_causal.tex` | 逐层恢复率按层区间聚合（n=14） | `causal_tracing_*.json` |
| 4 | `table4_errors.tex` | 错误类型，**两位标注器并排** + 卡方 + kappa | `error_labels.json`、`error_labels_llama.json` |
| 5 | `table5_cross_task.tex` | MMLU 选择题 / SciBench / OlympiadBench 形态对照 | `eval_mmlu*`、`eval_olympiad*` |
| 6 | `table6_layer_parse_vs_acc.tex` | 逐层"可解析率 vs 正确率"，按 clean 对错分组 | `ct_all/*.json` |
| 7 | `table7_prompt_scaffold.tex` | 同预算 cot vs structured 配对 + 格式遵循统计 | `eval_scibench_*_n150_t2048_*.json` |

图：`results/sensitivity_comparison.png`（图 2）、`results/causal_tracing_Qwen2.5-7B.png`（图 3），
均由 `scripts/make_figures.py` 生成。

---

## 分母口径（全表必须一致）

- 正确率**只统计 scorable 样本**：参考答案解析不出来的题（`unscorable_ref`）被剔除，**不算模型错**。
  每张表报 `n_scorable` 与 `n_unscored`。
- `n_unparsed_pred`：模型没给出可解析答案，**计为错**，单独报。
- `n_no_answer_line`：输出里没有 `ANSWER:` 行。这是**格式遵循**指标，不是截断
  （实测输出均值约 2400 字符、最大约 4600 字符，远低于 2048 token 上限）。
- **SciBench 物理 323 题全部可判分**（`n_unscored = 0`），分母即全量。
- **OlympiadBench 仅 152/351（43.3%）可数值判分**，用该基准出表必须写明覆盖率，
  且**不可与 SciBench 数值横向比较**（子集偏易）。

## 子域映射（唯一真源）

`src/data.py: SCIBENCH_SUBJECT_DOMAIN`。323 题分布：
general(fund) 71 | thermodynamics(thermo) 66 | mechanics(class) 56 | math(diff) 50 | optics(matter) 47 | quantum(quan) 33。

抽样请用 `--max-problems`，它按科目**轮转抽样**，不是取前 N 题。

## 表 2 的三个变体（语义必须写清）

| 变体 | 改动 | 覆盖率 | 能报什么 |
|---|---|---|---|
| `numeric` | 缩放物理量数值（factor 2.0，≤2 个数，保护幂次） | 305/323 = 94.4% | **只报输出变化分布**（正确答案随物理关系改变且不可直接求解） |
| `wording` | 纯表层改写，物理与数据逐字不变 | 240/323 = 74.3% | **唯一可报配对准确率与 McNemar** |
| `swap` | 互换两个不同物理量的数值（制造物理不一致） | 36/323 = 11.1% | 辅助证据；覆盖率低是数据固有，如实报告不放宽判据 |

敏感度定义为 `|Δ|/(|base|+ε)`，基准答案接近 0 时均值会被单样本抬高，
**报 median 与 p90，不报 mean**。

## 表 3 / 表 6 的因果追踪口径

- 计入规则（表 3）：只有 **clean 答对** 且 **corrupt 确实掉分** 的题计入恢复率，
  否则分母趋近 0。报告 `n_valid` / `n_skipped_clean_wrong` / `n_skipped_corrupt_flat`。
- 表 6 用的是另一套口径（`--all-attempts`）：**全部题目**都跑逐层恢复，但**不做恢复率归一化**，
  只统计"是否产出可解析数值"与"数值是否正确"，并按 clean 对错分组 —— 因为 clean 本身就错时
  恢复率无定义。不要把表 6 的数字当成"恢复率"。
- 方法细节：C2 占位符替换"数字+单位"；逐层恢复只恢复目标层，其余层保持 corrupt；
  只在 prefill 生效；clean/corrupt 分词长度不同，hidden 按右端对齐。
- 只用开放权重模型，不对闭源 API 做隐态干预。

## 表 4 的标注口径

- 剔除 `unscorable_ref` 与 `unparsed_pred` 两类非物理错误样本。
- 标签按 (模型, 题 id) 复合键存于 `results/error_labels.json`（Qwen2.5-14B 同族标注）与
  `results/error_labels_llama.json`（Llama-3.1-8B 异族复核）。
- 两位标注器一致率 60.2%，**Cohen's kappa = 0.133**：
  **只声明粗粒度结论**（E2 为主、E3 少），**不报精确百分比**。见 `results/error_label_agreement.json`。
- 跨模型卡方为**未校正** Pearson（`correction=False`）。

## 表 7 的口径

- 同 150 题、同 2048 token 预算下比较 cot 与 structured，配对 McNemar。
- 早期在 1024 token 下的对比（0.183 vs 0.211）被输出长度差异污染，**不要引用**；
  引用匹配预算版本（0.240 → 0.260，p=0.70）。

---

## 投稿前统一检查

- [ ] 七张表全部由 `make_paper_tables.py` 生成，无手改数字。
- [ ] 每个数字带 95% CI 与 n；`meta.git_commit` / `prompt_fingerprint` 可追。
- [ ] `n_scorable` / `n_unscored` / `n_unparsed_pred` / `n_no_answer_line` 四者区分清楚。
- [ ] OlympiadBench 覆盖率（43.3%）与"不可横向比较"写进正文与 Limitations。
- [ ] `swap` 覆盖率 11.1%、标注 kappa 0.133、格式遵循 71%-90% 三条负结果都写进 Limitations。
- [ ] 不引用任何被污染或被撤回的数字（1024 预算下的 scaffolding 对比、n=1 的破坏集探针）。
- [ ] 九张 tex 自检通过：花括号配平、环境嵌套、无 `__占位符__`。
