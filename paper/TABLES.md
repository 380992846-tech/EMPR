# EMPR 论文表格骨架

**用途**：每张表对应哪条脚本、列结构、数字来源、填表前需确认的事项。
**原则**：所有数字只填 `run_*` 脚本写入 `results/` 的实际输出。

> 运行环境：GPU（见 `DEPLOY_CHECKLIST.md`）。所有表均带 bootstrap 95% CI（脚本内置，seed=42，n_iter=10000）。
> 填表前先执行 `python scripts/preflight.py --benchmark scibench`：一次性确认数据、判分自洽与变体覆盖率，
> 结论写入 `results/preflight_scibench.json`。体检未通过时不执行全量实验。

---

## 表 1 · 主结果：模型 × 子域 正确率 + 95% CI

- **来源脚本**：`scripts/run_eval.py --save`（每模型生成 `results/eval_*.json`）-> `scripts/make_table.py --dir results --benchmark scibench --format tex`。
- **列说明**：Model | n | Overall | mechanics | thermodynamics | optics | quantum | math | general（子域名以 `per_domain_acc` 实际键为准）。
- **子域从哪来**：`src/data.py: SCIBENCH_SUBJECT_DOMAIN`（唯一真源）。当前 SciBench 物理 6 科 -> 323 题的分布是：
  general(fund) 71 | thermodynamics(thermo) 66 | mechanics(class) 56 | math(diff) 50 | optics(matter) 47 | quantum(quan) 33。
  注意：子域由科目统一映射得到；`general` 仅对应 fund（综合基础题），当前为 71/323 题。
- **注意：分母口径（必须写进论文）**：正确率**只统计 scorable 样本**——参考答案无法解析的题（`unscorable_ref`）
  被剔除，不算模型错。每张表都要报 `n_scorable` 与 `n_unscored`；另外报 `n_unparsed_pred`
  （模型没给出可解析答案，计为错）以暴露提示词/判分问题。
  **SciBench 物理 323 题全部可判分（n_unscored = 0）**；
  **OlympiadBench 351 题里只有 152 题（43.3%）可数值判分**，用那个基准出表必须写明覆盖率
  （其余是符号/多变量解，`results/preflight_olympiadbench.json` 有记录）。
- **注意：抽样口径**：`--max-problems N` 默认**按科目轮转抽样**（不是取前 N 题），否则会拿到清一色单科目。
- **填空清单**：
  - [ ] 覆盖模型池 core + large（只填写实际运行过的模型）。
  - [ ] 每个数字带 CI（`acc [lo, hi]`）+ n。
  - [ ] Overall 与各子域数字来自 `results/eval_*.json` 的 `overall_acc` / `per_domain_acc`。
  - [ ] 记录 `meta.git_commit` 与 `prompt_fingerprint`（表注或附录）。

模板：见 `table1_main.tex`。

---

## 表 2 · 敏感性：记忆 vs 计算（numeric / wording / swap）

- **来源脚本**：`scripts/run_sensitivity.py --save` -> `results/sensitivity_summary_<model>.json`，
  单变体明细在 `results/sensitivity_<variant>_<model>.json`（含 `items[]` 逐题审计）。
- **覆盖率（干跑，0 模型成本）**：`scripts/run_sensitivity.py --coverage --benchmark scibench --max-problems 323 --save`
  -> `results/sensitivity_coverage_scibench.json`。**当前实测值（seed=42）**：
  | 变体 | 改动题数 | 覆盖率 |
  |---|---|---|
  | numeric | 305/323 | 94.4% |
  | wording | 240/323 | 74.3% |
  | swap | 36/323 | 11.1% |
- **列说明**：Variant | coverage | n_parsed | n_unparsed | median sensitivity | p90 | unchanged_share | response_rate。
- **注意：统计量选择**：敏感度定义为 `|delta|/(|base|+eps)`，当基准答案接近 0 时单个样本即可显著抬高均值
  （管道验证中 wording 的 `mean=2983.8` 而 `median=0.000`）。
  **论文报告 median 与 p90，不报告 mean**（脚本三者均输出，并在打印时标注该差异）。
- **三个变体的语义（论文里必须写清）**：
  - `numeric`：换数值（物理结构不变，factor=2.0，最多改 2 个数）——**主信号**。
    真算的模型输出应随物理关系变化；靠检索/模式匹配则可能纹丝不动。
    注意：其正确答案随物理关系变化且**不可直接求解**，所以这个变体**只报输出变化分布**
    （`unchanged_share` = 输出完全没变的占比、`response_rate` = 1 - unchanged），**不报"变体准确率"**。
  - `wording`：换表层措辞（物理、数据逐字不变）——模型输出应几乎不变，且**准确率不应显著下降**。
    这是**唯一**能报配对准确率的变体（参考答案仍成立）：脚本给 `accuracy_paired`
    （base_acc -> variant_acc、Δ、McNemar p）。
  - `swap`：互换两个不同物理量的数值（制造物理不一致题）——模型应明显变化 / 出错。
- **注意：诚实标注**：
  - `swap` 覆盖率仅 11.1% 是**数据固有**（很多题所有数同属一个物理量、或无可识别单位），
    如实报告覆盖率，不放宽判据以凑样本；相应内容写入 Limitations。
  - `n_unparsed`（输出抠不出答案）单独列出，**不要**按 0 敏感处理。
  - `numeric` 的扰动只动数值、不动幂次（`10^{3}` 的底数 10 与指数 3 都受保护），
    改动上限 2 个数——这些都在 `tests/test_variants.py` 里钉住。
- **填空清单**：见 `table2_sensitivity.tex`；变体题面需人工抽查（看 `items[].question_variant`）。

---

## 表 3 · 因果追踪：逐层恢复率（核心机制定位）

- **来源脚本**：`scripts/run_causal_tracing.py --save --plot` -> json（`average_curve` + `problems[*].curve`）+ png。
- **列说明**：Layer index | mean recovery ±std（跨题均值曲线）；或按子域分组的峰值层对比。
- **注意：计入规则（重要）**：只有 **clean 答对** 且 **corrupt 确实掉分** 的题才计入恢复率，
  否则分母趋近于 0，恢复率数值失去意义。脚本会报告 `n_valid` / `n_skipped_clean_wrong` /
  `n_skipped_corrupt_flat`，论文中须一并给出（并说明这是筛选后的样本量）。
- **方法细节（写进 Method）**：C2 占位符替换"数字+单位"；逐层恢复时**只恢复目标层**（其余层保持 corrupt）；
  恢复发生在 prefill，解码步不干预；clean 与 corrupt 的 token 长度不同，hidden 按**右端对齐**。
- **注意：只用开放权重模型**（8B / 14B，70B 可选）。**不要**对闭源 API 模型做隐藏态干预。
- 默认 `--mode direct`（每题要多次前向，CoT 太贵）；层采样默认 ~16 个点（`--layer-step` 可调）。
- 模板：见 `table3_causal.tex`。

---

## 表 4 · 错误类型分布（E1 定律误用 / E2 代数 / E3 约束违背）

- **来源脚本**：`scripts/classify_errors.py --file results/eval_*.json --report [--compare-file …]`。
- **列说明**：Error type | 数量 | 占比 | 按 domain 分布。
- **注意：口径**：脚本会自动剔除两类**非物理错误**样本——`unscorable_ref`（参考答案无法解析）与
  `unparsed_pred`（模型未给出可解析答案，记为 NP），避免污染 E1/E2/E3 分布。
- **标签与检验**：标签按 (模型, 题id) 复合键存于 `results/error_labels.json`；跨模型卡方使用各自模型的键。
  卡方为**未校正** Pearson（scipy 对 2x2 默认启用 Yates 校正，对比时使用 `correction=False`）；样本量小时谨慎解读。
- 模板：建议做成 `E1/E2/E3 × domain` 小表（用 `--report` 输出直接改）。

---

## 其它图
- 图 1：规模-性能曲线（log 尺度）+ 按组合深度退化曲线 —— 由表 1 数据派生。
- 图 2：敏感性分布（numeric vs wording 的 `unchanged_share` / 敏感度核密度）—— 由 `sensitivity_*.json` 派生。
- 图 3：recovery-vs-layer 曲线（含 ±1 std 带与峰值层）—— `run_causal_tracing --plot` 直接输出。

---

## 投稿前统一检查
- [ ] 所有表格数字来自 `results/`，带 95% CI 与 n；并附 `meta.git_commit` / `prompt_fingerprint`。
- [ ] 明确区分 `n_scorable` / `n_unscored` / `n_unparsed_pred`，且分母口径全表一致。
- [ ] 子域名、模型名、benchmark 前缀（`scibench/...`）全文一致；`general` 桶仅对应 fund（当前 71/323 题）。
- [ ] 每个变体的覆盖率如实报告；`swap` 覆盖率偏低写入 Limitations。
- [ ] 因果追踪报告 `n_valid` 与跳过原因计数；不涉及闭源模型。
- [ ] 删除所有占位符（`[Figure about here]`、空表格），确认每个数字均由本流程产出。
