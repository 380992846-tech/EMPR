# EMPR — Emergent Physical Reasoning in Open LLMs

面向开放权重 LLM 的**多步物理推理**评测与机理分析框架。围绕三个可证伪的问题组织：

| 阶段 | 研究问题 | 方法 |
|---|---|---|
| Week 1 | 性能规律是什么？ | SciBench 物理全量 x 多模型 x 子域正确率 + bootstrap 95% CI |
| Week 2 | 模型在**计算**还是在**检索**？ | 三路变体探针：数值扰动 / 表层改写 / 物理量互换 |
| Week 3 | 若在计算，**计算发生在哪些层**？ | 逐层恢复率（causal tracing），定位关键层区间 |

框架的结论预设是**诚实的**：若数据支持"sophisticated pattern matching"而非真实物理理解，就照数据写。

---

## 目录

- [核心设计](#核心设计)
- [验证状态](#验证状态)
- [数据构成](#数据构成)
- [快速开始](#快速开始)
- [GPU 时间预算](#gpu-时间预算)
- [结果文件说明](#结果文件说明)
- [方法学说明与限制](#方法学说明与限制)
- [相关项目](#相关项目)

---

## 核心设计

### 判分：参考答案严进，模型输出宽出

正确率是整个项目的落点，判分必须先保证**不产生系统性偏差**。

- `parse_reference`（`src/grading.py`）：**严格**解析参考答案，要求整串就是一个数值
  （允许单位、百分号、分数、科学计数法、`\frac{}{}`、`\sqrt{}`、`\pi`/`\cot` 等 LaTeX 数值形式）。
  无法确定是数值时返回 `None`，该题标记 `unscorable_ref` 并**从分母中剔除**，不算模型答错。
- `extract_number`（`src/grading.py`）：**宽松**解析模型输出，按可信度依次尝试
  `ANSWER:` 行（取最后一段）-> `\boxed{}` -> "answer is X" -> 末行数字 -> 全文末数字，
  并对 Unicode 减号、上标、`\times`、千分位逗号做规范化。
- 判定结果分三类口径：`correct` / `incorrect`（含模型未给出可解析答案）/ `unscorable_ref`（剔除）。
  每个结果文件都记录 `n_scorable`、`n_unscored`、`n_unparsed_pred`，论文中三者必须同时报告。

设计依据：若用同一个宽松解析器处理参考答案，`x^2 + 2x` 会被读成 `2`、`\frac{1}{2}mv^2` 会被读成 `1`，
样本被误判为"可判分"，分母被污染且无法从结果中察觉。因此两侧必须使用不同强度的解析器。

### 推理：三条必须成立的约束

- **批量生成使用左填充**。decoder-only 模型右填充时会在 pad 之后继续生成，
  `transformers` 仅给出 warning 而不报错，批量结果会整体失真。该约束在代码与测试中固定。
- **instruct 模型走 chat template**（`chat=auto`，tokenizer 自带模板即启用），
  否则模型倾向于"续写指令"而非回答问题；是否启用记录在结果的 `meta.chat_template` 中。
- **输出条数与题数必须一致**，否则直接抛错。`zip(rows, outputs)` 一旦错位，
  A 题的输出会被判到 B 题的答案上，且不会报错。

### 数据：单一真源与确定性加载

- **子域映射的唯一真源**是 `src/data.py: SCIBENCH_SUBJECT_DOMAIN`，
  数据下载与加载路径都从这里取值，避免两处映射不一致导致子域统计失真。
- **本地数据优先**：`data/scibench/` 存在时直接使用，不先尝试 HF 镜像。
  HF 镜像的名称、字段与切分随时可能变化，会让同一份代码在不同机器上加载到不同数据集。
  可用 `EMPR_PREFER_HF=1` 强制走 HF。
- **`--max-problems N` 默认按科目轮转抽样**（不是取前 N 题）。直接切片只会取到字母序最前的科目，
  实测 `--max-problems 40` 会得到清一色 mechanics，随后被当作"在 SciBench 上 40 题"使用。

### 统计：可复现优先

- bootstrap CI 用纯 Python 实现（`random.Random` + 顺序统计量），
  保证"环境是否安装 numpy"不影响同一 seed 下的 CI，避免结果跨机器漂移。
- 配对显著性用 McNemar；错误类型分布对比用**未校正** Pearson 卡方
  （scipy 对 2x2 默认启用 Yates 校正，对比时注意 `correction=False`）。
- 统一种子 42、贪心解码（temperature=0）。
- 每个结果文件写入 `meta`：`git_commit`（脏工作区标记 `+dirty`）、运行环境、
  `prompt_fingerprint`、`max_new_tokens`、`chat_template`。结果不一致时先比对 `meta`。

### 工程化验证

- `scripts/preflight.py`：一条命令完成依赖检查、数据加载、判分自洽、变体覆盖率四项体检，
  未通过则以非零码退出。可在 CI 或批量任务前调用。
- `tests/`：83 项测试，覆盖判分回归、统计可复现性、变体探针的语义不变量、
  数据与提示词，以及**微型模型上的因果追踪集成测试**（验证 hook、张量对齐与恢复语义）。
- 所有结果文件均为严格合法 JSON（非有限值写 `null`）。

---

## 验证状态

所有数字均来自 `results/` 中可复现的输出。

| 项目 | 状态 | 证据 |
|---|---|---|
| 判分有效性（参考答案自判） | 已完成 | `results/selfcheck_scibench.json`：323/323 = 1.0000，0 条不可解析 |
| 判分回归用例 | 已完成 | `python -m src.grading` 输出 23/23 通过 |
| 一键体检 | 已完成 | `results/preflight_scibench.json`、`results/preflight_olympiadbench.json` 均 PASS |
| 单元 + 集成测试 | 已完成 | `pytest tests -q`，83 passed |
| 变体覆盖率（Week 2 前置） | 已完成 | SciBench：numeric 94.4%、wording 74.3%、swap 11.1%；OlympiadBench：57.8% / 51.3% / 2.8% |
| 第二基准可判分率 | 已完成 | `results/preflight_olympiadbench.json`：152/351 = 43.3% |
| 提示词长度实测 | 已完成 | `results/prompt_token_stats.json`：SciBench cot 平均 127 tokens（p90 183） |
| GPU 时间预算 | 已完成 | `results/gpu_budget_estimate.json`，由 `scripts/gpu_budget.py` 复算 |
| 管道端到端验证 | 已完成 | `results/smoke/`，见下 |
| Week 1 主表（正式模型） | 待 GPU | 需要 vllm 全量推理 |
| Week 2 敏感性（正式模型） | 待 GPU | 探针与审计口径已就绪 |
| Week 3 因果追踪（正式模型） | 待 GPU | hook 与恢复语义已在微型模型与 0.5B 上验证 |
| Week 4 错误分类 | 待人工标注 | 工具就绪：`scripts/classify_errors.py` |

### 管道验证产物（`results/smoke/`）

本机为 CPU 环境，用 0.5B 量级模型验证"提示词 -> 生成 -> 提取 -> 判定 -> CI -> 表格"整条链路可运行。
这些产物**不用于论文数字**，单独存放在 `results/smoke/`，避免与正式实验结果混淆
（同时也使 `make_table.py --dir results` 不会误把小模型混入论文主表）。

| 内容 | 命令要点 | 结果 |
|---|---|---|
| Week 1 direct | `--model Qwen/Qwen2.5-0.5B-Instruct --mode direct --max-problems 12 --device cpu` | overall 0.083 [0.000, 0.250]，`unparsed_pred` 1/12 |
| Week 1 cot | 同上，`--mode cot --max-problems 8 --max-new-tokens 192` | overall 0.000（0.5B 不具备 CoT 能力，符合预期） |
| Week 2 敏感性 | `run_sensitivity.py --max-problems 8 --mode direct` | 三变体分别产出覆盖率、敏感度与配对口径 |
| Week 3 因果追踪 | `run_causal_tracing.py --max-problems 6 --mode direct --layer-step 4` | 6 题中 1 题满足计入条件：clean 0.96 -> corrupt 0.00，峰值恢复 +0.43 @ layer 12/24 |
| MMLU 选择题 | `--benchmark mmlu_physics --mode direct --max-problems 12` | overall 0.167，验证选项字母判定链路 |
| 表格管道 | `make_table.py --dir results/smoke --format md` | 输出 `results/smoke/table_smoke.md` |

因果追踪中 5 题未计入，原因是 clean 运行本身未答对，恢复率无定义——这是筛选规则的正常结果，
正式模型中有效题数会显著提高。**有效题数必须写入论文。**

---

## 数据构成

### SciBench 物理（主基准，323 题）

| 子域 | 科目 | n | 占比 |
|---|---|---|---|
| general | fund（综合基础题） | 71 | 22% |
| thermodynamics | thermo | 66 | 20% |
| mechanics | class | 56 | 17% |
| math | diff（数学工具题） | 50 | 15% |
| optics | matter（波动与近代物理） | 47 | 15% |
| quantum | quan | 33 | 10% |

323 题参考答案全部可数值判分（`n_unscored = 0`），主表分母口径因此不受剔除影响。

### OlympiadBench（第二基准，351 题）

竞赛级物理，英文 236 题 + 中文 115 题，附官方解题步骤。
参考答案中**152 题（43.3%）** 可数值判分，其余为符号解或多变量解
（如 `\lambda=i\Omega`、`x(t)=A\cos(\Omega t+\phi)`），标记 `unscorable_ref` 后剔除出分母。
使用该基准出表时必须同时报告可判分覆盖率。

### MMLU 物理（选择题对照，488 题）

`high_school_physics` 151 + `college_physics` 102 + `conceptual_physics` 235，
判定走选项字母精确比对，题干含选项时自动切换为字母输出模板。

---

## 快速开始

```bash
pip install -r requirements.txt
python -m pytest tests -q                      # 83 项
python scripts/download_data.py                # SciBench 物理 6 科 -> data/scibench/*.jsonl
```

### 第一步：零成本体检（建议在调用正式模型前完成）

```bash
python scripts/preflight.py --benchmark scibench
        # 依赖 + 数据 + 判分自洽 + 变体覆盖率，结论写入 results/preflight_scibench.json
python -m src.grading                          # 判分回归用例明细
python scripts/run_eval.py --self-check --benchmark scibench --save
        # 以参考答案作为"模型输出"跑判分，期望 323/323 一致
python scripts/run_sensitivity.py --coverage --benchmark scibench --max-problems 323 --save
        # 三路变体覆盖率（不调用模型）
```

### 第二步：CPU 管道验证（可选）

```bash
python scripts/run_eval.py --model Qwen/Qwen2.5-0.5B-Instruct \
    --benchmark scibench --mode direct --max-problems 12 \
    --backend transformers --device cpu --max-new-tokens 64 --save --out-dir results/smoke
python scripts/check_grading.py --file results/smoke/eval_scibench_direct_Qwen__Qwen2.5-0.5B-Instruct.json
python scripts/make_table.py --dir results/smoke --format md
```

### 第三步：正式实验（GPU）

```bash
# Week 1：主表（模型 x 子域 正确率 + CI）
python scripts/run_eval.py --models core --benchmark scibench --mode cot \
    --max-problems 323 --backend vllm --gpus 1 --save
python scripts/check_grading.py --file results/eval_scibench_cot_*.json
python scripts/make_table.py --dir results --benchmark scibench --mode cot --format md --out results/table1.md

# Week 2：敏感性（记忆 vs 计算）
python scripts/run_sensitivity.py --model Qwen/Qwen2.5-7B-Instruct \
    --benchmark scibench --max-problems 323 --mode cot --backend vllm --save

# Week 3：因果追踪（逐层恢复率）
python scripts/run_causal_tracing.py --model Qwen/Qwen2.5-7B-Instruct \
    --benchmark scibench --max-problems 12 --mode direct --save --plot

# Week 4：错误类型（E1 定律误用 / E2 代数 / E3 约束违背）
python scripts/classify_errors.py --file results/eval_scibench_cot_*.json --report
python scripts/classify_errors.py --file A.json --report --compare-file B.json
```

部署与首跑细节见 [DEPLOY_CHECKLIST.md](DEPLOY_CHECKLIST.md)；论文表格对应关系见 [paper/TABLES.md](paper/TABLES.md)。
云上一键流程：`bash cloud_first_day.sh`（体检 + 小样本），确认后 `bash cloud_first_day.sh --full` 执行全量。

---

## GPU 时间预算

`scripts/gpu_budget.py` 将各实验折算为 prefill tokens、decode tokens 与启动次数，再按吞吐假设换算卡时。
其中 **prompt 长度是实测的**（`scripts/prompt_stats.py` 对真实题面分词），生成长度与吞吐是假设值，
可在脚本顶部常量或 `--throughput-scale` 中按现场实测调整（每个结果 json 都记录 `wall_time_s`，可用于反推）。

实测提示词长度（Qwen2.5 分词器，题意：SciBench 题面很短，成本主要来自生成）：

| 基准 / 模式 | mean | p50 | p90 | max |
|---|---|---|---|---|
| scibench / cot | 127 | 116 | 183 | 456 |
| scibench / direct | 122 | 111 | 178 | 451 |
| olympiadbench / cot | 153 | 142 | 252 | 817 |
| mmlu_physics / cot | 106 | 98 | 152 | 295 |

假设：direct 生成 30 tokens/题，cot 生成 450 tokens/题；vLLM bf16 单张 A800-80G 聚合输出吞吐
7B/8B 2500 tok/s、14B 1500 tok/s、32B 800 tok/s、70B/72B 400 tok/s（TP=2 时每卡减半）；
每次启动含 2-8 分钟模型加载开销。

主要实验的卡时估算：

| 实验 | 规模 | 卡数 | 卡时 |
|---|---|---|---|
| Week 1 主表（4 个 7B/8B/14B 模型 x cot + direct） | 323 题 | 1 | 约 0.4 |
| Week 1 补充（R1-Distill-14B） | 323 题 | 1 | 约 0.1 |
| Week 1 补充（Llama-70B、Qwen2.5-72B，各 cot） | 323 题 | 2 | 约 1.4 |
| 跨基准（OlympiadBench、MMLU 物理） | 152 / 488 题 | 1 | 约 0.1 |
| Week 2 敏感性（7B、14B，全量 323 题 x 三变体） | 生成量 x2.80 | 1 | 约 0.2 |
| Week 3 因果追踪（7B、14B，12 题） | 每题 18 次前向（HF 路径） | 1 | 约 0.1 |
| Week 4 错误分类（14B 作为标注器） | 约 240 条 | 1 | 约 0.1 |
| **合计** | 18 个任务 | — | **约 2.3 卡时**（含 1.5x 余量约 3.4 卡时） |

结论与建议：

- 完整实验的净计算量约 **2.3 A800 卡时**；计入失败重跑、参数调整与模型首次下载，按 **4-6 卡时** 规划即可。
- **1 张 A800-80G 可覆盖除 70B/72B 以外的全部实验**，一个工作日内可跑完。
- 70B / 72B 的 bf16 权重约 140 GB，需 **2 张卡做 TP=2**（A800 的 NVLink 带宽低于 A100，
  TP=2 效率略低），或改用 4-bit 量化在**单卡**上运行。
- 建议的并行方式：同时开 2 张卡，每张独立跑一个模型，主表 8 个任务约半小时完成；
  70B/72B 阶段再合为 TP=2。**不建议**为了并行而减少题量——CI 会明显变宽。
- 时间主要花在**模型加载**而非推理：单次 cot 全量推理仅 1-2 分钟，
  但每次启动需 2-8 分钟加载权重。因此应尽量合并模型到同一批任务、避免频繁重启。

复算方式：

```bash
python scripts/prompt_stats.py                 # 实测各基准 prompt 长度
python scripts/gpu_budget.py                   # 打印预算表并写入 results/gpu_budget_estimate.json
python scripts/gpu_budget.py --throughput-scale 1.5   # 现场吞吐更快时按比例调整
```

---

## 结果文件说明

| 文件 | 内容 | 关键字段 |
|---|---|---|
| `results/preflight_<bench>.json` | 一键体检结论 | `passed`、`data`、`grading_self_test`、`judge_self_consistency`、`variant_coverage` |
| `results/selfcheck_<bench>.json` | 判分自检 | `judge_self_consistency`、`n_inconsistent` |
| `results/eval_<bench>_<mode>_<model>.json` | 主结果 | `overall_acc`、`per_domain_acc`、`n_scorable`、`n_unscored`、`n_unparsed_pred`、`detail[]`、`meta` |
| `results/sensitivity_coverage_<bench>.json` | 变体覆盖率 | `coverage.{numeric,wording,swap}` |
| `results/sensitivity_<variant>_<model>.json` | 单变体结果 | `unchanged_share`、`median/p90/mean_sensitivity`、`items[]`（逐题审计） |
| `results/sensitivity_summary_<model>.json` | 三变体汇总 | `variants.*`；wording 含配对准确率与 McNemar |
| `results/causal_tracing_<bench>_<model>.json` | 逐层恢复率 | `average_curve`、`problems[*].curve`、`n_valid` 与跳过计数 |
| `results/prompt_token_stats.json` | 提示词长度实测 | `prompt_tokens.<bench>/<mode>` |
| `results/gpu_budget_estimate.json` | GPU 预算估算 | `assumptions`、`summary`、`runs` |
| `results/smoke/` | 管道验证产物 | 见上节 |

---

## 方法学说明与限制

1. **swap 变体覆盖率 11.1%（36/323）**：多数题目的数值同属一个物理量，或缺少可识别单位，
   无法构造"互换两个不同物理量"的变体。这是数据固有属性，如实报告，不通过放宽规则凑样本。
2. **wording 变体为规则改写（覆盖 74.3%）**：只替换框架词与同义短语，不触碰数字与单位。
   更强的一致性改写可交给模型，但会引入"改写质量"这一额外变量。
3. **敏感度统计量**：敏感度定义为 `|delta| / (|base| + eps)`，当 base 接近 0 时单个样本即可显著抬高均值
   （0.5B 验证中 wording 的 mean 为 2983.8，median 为 0.000）。
   论文报告 median、p90 与 `unchanged_share`，不报告 mean。
4. **numeric 变体不报告"变体准确率"**：其正确答案随物理关系变化且不可直接求解，
   仅报告输出变化分布；只有 wording 变体（物理与数据完全不变）可报告配对准确率与 McNemar 检验。
5. **因果追踪仅对开放权重模型做隐藏态干预**，闭源 API 不可复现。
   仅"clean 答对且 corrupt 确实掉分"的题目计入恢复率，报告时须给出 `n_valid` 与跳过原因计数。
6. **可判分覆盖率**：判定基于数值比对。SciBench 物理 323 题全部可判分；
   OlympiadBench 仅 43.3% 可数值判分，其正确率在可判分子集上计算，覆盖率必须一并报告。
   MMLU 物理为选择题，不涉及该问题。
7. **正式实验需 GPU**：本仓库在 CPU 环境完成管道验证、单元测试与全部零成本检查；
   主表、敏感性与因果追踪的正式数字需按上节预算在 GPU 上运行。

---

## 环境

| 组件 | 版本要求 | 说明 |
|---|---|---|
| Python | 3.10+ | 本仓库在 3.14 上验证 |
| numpy / scipy / sympy | >=1.24 / >=1.10 / >=1.11 | 统计与判分；sympy 缺失时判分会退化但不报错 |
| pyyaml | >=6.0 | 读取 `config.yaml` |
| pytest | >=7.4 | 单元与集成测试 |
| torch | >=2.0 | 推理后端 |
| transformers | >=4.40 | 4.x 与 5.x 均可；精度参数名按主版本自动适配（`torch_dtype` / `dtype`） |
| vllm | >=0.4 | GPU 推理加速；缺失时自动回退 transformers |
| matplotlib | >=3.7 | 因果追踪曲线 |

---

## 相关项目

- **量子纠错解码（Ising Decoding）**：<https://github.com/380992846-tech/Ising-Decoding>
