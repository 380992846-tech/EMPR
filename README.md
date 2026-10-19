# EMPR — Emergent Physical Reasoning in Open LLMs

面向开放权重 LLM 的**多步物理推理**评测与机理分析框架。围绕三个可证伪的问题组织，并在
SciBench 物理全量（323 题）上给出实测结果：

| 阶段 | 研究问题 | 方法 |
|---|---|---|
| Week 1 | 性能规律是什么？ | SciBench 物理 323 题 x 多模型 x 子域正确率 + bootstrap 95% CI |
| Week 2 | 模型在**计算**还是在**检索**？ | 三路变体探针：数值扰动 / 表层改写 / 物理量互换 |
| Week 3 | 若在计算，**计算发生在哪些层**？ | 逐层恢复率（causal tracing），定位关键层区间 |

---

## 实验结果

环境：单卡 NVIDIA A800-80G，bf16，贪心解码（temperature=0），seed=42。
SciBench 物理 323 题，**参考答案全部可判分（n_unscored = 0），模型输出全部可解析（n_unparsed = 0）**，
因此下表分母即全量 323 题。

### 1. 主结果：模型 x 提示策略

| 模型 | cot | direct | cot / direct |
|---|---|---|---|
| Qwen2.5-7B-Instruct | **0.211** [0.167, 0.257] | 0.087 [0.056, 0.118] | 2.43x |
| Qwen2.5-14B-Instruct | **0.257** [0.211, 0.307] | 0.118 [0.084, 0.155] | 2.18x |
| DeepSeek-R1-Distill-Qwen-7B | 0.111 [0.077, 0.146] | — | — |

**发现**

1. **思维链是最强的单一因素**：cot 相对 direct 提升 2.2-2.4 倍，远大于模型规模带来的增益。
2. **规模收益递减**：7B -> 14B（参数量 2 倍）在 cot 下仅 +4.6pp（+22% 相对），direct 下 +3.1pp。
   单靠放大模型难以突破 26% 的正确率。
3. **推理蒸馏模型在此基准上并未占优**：DeepSeek-R1-Distill-Qwen-7B（0.111）显著低于同规模
   Qwen2.5-7B-Instruct（0.211）。需注意其输出被 1536 token 上限截断，长推理链可能未完成——
   这一结果提示"长思维链"与"该基准的数值作答格式"之间存在结构性摩擦。
4. **子域难度不均**：cot 下数学工具题（diff）最高（7B 0.340 / 14B 0.400），综合基础题（fund）
   最低（0.169 / 0.197）。

### 2. 子域明细（cot，正确率 [95% CI]）

| 模型 | math | mechanics | thermodynamics | optics | quantum | general |
|---|---|---|---|---|---|---|
| Qwen2.5-14B | **0.400** [0.260, 0.540] | 0.268 [0.161, 0.393] | 0.242 [0.152, 0.348] | 0.234 [0.128, 0.362] | 0.212 [0.091, 0.364] | 0.197 [0.113, 0.296] |
| Qwen2.5-7B | 0.340 [0.220, 0.480] | 0.214 [0.107, 0.321] | 0.212 [0.121, 0.318] | 0.170 [0.064, 0.277] | 0.152 [0.030, 0.273] | 0.169 [0.085, 0.254] |
| R1-Distill-7B | 0.120 [0.040, 0.220] | 0.071 [0.018, 0.143] | 0.106 [0.045, 0.182] | 0.085 [0.021, 0.170] | 0.182 [0.061, 0.333] | 0.127 [0.056, 0.211] |

子域 n：math 50 / mechanics 56 / thermodynamics 66 / optics 47 / quantum 33 / general 71。

direct 模式下量子子域出现异常高点（Qwen2.5-14B：0.303 [0.152, 0.455]，是其 all-domain 最高），
与 cot 下的排序相反。该子域仅 33 题、CI 宽，且不排除少量题目与训练分布重合，**不作为结论使用**，
在论文中作为 caveat 说明。

### 3. 敏感性探针：数值变化 vs 表层改写（Qwen2.5-7B，direct，323 题基准）

| 变体 | 覆盖题数 | 输出改变比例 | median 敏感度 | p90 敏感度 |
|---|---|---|---|---|
| numeric（改数值） | 305 / 323 | **77.7%** | 0.587 | 8.96 |
| wording（纯表层改写，物理与数据完全不变） | 240 / 323 | **30.4%** | 0.000 | 0.95 |
| swap（互换两个物理量） | 36 / 323 | 69.4% | 0.371 | 15.06 |

wording 变体的配对准确率：8.33% -> 9.58%（Δ=+1.25pp，McNemar p=0.371，不显著）。

**发现**

1. 改数值后 77.7% 的输出发生变化，说明模型**并非纯粹查表**，输入确实参与了计算。
2. 但**在物理与数据一字不改、仅替换措辞的情况下，30.4% 的最终答案发生改变**。
   这是表层形式敏感的直接影响：答案的稳定性依赖于题面的具体表述，而不是物理结构本身。
3. 表层改写整体上不改变正确率（+1.25pp，p=0.371），即这些翻转大致是**对称的错误波动**，
   而非"改写让它变难"。综合 (1)(2)：模型的输出来自计算，但计算过程对无关的表层形式高度不稳定，
   行为更接近 "sophisticated pattern matching" 而非稳健的物理求解。
4. `swap` 覆盖率仅 11.1%（36/323），是数据固有属性（多数题目的数值同属一个物理量或缺少可识别单位），
   结论仅作辅助，不作为主证据。

### 4. 因果追踪：逐层恢复率（Qwen2.5-7B，28 层）

方法：把物理量替换为无语义占位符得到 corrupt 输入，再逐层把 clean 的 hidden state 恢复回去，
量测正确性回升幅度。仅当 **clean 答对且 corrupt 确实掉分** 时该题才计入
（150 题中有效 **14 题**，跳过 clean 答错 134 题、corrupt 未掉分 2 题）。

| 层区间 | 恢复率（均值 ± std，n=14） |
|---|---|
| 0-2 | +0.43 ± 0.91 |
| **4-16（平台区）** | **+0.52 ~ +0.84**，峰值 **layer 8：+0.837 ± 0.251** |
| 18-20 | +0.42 -> +0.20（快速衰减） |
| 22-26 | +0.04 ~ +0.10（基本为零） |

**发现**：可恢复的行为信息集中在**网络前中段（约第 4-16 层，共 28 层）**，第 18 层之后恢复干预
迅速失效。这与 causal tracing 文献中"中段层承载可迁移表征"的结论方向一致。
layer 0 的高方差（±0.91）来自 embedding 级恢复的不稳定性，不应解读为有效恢复点。

---

## 复现

```bash
pip install -r requirements.txt
python -m pytest tests -q                  # 83 项
python scripts/download_data.py            # SciBench 物理 6 科 -> data/scibench/*.jsonl
python scripts/preflight.py --benchmark scibench    # 零成本体检，未通过则不必上 GPU
```

正式实验（GPU）：

```bash
# Week 1 主表
python scripts/run_eval.py --model Qwen/Qwen2.5-7B-Instruct --benchmark scibench \
    --mode cot --max-problems 323 --backend vllm --gpus 1 --save
python scripts/run_eval.py --model Qwen/Qwen2.5-14B-Instruct --benchmark scibench \
    --mode cot --max-problems 323 --backend vllm --gpus 1 --save
python scripts/make_table.py --dir results --benchmark scibench --mode cot --format md --out results/table1_cot.md

# Week 2 敏感性
python scripts/run_sensitivity.py --model Qwen/Qwen2.5-7B-Instruct --benchmark scibench \
    --mode direct --max-problems 323 --backend vllm --save

# Week 3 因果追踪（需 transformers 后端：要读取隐藏态）
python scripts/run_causal_tracing.py --model Qwen/Qwen2.5-7B-Instruct --benchmark scibench \
    --max-problems 150 --mode direct --layer-step 2 --device cuda --save

# Week 4 错误类型
python scripts/classify_errors.py --file results/eval_scibench_cot_*.json --report
```

本次实验所用的完整命令、日志口径与超参记录在 `results/*.json` 的 `meta` 中
（`git_commit`、`prompt_fingerprint`、`max_new_tokens`、`chat_template`、`wall_time_s`）。
部署细节见 [DEPLOY_CHECKLIST.md](DEPLOY_CHECKLIST.md)，表格与论文对齐关系见 [paper/TABLES.md](paper/TABLES.md)。

---

## GPU 时间预算（实测校准）

`scripts/gpu_budget.py` 给出可复算的预算表；其中 prompt 长度由 `scripts/prompt_stats.py` 实测：

| 基准 / 模式 | mean | p50 | p90 | max |
|---|---|---|---|---|
| scibench / cot | 127 | 116 | 183 | 456 |
| scibench / direct | 122 | 111 | 178 | 451 |
| olympiadbench / cot | 153 | 142 | 252 | 817 |
| mmlu_physics / cot | 106 | 98 | 152 | 295 |

本次实验的**实测吞吐**（A800-80G，bf16，transformers 后端，batch 16-32，贪心解码）：

| 模型 / 模式 | 323 题耗时 | 单题耗时 | 聚合输出吞吐 |
|---|---|---|---|
| Qwen2.5-7B / cot | 494 s | 1.53 s | 294 tok/s |
| Qwen2.5-14B / cot | 1388 s | 4.30 s | 105 tok/s |
| Qwen2.5-7B / direct | 15.5 s | 0.05 s | 625 tok/s |
| Qwen2.5-14B / direct | 34.5 s | 0.11 s | 281 tok/s |
| R1-Distill-7B / cot (1536 tok) | 926 s | 2.87 s | 536 tok/s |

结论：

- **本轮全部实验的净 GPU 计算约 0.9 卡时**（3 个模型、5 次全量或大规模推理）。
- 与预算脚本的差异主要来自后端：`transformers.generate` 无连续批处理，7B 实测 294 tok/s、
  14B 105 tok/s，而 vLLM 通常快 5-10 倍。
- **时间主要花在模型下载**（约 55 分钟，走 ModelScope 约 14 MB/s），而非推理本身。
  调度上应把同一模型的多个实验排在一起，减少重复下载。
- 若全部改用 vLLM，同样组合估计在 **0.2 卡时**内完成；`--backend auto` 会优先尝试 vLLM。

复算：

```bash
python scripts/prompt_stats.py                  # 实测 prompt 长度
python scripts/gpu_budget.py                    # 预算表（写入 results/gpu_budget_estimate.json）
python scripts/gpu_budget.py --throughput-scale 3   # 换成 vLLM 实测吞吐时按比例缩放
```

---

## 核心设计

### 判分：参考答案严进，模型输出宽出

- `parse_reference`（`src/grading.py`）**严格**解析参考答案，要求整串就是一个数值
  （允许单位、百分号、分数、科学计数法、`\frac{}{}`、`\sqrt{}`、`\pi`/`\cot` 等 LaTeX 数值形式）。
  无法确定时返回 `None`，该题标记 `unscorable_ref` 并**从分母剔除**，不算模型答错。
- `extract_number`（`src/grading.py`）**宽松**解析模型输出，按可信度依次尝试
  `ANSWER:` 行（取最后一段）-> `\boxed{}` -> "answer is X" -> 末行数字 -> 全文末数字，
  并规范化 Unicode 减号、上标、`\times`、千分位逗号。
- 结果分三类口径：`correct` / `incorrect` / `unscorable_ref`。每个结果文件都记录
  `n_scorable`、`n_unscored`、`n_unparsed_pred`，三者必须同时报告。

若两侧共用同一个宽松解析器，`x^2 + 2x` 会被读成 `2`、`\frac{1}{2}mv^2` 会被读成 `1`，
样本被误判为可判分且无从察觉，因此两侧的解析强度必须不同。

### 推理：三条必须成立的约束

- **批量生成使用左填充**。decoder-only 右填充时会在 pad 之后继续生成，`transformers` 仅给 warning
  而不报错，批量结果整体失真。
- **instruct 模型走 chat template**（`chat=auto`），否则模型倾向于"续写指令"而非回答问题；
  是否启用记录在结果的 `meta.chat_template`。
- **输出条数与题数必须一致**，否则直接抛错——`zip(rows, outputs)` 一旦错位，
  A 题的输出会被判到 B 题的答案上且不会报错。

### 数据：单一真源与确定性加载

- 子域映射的唯一真源是 `src/data.py: SCIBENCH_SUBJECT_DOMAIN`，下载与加载路径均由此取值。
- **本地数据优先**：`data/scibench/` 存在时直接使用，不先尝试 HF 镜像（镜像的字段与切分可能变化，
  会让同一份代码在不同机器加载到不同数据集）；`EMPR_PREFER_HF=1` 可强制走 HF。
- `--max-problems N` 默认**按科目轮转抽样**：直接切片只会取到字母序最前的科目，
  例如 `--max-problems 40` 得到的是清一色 mechanics。

### 统计：可复现优先

- bootstrap CI 用纯 Python 实现（`random.Random` + 顺序统计量），保证环境是否安装 numpy
  不影响同一 seed 下的 CI。
- 配对显著性用 McNemar；错误类型分布用**未校正** Pearson 卡方（scipy 对 2x2 默认 Yates 校正）。
- 统一 seed=42、贪心解码。每个结果文件写入 `meta`（commit / 环境 / 提示词指纹 / 解码参数），
  结果不一致时先比对 `meta`。

### 工程化验证

- `scripts/preflight.py`：一次完成依赖、数据、判分自洽、变体覆盖率四项体检，未通过则非零退出。
- `tests/`：83 项测试，覆盖判分回归、统计可复现性、变体探针的语义不变量、数据与提示词，
  以及**微型模型上的因果追踪集成测试**（验证 hook、张量对齐与恢复语义）。
- 所有结果文件均为严格合法 JSON（非有限值写 `null`）。

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

### OlympiadBench（第二基准，351 题）

竞赛级物理（英 236 / 中 115），附官方解题步骤。参考答案中 **152 题（43.3%）可数值判分**，
其余为符号解或多变量解（如 `\lambda=i\Omega`、`x(t)=A\cos(\Omega t+\phi)`），
标记 `unscorable_ref` 后剔除，使用时必须同时报告可判分覆盖率。

### MMLU 物理（选择题对照，488 题）

`high_school_physics` 151 + `college_physics` 102 + `conceptual_physics` 235，
判定走选项字母精确比对，题干含选项时自动切换为字母输出模板。

---

## 结果文件说明

| 文件 | 内容 |
|---|---|
| `results/eval_<bench>_<mode>_<model>.json` | 主结果：`overall_acc`、`per_domain_acc`、`n_scorable`、`n_unscored`、`n_unparsed_pred`、`detail[]`、`meta` |
| `results/table1_cot.md` / `.tex` | 由结果自动生成的主表（cot） |
| `results/sensitivity_summary_<model>.json` | 三变体汇总；wording 含配对准确率与 McNemar |
| `results/sensitivity_<variant>_<model>.json` | 单变体明细，含 `items[]` 逐题审计 |
| `results/causal_tracing_<bench>_<model>.json` | 逐层恢复率：`average_curve`、`problems[*].curve`、`n_valid` 与跳过计数 |
| `results/preflight_<bench>.json` | 一键体检结论 |
| `results/selfcheck_<bench>.json` | 判分自检（参考答案自判） |
| `results/prompt_token_stats.json` | 提示词长度实测 |
| `results/gpu_budget_estimate.json` | GPU 时间预算 |
| `results/smoke/` | 管道验证产物（0.5B / CPU），不用于结论 |

---

## 方法学说明与限制

1. **swap 变体覆盖率 11.1%**：多数题目数值同属一个物理量或缺少可识别单位，无法构造
   "互换两个不同物理量"的变体。如实报告覆盖率，不放宽判据凑样本。
2. **wording 变体为规则改写（覆盖 74.3%）**：只替换框架词与同义短语，不触碰数字与单位。
   更强的一致性改写可由模型完成，但会引入"改写质量"这一额外变量。
3. **敏感度统计量**：定义为 `|delta|/(|base|+eps)`，基准答案接近 0 时单个样本即可显著抬高均值
   （管道验证中曾出现 mean 2983.8 而 median 0.000）。因此报告 median、p90 与输出改变比例，
   不报告 mean。
4. **numeric 变体不报告"变体准确率"**：其正确答案随物理关系变化且不可直接求解，
   仅报告输出变化分布；只有 wording 变体（物理与数据完全不变）可报告配对准确率与 McNemar。
5. **因果追踪仅对开放权重模型做隐藏态干预**，闭源 API 不可复现。仅"clean 答对且 corrupt 确实掉分"
   的题目计入，报告时给出 `n_valid` 与跳过原因计数（本次 150 题中 14 题有效，样本量偏小，
   layer 0-2 的高方差需谨慎解读）。
6. **可判分覆盖率**：SciBench 物理 323 题全部可判分；OlympiadBench 仅 43.3% 可数值判分，
   其正确率在可判分子集上计算，覆盖率必须一并报告。
7. **后端差异**：本轮使用 transformers 后端（A800），未安装 vLLM；吞吐比 vLLM 低 5-10 倍。
   结论不受影响，但复现时耗时会有差异。

---

## 环境

| 组件 | 版本要求 | 说明 |
|---|---|---|
| Python | 3.10+ | 在 3.12 / 3.14 上验证 |
| numpy / scipy / sympy | >=1.24 / >=1.10 / >=1.11 | 统计与判分；sympy 缺失时判分会退化但不报错 |
| pyyaml | >=6.0 | 读取 `config.yaml` |
| pytest | >=7.4 | 单元与集成测试 |
| torch | >=2.0 | 本轮实测 2.11+cu128 |
| transformers | >=4.40 | 4.x 与 5.x 均可；精度参数名按主版本自动适配（`torch_dtype` / `dtype`） |
| vllm | >=0.4 | 可选，GPU 推理加速；缺失时自动回退 transformers |
| matplotlib | >=3.7 | 因果追踪曲线 |

注：若镜像内 `torchvision` 与 torch 版本不匹配（报 `operator torchvision::nms does not exist`），
会导致 transformers 无法导入 Qwen2 模型类；本项目不需要视觉库，直接
`pip uninstall -y torchvision` 即可。

---

## 相关项目

- **量子纠错解码（Ising Decoding）**：<https://github.com/380992846-tech/Ising-Decoding>
