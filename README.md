# EMPR — Emergent Physical Reasoning in Open LLMs

面向开放权重 LLM 的**多步物理推理**评测与机理分析框架。围绕三个可证伪的问题组织，并在
SciBench 物理全量（323 题）上给出实测结果：

| 阶段 | 研究问题 | 方法 |
|---|---|---|
| Week 1 | 性能规律是什么？ | SciBench 物理 323 题 x 多模型 x 子域正确率 + bootstrap 95% CI |
| Week 2 | 模型在**计算**还是在**检索**？ | 三路变体探针：数值扰动 / 表层改写 / 物理量互换 |
| Week 3 | 若在计算，**计算发生在哪些层**？ | 逐层恢复率（causal tracing），定位关键层区间 |
| Week 4 | 错在哪一类环节？ | 错误类型标注 + 异族标注器复核（E1/E2/E3 + kappa） |

**论文产物以什么为准**（本仓库没有完整论文草稿，只有表格与章节片段）：

| 需要的东西 | 以哪个文件为准 |
|---|---|
| 任何数字 | `results/*.json`（原始结果，带 `meta`） |
| 论文表格 | `paper/table1..7_*.tex`，由 `scripts/make_paper_tables.py` 从 results 自动生成，**禁止手改** |
| 中文结论叙述 | 本 README 第 1-9 节 |
| 英文叙述 | `paper/discussion.tex`、`paper/reproducibility_statement.tex` |
| 填表口径与分母定义 | `paper/TABLES.md` |

---

## 实验结果

环境：单卡 NVIDIA A800-80G，bf16，贪心解码（temperature=0），seed=42。
SciBench 物理 323 题，**参考答案全部可判分（n_unscored = 0），模型输出全部可解析（n_unparsed = 0）**，
因此下表分母即全量 323 题。

### 1. 主结果：5 个模型 x 2 种提示策略

| 模型 | cot | direct | cot / direct |
|---|---|---|---|
| Qwen2.5-3B-Instruct | 0.121 [0.087, 0.158] | 0.053 [0.031, 0.077] | 2.28x |
| Qwen2.5-7B-Instruct | 0.211 [0.167, 0.257] | 0.087 [0.056, 0.118] | 2.43x |
| Qwen2.5-14B-Instruct | **0.257** [0.211, 0.307] | **0.118** [0.084, 0.155] | 2.18x |
| Meta-Llama-3.1-8B-Instruct | 0.099 [0.068, 0.133] | 0.077 [0.050, 0.108] | 1.29x |
| DeepSeek-R1-Distill-Qwen-7B | 0.111 [0.077, 0.146] | — | — |

**发现**

1. **规模收益递减**（Qwen 家族内部）：3B 0.121 -> 7B 0.211 -> 14B 0.257。
   从 3B 到 7B 提升 +9.0pp，7B 到 14B（参数翻倍）仅 +4.6pp，且 14B 的 95% CI 下界（0.211）
   与 7B 的点估计重合——规模接近饱和。
2. **CoT 的收益依赖模型家族，而非普遍规律**：Qwen 家族 cot/direct 稳定在 2.18-2.43 倍，
   而 Llama-3.1-8B 仅 1.29 倍（0.099 vs 0.077）。"让模型一步步写"并不自动带来大幅提升。
3. **家族差异大于规模差异**：参数量仅 3B 的 Qwen2.5-3B（0.121）高于 8B 的 Llama-3.1-8B（0.099）。
   在这类数值物理题上，模型家族的影响超过 2.7 倍的参数差距。
4. **推理蒸馏模型在此基准上并未占优**：DeepSeek-R1-Distill-Qwen-7B（0.111）低于同规模
   Qwen2.5-7B-Instruct（0.211）。需注意其输出被 1536 token 上限截断，长推理链可能未完成——
   提示"长思维链"与"该基准的数值作答格式"之间存在结构性摩擦。
5. **子域难度不均**：cot 下数学工具题（diff）最高（3B 0.200 / 7B 0.340 / 14B 0.400），
   综合基础题最低（0.099 / 0.169 / 0.197）；mechanics 在 3B 上是唯一低于 0.1 的子域（0.054）。

### 2. 子域明细（cot，正确率 [95% CI]）

| 模型 | math | mechanics | thermodynamics | optics | quantum | general |
|---|---|---|---|---|---|---|
| Qwen2.5-14B | **0.400** [0.260, 0.540] | 0.268 [0.161, 0.393] | 0.242 [0.152, 0.348] | 0.234 [0.128, 0.362] | 0.212 [0.091, 0.364] | 0.197 [0.113, 0.296] |
| Qwen2.5-7B | 0.340 [0.220, 0.480] | 0.214 [0.107, 0.321] | 0.212 [0.121, 0.318] | 0.170 [0.064, 0.277] | 0.152 [0.030, 0.273] | 0.169 [0.085, 0.254] |
| Qwen2.5-3B | 0.200 [0.100, 0.320] | 0.054 [0.000, 0.125] | 0.121 [0.045, 0.212] | 0.106 [0.021, 0.213] | 0.182 [0.061, 0.333] | 0.099 [0.028, 0.169] |
| Llama-3.1-8B | 0.140 [0.060, 0.240] | 0.089 [0.018, 0.179] | 0.121 [0.045, 0.197] | 0.064 [0.000, 0.149] | 0.061 [0.000, 0.152] | 0.099 [0.028, 0.169] |
| R1-Distill-7B | 0.120 [0.040, 0.220] | 0.071 [0.018, 0.143] | 0.106 [0.045, 0.182] | 0.085 [0.021, 0.170] | 0.182 [0.061, 0.333] | 0.127 [0.056, 0.211] |

子域 n：math 50 / mechanics 56 / thermodynamics 66 / optics 47 / quantum 33 / general 71。

主表与子域表均可由 `results/table1_cot.md` / `.tex` 复现（`scripts/make_table.py` 生成）。

direct 模式下量子子域出现异常高点（Qwen2.5-14B：0.303 [0.152, 0.455]，是其 all-domain 最高），
与 cot 下的排序相反。该子域仅 33 题、CI 宽，且不排除少量题目与训练分布重合，**不作为结论使用**，
在论文中作为 caveat 说明。

### 3. 敏感性探针：数值变化 vs 表层改写（4 个模型，direct，323 题基准）

先看 Qwen2.5-7B 的完整口径：

| 变体 | 覆盖题数 | 输出改变比例 | median 敏感度 | p90 敏感度 |
|---|---|---|---|---|
| numeric（改数值） | 305 / 323 | **77.7%** | 0.587 | 8.96 |
| wording（纯表层改写，物理与数据完全不变） | 240 / 323 | **30.4%** | 0.000 | 0.95 |
| swap（互换两个物理量） | 36 / 323 | 69.4% | 0.371 | 15.06 |

四个模型的横向对比（数字为"输出发生改变的比例"）：

| 模型 | direct 正确率 | numeric 改变 | wording 改变 | swap 改变 |
|---|---|---|---|---|
| Qwen2.5-3B | 0.053 | 77.0% | 33.8% | 83.3% |
| Qwen2.5-7B | 0.087 | 77.7% | 30.4% | 69.4% |
| Qwen2.5-14B | 0.118 | **83.9%** | 31.7% | 80.6% |
| Llama-3.1-8B | 0.077 | **57.4%** | 24.4% | 63.9% |

（`swap` 覆盖仅 36 题，比例波动大，仅作辅助证据。）

![Output sensitivity](<results/sensitivity_comparison.png>)

**发现**

1. **模型会随数值变化**：numeric 变体下 57%-84% 的输出发生改变，说明并非纯粹查表，
   输入数值确实参与计算。
2. **但表层措辞同样能改变 24%-34% 的最终答案**——此时物理与数据一字未改，改变的只有问法与措辞。
   四个模型**无一例外**，说明这是开放权重模型在该基准上的普遍行为，而非个别模型的怪癖。
3. **表层改写不改变正确率**（4 个模型的配对 McNemar 检验 p 值均 > 0.13：
   3B +0.0pp / 7B +1.3pp / 14B +2.1pp / Llama -0.4pp）。即这些翻转是对称的错误波动，
   而不是"改写让题目变难"。综合 (1)(2)：输出确实来自计算，但计算过程对与物理无关的表层形式
   高度不稳定 —— 这正是 "sophisticated pattern matching" 的操作性刻画。
4. **对数值的响应强度与正确率方向一致**：Qwen 家族内部 3B 77.0% -> 7B 77.7% -> 14B 83.9%，
   模型越大越"跟着数值算"，正确率也越高；而 Llama-3.1-8B 是唯一低于 60% 的（57.4%），
   同时也是 cot 正确率最低的模型（0.099）。n=4 下该趋势只作定性描述，不作为定量结论。

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

![Causal tracing recovery curve](<results/causal_tracing_Qwen2.5-7B.png>)

### 5. 识别 vs 生成：MMLU 物理对照（Qwen2.5-7B，488 题选择题）

同一个模型、同样属于物理知识，把答案形式从"自由作答数字"换成"四选一"：

| 任务 | 题量 | 作答形式 | 正确率 |
|---|---|---|---|
| MMLU 物理（high_school / college / conceptual） | 488 | 选出 A/B/C/D | **0.666** [0.623, 0.707] |
| SciBench 物理 cot | 323 | 写出最终数值 | 0.211 [0.167, 0.257] |
| SciBench 物理 direct | 323 | 直接写数值 | 0.087 [0.056, 0.118] |

**发现**：同一模型在选择题上 **66.6%**，在需要自行产出数值时只有 **21.1%**，相差 3.2 倍。
这排除了"模型缺乏物理知识"这一解释 —— 知识在，**从候选项里认出正确答案可以，独立把数算出来不行**。
这是本项目"pattern matching 而非物理求解"定位最直接的一组对照证据。
（MMLU 488 题全部可判分、模型输出全部可解析，`unparsed = 0`。）

### 6. 第二基准：OlympiadBench（竞赛物理，351 题）

| 模型 | 可判分题数 | cot 正确率 |
|---|---|---|
| Qwen2.5-7B-Instruct | 152 | 0.237 [0.171, 0.303] |
| Qwen2.5-14B-Instruct | 152 | 0.303 [0.230, 0.375] |

**结论**：14B 优于 7B（+6.6pp），与 SciBench 上的 +4.6pp 方向一致，说明规模趋势不是单一基准的偶然。

**可比性限制**：OlympiadBench 351 题中只有 **152 题（43.3%）** 的参考答案可以数值判分，
其余 199 题是符号解/多变量解（如 `x(t)=A\cos(\Omega t+\phi)`）被剔除。
能留下纯数值答案的题目本身就偏简单，因此**不能把 OlympiadBench 的 30.3% 与 SciBench 的 25.7% 直接比较**，
只能在同一基准内部比较模型。子域上两个模型在 optics 均为 0.000，但该子域仅 6 题可判分，
置信区间退化为 [0, 0]，不作为结论。

### 7. 表征与输出是否解耦：逐层分离"写成数字"与"算对数字"

第 4 节只统计"clean 答对且 corrupt 掉分"的题（14 题），因为恢复率需要正确的参照。
另一个不依赖该假设的问题值得单独测：**逐层恢复时，"产出可解析数值"与"数值正确"
是不是同一条通路？** 为此对全部 100 道尝试题逐层恢复，同时记录这两个指标：

| 条件 | 可解析率（输出执行） | 正确率（语义） |
|---|---|---|
| corrupt（不恢复） | 96.0% | 1.0% |
| clean（不破坏） | 100.0% | 10.0% |
| 逐层恢复 layer 0-24（间隔 4） | **97.0% - 100.0%**（几乎持平） | 1.0% - 9.0%（波动） |

**按 clean 是否正确拆开看**（关键）：

| 恢复层 | clean 答对的 10 题 | clean 答错的 90 题 |
|---|---|---|
| 0 | **70%** | 1.1% |
| 4 | 40% | 3.3% |
| 8 | 50% | 0.0% |
| 12 | **70%** | 0.0% |
| 16 | 60% | 3.3% |
| 20 | 30% | 2.2% |
| 24 | **10%** | 0.0% |

**发现**

1. **输出执行与层无关**：corrupt 状态下仍有 96% 产出可解析数值（模型不会说"我不知道"），
   逐层恢复后稳定在 97%-100%。**"把答案写成数字"不依赖任何特定层。**
2. **数值正确性依赖中段状态**：对本来能解的题，恢复第 0/12 层把正确率从 1% 拉回 **70%**，
   4-16 层维持 40%-70%，到第 24 层只剩 10%。即**数值由前中段状态承载，晚层恢复基本无效**。
3. **对本就不会的题，任何单层恢复都救不回**：90 题逐层正确率最高仅 **3.3%**。

**结论**：格式通路与数值通路确实分开（第 1 点 vs 第 2 点），
但"会不会做"不由某一层决定，因此**不能**靠表征修补解决不会做的题。
这条负结果同样有价值：它排除了"模型其实会算、只是输出格式错了"这一解释 —— corrupt 时
96% 的样本都能正常输出数字，错的不是格式。

### 8. Week 4：错误类型分布（E1 定律误用 / E2 代数计算 / E3 约束违背）
标注方式：以 Qwen2.5-14B 作为标注器，对 7B 与 14B 在 SciBench cot 下的全部错误样本分类。
已剔除 `unscorable_ref`（参考答案不可判分）与 `unparsed_pred`（模型未给出可解析答案）。

| 模型 | 错误样本 | E1 定律误用 | E2 代数/计算 | E3 约束违背 | N 判定误报 |
|---|---|---|---|---|---|
| Qwen2.5-7B | 255 | 29.4% | **57.6%** | 10.2% | 2.7% |
| Qwen2.5-14B | 240 | 22.5% | **65.0%** | 8.3% | 4.2% |

两模型错误分布卡方检验：chi2 = 4.55, df = 3, **p = 0.208**（无显著差异）。

**发现**：错误的主体是 **E2（代数/计算，58%-65%）**，而非 E1（选错物理定律，22%-29%）。
即模型多数情况下**选对了物理关系，却在具体数值推导上失败**。
且 14B 与 7B 的错误类型分布无显著差异 —— 14B 的提升来自"更多题做对"，而不是"换了失败模式"。
这一结果与第 5 节的 MMLU 对照互为印证：知识层面（选定律、认选项）较强，数值执行层面薄弱。

#### 8.1 异族标注器复核（关键：这条结论有多稳？）

同族标注器可能有自我偏好，因此用**异族**的 Llama-3.1-8B 对**同一批 495 条样本**独立重标：

| 标注器 | E1 | E2 | E3 | N |
|---|---|---|---|---|
| Qwen2.5-14B（同族） | 26.1% | 61.2% | 9.3% | 3.4% |
| Llama-3.1-8B（异族） | 8.3% | **83.6%** | 8.1% | 0.0% |

一致性：**观测一致率 60.2%（298/495），Cohen's kappa = 0.133**（按 Landis-Koch 分级属"一致性偏低"）。

**双向解读（都写进论文）**：

- **稳健的部分**：两位标注器的众数类别**都是 E2**，且 E3 都稳定在 8%-9% 的低位。
  因此"错误主体是代数/计算而非选错定律""E3 占比小"这两条结论在换标注器后依然成立。
- **不稳健的部分**：E1 与 E2 的具体比例高度依赖标注器（同族给 E1 26%、异族只给 8%），
  kappa 0.133 说明细粒度归因基本没有超出随机的共识。原因有二：
  (a) E1/E2 的边界本身模糊 —— 用错定律往往也表现为算错；
  (b) 异族标注器从不输出 N（识别不出判分误报），并把大量 E1 并入 E2。

**因此本项目的 Week 4 结论只做粗粒度表述**（"E2 为主、E3 小、7B/14B 分布无显著差异"），
**不引用精确百分比**。这也是 `paper/table4_errors.tex` 同时给出两位标注器两行的原因。

#### 8.2 把低一致性当成贡献，而不是缺陷

这个 kappa 值不只是"我们自己标注没标好"，它揭示的是**错误分类体系本身的上限**：
"用错定律"与"算错代数"的边界天然模糊 —— 定律选错时算术必然也错，所以输出文本同时兼容两种标签；
反过来，正确推导里的算术滑步又与"选错关系式"在文本上难以区分。只要不测量一致性，
这种依赖标注器的天花板是完全不可见的。

因此本仓库把**双标注数据集**作为一项独立产出发布（`results/error_labels.json` +
`results/error_labels_llama.json`，495 条错误样本 × 2 位标注器）。
它的用途有两个：既是物理推理的错误分析素材，也是 **LLM-as-a-Judge 归因鲁棒性**的评测素材 ——
在这里，标注器之间的**分歧本身**才是信号，而不是噪声。

### 9. 结构性提示能否弥补能力（反事实探针）：否

第 3 节显示表层改写能翻转 24%-34% 的答案。一个自然的工程追问是：
**在提示里强制"先列出已知量与未知量"能不能压住这种不稳定？**

同一 150 题分层子集、同样 2048 token 预算下的配对比较：

| 提示 | 正确率 | 配对 Δ | McNemar p |
|---|---|---|---|
| cot（标准） | 0.240 [0.173, 0.313] | — | — |
| **structured（先列已知/未知）** | 0.260 [0.193, 0.333] | **+2.0pp** | **0.700（不显著）** |

**结论：结构化解题脚手架没有可测量的收益。** 结论方向在两种预算下还翻转了
（1024 token 预算下 structured 为 0.183 vs cot 0.211，但那是被输出长度差异污染的比较：
structured 的推导更长，同样预算下更吃亏；换成匹配预算后差异消失）。
因此本项目不主张"用提示工程弥补物理数值推理的不稳定"。

#### 9.1 附带发现：格式遵循是隐藏的天花板

| 提示 | 输出中含 `ANSWER:` 行 | 无该行时正确率 | 有该行时正确率 | 输出长度 mean / max |
|---|---|---|---|---|
| cot | 29% | 0.252 | 0.209 | 2377 / 4224 字符 |
| structured | 10% | 0.267 | 0.200 | 2373 / 4639 字符 |

最长输出约 1550 token，**远低于 2048 上限，所以不是截断**，而是模型没有遵守
"把最终答案写在 `ANSWER:` 行"的格式要求（它更倾向以 `\boxed{}` 或叙述句收尾）。
两端正确率相近（0.20-0.27），说明判分器的回退抽取没有系统性抬高或压低结果 ——
但这条应当如实披露：**约七成至九成的输出并不符合所要求的输出格式**，
任何依赖严格格式的评测都会被这一项系统性削弱。

> 说明：`scripts/run_repair_probe.py` 实现了"改写破坏集 + 脚手架修复"的探针，
> 但其 structured 分支同样受上述格式问题影响，未达到可解释的测量条件；
> 因此该探针结果**不作为结论发布**，仅保留脚本供后续在匹配预算下复用。

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

# Week 4 错误类型（用 14B 作为标注器给 7B / 14B 的错误样本分类）
python scripts/classify_errors.py --file results/eval_scibench_cot_Qwen2.5-7B-Instruct.json \
    --judge-model Qwen/Qwen2.5-14B-Instruct --backend vllm
python scripts/classify_errors.py --file results/eval_scibench_cot_Qwen2.5-7B-Instruct.json \
    --report --compare-file results/eval_scibench_cot_Qwen2.5-14B-Instruct.json

# Week 4 异族标注器复核（写入独立标签文件，避免覆盖同族标签）
python scripts/classify_errors.py --file results/eval_scibench_cot_Qwen2.5-7B-Instruct.json \
    --judge-model meta-llama/Llama-3.1-8B-Instruct --backend vllm \
    --labels-file results/error_labels_llama.json
python scripts/compare_labels.py --a results/error_labels.json \
    --b results/error_labels_llama.json --json results/error_label_agreement.json

# 跨任务对照：MMLU 物理选择题（488 题，判定走选项字母精确比对）
python scripts/run_eval.py --model Qwen/Qwen2.5-7B-Instruct --benchmark mmlu_physics \
    --mode direct --max-problems 488 --backend vllm --save

# 第二基准：OlympiadBench（351 题，仅 152 题可数值判分）
python scripts/run_eval.py --model Qwen/Qwen2.5-7B-Instruct --benchmark olympiadbench \
    --mode cot --max-problems 351 --backend vllm --save

# 图 2 / 图 3
python scripts/make_figures.py
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
| Qwen2.5-3B / cot | 321 s | 0.99 s | 452 tok/s |
| Qwen2.5-7B / cot | 494 s | 1.53 s | 294 tok/s |
| Qwen2.5-14B / cot | 1388 s | 4.30 s | 105 tok/s |
| Llama-3.1-8B / cot | 635 s | 1.97 s | 229 tok/s |
| R1-Distill-7B / cot (1536 tok) | 926 s | 2.87 s | 536 tok/s |
| Qwen2.5-7B / direct | 15.5 s | 0.05 s | 625 tok/s |
| Qwen2.5-14B / direct | 34.5 s | 0.11 s | 281 tok/s |

结论：

- **本轮全部实验的净 GPU 计算约 2.0 卡时**（12 次评测 run + 4 次敏感性 + 2 轮因果追踪 + 4 次错误标注，
  明细见 `paper/reproducibility_statement.tex` 的 compute 表）。
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
| `results/table1_cot.md` / `.tex`、`table1_direct.md` | 自动生成的主表（SciBench） |
| `results/table_olympiad_cot.md`、`table_mmlu.md` | 第二基准与 MCQ 对照的自动表格 |
| `results/error_labels.json` | Week 4 错误类型标签（Qwen2.5-14B 标注），键为"(模型, 题 id)" |
| `results/error_labels_llama.json` | 同一批样本的异族标注（Llama-3.1-8B），用于一致性复核 |
| `results/error_label_agreement.json` | 两位标注器的一致率、Cohen's kappa 与分布对比 |
| `paper/table1_main.tex` … `table5_cross_task.tex` | 由 `scripts/make_paper_tables.py` 从 results 生成的论文表格（数字不手抄） |
| `results/causal_tracing_Qwen2.5-7B.png` | 图 3：逐层恢复率曲线（均值 ±1 std，标峰值层与平台区） |
| `results/sensitivity_comparison.png` | 图 2：各模型 numeric vs wording 的输出改变比例 |
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
6. **可判分覆盖率**：SciBench 物理 323 题全部可判分；OlympiadBench 仅 43.3%（152/351）可数值判分，
   其正确率在可判分子集上计算，覆盖率必须一并报告，且不可与 SciBench 的数值直接横向比较。
7. **后端差异**：本轮使用 transformers 后端（A800），未安装 vLLM；吞吐比 vLLM 低 5-10 倍。
   结论不受影响，但复现时耗时会有差异。
8. **错误类型标注存在标注器依赖**：Week 4 用 Qwen2.5-14B（同族）与 Llama-3.1-8B（异族）
   两条独立标注链复核同一批 495 条样本，观测一致率 60.2%、**Cohen's kappa = 0.133**。
   两位标注器都认定 E2 为主、E3 占比小，但 E1/E2 的具体比例差异大（26.1% vs 8.3%）。
   因此结论按粗粒度表述（"E2 为主、E3 小、两模型分布无显著差异"），不引用精确百分比。
   E1/E2 边界本身模糊（用错定律往往也表现为算错），这是该分类体系的固有局限。
9. **MMLU 与 SciBench 是不同任务形态**：前者四选一、后者自由作答，两者正确率不能直接相减，
   第 5 节的对照仅用于说明"同一模型在不同作答形式下的表现差异"这一方向性结论。

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
