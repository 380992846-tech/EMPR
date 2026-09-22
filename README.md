# EMPR — Emergent Physical Reasoning in Open LLMs

用真实实验检验开放权重 LLM 的**多步物理推理**：它到底在算，还是在做精细的模式匹配？

三个可证伪的问题，对应三周实验：

| 周 | 问题 | 手段 |
|---|---|---|
| Week 1 | 性能规律是什么？ | SciBench 物理全量 × 多模型 × 子域正确率 + 95% CI |
| Week 2 | 它在**算**还是在**记**？ | 三个变体探针：改数值 / 改措辞 / 互换物理量 |
| Week 3 | 若是算，**算在哪几层**？ | 逐层恢复率（causal tracing），定位关键层区间 |

最终框定是**诚实的**：预期结论是"sophisticated pattern matching，不是真物理理解"——
如果数据不支持，就照数据写。

---

## 进度（只写已经真跑出来、能在 `results/` 找到证据的）

| 项目 | 状态 | 证据 / 数字 |
|---|---|---|
| 判分有效性（参考答案自判） | ✅ 已跑 | `results/selfcheck_scibench.json`：**323/323 = 1.0000**，0 条参考答案不可解析 |
| 判分回归用例 | ✅ 已跑 | `python -m src.grading` → 23/23 |
| 一键体检 | ✅ 已跑 | `results/preflight_scibench.json`：PASS（依赖/数据/判分/覆盖率四项全过） |
| 单元 + 集成测试 | ✅ 已跑 | `pytest tests -q` → **79 passed**（含微型模型上的因果追踪集成测试） |
| 变体覆盖率（Week 2 前置） | ✅ 已跑 | `results/sensitivity_coverage_scibench.json`：numeric 94.4%、wording 74.3%、swap 11.1% |
| 链路端到端（0.5B · CPU） | ✅ 已跑 | `results/eval_scibench_{direct,cot}_Qwen__Qwen2.5-0.5B-Instruct.json` 等，见下方「冒烟」 |
| Week 1 主表（真模型） | ⏳ 待云 GPU | 本机无 CUDA（torch 2.11+cpu）；冒烟只证明链路可跑通 |
| Week 2 敏感性（真模型） | ⏳ 待云 GPU | 探针与审计口径已就绪并已在 0.5B 上跑通 |
| Week 3 因果追踪（真模型） | ⏳ 待云 GPU | hook/恢复/对齐已在微型模型与 0.5B 上验证 |
| Week 4 错误分类 | ⏳ 待人工标注 | 工具就绪（`scripts/classify_errors.py`） |

### 本机 CPU 冒烟（**仅验证链路，不是论文数字**）

本机无 GPU，所以只能跑 0.5B 量级模型；唯一目的是证明
「提示词 → 生成 → 提取 → 判定 → CI → 表格」整条链路真的能跑通，避免上云才发现管道坏了。

| 跑的东西 | 命令要点 | 结果 |
|---|---|---|
| Week 1 direct | `--model Qwen/Qwen2.5-0.5B-Instruct --mode direct --max-problems 12 --device cpu` | overall 0.083 [0.000, 0.250]，`unparsed_pred` 1/12 |
| Week 1 cot | 同上，`--mode cot --max-problems 8 --max-new-tokens 192` | overall 0.000 [0.000, 0.000]（0.5B 做不了 CoT，符合预期） |
| Week 2 敏感性 | `run_sensitivity.py --max-problems 8 --mode direct`（8 题基准） | numeric：覆盖 7/8、`unchanged_share` 0.143、median 2.20 / p90 26.1；wording：覆盖 8/8、`unchanged_share` 0.625、配对准确率 0.125→0.125；swap：覆盖 1/8 |
| Week 3 因果追踪 | `run_causal_tracing.py --max-problems 6 --mode direct --layer-step 4` | 6 题里只 1 题有效（clean 答对且 corrupt 掉分）：clean 0.96 → corrupt 0.00，峰值恢复 **+0.43 @ layer 12/24**；跳过 clean 答错 5 题 |
| 表格 | `make_table.py --dir results --format md --out results/table_smoke.md` | 仅 1 个模型、每格 n=2，CI 无参考价值——文件名带 `smoke` 以免误用 |

> 冒烟里最有价值的两个副产品：
> 1. **因果追踪的筛选规则在工作**——6 题里 5 题因"clean 都没答对"被跳过；0.5B 本来就没在算，
>    恢复率无从谈起。真模型（7B/14B）有效题数会高得多，**有效题数必须写进论文**。
> 2. **`mean sensitivity` 不可用**：wording 的 mean=2983.8 而 median=0.000——
>    因为敏感度是 `|Δ|/(|base|+ε)`，base≈0 时单个样本就能把它带飞。论文请报 median / p90
>    与 `unchanged_share`（脚本已同时输出这三者，并在打印时标注这一点）。

> **论文数字必须来自云 GPU 的 vllm 全量跑**，且结果 json 里的 `meta.git_commit` /
> `prompt_fingerprint` 要一致。`results/table_smoke.md` 不得进论文。

---

## 数据构成（SciBench 物理 6 科，323 题）

| 子域 | 科目 | n | 占比 |
|---|---|---|---|
| general | fund（综合基础题） | 71 | 22% |
| thermodynamics | thermo | 66 | 20% |
| mechanics | class | 56 | 17% |
| math | diff（数学工具题） | 50 | 15% |
| optics | matter（波动与近代物理） | 47 | 15% |
| quantum | quan | 33 | 10% |

子域映射的**唯一真源**是 `src/data.py: SCIBENCH_SUBJECT_DOMAIN`（`download_data.py` 与加载路径都从它取）。
早期版本两处各写一套，把 quan/diff 落成 `general`，导致 154/323 题挤进同一个桶、子域主表失真——
现在 `tests/test_data_prompts.py` 里有测试钉住这一点。

---

## 快速开始

```bash
pip install -r requirements.txt
python -m pytest tests -q                      # 79 项，几秒钟
python scripts/download_data.py                # SciBench 物理 6 科 → data/scibench/*.jsonl
```

### 第 0 步（免费，必做）：先确认判定可靠，再花钱跑模型

```bash
python scripts/preflight.py --benchmark scibench
        # 一条命令跑完下面四项，结论写进 results/preflight_scibench.json；不过就以非零码退出
python -m src.grading                          # 判分回归：期望 23/23
python scripts/run_eval.py --self-check --benchmark scibench --save
        # 拿参考答案当"模型输出"跑判分：必须 323/323 一致；任何不一致 = 判分 bug
python scripts/run_sensitivity.py --coverage --benchmark scibench --max-problems 323 --save
        # 三个变体的覆盖率（干跑，0 模型成本）
```

### 没 GPU 也想验证链路（CPU，0.5B 冒烟）

```bash
python scripts/run_eval.py --model Qwen/Qwen2.5-0.5B-Instruct \
    --benchmark scibench --mode direct --max-problems 12 \
    --backend transformers --device cpu --max-new-tokens 64 --save
python scripts/check_grading.py --file results/eval_scibench_direct_Qwen__Qwen2.5-0.5B-Instruct.json
python scripts/make_table.py --dir results --format md
```

### 云 GPU 正式评估

```bash
# Week 1：主表（模型 × 子域 正确率 + CI）
python scripts/run_eval.py --model Qwen/Qwen2.5-14B-Instruct \
    --benchmark scibench --cot --max-problems 323 --backend vllm --gpus 1 --save
python scripts/check_grading.py --file results/eval_scibench_cot_*.json   # 判定质量抽查
python scripts/make_table.py --dir results --benchmark scibench --mode cot --format md --out results/table1.md

# Week 2：敏感性（记忆 vs 计算）
python scripts/run_sensitivity.py --model Qwen/Qwen2.5-7B-Instruct \
    --benchmark scibench --max-problems 60 --mode cot --backend vllm --save

# Week 3：因果追踪（逐层恢复率；题数给小）
python scripts/run_causal_tracing.py --model Qwen/Qwen2.5-7B-Instruct \
    --benchmark scibench --max-problems 12 --mode direct --save --plot

# Week 4：错误类型（E1 定律误用 / E2 代数 / E3 约束违背）
python scripts/classify_errors.py --file results/eval_scibench_cot_*.json --report
python scripts/classify_errors.py --file A.json --report --compare-file B.json   # 两模型卡方
```

部署与首跑细节见 [DEPLOY_CHECKLIST.md](DEPLOY_CHECKLIST.md)；论文表格对应关系见 [paper/TABLES.md](paper/TABLES.md)。
上云后想要"一条龙"：`bash cloud_first_day.sh`（体检 + 小样本），确认后 `bash cloud_first_day.sh --full` 跑全量。

---

## 目录结构

```
EMPR/
  config.yaml              # 模型池 / 基准 / 超参（唯一的配置源；提示词在 src/prompts.py）
  requirements.txt
  DEPLOY_CHECKLIST.md      # 上云部署 + 首跑检查清单
  data/                    # download_data.py 落盘处（不进库，可重建）
  src/
    data.py                # 基准加载 + 子域映射唯一真源 + 分层抽样
    grading.py             # 答案提取与判定（严格解析参考答案 / 宽松解析模型输出）
    stats.py               # bootstrap CI、McNemar、卡方（纯 Python 可复现）
    llm.py                 # vllm / transformers 封装（左填充 + chat template + 进度）
    prompts.py             # 提示词模板（所有模式都要求 ANSWER: 行）
    meta.py                # 运行元数据：commit / 环境 / 提示词指纹
  scripts/
    preflight.py           # 一键体检：依赖/数据/判分/覆盖率，PASS 才建议上云
    run_eval.py            # Week 1 主入口（含 --self-check 判分自检）
    run_sensitivity.py     # Week 2 三变体（含 --coverage 干跑、逐题审计）
    run_causal_tracing.py  # Week 3 逐层恢复率 + 曲线图
    classify_errors.py     # Week 4 错误分类 + 卡方
    check_grading.py       # 判定质量抽查（status 分布 + 分层 + 抽样）
    make_table.py          # results/ → 论文表格（md/tex/csv）
    download_data.py       # SciBench(GitHub) + OlympiadBench(HF)
  tests/                   # pytest：判分 / 统计 / 变体 / 数据 / 提示词
  results/                 # 结果 json（进库）+ logs/（不进库）
  paper/                   # 表格骨架（TABLES.md / *.tex）
```

---

## 结果文件怎么读

| 文件 | 内容 | 关键字段 |
|---|---|---|
| `results/selfcheck_scibench.json` | 判分自检 | `judge_self_consistency`、`n_inconsistent` |
| `results/eval_<bench>_<mode>_<model>.json` | 主结果 | `overall_acc`、`per_domain_acc`、`n_scorable`、`n_unscored`、`n_unparsed_pred`、`detail[]`、`meta` |
| `results/sensitivity_coverage_<bench>.json` | 变体覆盖率（干跑） | `coverage.{numeric,wording,swap}` |
| `results/sensitivity_<variant>_<model>.json` | 单变体结果 | `unchanged_share`、`median/p90/mean_sensitivity`、`items[]`（逐题审计） |
| `results/sensitivity_summary_<model>.json` | 三变体汇总 | `variants.*`；wording 额外有配对准确率 + McNemar |
| `results/causal_tracing_<bench>_<model>.json` | 逐层恢复率 | `average_curve`、`problems[*].curve`、`n_valid` / 跳过计数 |

每个 json 都带 `meta`：`git_commit`（脏工作区会标 `+dirty`）、`env`；跑模型的还带
`prompt_fingerprint`、`max_new_tokens`、`chat_template`。**两份结果对不上时，先比 `meta`。**
所有结果文件都是严格合法 JSON（非有限值写 `null`，不会出现裸 `NaN`）。

---

## 设计约定（以及为什么）

**判定（`src/grading.py`）**
- **参考答案严进、模型输出宽出**：`parse_reference` 要求整串就是个数值（可带单位/百分号/分数/科学计数法），
  否则该题标记 `unscorable_ref` 并**剔除出分母**；模型输出则用宽松的 `extract_number`
  （`ANSWER:` 行 → `\boxed{}` → "answer is X" → 末行数 → 全文末数）。
  理由：宽松解析会把 `x^2 + 2x` 读成 `2`、把 `\frac{1}{2}mv^2` 读成 `1`，
  再拿模型答案跟这个假数字比——等于悄悄污染分母，而错误完全看不出来。
- 判定只分三类口径：`correct` / `incorrect`（含模型没给出可解析答案）/ `unscorable_ref`（剔除）。
  论文里必须同时报 `n_unscored` 与 `n_unparsed_pred`，不能只报一个正确率。
- 容差：相对误差 ≤5%，近零用绝对差 0.01。

**提示词（`src/prompts.py`）**
- **所有模式都必须要求 `ANSWER:` 行**。早期 `direct` 模板只写 "Give only the final numeric answer"，
  判定只能靠"全文最后一个数"兜底——而 Week 2 默认跑 direct，等于让一半实验的判定精度白掉一档。
- 选择题（题干含 ≥2 个 `A. xxx` 选项）自动改用要求"选项字母"的模板。

**推理（`src/llm.py`）**
- **批量生成必须左填充**：decoder-only 右填充会让模型在 pad 之后续写，transformers 只打 warning 不报错。
  这是静默产生垃圾结果的经典坑，代码注释与测试都钉住了。
- **instruct 模型走 chat template**（`chat=auto`，有模板就用），否则模型会去"续写指令"而不是答题。
- 输出条数必须等于题数，否则直接报错——`zip(rows, outputs)` 错位会把 A 题的输出判到 B 题答案上。

**数据（`src/data.py`）**
- 子域映射唯一真源；未知科目原样返回，不静默并桶。
- **本地数据优先**：`data/scibench/` 存在就直接用，不再先试 HF。HF 镜像的名字/字段/切分随时会变，
  同一份代码在不同机器上可能悄悄加载到不同数据集（题目数、子域分布、答案格式全变，结果不可比）。
  要强制走 HF：`EMPR_PREFER_HF=1`。
- `--max-problems N` **默认按科目轮转抽样**（不是取前 N 题）。直接切片只会拿到字母序最前的科目，
  实测 `--max-problems 40` 会得到清一色 mechanics，然后被写成"在 SciBench 上 40 题"。

**统计（`src/stats.py`）**
- bootstrap CI 一律用纯 Python（`random.Random` + 顺序统计量），保证"装没装 numpy"不影响 CI；
  早期 numpy/纯 Python 双路径会让同一 seed 在不同机器上给出不同 CI。
- 卡方是未校正 Pearson（scipy 对 2×2 默认做 Yates 校正，对比时用 `correction=False`）。
- 统一 seed=42、贪心解码（temperature=0）。

**运行记录（`src/meta.py`）**
- 每个结果 json 都带 commit / 环境 / 提示词指纹，便于回溯"这批数字怎么来的"。

---

## 诚实标注与已知限制（论文 Limitations 可直接用）

1. **swap 覆盖率只有 11.1%（36/323）**：很多题的所有数值同属一个物理量、或没有可识别单位，
   构不出"互换两个不同物理量"的变体。这是数据固有属性，如实报告，不硬凑样本。
2. `wording` 变体是**规则改写**（74.3% 覆盖），只换框架词/同义短语、绝不碰数字；
   更强的做法是用一致改写模型，但那会把"改写质量"引入变量。
3. `numeric` 变体改的是题干数值，**其正确答案随物理关系变化且不可直接求解**，
   因此该变体只报 `unchanged_share` / 敏感度分布，**不报"变体准确率"**；
   只有 `wording`（物理与数据完全不变）能报配对准确率 + McNemar。
4. 因果追踪**只对开放权重模型**做隐藏态干预；闭源 API 不可复现。
   且只有"clean 答对且 corrupt 确实掉分"的题才计入恢复率（脚本会报跳过数量）。
5. 本机无 CUDA：真模型数字必须来自云 GPU；README 里的 CPU 冒烟只证明链路能跑通。
6. 判定基于数值比对，**符号型参考答案**（OlympiadBench 常见）会被剔除出分母，
   那些基准上的正确率是在"可判分子集"上算的，覆盖率必须一并报告。

---

## 相关项目

- **量子纠错解码（Ising Decoding）**：<https://github.com/380992846-tech/Ising-Decoding>
