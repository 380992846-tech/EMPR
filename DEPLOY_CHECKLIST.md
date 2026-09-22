# EMPR 部署与首跑检查清单

目标：在 GPU 机器上一次性把评估管道跑对。按顺序执行，每一步确认通过后再进入下一步。

> 原则：**先在零成本步骤上暴露问题，再执行 GPU 全量实验。**
> 判分自检、变体覆盖率、单元测试都不需要 GPU，可在投入算力之前完成。

> 一键流程：上云后先执行 `bash cloud_first_day.sh`（依赖 -> 测试 -> 体检 -> 小样本基线），
> 全部通过后加 `--full` 执行全量；指定模型用 `--models=A,B`。本文档为手动版，便于排错。

---

## 0. 上云之前：本地零成本检查（约 5 分钟）

```bash
python -m pip install -r requirements.txt      # 最小依赖：pyyaml sympy numpy scipy pytest
python -m pytest tests -q                       # 83 项：判分 / 统计 / 变体 / 数据 / 提示词
python -m src.grading                           # 判分回归用例，期望 23/23
python scripts/download_data.py                 # SciBench 物理 6 科，约 323 题
python scripts/run_eval.py --self-check --benchmark scibench --save
        # 以参考答案作为模型输出跑判分，期望 323/323 一致
python scripts/run_sensitivity.py --coverage --benchmark scibench --max-problems 323 --save
        # 变体覆盖率：numeric 94.4% / wording 74.3% / swap 11.1%
python scripts/preflight.py --benchmark olympiadbench
        # 使用第二基准时：确认参考答案可判分率（当前 152/351 = 43.3%）与变体覆盖率
python scripts/prompt_stats.py && python scripts/gpu_budget.py
        # 实测提示词长度，并输出 GPU 时间预算（见 README 的 GPU 时间预算一节）
```

CPU 环境验证整条链路（可选，产物写入 `results/smoke/`，仅用于管道验证）：

```bash
python scripts/run_eval.py --model Qwen/Qwen2.5-0.5B-Instruct \
    --benchmark scibench --mode direct --max-problems 12 \
    --backend transformers --device cpu --save --out-dir results/smoke
python scripts/check_grading.py --file results/smoke/eval_scibench_direct_*.json
python scripts/make_table.py --dir results/smoke --format md
```

---

## 1. 选机器与准备

- [ ] 算力：**A800 / A100-80G** 可覆盖全部实验（含 70B 的 TP=2）；只跑 8B/14B 时 **4090-24G 足够**。
- [ ] 磁盘 ≥ 100G（Qwen-14B 权重约 30G，70B 约 140G）。
- [ ] 安装 `vllm`（CUDA 版本匹配）为佳；无法安装时使用 `--backend transformers`（速度较慢但可运行）。
- [ ] 代码先提交到仓库，云上 `git clone` 获取，避免手动传输文件。

## 2. 环境

- [ ] `git clone <repo> && cd EMPR`
- [ ] `python -m venv .venv && source .venv/bin/activate`
- [ ] `pip install -r requirements.txt`
- [ ] 验证：`python -c "import torch, transformers; print(torch.cuda.is_available())"`
      - 若 `vllm` 导入报错，先对齐版本，或直接使用 transformers 后端。

## 3. 数据

- [ ] `python scripts/download_data.py`（默认物理 6 科 -> `data/scibench/*.jsonl`）
- [ ] （可选）`python scripts/download_data.py --benchmark olympiadbench`
      - 竞赛物理的参考答案中含大量符号解：**仅 43.3%（152/351）可数值判分**，
        其余标记为 `unscorable_ref` 并从分母中剔除。报告正确率时必须同时报告该覆盖率。
      - 答案字段为列表时必须取标量后再写入：`final_answer` 形如 `['9.26']`，
        若直接 `str()` 会得到 `"['9.26']"`，参考答案将无法被解析。
- [ ] 校验：`python -c "from src.data import load_benchmark, format_composition as f; d=load_benchmark('scibench'); print(len(d)); print(f(d))"`
      - 期望：323 题；`general=71 | thermodynamics=66 | mechanics=56 | math=50 | optics=47 | quantum=33`
      - 子域映射的唯一真源是 `src/data.py: SCIBENCH_SUBJECT_DOMAIN`，不要在别处复制一份。

## 4. 首跑前：判定自测（务必先做）

- [ ] `python -m src.grading`，期望 `自检：23/23 通过`
- [ ] `python -m pytest tests -q`，期望全部通过
- [ ] `python scripts/run_eval.py --self-check --benchmark scibench`，期望 `323/323 (1.0000)`
- [ ] 小样本实跑并抽查判定质量：
      ```bash
      python scripts/run_eval.py --model Qwen/Qwen2.5-7B-Instruct --benchmark scibench \
          --mode cot --max-problems 20 --backend vllm --gpus 1 --save
      python scripts/check_grading.py --file results/eval_scibench_cot_Qwen__Qwen2.5-7B-Instruct.json
      ```
      - 关注 `unparsed_pred`（模型未给出可解析答案）比例：超过 10% 时先检查提示词与判分，再执行全量
      - 关注 `unscorable_ref`：这些题已从分母剔除，论文中须说明覆盖率

## 5. Week 1 正式评估

- [ ] 单模型全量：
      ```bash
      python scripts/run_eval.py --model Qwen/Qwen2.5-14B-Instruct \
          --benchmark scibench --mode cot --max-problems 323 --backend vllm --gpus 1 --save
      ```
- [ ] 多模型：`--models core`（config 中的核心池）或 `--models-all`（core + large）
- [ ] 70B/72B：权重约 140 GB，使用 `--gpus 2`（tensor parallel），或改用 4-bit 量化在单卡运行
- [ ] 生成主表：
      ```bash
      python scripts/make_table.py --dir results --benchmark scibench --mode cot \
          --format md --out results/table1.md
      ```
      - 表格只统计 `scorable` 样本；`n < 30` 的单元格会提示置信区间参考价值有限

## 6. Week 2 / 3 / 4（第 5 步通过后再执行）

- [ ] 敏感性：
      ```bash
      python scripts/run_sensitivity.py --model Qwen/Qwen2.5-7B-Instruct \
          --benchmark scibench --max-problems 323 --mode cot --backend vllm --save
      ```
      - 先查看逐题审计（`results/sensitivity_numeric_*.json` 的 `items`），确认变体题面合理
      - 报告口径：`coverage` / `n_unparsed` / `median_sensitivity` / `p90` / `unchanged_share`；
        wording 变体另报告配对准确率与 McNemar 检验
      - 不报告 `mean_sensitivity`（当基准答案接近 0 时会被单个样本显著抬高）
- [ ] 因果追踪（逐层前向，题数控制在 8-20）：
      ```bash
      python scripts/run_causal_tracing.py --model Qwen/Qwen2.5-7B-Instruct \
          --benchmark scibench --max-problems 12 --mode direct --save --plot
      ```
      - 仅"clean 答对且 corrupt 确实掉分"的题目计入恢复率；脚本会报告 `n_valid` 与跳过原因计数
      - 该实验需要读取隐藏态，只能使用 transformers 后端（不支持 vllm）
- [ ] 错误分类：
      ```bash
      python scripts/classify_errors.py --file results/eval_scibench_cot_Qwen__Qwen2.5-7B-Instruct.json --report
      ```

## 7. 常见问题排查

| 现象 | 原因 / 处理 |
|---|---|
| 批量生成结果重复或错乱 | tokenizer 右填充：decoder-only 批量必须左填充（`src/llm.py` 已设 `padding_side="left"`） |
| instruct 模型输出像是在续写指令 | 未走 chat template。默认 `chat=auto` 会使用 tokenizer 自带模板；确认结果 json 中 `meta.chat_template=true` |
| `ModuleNotFoundError: vllm` | 使用 `--backend transformers`，或安装与 CUDA 匹配的 vllm |
| HF 下载超时 | 使用本地 `download_data.py`；或 `export HF_ENDPOINT=https://hf-mirror.com` |
| `unparsed_pred` 比例偏高 | 先检查 `src/prompts.py` 是否要求 `ANSWER:` 行，再检查 `src/grading.py` |
| `unscorable_ref` 较多 | 参考答案为符号或表达式（OlympiadBench 常见）：这些题不计入分母，报告中说明覆盖率 |
| OOM | 换更小模型 / 降低 `--max-new-tokens` / 使用 `--gpus 2` / 量化 |
| 置信区间过宽 | 题量不足；增大 `--max-problems`（注意默认按科目轮转抽样，不是取前 N 题） |
| 两次结果不一致 | 对比结果 json 的 `meta.git_commit` / `prompt_fingerprint` / `max_new_tokens` 是否变化 |

## 8. 记录与提交

- [ ] 每次运行后提交 `results/*.json`（论文数字的唯一来源）。
- [ ] 日志目录 `results/logs/` 不纳入版本控制。
- [ ] 使用 `results/` 中的实际输出填充 `paper/` 下的表格与图。
- [ ] 更新 README 的进度表：已完成的模型与基准、样本量、置信区间。
