# EMPR — Emergent Physical Reasoning in Open LLMs

**目标**：用真实实验支撑开放 LLM 在多步物理推理上的
(1) 性能规律、(2) 数值敏感性（区分记忆 vs 计算）、(3) 因果追踪（计算定位到哪几层）。
最终诚实框定为 "sophisticated pattern matching，不是真物理理解"。

**现状**：本仓库已包含 Week 1（评估管道）+ Week 2（敏感性，三种变体）+ Week 3（因果追踪，逐层恢复率曲线）+ Week 4（错误分类）的可运行骨架。


---

## 配置源（config.yaml）

- 模型池分 **core（默认，4090 友好）** 和 **large（可选，需 A100）** 两组。
- 默认基准/模式/模型在 `config.yaml → defaults`；提示词模板在 `src/prompts.py`（**不在 config.yaml**）。
- 改模型/超参优先改 `config.yaml`，命令行 `--flag` 可覆盖。

---

## 快速开始

```bash
# 1. 建环境（在云 GPU 机器上）
pip install -r requirements.txt

# 2. 下载基准（直接从源拉取，无需手动下载）：
#    python scripts/download_data.py                       # SciBench 物理 6 科 ~323 题（GitHub）
#    python scripts/download_data.py --benchmark olympiadbench   # 竞赛物理（HF，含官方解题）
#    python scripts/download_data.py --benchmark all       # 两者都下
#    落地 data/scibench/*.jsonl、data/olympiadbench/*.jsonl，src/data.py 自动扫描。

# 3. 跑评估（示例：Qwen2.5-14B，零样本 + CoT，前 100 题）
python scripts/run_eval.py --model Qwen/Qwen2.5-14B-Instruct \
    --benchmark scibench --cot --max-problems 100 \
    --backend vllm --gpus 1 --save

# 4. 判定质量抽查（重要！先确认对错判定可靠再全量跑）
python scripts/check_grading.py --file results/eval_*.json

# 5. 敏感性（三种变体：数值/措辞/物理量互换）
python scripts/run_sensitivity.py --model Qwen/Qwen2.5-7B-Instruct \
    --benchmark scibench --max-problems 40 --backend vllm --save

# 6. 因果追踪（逐层恢复率；题数给小，8-20）
python scripts/run_causal_tracing.py --model Qwen/Qwen2.5-7B-Instruct \
    --benchmark scibench --max-problems 8 --save --plot

# 7. 自动生成论文主表（模型 × 子域 × 正确率+CI）
python scripts/make_table.py --dir results --benchmark scibench --mode cot --format md
python scripts/make_table.py --dir results --format md --out results/table.md

# 8. Week 4 错误类型分类（E1 定律误用 / E2 代数 / E3 约束违背）
python scripts/classify_errors.py --file results/eval_*.json           # 手动标注
python scripts/classify_errors.py --file results/eval_*.json --report   # 统计报告
python scripts/classify_errors.py --file A.json --report --compare-file B.json  # 两模型卡方

# 输出结果表（含 bootstrap 95% CI）到 results/
```

> 部署 + 首跑步骤见 [DEPLOY_CHECKLIST.md](DEPLOY_CHECKLIST.md)。

---

## 目录结构

```
EMPR/
  config.yaml              # 默认模型/基准/超参
  requirements.txt
  data/                    # 手动放 JSON；HF 不可用时的回退（src/data.py 会扫 data/<name>/）
  src/
    data.py                # 基准加载：多 HF 候选 + 本地目录回退，字段防御式提取
    grading.py             # 答案提取 + 对错判定（sympy/纯 Python，已不硬依赖 sympy）
    stats.py               # bootstrap 95% CI、McNemar、卡方
    llm.py                 # vllm / transformers 推理封装
    prompts.py             # 提示词模板（zero-shot CoT / direct）
  scripts/
    run_eval.py            # [可运行] Week 1 评估主表（模型 × 子域正确率 + CI）
    check_grading.py       # [可运行] 判定质量抽查（null 率 + 样例人工核对）
    make_table.py          # [可运行] 从 results/ 生成论文主表（md/tex/csv）
    classify_errors.py     # [可运行] Week 4 错误类型 E1/E2/E3 分类 + 卡方
    run_sensitivity.py     # [可运行] Week 2 敏感性：numeric / wording / swap 三变体
    run_causal_tracing.py  # [可运行] Week 3 因果追踪：逐层恢复率曲线 + 热力图
    download_data.py       # 数据下载：SciBench(GitHub) + OlympiadBench(HF)
  results/                 # 输出（json + 表格 + png）
```

---

## 设计约定

- **确定性**：greedy（temperature=0）保证可复现；随机种子统一 42。
- **对错判定**：默认用符号/数值比对（sympy，相对误差 ≤5%，近零用绝对差）；LLM-judge 仅做辅助。
- **可复现**：所有结果带 bootstrap 95% CI（10,000 次）。

---

## 相关项目

- **量子纠错解码（Ising Decoding）**：<https://github.com/380992846-tech/Ising-Decoding>

