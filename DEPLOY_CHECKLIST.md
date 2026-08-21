# EMPR 云上部署 + 首跑检查清单

目标：在租的云 GPU 上，把 Week 1 评估管道完整跑通，并确认对错判定可靠。
按顺序勾选。**每条都先确认能过，再进下一条**——首跑最重要是"确定性 + 判定可靠"，不是跑得多。

---

## 0. 选机器 / 提前准备（租机前）
- [ ] 确定算力：**A100-80G** 最佳（能跑 70B 推理 + 因果追踪）；若只跑 8B/14B，**4090-24G 够用**。
- [ ] 确认机器**盘 ≥ 100G**（模型权重下载占空间，Qwen-14B ~30G、70B ~140G）。
- [ ] 确认能装 `vllm`（CUDA 版本匹配）。装不了就退回 transformers（`--backend transformers`），慢但能跑。
- [ ] 本地先把骨架提交到 Git 仓库（`git init && add -A && commit`），云上 `git clone` 拉下来，避免手动传文件。

---

## 1. 环境
- [ ] 云上 `git clone <你的仓库>` 并 `cd EMPR`
- [ ] `python -m venv .venv && source .venv/bin/activate`（或 conda）
- [ ] `pip install -r requirements.txt`
- [ ] 验证关键库：`python -c "import torch, transformers, vllm; print('ok')"`
      - 若 vllm 报错，先 `pip install vllm` 对版本，或改用 transformers。

---

## 2. 数据（SciBench / OlympiadBench）
- [ ] 下载物理数据：`python scripts/download_data.py`（默认 6 个物理科目 matter/class/fund/diff/quan/thermo）
      - 输出到 `data/scibench/*.jsonl`
- [ ] （可选）补竞赛物理 + 官方解题：`python scripts/download_data.py --benchmark olympiadbench`
      - 输出到 `data/olympiadbench/*.jsonl`，带 `solution` 解题步骤
- [ ] 校验加载：`python -c "from src.data import load_benchmark; d=load_benchmark('scibench'); print(len(d)); print(d[0]['answer'])"`
      - 期望：打印题目数（六科合计约 300+），且能看到一道题的答案。
- [ ] 若 HF 也装了、想用 HF 数据：改 `src/data.py` 的 `HF_CANDIDATES` 确认名字；不装就全程走本地（推荐）。

---

## 3. 首跑前 · 判定自测（最重要，别跳过）
- [ ] `python -m src.grading` —— 期望 7/7 全部 `[OK]`。
- [ ] 手动抽查几道题的模型输出，确认 `extract_number` 能稳定抠出最终答案：
      - 先跑极小样例：`python scripts/run_eval.py --model Qwen/Qwen2.5-7B-Instruct --benchmark scibench --cot --max-problems 10 --backend transformers --save`
      - 用判定抽查脚本：`python scripts/check_grading.py --file results/eval_*.json`
      - **检查重点**：`pred_val=null` 占比高不高？`ref` 有没有很多是符号/表达式导致数值判定失效？
      - 若命中率高，继续；命中率低，**先调 `src/grading.py` 再全量跑**（不要拿坏判定跑几千题）。

---

## 4. Week 1 正式评估
- [ ] 换 vllm 加速：`python scripts/run_eval.py --model Qwen/Qwen2.5-14B-Instruct --benchmark scibench --cot --max-problems 200 --backend vllm --gpus 1 --save`
- [ ] 多模型：加 `--model` 换成其他，或 `--models Qwen/...,meta-llama/Llama-3.1-8B-Instruct,...`。
      - 70B 需大显存：`--gpus 1`（A100-80G）或 `--gpus 2`（多卡 tensor parallel）。
- [ ] 产出 `results/eval_*.json`，确认每行带 `overall_acc` 和 95% CI。
- [ ] **生成主表**：整理成 模型 × 子域 的表格（可手动或小脚本），数字都要带 CI。

---

## 5. Week 2/3（可选，第 4 步通过后再碰）
- [ ] 敏感性：`python scripts/run_sensitivity.py --model Qwen/Qwen2.5-7B-Instruct --benchmark scibench --max-problems 40 --backend vllm --save`
      - 先抽查 `wording`/`swap` 生成的变体题面是否合理（`results/sensitivity_*.json`）。
- [ ] 因果追踪（慢，题数给小）：`python scripts/run_causal_tracing.py --model Qwen/Qwen2.5-7B-Instruct --benchmark scibench --max-problems 8 --save --plot`

---

## 6. 常见坑排查
| 现象 | 处理 |
|---|---|
| `ModuleNotFoundError: vllm` | 装不了就用 `--backend transformers`；或 `pip install vllm` 换 CUDA 匹配版本 |
| HF 下载超时/报错 | 走本地：`scripts/download_data.py` 落 `data/scibench/`，改走本地加载 |
| 模型下载慢/断 | 用国内镜像（HF_ENDPOINT=https://hf-mirror.com 或 ModelScope），或预下载 |
| 大量 `pred_val=null` | 调 `src/grading.py` 的 `extract_number`，别硬跑 |
| OOM | 换小模型 / 减小 `--max-new-tokens` / `--gpus` 多卡 / 用 8bit 量化 |
| 结果 CI 太宽 | 题数太少（min(200)），加大 `--max-problems` |

---

## 7. 记录 & 提交
- [ ] 每次跑完把 `results/*.json` 提交进仓库。
- [ ] 用 `results` 里的真实数字开始填论文表格。
- [ ] 每周结束更新 README 的进度一节，写清哪些模型/基准已跑、CI 是多少。
