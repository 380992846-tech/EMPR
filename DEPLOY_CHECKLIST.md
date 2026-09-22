# EMPR 部署 + 首跑检查清单

目标：在租的云 GPU 上，把评估管道**一次跑对**，并在花钱全量推理之前确认"数据可加载 +
判定可靠"。按顺序勾选，每条先确认能过再进下一条。

> 原则：**先在本地把零成本检查跑完（判分自检 + 变体覆盖率 + 单元测试），再上云烧卡。**
> 这三步不需要 GPU，能挡掉绝大多数"跑完 300 题才发现判定是错的"事故。

> **懒人路径**：上云后先 `bash cloud_first_day.sh`（依赖 → 测试 → 一键体检 → 小样本基线），
> 全部通过再加 `--full` 跑全量；指定模型用 `--models=A,B`。下面是手动版清单，适合排错。

---

## 0. 上云之前：本地零成本体检（强烈建议，5 分钟）

```bash
python -m pip install -r requirements.txt      # 或只用最小依赖：pyyaml sympy numpy scipy pytest
python -m pytest tests -q                       # 74 项：判分/统计/变体/数据/提示词
python -m src.grading                           # 判分回归用例（应 23/23 通过）
python scripts/download_data.py                 # 拉 SciBench 物理 6 科（~323 题）
python scripts/run_eval.py --self-check --benchmark scibench --save
        # 用参考答案当预测跑判分：必须 323/323 一致，否则先修 src/grading.py
python scripts/run_sensitivity.py --coverage --benchmark scibench --max-problems 323 --save
        # 变体覆盖率：numeric 94.4% / wording 74.3% / swap 11.1%（swap 低是数据固有，如实写）
```

没 GPU 也想验证"模型→判定→表格"整条链路（可选，结果不可当论文数字）：

```bash
python scripts/run_eval.py --model Qwen/Qwen2.5-0.5B-Instruct \
    --benchmark scibench --mode direct --max-problems 12 \
    --backend transformers --device cpu --save
python scripts/make_table.py --dir results
```

---

## 1. 选机器 / 提前准备（租机前）
- [ ] 算力：**A100-80G** 最佳（能跑 70B + 因果追踪）；只跑 8B/14B 则 **4090-24G 够用**。
- [ ] 磁盘 ≥ 100G（Qwen-14B 权重 ~30G，70B ~140G）。
- [ ] 能装 `vllm`（CUDA 匹配）最好；装不上退回 `--backend transformers`（慢但能跑）。
- [ ] 代码先提交推仓库，云上 `git clone`，别手动传文件。

## 2. 环境
- [ ] `git clone <repo> && cd EMPR`
- [ ] `python -m venv .venv && source .venv/bin/activate`
- [ ] `pip install -r requirements.txt`
- [ ] 验证：`python -c "import torch, transformers; print(torch.cuda.is_available())"`
      - `vllm` 报错就先对版本，或直接用 transformers。

## 3. 数据
- [ ] `python scripts/download_data.py`（默认物理 6 科 → `data/scibench/*.jsonl`）
- [ ] （可选）`python scripts/download_data.py --benchmark olympiadbench`
- [ ] 校验：`python -c "from src.data import load_benchmark, format_composition as f; d=load_benchmark('scibench'); print(len(d)); print(f(d))"`
      - 期望：323 题；`general=71 | thermodynamics=66 | mechanics=56 | math=50 | optics=47 | quantum=33`
      - 子域映射的唯一真源是 `src/data.py: SCIBENCH_SUBJECT_DOMAIN`（不要另写一套）

## 4. 首跑前 · 判定自测（最重要，别跳过）
- [ ] `python -m src.grading` → 期望 `自检：23/23 通过`
- [ ] `python -m pytest tests -q` → 期望全绿
- [ ] `python scripts/run_eval.py --self-check --benchmark scibench` → 期望 `323/323 (1.0000)`
- [ ] 小样本真跑一段，抽查判定质量：
      ```bash
      python scripts/run_eval.py --model Qwen/Qwen2.5-7B-Instruct --benchmark scibench \
          --cot --max-problems 20 --backend vllm --gpus 1 --save
      python scripts/check_grading.py --file results/eval_scibench_cot_Qwen__Qwen2.5-7B-Instruct.json
      ```
      - 看 `unparsed_pred`（模型输出抠不出数值）比例：>10% 就先查提示词/判分，别全量跑
      - 看 `unscorable_ref`：这些题已自动剔除出分母，论文里要写明覆盖率

## 5. Week 1 正式评估
- [ ] 单模型全量：
      ```bash
      python scripts/run_eval.py --model Qwen/Qwen2.5-14B-Instruct \
          --benchmark scibench --cot --max-problems 323 --backend vllm --gpus 1 --save
      ```
- [ ] 多模型：`--models a,b,c` 或 `--models-all`（config 的 core+large）
- [ ] 70B：`--gpus 1`（A100-80G）或 `--gpus 2`（tensor parallel）
- [ ] 生成主表：`python scripts/make_table.py --dir results --benchmark scibench --mode cot --format md --out results/table1.md`
      - 表格只统计 `scorable` 样本；n<30 的格子会提示 CI 参考价值低

## 6. Week 2 / 3 / 4（第 5 步通过后再碰）
- [ ] 敏感性：`python scripts/run_sensitivity.py --model Qwen/Qwen2.5-7B-Instruct --benchmark scibench --max-problems 60 --backend vllm --save`
      - 先看逐题审计（`results/sensitivity_numeric_*.json` 的 `items`）确认变体题面合理
      - 报告口径：`coverage` / `n_unparsed` / `unchanged_share` / （wording 才有）配对准确率+McNemar
- [ ] 因果追踪（慢，题数给小）：
      ```bash
      python scripts/run_causal_tracing.py --model Qwen/Qwen2.5-7B-Instruct \
          --benchmark scibench --max-problems 12 --mode direct --save --plot
      ```
      - 只有"clean 答对且 corrupt 确实掉分"的题才计入；脚本会报跳过数量
- [ ] 错误分类：`python scripts/classify_errors.py --file results/eval_*.json --report`

## 7. 常见坑排查
| 现象 | 原因 / 处理 |
|---|---|
| 批量生成结果是重复/乱码 | **tokenizer 右填充**：decoder-only 批量必须左填充（`src/llm.py` 已设 `padding_side="left"`）。自己改推理代码时注意 |
| instruct 模型输出像"续写指令" | 没走 chat template。默认 `chat=auto` 会用 tokenizer 自带模板；确认结果 json 里 `meta.chat_template=true` |
| `ModuleNotFoundError: vllm` | `--backend transformers`，或装匹配 CUDA 的 vllm |
| HF 下载超时 | 走本地 `download_data.py`；或 `export HF_ENDPOINT=https://hf-mirror.com` |
| 大量 `unparsed_pred` | 先查 `src/prompts.py` 是否仍要求 `ANSWER:` 行，再查 `src/grading.py` |
| `unscorable_ref` 很多 | 参考答案是符号/表达式（OlympiadBench 常见）：这些题不进分母，论文里报覆盖率 |
| OOM | 换小模型 / 降 `--max-new-tokens` / `--gpus` 多卡 / 量化 |
| CI 太宽 | 题数太少；加大 `--max-problems`（注意默认按科目轮转抽样，不是取前 N 题） |
| 结果对不上上一次 | 看结果 json 的 `meta.git_commit` / `prompt_fingerprint` / `max_new_tokens` 是否变了 |

## 8. 记录 & 提交
- [ ] 每次跑完提交 `results/*.json`（论文数字的唯一来源，已不再被 .gitignore 挡住）。
- [ ] 日志（`results/logs/*.log`）不入库。
- [ ] 用 `results/` 里的真实数字填 `paper/`，不要沿用任何早期草案数值。
- [ ] 每周更新 README 的「进度」表：哪些模型/基准跑完了、n 是多少、CI 是多少。
