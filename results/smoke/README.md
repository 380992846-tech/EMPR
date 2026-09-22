# 管道验证产物（pipeline validation）

本目录是**管道验证**（pipeline validation）的原始输出，用于证明
"提示词 -> 生成 -> 答案提取 -> 判定 -> 置信区间 -> 表格"整条链路可以端到端运行。
生成环境为 CPU，模型为 `Qwen/Qwen2.5-0.5B-Instruct`（0.5B）。

**这些数字不用于论文或任何结论**，原因：

1. 0.5B 模型不具备多步物理推理能力，正确率接近随机（direct 0.083、cot 0.000、MMLU 0.167）；
2. 题量很小（8-12 题），95% 置信区间宽到没有解释力；
3. 解码参数为 CPU 友好的小值（`max_new_tokens` 48-192），与正式实验不一致。

正式实验的数字写入上级目录 `results/`，需在 GPU 上用 vllm 按
[DEPLOY_CHECKLIST.md](../../DEPLOY_CHECKLIST.md) 的流程运行。脚本支持 `--out-dir`，
用于把验证产物与正式结果分开存放：

```bash
python scripts/run_eval.py --model Qwen/Qwen2.5-0.5B-Instruct \
    --benchmark scibench --mode direct --max-problems 12 \
    --backend transformers --device cpu --save --out-dir results/smoke
```

## 文件说明

| 文件 | 内容 | 关键结果 |
|---|---|---|
| `eval_scibench_direct_*.json` | SciBench direct，12 题 | overall 0.083 [0.000, 0.250]，`unparsed_pred` 1/12 |
| `eval_scibench_cot_*.json` | SciBench cot，8 题 | overall 0.000 |
| `eval_mmlu_physics_direct_*.json` | MMLU 物理选择题，12 题 | overall 0.167（验证选项字母判定链路） |
| `sensitivity_{numeric,wording,swap}_*.json` | 三路变体，8 题基准 | 逐题审计 `items[]`，含 `unchanged_share` 与配对口径 |
| `sensitivity_summary_*.json` | 三变体汇总 | wording 覆盖 8/8，`unchanged_share` 0.625 |
| `causal_tracing_scibench_*.json` | 逐层恢复率，6 题 | 1 题满足计入条件（clean 0.96 -> corrupt 0.00，峰值 +0.43 @ layer 12/24） |
| `table_smoke.md` | 表格管道输出 | 每格 n=2，CI 无参考价值 |

因果追踪中 6 题只有 1 题计入，原因是其余题目的 clean 运行本身未答对，恢复率无定义。
该筛选规则在正式实验中同样生效，报告的 `n_valid` 与跳过原因计数必须写入论文。

复现命令与详细日志见上级目录及 `results/logs/`（日志不入库）。
