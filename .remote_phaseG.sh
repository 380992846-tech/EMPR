#!/usr/bin/env bash
# 阶段 G：A(OlympiadBench 7B+14B) / B(MMLU 7B MCQ) / C(Week4 错误分类, 14B 当标注器)
set -uo pipefail
export PATH=/root/miniconda3/bin:$PATH
export HF_HOME=/root/autodl-tmp/hf
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false
cd /root/EMPR || exit 1
STATUS=/root/EMPR/results/logs/STATUS6.txt
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$STATUS"; }
free_gb() { df -BG --output=avail /root/autodl-tmp | tail -1 | tr -dc '0-9'; }
dl() {
  [ -f "$2/config.json" ] && { say "model ready: $2"; return 0; }
  say "downloading $1 (free $(free_gb)G)"
  timeout 7200 python - "$1" "$2" <<'PY' 2>&1 | tail -2 | tee -a "$STATUS"
import sys
from modelscope import snapshot_download
print("DOWNLOADED:", snapshot_download(sys.argv[1], cache_dir="/root/autodl-tmp/ms_cache", local_dir=sys.argv[2]))
PY
  [ -f "$2/config.json" ] || { say "DL FAIL $1"; return 1; }
}
stage() {
  local tag=$1 tmo=$2; shift 2
  say "START $tag (free $(free_gb)G)"
  timeout "$tmo" "$@" > "results/logs/$tag.log" 2>&1
  local rc=$?
  grep -E "OVERALL|save|error|Traceback|已标注|E1|卡方|chi2" "results/logs/$tag.log" | tail -6 | sed 's/^/    /' | tee -a "$STATUS"
  say "END $tag rc=$rc"
  return $rc
}
EV=scripts/run_eval.py

# ================= 7B: B(MMLU) + A(OlympiadBench) =================
S7=/root/autodl-tmp/models/Qwen2.5-7B-Instruct
if dl qwen/Qwen2.5-7B-Instruct "$S7"; then
  # B: MMLU 选择题（HF 不可达 -> 走本地 data/mmlu_physics/）
  stage g1_7b_mmlu 3600 env HF_HUB_OFFLINE=1 python "$EV" --model "$S7" --benchmark mmlu_physics \
    --mode direct --max-problems 488 --backend transformers --device cuda \
    --max-new-tokens 64 --batch-size 48 --save
  # A: OlympiadBench 竞赛物理（152 题可判分）
  stage g2_7b_olympiad 7200 python "$EV" --model "$S7" --benchmark olympiadbench \
    --mode cot --max-problems 351 --backend transformers --device cuda \
    --max-new-tokens 1024 --batch-size 32 --save
  say "removing 7B"; rm -rf "$S7"; say "free now $(free_gb)G"
fi

# ================= 14B: A(OlympiadBench) + C(错误分类) =================
S14=/root/autodl-tmp/models/Qwen2.5-14B-Instruct
if dl qwen/Qwen2.5-14B-Instruct "$S14"; then
  stage g3_14b_olympiad 10800 python "$EV" --model "$S14" --benchmark olympiadbench \
    --mode cot --max-problems 351 --backend transformers --device cuda \
    --max-new-tokens 1024 --batch-size 16 --save
  # C: Week 4 错误分类 —— 用 14B 给 7B 与 14B 自己的错误样本打 E1/E2/E3/N 标签
  for src in results/eval_scibench_cot_Qwen2.5-7B-Instruct.json results/eval_scibench_cot_Qwen2.5-14B-Instruct.json; do
    [ -f "$src" ] || continue
    tag="g4_judge_$(basename "$src" .json)"
    stage "$tag" 7200 python scripts/classify_errors.py --file "$src" --judge-model "$S14" --backend transformers
  done
  # 两个模型的错误分布卡方对比
  stage g5_judge_report 1800 python scripts/classify_errors.py \
    --file results/eval_scibench_cot_Qwen2.5-7B-Instruct.json --report \
    --compare-file results/eval_scibench_cot_Qwen2.5-14B-Instruct.json
  say "removing 14B"; rm -rf "$S14"; say "free now $(free_gb)G"
fi

say "PHASE G DONE free=$(free_gb)G"
ls -la results/eval_*.json results/error_labels.json 2>/dev/null | tail -14
