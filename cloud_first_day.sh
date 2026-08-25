#!/usr/bin/env bash
# ============================================================================
# EMPR 云上首日 · 一条龙脚本
#
# 目标：租到机器后，用最小成本把"评估管道"整体跑通，并确认"判定可靠 + 数据可加载"，
#       为 Week 1 全量评估扫清障碍。默认只做便宜的步骤；全量推理用 --full 另行开启。
#
# 用法：
#   bash cloud_first_day.sh                  # 只做免费/廉价的检查 + 7B 小样本基线
#   bash cloud_first_day.sh --tiny-off       # 连 7B 小样本基线也不跑（只做环境+数据+判分+覆盖率）
#   bash cloud_first_day.sh --full           # 在每次检查通过后，额外跑 1~2 个模型的全量 vllm 评估
#
# 前置：已 `git clone https://github.com/380992846-tech/EMPR` 并在仓库根目录执行本脚本。
# 若网络下载慢，可 export HF_ENDPOINT=https://hf-mirror.com
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

FULL=0
TINY=1
for arg in "$@"; do
  case "$arg" in
    --full) FULL=1 ;;
    --tiny-off) TINY=0 ;;
    *) echo "unknown arg: $arg"; exit 2 ;;
  esac
done

banner() { printf "\n\033[1;36m==== %s ====\033[0m\n" "$*"; }
step()   { printf "\n\033[1m[ %s ]\033[0m %s\n" "$1" "$2"; }
die()    { printf "\n\033[1;31m[FAIL] %s\033[0m\n" "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 0. 环境检查
# ---------------------------------------------------------------------------
banner "0/8  环境检查"
command -v python3 >/dev/null 2>&1 || die "未找到 python3"
command -v nvidia-smi >/dev/null 2>&1 || echo "  [warn] 未检测到 nvidia-smi（无 GPU/驱动？）"
echo "  python3: $(python3 --version)"

if [ ! -d .venv ]; then
  step "venv" "创建虚拟环境 .venv"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# ---------------------------------------------------------------------------
# 1. 依赖
# ---------------------------------------------------------------------------
banner "1/8  安装依赖"
step "pip" "pip install -r requirements.txt（vllm 装不上会自动提示改用 transformers）"
pip install --upgrade pip >/dev/null 2>&1 || true
pip install -r requirements.txt || die "依赖安装失败（可先装 torch 匹配 CUDA，见 DEPLOY_CHECKLIST）"

# 校验关键库 + vllm 是否可用
python - <<'PY' || true
try:
    import torch, transformers; print("  torch", torch.__version__, "| transformers", transformers.__version__)
except Exception as e:
    print("  [warn] torch/transformers 导入失败:", e)
try:
    import vllm; print("  vllm 可用:", vllm.__version__)
except Exception:
    print("  [warn] vllm 不可用 -> 后续用 --backend transformers（慢但能跑）")
PY

# ---------------------------------------------------------------------------
# 2. 数据
# ---------------------------------------------------------------------------
banner "2/8  下载基准数据"
step "data" "SciBench 物理 6 科（~323 题）"
python scripts/download_data.py || die "SciBench 下载失败（可手动放 data/scibench/，见 README）"
step "data" "OlympiadBench 物理（可选，含官方解题）"
python scripts/download_data.py --benchmark olympiadbench || echo "  [warn] OlympiadBench 下载失败，先跳过（主基准 SciBench 已够）"

banner "3/8  数据加载自检"
python - <<'PY' || die "数据加载/答案自检失败（请检查 data/ 或 download_data.py）"
from src.data import load_benchmark
d = load_benchmark("scibench")
print("  SciBench 物理题数:", len(d))
if d:
    print("  样例答案:", d[0]["answer"], "| 域:", d[0]["domain"])
PY

# ---------------------------------------------------------------------------
# 3. 判分自检
# ---------------------------------------------------------------------------
banner "4/8  判分自检（重要：先确认对错判定可靠再全量跑）"
python -m src.grading || die "判分自检未通过，请先修 src/grading.py"

# ---------------------------------------------------------------------------
# 4. 覆盖率干跑（不调模型，确认三个变体生成器可靠）
# ---------------------------------------------------------------------------
banner "5/8  敏感性·覆盖率干跑（0 模型成本，确认变体生成器可用）"
python scripts/run_sensitivity.py --coverage --benchmark scibench --max-problems 323 || \
  echo "  [warn] 覆盖率干跑失败（不影响评估主表，但 Week 2 需要它）"

# ---------------------------------------------------------------------------
# 5. 小样本基线（7B · transformers · 10 题）—— 便宜，确认推理+判分链路 OR 依 --tiny-off 跳过
# ---------------------------------------------------------------------------
if [ "$TINY" = "1" ]; then
  banner "6/8  小样本基线（7B · transformers · 10 题）"
  python scripts/run_eval.py --model Qwen/Qwen2.5-7B-Instruct \
    --benchmark scibench --cot --max-problems 10 --backend transformers --save || \
    die "小样本基线失败（先看报错；模型下载慢可设 HF_ENDPOINT 镜像）"
  banner "7/8  判定质量抽查（重点看 pred_val=null 占比）"
  python scripts/check_grading.py --file results/eval_*.json || \
    echo "  [warn] 请人工核对 check_grading 输出"
else
  banner "6/8  7B 小样本基线（已按 --tiny-off 跳过）"
fi

# ---------------------------------------------------------------------------
# 6. 全量推理（仅 --full）
# ---------------------------------------------------------------------------
if [ "$FULL" = "1" ]; then
  banner "8/8  全量评估（vllm，2 个模型 · SciBench 全量）"
  python scripts/run_eval.py --model Qwen/Qwen2.5-7B-Instruct \
    --benchmark scibench --cot --max-problems 323 --backend vllm --gpus 1 --save || die "7B 全量失败"
  python scripts/run_eval.py --model Qwen/Qwen2.5-14B-Instruct \
    --benchmark scibench --cot --max-problems 323 --backend vllm --gpus 1 --save || die "14B 全量失败"
  banner "8/8  生成主表"
  python scripts/make_table.py --dir results --benchmark scibench --mode cot --format md \
      --out results/table.md || echo "  [warn] 生成主表失败（可稍后手动整理）"
else
  banner "8/8  跳过全量推理"
  echo "  首日小管道已跑通。确认结果 OK 后，再用 --full 开启全量（会花 GPU 钱）："
  echo "    bash cloud_first_day.sh --full"
fi

banner "完成 ✓"
echo "  下一步："
echo "    1. 看 results/eval_*.json 的 overall_acc 与 95% CI；"
echo "    2. 用 results/ 真实数字填 paper/ 里的表格（TABLES.md 有对应关系）；"
echo "    3. 需要更多模型/敏感/因果追踪，按 DEPLOY_CHECKLIST.md 第 5 步执行。"
