#!/usr/bin/env bash
# ============================================================================
# EMPR 云上首日 · 一条龙脚本
#
# 目标：租到机器后，用最小成本把"评估管道"整体跑通，确认"环境 OK + 数据可加载 +
#       判定可靠 + 变体可用"，为 Week 1 全量评估扫清障碍。
#       默认只做免费/廉价的步骤；全量推理用 --full 另行开启。
#
# 用法：
#   bash cloud_first_day.sh                  # 免费体检 + 7B 小样本基线
#   bash cloud_first_day.sh --tiny-off       # 连小样本基线也不跑（只做环境+数据+判分+覆盖率）
#   bash cloud_first_day.sh --full           # 检查通过后，额外跑 2 个模型的全量 vllm 评估
#   bash cloud_first_day.sh --models "Qwen/Qwen2.5-14B-Instruct"   # 指定模型
#
# 前置：已 `git clone https://github.com/380992846-tech/EMPR` 并在仓库根目录执行本脚本。
# 若网络下载慢：export HF_ENDPOINT=https://hf-mirror.com
#
# 设计原则：**先在免费步骤上失败，别在烧卡的步骤上失败。**
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

FULL=0
TINY=1
FULL_MODELS=("Qwen/Qwen2.5-7B-Instruct" "Qwen/Qwen2.5-14B-Instruct")
for arg in "$@"; do
  case "$arg" in
    --full) FULL=1 ;;
    --tiny-off) TINY=0 ;;
    --models=*) IFS=',' read -r -a FULL_MODELS <<< "${arg#--models=}" ;;
    --models) echo "用法：--models=A,B（用等号，避免 shell 分词问题）"; exit 2 ;;
    *) echo "unknown arg: $arg"; exit 2 ;;
  esac
done

banner() { printf "\n\033[1;36m==== %s ====\033[0m\n" "$*"; }
step()   { printf "\n\033[1m[ %s ]\033[0m %s\n" "$1" "$2"; }
die()    { printf "\n\033[1;31m[FAIL] %s\033[0m\n" "$*" >&2; exit 1; }
warn()   { printf "\n\033[1;33m[warn] %s\033[0m\n" "$*"; }

# 取最新一个匹配的 eval json（避免 results/eval_*.json 展开成多个参数传给 --file）
latest_eval() { ls -t results/eval_*.json 2>/dev/null | head -n 1; }

# ---------------------------------------------------------------------------
banner "0/9  环境检查"
command -v python3 >/dev/null 2>&1 || die "未找到 python3"
echo "  python3: $(python3 --version)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/  GPU: /'
else
  warn "未检测到 nvidia-smi：只能跑 CPU 冒烟，别指望全量"
fi

if [ ! -d .venv ]; then
  step "venv" "创建虚拟环境 .venv"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# ---------------------------------------------------------------------------
banner "1/9  安装依赖"
pip install --upgrade pip >/dev/null 2>&1 || true
pip install -r requirements.txt || die "依赖安装失败（可先装匹配 CUDA 的 torch，见 DEPLOY_CHECKLIST 第 2 节）"
python - <<'PY' || true
try:
    import torch, transformers
    print(f"  torch {torch.__version__} | transformers {transformers.__version__} | "
          f"cuda={torch.cuda.is_available()}")
except Exception as e:
    print("  [warn] torch/transformers 导入失败:", e)
try:
    import vllm; print("  vllm 可用:", vllm.__version__)
except Exception:
    print("  [warn] vllm 不可用 -> 后续用 --backend transformers（慢但能跑）")
PY

# ---------------------------------------------------------------------------
banner "2/9  数据下载"
python scripts/download_data.py || die "SciBench 下载失败（见 README / DEPLOY_CHECKLIST）"
python scripts/download_data.py --benchmark olympiadbench || warn "OlympiadBench 下载失败，先跳过"

# ---------------------------------------------------------------------------
banner "3/9  单元测试（判分/统计/变体/数据/提示词）"
python -m pytest tests -q || die "单元测试失败：先修代码，别往下跑"

# ---------------------------------------------------------------------------
banner "4/9  一键体检：依赖 + 数据 + 判分自洽 + 变体覆盖率"
python scripts/preflight.py --benchmark scibench || die "体检未通过：按上面的 FAIL 项修完再来"

# ---------------------------------------------------------------------------
banner "5/9  判分回归用例明细（可选，看逐条行为）"
python -m src.grading || warn "判分回归有 FAIL，请先修 src/grading.py"

# ---------------------------------------------------------------------------
if [ "$TINY" = "1" ]; then
  banner "6/9  小样本基线（7B + 0.5B · 便宜 · 确认推理链路）"
  python scripts/run_eval.py --model Qwen/Qwen2.5-7B-Instruct \
    --benchmark scibench --mode direct --max-problems 10 --backend auto --save \
    || die "7B 小样本失败（模型下载慢可设 HF_ENDPOINT 镜像）"
  python scripts/run_eval.py --model Qwen/Qwen2.5-0.5B-Instruct \
    --benchmark scibench --mode cot --max-problems 8 --max-new-tokens 192 \
    --backend transformers --save || warn "0.5B 冒烟失败（不影响主流程）"

  banner "7/9  判定质量抽查（重点看 unparsed_pred 与 unscorable_ref）"
  F="$(latest_eval || true)"
  if [ -n "${F:-}" ]; then
    python scripts/check_grading.py --file "$F" --samples 6 || warn "抽查脚本报错，请人工看 $F"
  else
    warn "没找到 results/eval_*.json，跳过抽查"
  fi
else
  banner "6/9~7/9  小样本基线（已按 --tiny-off 跳过）"
fi

# ---------------------------------------------------------------------------
if [ "$FULL" = "1" ]; then
  banner "8/9  全量评估（vllm · SciBench 323 题 · cot）"
  for M in "${FULL_MODELS[@]}"; do
    step "eval" "$M"
    python scripts/run_eval.py --model "$M" --benchmark scibench --mode cot \
      --max-problems 323 --backend vllm --gpus 1 --save || die "$M 全量失败"
  done
  banner "8/9  生成主表"
  python scripts/make_table.py --dir results --benchmark scibench --mode cot \
    --format md --out results/table1.md || warn "主表生成失败，可稍后手动整理"
  python scripts/make_table.py --dir results --benchmark scibench --mode cot \
    --format tex --out results/table1.tex || true
else
  banner "8/9  跳过全量推理"
  echo "  首日小管道已跑通。确认 OK 后再开全量（会花 GPU 钱）："
  echo "    bash cloud_first_day.sh --full"
  echo "    bash cloud_first_day.sh --full --models=Qwen/Qwen2.5-14B-Instruct"
fi

# ---------------------------------------------------------------------------
banner "9/9  完成"
echo "  下一步："
echo "    1. 看 results/eval_*.json 的 overall_acc / per_domain_acc / n_unscored；"
echo "    2. 用 results/ 真实数字填 paper/（对应关系见 paper/TABLES.md）；"
echo "    3. Week 2/3/4 的执行顺序见 DEPLOY_CHECKLIST.md 第 6 节。"
