#!/usr/bin/env bash
# 极端暴雨场景 — 测试 checkpoint（环境说明 + L1/L2/L3 I/O + RL rollout + 组网网格落点）
# 组网交付物会为每个节点决策 grid_index / 归一化坐标（有残余与无残余两种模式各一套）
#
# 用法:
#   cd HMARL
#   ./run_rescuenet_test_extreme_rainstorm.sh
#   ./run_rescuenet_test_extreme_rainstorm.sh checkpoints/extreme_rainstorm
#   ./run_rescuenet_test_extreme_rainstorm.sh checkpoints/extreme_rainstorm_best --skip-eval
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"
cd "$ROOT"

if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$REPO_ROOT/.venv/bin/activate"
fi

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  PYTHON_BIN="python3"
fi

CHECKPOINT="checkpoints/extreme_rainstorm_best"
EXTRA_ARGS=()
for arg in "$@"; do
  case "$arg" in
    checkpoints/*) CHECKPOINT="$arg" ;;
    *) EXTRA_ARGS+=("$arg") ;;
  esac
done

echo "[networking] 导出极端暴雨组网方案 outputs/rainstorm_{with,no}_residual ..."
"$PYTHON_BIN" -m Networking_plan.deployment.export_plan --scenario extreme_rainstorm --mode with_residual
"$PYTHON_BIN" -m Networking_plan.deployment.export_plan --scenario extreme_rainstorm --mode no_residual

exec "$PYTHON_BIN" rescuenet/test_checkpoint.py \
  --checkpoint-dir "$CHECKPOINT" \
  --scenario-alias extreme_rainstorm \
  --episodes 1 \
  "${EXTRA_ARGS[@]}"
