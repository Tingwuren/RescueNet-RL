#!/usr/bin/env bash
# HMARL 训练（逐 step 打印 loss + 每个 Update 评估一次）
#
# 用法:
#   cd HMARL
#   ./run_rescuenet_train.sh --scenario super_typhoon
#   ./run_rescuenet_train.sh --scenario extreme_rainstorm --total-timesteps 5000
#
# 说明:
# - 环境步 (EnvStep): 每与环境交互 1 步打印 reward/coverage（--env-step-log-interval 1）
# - PPO 梯度步 (OptStep): 每个 minibatch 的 loss（--step-loss-interval 1；在 rollout 收集完之后）
# - rollout-steps=1024 表示每 1024 个环境步才做一次策略更新，不是“每 1024 步才打印一次”
# - Update 测试: eval-interval=1，保证每个 Update 都做一次测试
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"
cd "$ROOT"

if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$REPO_ROOT/.venv/bin/activate"
fi

exec python rescuenet/train.py \
  --scenario super_typhoon \
  --total-timesteps 500000 \
  --rollout-steps 1024 \
  --log-interval 1 \
  --eval-interval 1 \
  --eval-episodes 1 \
  --hierarchy-report-interval 1 \
  --env-step-log-interval 1 \
  --step-loss-interval 1 \
  "$@"

