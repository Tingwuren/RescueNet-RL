#!/usr/bin/env bash
# 超强台风场景 — RescueNet HMARL 训练（产物: checkpoints/super_typhoon/）
#
# 用法:
#   cd HMARL
#   ./run_rescuenet_train_super_typhoon.sh
#   ./run_rescuenet_train_super_typhoon.sh --total-timesteps 5000
#
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
  --total-timesteps 50000 \
  --rollout-steps 1024 \
  --log-interval 1 \
  --eval-interval 5 \
  --eval-episodes 3 \
  --hierarchy-report-interval 1 \
  "$@"
