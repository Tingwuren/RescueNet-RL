#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[1/3] Updating base dependencies in conda env: pytorch"
conda run -n pytorch pip install -r "${REPO_ROOT}/requirements.txt"

echo "[2/3] Updating ns-3 integration dependencies"
conda run -n pytorch pip install -r "${REPO_ROOT}/ns-3.46.1/requirements.txt"

echo "[3/3] Verifying key modules"
conda run -n pytorch python -c "import fastapi, uvicorn, sumolib, torch, gymnasium; print('ok')"

echo "Environment 'pytorch' is ready for RescueNet-RL + ns-3.46.1."
