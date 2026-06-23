#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export USERS="${USERS:-1500}"

exec "${SCRIPT_DIR}/run_jmeter_6_endpoints_1200.sh" "$@"
