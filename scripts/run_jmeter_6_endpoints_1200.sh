#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

USERS="${USERS:-1200}"
RAMP="${RAMP:-60}"
LOOPS="${LOOPS:-1}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
PROTOCOL="${PROTOCOL:-http}"
TIMEOUT_MS="${TIMEOUT_MS:-30000}"
PLAN="${PLAN:-jmeter/rescuenet_single_endpoint_paced_1200.jmx}"
PREWARM="${PREWARM:-1}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-artifacts/load_tests/jmeter/separate_${USERS}_${TS}}"

usage() {
  cat <<USAGE
Usage:
  scripts/run_jmeter_6_endpoints_1200.sh [options]

This script runs six independent JMeter pressure tests sequentially.
Each endpoint is tested with the same virtual-user count, so the default
run is ${USERS} concurrent users per endpoint.

Options:
  --users N        Virtual users / threads per endpoint. Default: ${USERS}
  --ramp N         Ramp-up seconds. Default: ${RAMP}
  --loops N        Loop count. Default: 1
  --host HOST      Target host. Default: ${HOST}
  --port PORT      Target port. Default: 8000
  --protocol P     http or https. Default: http
  --timeout MS     Connect and response timeout. Default: 30000
  --out-dir DIR    Output directory. Default: artifacts/load_tests/jmeter/separate_<users>_<timestamp>
  --no-prewarm     Skip health/scenarios warm-up requests.
  -h, --help       Show this help.

Environment overrides:
  USERS, RAMP, LOOPS, HOST, PORT, PROTOCOL, TIMEOUT_MS, OUT_DIR, HEAP
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --users)
      USERS="$2"
      shift 2
      ;;
    --ramp)
      RAMP="$2"
      shift 2
      ;;
    --loops)
      LOOPS="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --protocol)
      PROTOCOL="$2"
      shift 2
      ;;
    --timeout)
      TIMEOUT_MS="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --no-prewarm)
      PREWARM="0"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v jmeter >/dev/null 2>&1; then
  echo "jmeter is not installed or not in PATH." >&2
  exit 1
fi

if [[ ! -f "$PLAN" ]]; then
  echo "JMeter plan not found: $PLAN" >&2
  exit 1
fi

BASE_URL="${PROTOCOL}://${HOST}:${PORT}"
DISPLAY_TARGET="${TARGET_LABEL:-被测系统本机服务}"
HEALTH_TIMEOUT="$(( (TIMEOUT_MS + 999) / 1000 ))"
mkdir -p "$OUT_DIR"

health_check() {
  local label="$1"
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl is not installed or not in PATH." >&2
    exit 1
  fi

  local code
  echo "${label} health check command:"
  echo "  curl -sS -o /dev/null -w \"%{http_code}\" --max-time ${HEALTH_TIMEOUT} \"\$BASE_URL/api/health\""
  code="$(curl -sS -o /dev/null -w "%{http_code}" --max-time "$HEALTH_TIMEOUT" "${BASE_URL}/api/health" || true)"
  echo "${label} health check output: HTTP ${code}"
  if [[ "$code" != "200" ]]; then
    echo "${label} health check failed for ${BASE_URL}/api/health" >&2
    exit 1
  fi
}

if [[ "$PREWARM" == "1" ]] && command -v curl >/dev/null 2>&1; then
  health_check "Pre-test"
  echo "Prewarm command:"
  echo "  curl -fsS -o /dev/null \"\$BASE_URL/api/scenarios\""
  echo "Prewarming 场景列表接口"
  curl -fsS -o /dev/null "${BASE_URL}/api/scenarios" || true
fi

ulimit -n 65535 2>/dev/null || true
export HEAP="${HEAP:--Xms1g -Xmx6g -XX:+UseG1GC -XX:-HeapDumpOnOutOfMemoryError}"
# Prevent Git Bash/MSYS2 from rewriting JMeter properties like /api/health
# into Windows paths such as D:/Program files/Git/api/health.
export MSYS2_ARG_CONV_EXCL="${MSYS2_ARG_CONV_EXCL:-*}"

NAMES=(
  "health"
  "devices"
  "runtime_status_1"
  "runtime_status_2"
  "replay_sessions"
  "scenarios"
)

PATHS=(
  "/api/health"
  "/api/devices"
  "/api/mahimahi/status"
  "/api/ns3/status"
  "/api/replay/sessions?limit=5"
  "/api/scenarios"
)

LABELS=(
  "健康检查接口"
  "设备列表接口"
  "运行状态接口一"
  "运行状态接口二"
  "回放会话列表接口"
  "场景列表接口"
)

echo "Running JMeter pressure tests"
echo "  plan:     $PLAN"
echo "  target:   $DISPLAY_TARGET"
echo "  users:    $USERS per endpoint"
echo "  output:   $OUT_DIR"

for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  path="${PATHS[$i]}"
  label="${LABELS[$i]}"
  endpoint_dir="${OUT_DIR}/${name}"
  mkdir -p "$endpoint_dir"

  echo
  echo "[$((i + 1))/${#NAMES[@]}] Testing ${label}"

  jmeter -n \
    -t "$PLAN" \
    -Jusers="$USERS" \
    -Jramp="$RAMP" \
    -Jloops="$LOOPS" \
    -Jsync_users="$USERS" \
    -Jsync_timeout="$TIMEOUT_MS" \
    -Jhost="$HOST" \
    -Jport="$PORT" \
    -Jprotocol="$PROTOCOL" \
    -Jconnect_timeout="$TIMEOUT_MS" \
    -Jresponse_timeout="$TIMEOUT_MS" \
    -Jmethod="GET" \
    -Jpath="$path" \
    -Jlabel="$label" \
    -l "${endpoint_dir}/results.jtl" \
    -e -o "${endpoint_dir}/html" \
    -j "${endpoint_dir}/jmeter.log"
done

python3 - "$OUT_DIR" "$DISPLAY_TARGET" "$USERS" <<'PY'
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

out_dir = Path(sys.argv[1])
display_target = sys.argv[2]
users = int(sys.argv[3])
json_path = out_dir / "summary.json"
report_path = out_dir / "report.md"

def pct(values, q):
    if not values:
        return 0
    values = sorted(values)
    k = (len(values) - 1) * q
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(values[int(k)])
    return values[f] + (values[c] - values[f]) * (k - f)

groups = defaultdict(list)
for jtl_path in sorted(out_dir.glob("*/results.jtl")):
    with jtl_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["_jtl"] = str(jtl_path)
            groups[row["label"]].append(row)

summary = []
for label, rows in groups.items():
    elapsed = [int(row["elapsed"]) for row in rows]
    total = len(rows)
    failures = sum(1 for row in rows if row["success"].lower() != "true")
    start = min(int(row["timeStamp"]) for row in rows)
    end = max(int(row["timeStamp"]) + int(row["elapsed"]) for row in rows)
    duration_s = max((end - start) / 1000, 0.001)
    codes = Counter(row["responseCode"] for row in rows)
    summary.append({
        "endpoint": label,
        "samples": total,
        "success": total - failures,
        "failures": failures,
        "error_pct": round(failures * 100 / total, 4) if total else 0,
        "avg_ms": round(sum(elapsed) / total, 2) if total else 0,
        "min_ms": min(elapsed) if elapsed else 0,
        "p50_ms": round(pct(elapsed, 0.50), 2),
        "p90_ms": round(pct(elapsed, 0.90), 2),
        "p95_ms": round(pct(elapsed, 0.95), 2),
        "p99_ms": round(pct(elapsed, 0.99), 2),
        "max_ms": max(elapsed) if elapsed else 0,
        "throughput_req_s": round(total / duration_s, 2),
        "status_counts": dict(codes),
    })

order = {
    "健康检查接口": 0,
    "设备列表接口": 1,
    "运行状态接口一": 2,
    "运行状态接口二": 3,
    "回放会话列表接口": 4,
    "场景列表接口": 5,
}
summary.sort(key=lambda item: order.get(item["endpoint"], 999))

total_samples = sum(item["samples"] for item in summary)
total_failures = sum(item["failures"] for item in summary)
overall = {
    "target": display_target,
    "users_per_endpoint": users,
    "samples": total_samples,
    "success": total_samples - total_failures,
    "failures": total_failures,
    "error_pct": round(total_failures * 100 / total_samples, 4) if total_samples else 0,
    "endpoints": summary,
}

json_path.write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")

lines = [
    "# JMeter Separate Endpoint Pressure Test",
    "",
    f"- Target: `{display_target}`",
    f"- Virtual users per endpoint: `{users}`",
    f"- Total samples: `{overall['samples']}`",
    f"- Success: `{overall['success']}`",
    f"- Failures: `{overall['failures']}`",
    f"- Error %: `{overall['error_pct']}`",
    "",
    "| Endpoint | Samples | Success | Error % | Avg ms | P50 ms | P90 ms | P95 ms | P99 ms | Max ms | Throughput req/s |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for item in summary:
    lines.append(
        f"| `{item['endpoint']}` | {item['samples']} | {item['success']} | "
        f"{item['error_pct']:.2f} | {item['avg_ms']:.2f} | {item['p50_ms']:.2f} | "
        f"{item['p90_ms']:.2f} | {item['p95_ms']:.2f} | {item['p99_ms']:.2f} | "
        f"{item['max_ms']} | {item['throughput_req_s']:.2f} |"
    )
report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("\nResult summary")
print("=" * 72)
print(f"Target: {display_target}")
print(f"Users per endpoint: {users}")
print(f"Samples: {overall['samples']}  Success: {overall['success']}  Failures: {overall['failures']}  Error%: {overall['error_pct']}")
for item in summary:
    print(
        f"{item['endpoint']}: samples={item['samples']} success={item['success']} "
        f"fail={item['failures']} avg={item['avg_ms']}ms p95={item['p95_ms']}ms "
        f"throughput={item['throughput_req_s']}/s"
    )
print(f"\nWrote {json_path}")
print(f"Wrote {report_path}")
PY

health_check "Post-test"

echo
echo "JMeter artifacts:"
echo "  Output:  $OUT_DIR"
echo "  Summary: $OUT_DIR/summary.json"
echo "  Report:  $OUT_DIR/report.md"
echo "  HTML:    $OUT_DIR/<endpoint>/html/index.html"
echo "  JTL:     $OUT_DIR/<endpoint>/results.jtl"
