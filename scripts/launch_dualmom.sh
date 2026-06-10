#!/usr/bin/env bash
# Wrapper used by systemd (dualmom.service) to launch the monthly dual-momentum
# runner on Linux. Fires every weekday morning; the runner self-gates to the
# first trading day of the month (no-ops otherwise) and refuses live orders
# unless DUALMOM_ALPACA_API_KEY is set. Captures output to
# logs/dualmom_task_<date>.log.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y-%m-%d)"
WRAP_LOG="$LOG_DIR/dualmom_task_${STAMP}.log"

PY="$ROOT/.venv/bin/python"
SCRIPT="$ROOT/live/run_dualmom.py"

{
  echo "=== DualMom launch at $(date '+%Y-%m-%d %H:%M:%S %z') ==="
  "$PY" "$SCRIPT" "$@"
  EXIT_CODE=$?
  echo "=== DualMom end at $(date '+%Y-%m-%d %H:%M:%S %z') (exit=$EXIT_CODE) ==="
  exit $EXIT_CODE
} >> "$WRAP_LOG" 2>&1
