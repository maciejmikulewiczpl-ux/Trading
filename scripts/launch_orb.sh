#!/usr/bin/env bash
# Wrapper used by systemd (orb.service) to launch the ORB paper runner on Linux.
# Mirrors scripts/launch_orb.ps1: captures combined stdout+stderr to
# logs/task_<date>.log so any pre-startup Python errors are still captured even
# if the runner can't open its own log file. The runner itself also writes
# logs/orb_<date>.log via Python logging.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y-%m-%d)"
WRAP_LOG="$LOG_DIR/task_${STAMP}.log"

PY="$ROOT/.venv/bin/python"
SCRIPT="$ROOT/live/paper_orb.py"

{
  echo "=== Task launch at $(date '+%Y-%m-%d %H:%M:%S %z') ==="
  "$PY" "$SCRIPT" "$@"
  EXIT_CODE=$?
  echo "=== Task end at $(date '+%Y-%m-%d %H:%M:%S %z') (exit=$EXIT_CODE) ==="
  exit $EXIT_CODE
} >> "$WRAP_LOG" 2>&1
