#!/usr/bin/env bash
# meanrev.service: daily EOD mean-reversion forward-test lab (2026-08-07). DATA-ONLY, no account /
# no orders -- it maintains a paper book (logs/meanrev_positions.json) and appends closed trades to
# logs/meanrev_trades.csv = an out-of-sample track record for the backtested Connors RSI-2 edge
# (+0.47%/trade, 66% win). The last-date guard makes re-runs safe. Uses .env keys for DATA only.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
LOG_DIR="$ROOT/logs"; mkdir -p "$LOG_DIR"
STAMP="$(TZ=America/New_York date +%F)"
PY="$ROOT/.venv/bin/python"
{
  echo "=== meanrev-lab $(date '+%F %T %z') ==="
  git pull --ff-only --no-edit >/dev/null 2>&1 || true
  "$PY" "$ROOT/experiments/meanrev/lab.py" scan
  echo "=== end $(date '+%F %T %z')  (exit=$?) ==="
} >> "$LOG_DIR/task_meanrev_${STAMP}.log" 2>&1
