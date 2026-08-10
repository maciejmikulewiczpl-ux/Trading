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
  # Back up the VM-written analytical data to GitHub (VM has push auth as of 2026-08-10). Only the
  # meanrev lab (data-only, EOD, low-risk) commits these -- the trading-bot launchers stay pull-only.
  # rebase before push = race-safe vs the Surface's pushes.
  git add logs/lottery_execution.csv logs/meanrev_trades.csv 2>/dev/null || true
  if ! git diff --cached --quiet; then
    git commit -q -m "data: VM analytical logs $(TZ=America/New_York date +%F)" \
      && git pull --rebase -q >/dev/null 2>&1 && git push -q origin main >/dev/null 2>&1 || true
  fi
  echo "=== end $(date '+%F %T %z')  (exit=$?) ==="
} >> "$LOG_DIR/task_meanrev_${STAMP}.log" 2>&1
