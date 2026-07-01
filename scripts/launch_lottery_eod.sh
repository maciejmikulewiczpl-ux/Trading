#!/usr/bin/env bash
# systemd wrapper (lottery-eod.service): the SAME-DAY EXIT run for the lottery/hype bot.
# Fires ~15:55 ET on US weekdays and force-closes EVERY tracked position opened today
# (cancels its GTC trailing stop first, then closes at market) so the bot sleeps in cash.
#
# Rationale (2026-06-30): our own logged picks show the top-3 average +1.04% from 09:45->close
# but -1.49% by T+3 -- the multi-day hold gave back the edge to the ~-0.64%/night overnight gap
# bleed a trailing stop cannot cap. So we capture the intraday move and exit same-day.
#
# No board wait needed (this only closes existing positions). Idle exit 0 if nothing is tracked
# or the market was closed.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
STAMP="$(TZ=America/New_York date +%F)"
WRAP_LOG="$LOG_DIR/task_lottery_eod_${STAMP}.log"
PY="$ROOT/.venv/bin/python"

{
  echo "=== lottery-eod (same-day close) $(date '+%F %T %z')  (ET date $STAMP) ==="
  git pull --ff-only --no-edit >/dev/null 2>&1 || true
  "$PY" "$ROOT/scripts/run_lottery_bot.py" --eod-close || true
  echo "=== lottery-eod end $(date '+%F %T %z')  (exit=$?) ==="
} >> "$WRAP_LOG" 2>&1
