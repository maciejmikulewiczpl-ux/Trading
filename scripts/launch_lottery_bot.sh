#!/usr/bin/env bash
# systemd wrapper (lottery-bot.service): run the LOTTERY paper bot on the VM, on the
# repurposed dual-momentum account (keys in .env.lottery), buying the top-3 premarket
# GAP-UPS (gap_pct >= 10%, selector switched 2026-10-18) from the laptop morning board.
#
# Flow: the laptop board run (~6:24am PT = ~09:24 ET) writes
# experiments/lottery/picks/<ET-date>.json and git-pushes it. This wrapper fires at
# ~09:44 ET, pulls, WAITS briefly for today's board, then runs run_lottery_bot.py
# ($2000 notional each + 10% native trailing stop, T+3 time-stop).
#
# If no board arrives (laptop asleep -- happened 10 straight days 10-05..10-16), it builds a
# gap-only fallback board on the VM (experiments/lottery/gap_fallback.py, same pm_gappers()
# signal) and trades that instead of sitting idle.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
STAMP="$(TZ=America/New_York date +%F)"          # ET date == how board files are named
WRAP_LOG="$LOG_DIR/task_lottery_${STAMP}.log"
PY="$ROOT/.venv/bin/python"
BOARD="$ROOT/experiments/lottery/picks/${STAMP}.json"

{
  echo "=== lottery-bot launch $(date '+%F %T %z')  (ET date $STAMP) ==="
  # Wait up to ~2 min for the laptop board (it normally lands ~09:25 ET, well before 09:44);
  # longer waits only delay the entry past the 09:45 the edge was measured at.
  found=0
  for i in $(seq 1 4); do
    git pull --ff-only --no-edit >/dev/null 2>&1 || true
    if [ -f "$BOARD" ]; then echo "today's board present: $BOARD"; found=1; break; fi
    echo "[$i/4] waiting for today's board ($BOARD) ..."
    sleep 30
  done
  if [ "$found" -ne 1 ]; then
    echo "no board pushed for $STAMP -> building VM gap fallback."
    "$PY" "$ROOT/experiments/lottery/gap_fallback.py" "$STAMP" || true
  fi
  "$PY" "$ROOT/scripts/run_lottery_bot.py"
  echo "=== lottery-bot end $(date '+%F %T %z')  (exit=$?) ==="
} >> "$WRAP_LOG" 2>&1
