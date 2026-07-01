#!/usr/bin/env bash
# systemd wrapper (lottery-bot.service): run the LOTTERY paper bot on the VM, on the
# repurposed dual-momentum account (keys in .env.lottery), buying the top-3
# combined_score hype picks from the laptop morning board.
#
# Flow: the laptop board run (~6:24am PT = ~09:24 ET) writes
# experiments/lottery/picks/<ET-date>.json and git-pushes it. This wrapper fires at
# ~09:44 ET, pulls, WAITS for today's board to arrive, then runs run_lottery_bot.py
# (top-3 by combined_score, $2000 notional each + 10% native trailing stop). Positions are
# force-closed the SAME DAY at ~15:55 ET by the separate lottery-eod.timer (2026-06-30); the
# morning run's own close is now just a T+1 backstop if that EOD run was ever missed.
#
# If no board arrives (laptop off / no scan), it exits idle.
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
  # Wait up to ~10 min for the laptop board to push; git pull each tick.
  found=0
  for i in $(seq 1 20); do
    git pull --ff-only --no-edit >/dev/null 2>&1 || true
    if [ -f "$BOARD" ]; then echo "today's board present: $BOARD"; found=1; break; fi
    echo "[$i/20] waiting for today's board ($BOARD) ..."
    sleep 30
  done
  if [ "$found" -ne 1 ]; then
    echo "no board pushed for $STAMP after ~10 min -> lottery bot IDLE today."
    # still run time-stops in case prior positions need a T+3 close
    "$PY" "$ROOT/scripts/run_lottery_bot.py" --time-stops-only || true
    exit 0
  fi
  "$PY" "$ROOT/scripts/run_lottery_bot.py"
  echo "=== lottery-bot end $(date '+%F %T %z')  (exit=$?) ==="
} >> "$WRAP_LOG" 2>&1
