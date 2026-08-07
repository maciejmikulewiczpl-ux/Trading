#!/usr/bin/env bash
# systemd wrapper (news-hold.service): the NEWS bot RE-STRUCTURED 2026-08-07 from a same-day ORB
# bot into a MULTI-DAY CATALYST HOLD. It reuses the battle-tested hype engine (run_lottery_bot.py)
# via env overrides -- news account (.env.news), news_edge picks, catalyst selector (signal=1),
# $2000 notional + 10% native trailing stop + T+3 time-stop. Rationale: the same-day news signal
# was dead, but the analyst's direction call ORDERS the multi-day move (signal=1 ret_3d win 53% >>
# signal=0/-1); forward-test whether the trailing stop harvests it (like it did for the hype bot).
#
# Flow: the laptop news scan (~09:33 ET) writes experiments/news_edge/picks/<ET-date>.json + pushes.
# This wrapper fires ~09:45 ET, pulls, waits for today's picks, then runs the engine in news mode.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# --- news multi-day mode (all else defaults to the hype bot; engine is shared) ---
export LOTTERY_TAG=newshold
export LOTTERY_ENV_FILE=.env.news
export LOTTERY_PICKS_DIR=news_edge
export LOTTERY_SELECT=catalyst

LOG_DIR="$ROOT/logs"; mkdir -p "$LOG_DIR"
STAMP="$(TZ=America/New_York date +%F)"
WRAP_LOG="$LOG_DIR/task_newshold_${STAMP}.log"
PY="$ROOT/.venv/bin/python"
BOARD="$ROOT/experiments/news_edge/picks/${STAMP}.json"

{
  echo "=== news-hold launch $(date '+%F %T %z')  (ET date $STAMP) ==="
  found=0
  for i in $(seq 1 20); do
    git pull --ff-only --no-edit >/dev/null 2>&1 || true
    if [ -f "$BOARD" ]; then echo "today's news picks present: $BOARD"; found=1; break; fi
    echo "[$i/20] waiting for today's news picks ($BOARD) ..."
    sleep 30
  done
  if [ "$found" -ne 1 ]; then
    echo "no news picks pushed for $STAMP after ~10 min -> news-hold IDLE today (baseline unaffected)."
    "$PY" "$ROOT/scripts/run_lottery_bot.py" --time-stops-only || true
    exit 0
  fi
  "$PY" "$ROOT/scripts/run_lottery_bot.py"
  echo "=== news-hold end $(date '+%F %T %z')  (exit=$?) ==="
} >> "$WRAP_LOG" 2>&1
