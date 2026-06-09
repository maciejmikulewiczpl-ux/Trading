#!/usr/bin/env bash
# systemd wrapper (news-orb.service): run the NEWS-EDGE catalyst-ORB on the VM, on the
# 2nd paper account, using the watchlist the LAPTOP morning scan produced and pushed.
#
# Flow: the laptop scan (~9:33 ET) writes experiments/news_edge/picks/<ET-date>.json and
# git-pushes it. This wrapper fires a bit later (news-orb.timer ~09:40 ET), pulls, WAITS
# for today's picks to arrive (the laptop may still be scanning), then hands off to
# scripts/run_news_orb.py (which sets the 2nd account + filters off + isolated heartbeat/
# log, derives the watchlist from today's + picks, and runs the session to 15:55 ET).
#
# If no picks arrive (laptop off / no scan that day), it exits idle — the baseline ORB
# bot is untouched.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
STAMP="$(TZ=America/New_York date +%F)"          # ET date == how picks files are named
WRAP_LOG="$LOG_DIR/task_news_${STAMP}.log"
PY="$ROOT/.venv/bin/python"
PICKS="$ROOT/experiments/news_edge/picks/${STAMP}.json"

{
  echo "=== news-orb launch $(date '+%F %T %z')  (ET date $STAMP) ==="
  # Wait up to ~12 min for the laptop scan to push today's picks; git pull each tick.
  found=0
  for i in $(seq 1 24); do
    git pull --ff-only --no-edit >/dev/null 2>&1 || true
    if [ -f "$PICKS" ]; then echo "today's picks present: $PICKS"; found=1; break; fi
    echo "[$i/24] waiting for today's picks ($PICKS) ..."
    sleep 30
  done
  if [ "$found" -ne 1 ]; then
    echo "no picks pushed for $STAMP after ~12 min -> news bot IDLE today (baseline bot unaffected)."
    exit 0
  fi
  # Hand off to the engine (reads today's + picks, trades them on the 2nd account all day).
  "$PY" "$ROOT/scripts/run_news_orb.py"
  echo "=== news-orb end $(date '+%F %T %z')  (exit=$?) ==="
} >> "$WRAP_LOG" 2>&1
