#!/usr/bin/env bash
# systemd wrapper (mes-bot.service): run ONE pass of the MES paper bot on the VM. Fired every ~5 min
# during RTH by mes-bot.timer. Each pass is stateless (state in futures/state.json): builds the 09:30-
# 09:45 ET opening range, applies the tight-OR-day gate, enters on a breakout with a native trailing
# stop, and flattens at 15:55 ET. Requires IB Gateway logged into the PAPER account (a separate
# always-on process) with the API enabled. Keeps DRY_RUN per .env.futures.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
STAMP="$(TZ=America/New_York date +%F)"
WRAP_LOG="$LOG_DIR/task_mes_${STAMP}.log"
# ib_async + pandas + yfinance env. Adjust if the VM uses a dedicated futures venv.
PY="$ROOT/.venv/bin/python"

{
  echo "=== mes-bot pass $(date '+%F %T %z') ==="
  git pull --ff-only --no-edit >/dev/null 2>&1 || true
  "$PY" "$ROOT/futures/run_mes_bot.py"
  echo "=== mes-bot end $(date '+%F %T %z')  (exit=$?) ==="
} >> "$WRAP_LOG" 2>&1
