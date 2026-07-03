#!/usr/bin/env bash
# systemd wrapper (tsmom-rebalance.service): monthly cross-asset TSMOM rebalance on the VM (Alpaca
# paper, .env.tsmom). Idempotent -- fires on the 2nd/3rd/4th of the month; the first run rebalances,
# the rest find "already at target". Runs entirely on the VM = reliable (no laptop/TWS dependency).
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
LOG_DIR="$ROOT/logs"; mkdir -p "$LOG_DIR"
STAMP="$(TZ=America/New_York date +%Y-%m)"
PY="$ROOT/.venv/bin/python"
{
  echo "=== tsmom-rebalance $(date '+%F %T %z') ==="
  git pull --ff-only --no-edit >/dev/null 2>&1 || true
  "$PY" "$ROOT/tsmom/rebalance.py"
  echo "=== tsmom-rebalance end $(date '+%F %T %z')  (exit=$?) ==="
} >> "$LOG_DIR/tsmom_${STAMP}.log" 2>&1
