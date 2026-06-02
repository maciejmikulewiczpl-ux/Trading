#!/usr/bin/env bash
# One-shot Linux install / re-install for the trading runners on a fresh VPS
# (designed for Oracle Cloud Always-Free Ubuntu 22.04+ but works on any
# systemd-based distro with Python 3.11+).
#
# What this does:
#   1. Verifies you ran it AS THE USER who'll own the cron jobs (NOT root).
#   2. Installs OS prereqs (python3-venv, tzdata) via sudo apt.
#   3. Creates .venv (if missing) and pip-installs runtime deps.
#   4. Verifies .env exists and contains the required Alpaca + ntfy keys.
#   5. Makes the bash launchers executable.
#   6. Copies + activates the systemd timers (orb + dualmom).
#   7. Runs a smoke test of paper_orb so we know the wiring is good
#      before the next scheduled fire.
#
# Re-running this script is idempotent: timers are stop->copy->start each time,
# .venv is reused, deps are pip-upgraded.
#
# Usage (on the VM):
#     git clone <your-repo-url> ~/trading
#     scp .env <vm-user>@<vm-ip>:~/trading/.env   # from your laptop, separately
#     bash ~/trading/scripts/install_linux.sh
set -euo pipefail

# ----- 0. preflight -----
if [[ $EUID -eq 0 ]]; then
  echo "ERROR: do not run as root. Run as the user that owns ~/trading." >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USER_NAME="$(id -un)"
HOME_DIR="$HOME"

echo "==> Installing into $ROOT as user '$USER_NAME'"
if [[ "$ROOT" != "$HOME_DIR/trading" ]]; then
  echo "WARN: repo is at $ROOT, not $HOME_DIR/trading."
  echo "      Systemd units assume $HOME_DIR/trading -- they will be patched"
  echo "      to point at $ROOT below."
fi

# ----- 1. OS packages -----
echo "==> Installing OS prereqs (python3-venv, tzdata, ca-certificates)"
sudo apt-get update -y
sudo apt-get install -y python3-venv python3-pip tzdata ca-certificates

# ----- 2. venv + Python deps -----
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "==> Creating $ROOT/.venv"
  python3 -m venv "$ROOT/.venv"
fi
echo "==> Installing/upgrading Python runtime deps"
# pystray + pillow are skipped: the tray icon is Windows-only decoration, the
# Python import is already wrapped in try/except so the runner doesn't need them.
"$ROOT/.venv/bin/pip" install --upgrade pip
"$ROOT/.venv/bin/pip" install --upgrade "alpaca-py>=0.30.0" "pandas>=2.2.0"

# ----- 3. .env sanity check -----
if [[ ! -f "$ROOT/.env" ]]; then
  echo "ERROR: $ROOT/.env not found. scp it from your laptop:" >&2
  echo "       scp .env ${USER_NAME}@<this-vm-ip>:$ROOT/.env" >&2
  exit 1
fi
# Make sure .env isn't world-readable (it contains API secrets).
chmod 600 "$ROOT/.env"
for required in ALPACA_API_KEY ALPACA_SECRET_KEY; do
  if ! grep -q "^${required}=" "$ROOT/.env"; then
    echo "ERROR: $required missing from .env" >&2
    exit 1
  fi
done
if ! grep -q "^DUALMOM_ALPACA_API_KEY=" "$ROOT/.env"; then
  echo "WARN: DUALMOM_ALPACA_API_KEY not set in .env -- dual-momentum runner"
  echo "      will refuse to place live orders (smoke-test mode only)."
fi
if ! grep -q "^NTFY_TOPIC=" "$ROOT/.env"; then
  echo "WARN: NTFY_TOPIC not set -- push notifications will be silent."
fi

# ----- 4. logs dir + executable bits -----
mkdir -p "$ROOT/logs"
chmod +x "$ROOT/scripts/launch_orb.sh" "$ROOT/scripts/launch_dualmom.sh"

# ----- 5. systemd units -----
echo "==> Installing systemd units (sudo)"
SYS_DIR="/etc/systemd/system"
for unit in orb.service orb.timer dualmom.service dualmom.timer status.service; do
  src="$ROOT/scripts/systemd/$unit"
  dst="$SYS_DIR/$unit"
  # Substitute user + path if they don't match the unit's hardcoded defaults
  # ("ubuntu" + "/home/ubuntu/trading"). sed is a no-op when they DO match.
  sudo bash -c "sed -e 's|User=ubuntu|User=${USER_NAME}|g' \
                    -e 's|/home/ubuntu/trading|${ROOT}|g' \
                    '$src' > '$dst'"
done
sudo systemctl daemon-reload

# Stop any currently-running timers so we re-apply cleanly on re-install
sudo systemctl stop orb.timer dualmom.timer 2>/dev/null || true

echo "==> Enabling + starting timers"
sudo systemctl enable --now orb.timer dualmom.timer

# Always-on status web page. enable --now won't restart an already-running
# instance, so restart explicitly to pick up code changes on re-install.
echo "==> Enabling + (re)starting status web page service"
sudo systemctl enable status.service
sudo systemctl restart status.service

# ----- 6. smoke test -----
echo "==> Running paper_orb smoke test (account + data + ntfy)"
set +e
"$ROOT/.venv/bin/python" "$ROOT/live/paper_orb.py" --preflight-only
SMOKE_RC=$?
set -e
if [[ $SMOKE_RC -ne 0 ]]; then
  echo "WARN: smoke test exited $SMOKE_RC -- inspect the output above before relying on the next scheduled fire." >&2
fi

# ----- 7. show what's scheduled -----
echo
echo "==> Next scheduled fires:"
systemctl list-timers --all 'orb.timer' 'dualmom.timer'

echo
echo "Install complete."
echo "  Logs:           tail -F $ROOT/logs/orb_*.log"
echo "  Service status: systemctl status orb.timer dualmom.timer status.service"
echo "  Manual fire:    sudo systemctl start orb.service   # for testing only"
echo
echo "  Status page (bound to 127.0.0.1:8787 on this VM). From your laptop:"
echo "    ssh -i <key> -L 8787:localhost:8787 ${USER_NAME}@<this-vm-ip>"
echo "    then open http://localhost:8787 in your browser."
