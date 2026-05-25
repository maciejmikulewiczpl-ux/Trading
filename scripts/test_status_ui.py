"""Standalone test for the system-tray status indicator.

Runs the tray icon for ~60 seconds with a fake state that mutates over time
so you can verify:
  1. A green circle appears in your Windows system tray (bottom-right notification area).
  2. Right-clicking the icon shows a menu with "Show status..." and "Hide tray icon".
  3. Clicking "Show status..." opens a dark window with phase, equity, per-symbol info.
  4. The window auto-refreshes every 2 seconds and reflects the changing fake state.
  5. Closing the window leaves the tray icon running until the test exits (60s).

Run:
    .\\.venv\\Scripts\\python.exe scripts\\test_status_ui.py
"""
from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from live.status_ui import StatusController  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("status_ui_test")

WATCHLIST = ["SPY", "QQQ", "AAPL", "NVDA", "TSLA"]


class FakeRun:
    """Holds mutable state that walks through phases as if a real session were in progress."""
    def __init__(self):
        self.t0 = time.time()
        self.equity = 100_000.0

    def snapshot(self) -> dict:
        elapsed = time.time() - self.t0
        # Phase walks through: pre-market -> OR build -> hunting -> EOD over the 60-second test
        if elapsed < 10:
            phase = "pre-market (waiting for 9:30 ET)"
            sym_status = "(market not open yet)"
            or_locked = False
            entered = False
        elif elapsed < 25:
            phase = "building opening range"
            sym_status = "building OR..."
            or_locked = False
            entered = False
        elif elapsed < 40:
            phase = "hunting for breakouts"
            sym_status = "watching for breakout"
            or_locked = True
            entered = False
        elif elapsed < 55:
            phase = "hunting for breakouts"
            or_locked = True
            entered = True
        else:
            phase = "EOD flatten / done"
            or_locked = True
            entered = True

        snap = {
            "phase": phase,
            "equity": self.equity,
            "day_pnl": +12.34 if elapsed > 50 else 0.0,
            "halted": False,
            "last_update": datetime.now().strftime("%H:%M:%S"),
            "symbols": {},
        }
        # Build per-symbol fake state
        prices = {"SPY": 593.21, "QQQ": 511.40, "AAPL": 207.83, "NVDA": 143.12, "TSLA": 282.55}
        for sym in WATCHLIST:
            p = prices[sym]
            sym_data = {
                "or_high": round(p * 1.003, 2) if or_locked else None,
                "or_low":  round(p * 0.997, 2) if or_locked else None,
                "or_locked": or_locked,
                "entered": entered and sym in ("AAPL", "NVDA"),
            }
            if entered and sym == "AAPL":
                sym_data["status"] = "ENTERED (bracket live on Alpaca)"
                sym_data["entry_price"] = round(p * 1.004, 2)
                sym_data["stop_price"]  = round(p * 0.997, 2)
                sym_data["target_price"] = round(p * 1.018, 2)
                sym_data["shares"] = 48
            elif entered and sym == "NVDA":
                sym_data["status"] = "ENTERED (bracket live on Alpaca)"
                sym_data["entry_price"] = round(p * 1.005, 2)
                sym_data["stop_price"]  = round(p * 0.992, 2)
                sym_data["target_price"] = round(p * 1.031, 2)
                sym_data["shares"] = 52
            else:
                if elapsed < 10:
                    sym_data["status"] = "(market not open yet)"
                elif elapsed < 25:
                    sym_data["status"] = "building OR..."
                else:
                    sym_data["status"] = "watching for breakout"
            snap["symbols"][sym] = sym_data
        return snap


def main():
    log.info("Starting status UI test (60 seconds). Look for a GREEN circle in your system tray.")
    fake = FakeRun()
    ui = StatusController(get_status=fake.snapshot)
    if not ui.start():
        log.error("UI failed to start. pystray/Pillow probably missing — check the .venv install.")
        return 1
    log.info("Tray icon started. Right-click or click it to open the status window.")
    log.info("State will walk: pre-market -> OR-build -> hunting -> ENTERED (AAPL+NVDA) -> EOD")
    try:
        time.sleep(60)
    except KeyboardInterrupt:
        pass
    log.info("Test complete; stopping tray icon.")
    ui.stop()
    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
