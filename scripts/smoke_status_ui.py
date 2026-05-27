"""Headless smoke test for live/status_ui.py.

Verifies the Tk-on-dedicated-thread fix:
  - start() spawns the Tk thread; window initialized hidden
  - simulating 'Show status...' click -> window becomes visible
  - simulating X button (WM_DELETE_WINDOW) -> window hidden, not destroyed
  - re-show after hide works
  - stop() tears down cleanly

Run:
    .venv/Scripts/python.exe scripts/smoke_status_ui.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live.status_ui import StatusController, _AVAILABLE

if not _AVAILABLE:
    print("FAIL: pystray/Pillow not installed")
    sys.exit(1)
print(f"pystray + Pillow available: {_AVAILABLE}")


def fake_snapshot():
    return {
        "phase": "smoke test",
        "equity": 100000.0,
        "day_pnl": -50.0,
        "halted": False,
        "last_update": "12:34:56",
        "symbols": {
            "SPY": {"or_high": 750.0, "or_low": 749.0, "or_locked": True,
                    "entered": False, "status": "watching",
                    "entry_price": None, "stop_price": None,
                    "target_price": None, "shares": None},
        },
    }


def main() -> int:
    ui = StatusController(get_status=fake_snapshot)
    print("starting...")
    ok = ui.start()
    print(f"start() returned: {ok}")
    if not ok:
        return 1

    time.sleep(1.0)
    print(f"_root_ready: {ui._root_ready.is_set()}  _root is set: {ui._root is not None}")
    print(f"initial window state: {ui._root.state()!r}")
    if ui._root.state() != "withdrawn":
        print("FAIL: window did not start hidden")
        ui.stop()
        return 1

    print("simulating 'Show status...' click...")
    ui._on_show()
    time.sleep(2.0)
    print(f"window state after show: {ui._root.state()!r}")
    if ui._root.state() != "normal":
        print("FAIL: window did not show")
        ui.stop()
        return 1
    print("PASS: window shown")

    print("simulating X button (WM_DELETE_WINDOW)...")
    ui._root.after(0, ui._hide_window)
    time.sleep(1.5)
    print(f"window state after hide: {ui._root.state()!r}")
    if ui._root.state() != "withdrawn":
        print("FAIL: window did not hide")
        ui.stop()
        return 1
    print("PASS: window hidden via X button")

    print("simulating Show again (re-show after hide)...")
    ui._on_show()
    time.sleep(1.5)
    print(f"window state after reshow: {ui._root.state()!r}")
    if ui._root.state() != "normal":
        print("FAIL: window did not re-show")
        ui.stop()
        return 1
    print("PASS: re-show after hide works")

    print("set_state cycle...")
    ui.set_state("warning")
    time.sleep(0.3)
    ui.set_state("halted")
    time.sleep(0.3)

    print("stop()...")
    ui.stop()
    time.sleep(1.0)
    print("PASS: all checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
