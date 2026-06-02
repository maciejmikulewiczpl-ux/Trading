"""Liveness heartbeat — the runner's proof of life, decoupled from trade state.

The ORB runner writes `logs/heartbeat.json` on every loop iteration. The status
server reads it to answer one question: *is the script actually running right
now?* — which is independent of *what trades exist* (that comes from Alpaca,
the authoritative source).

Design rules, mirroring status_ui.py:
  - Pure instrumentation. Every write swallows its own errors; a failed
    heartbeat must never disturb the trading loop.
  - Atomic writes (temp file + os.replace) so a reader never sees a half-
    written file, even mid-write.
  - The runner declares its OWN cadence via `expected_next_by`. The server
    never hardcodes a staleness threshold — it just checks whether that
    deadline has passed. This makes pre-open waits and the 10s poll loop both
    work without special-casing on the reader side.
"""
from __future__ import annotations

import json
import os
import socket
import time as _time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
HEARTBEAT_PATH = ROOT / "logs" / "heartbeat.json"

_HOST = socket.gethostname()
_PID = os.getpid()


def write(expected_next_by: float, *, path: Path = HEARTBEAT_PATH, **fields: Any) -> None:
    """Write a heartbeat. Never raises.

    expected_next_by: epoch seconds by which the NEXT beat is due. The reader
    treats `now > expected_next_by` (with grace) as "no fresh beat".
    fields: arbitrary extra context (phase, equity, halted, ...) for display.
    """
    try:
        now = _time.time()
        rec = {
            "ts": now,
            "ts_iso": datetime.now().astimezone().isoformat(timespec="seconds"),
            "expected_next_by": expected_next_by,
            "pid": _PID,
            "host": _HOST,
            **fields,
        }
        path.parent.mkdir(exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rec), encoding="utf-8")
        os.replace(tmp, path)  # atomic on Windows + POSIX
    except Exception:
        pass  # heartbeat is decoration; never disturb the trading loop


def read(path: Path = HEARTBEAT_PATH) -> Optional[dict]:
    """Read the latest heartbeat, or None if missing/unreadable."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
