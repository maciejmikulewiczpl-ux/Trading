"""Send an ntfy phone push with today's news-edge picks summary.

Called by scripts/run_news_scan_headless.ps1 AFTER the scan+push (deterministic — not the
headless agent, so no outward-facing gating). Reads the logged picks and pushes the
frontrunner/avoid list to your phone via the same NTFY_TOPIC the trading bot uses.

Run:
    .venv/Scripts/python.exe experiments/news_edge/notify_picks.py [YYYY-MM-DD]
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from backtest.run_orb import load_env  # noqa: E402
from live.notify import notify  # noqa: E402

ET = ZoneInfo("America/New_York")


def main() -> int:
    load_env()
    date = sys.argv[1] if len(sys.argv) > 1 else datetime.now(ET).date().isoformat()
    f = ROOT / "experiments" / "news_edge" / "picks" / f"{date}.json"
    if not f.exists():
        print(f"no picks file for {date}; nothing to push")
        return 0
    picks = json.load(open(f)).get("picks", [])
    pos = [p for p in picks if p.get("signal", 0) > 0]
    neg = [p["symbol"] for p in picks if p.get("signal", 0) < 0]
    lines = []
    if pos:
        lines.append("LONG: " + " ".join(p["symbol"] for p in pos))
    if neg:
        lines.append("AVOID: " + " ".join(neg))
    lines.append("")
    for p in pos:   # one line of reason per frontrunner (what the VM bot trades)
        lines.append(f"+ {p['symbol']}: {str(p.get('reason',''))[:64]}")
    msg = "\n".join(lines) if lines else "no clear picks today"
    ok = notify(msg, title=f"News-edge {date}: {len(pos)} long, {len(neg)} avoid",
                priority=3, tags=["newspaper"])
    print("ntfy push sent" if ok else "ntfy NOT sent (NTFY_TOPIC unset in .env?)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
