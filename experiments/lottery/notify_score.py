"""After the close: score today's lottery board (ret_945_close), backfill prior days'
1d/3d, and push a short hit-rate summary to the phone via ntfy. Run by the LotteryScore
Windows task ~13:10 PT (just after the 15:55 ET measurement close).

Run:
    .venv/Scripts/python.exe experiments/lottery/notify_score.py [YYYY-MM-DD]
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
import experiments.lottery.outcomes as oc  # noqa: E402

ET = ZoneInfo("America/New_York")
PICKS_DIR = Path(__file__).resolve().parent / "picks"


def main() -> int:
    load_env()
    date = sys.argv[1] if len(sys.argv) > 1 else datetime.now(ET).date().isoformat()
    try:
        oc.score_day(date)          # fill ret_945_close for today
    except Exception as e:
        print(f"score_day failed: {e}")
    try:
        oc.backfill(7)              # fill ret_1d/ret_3d for prior days
    except Exception as e:
        print(f"backfill failed: {e}")

    f = PICKS_DIR / f"{date}.json"
    if not f.exists():
        notify(f"Lottery {date}: no board file.", title="Lottery score")
        return 0
    picks = json.load(open(f)).get("picks", [])
    scored = [p for p in picks if p.get("ret_945_close") is not None]
    if not scored:
        notify(f"Lottery {date}: outcomes not available yet.", title="Lottery score")
        return 0

    w1 = sum(1 for p in scored if p["ret_945_close"] >= 5.0)
    rand = [p for p in scored if p.get("basket") == "random"]
    avg = sum(p["ret_945_close"] for p in scored) / len(scored)
    best = max(scored, key=lambda p: p["ret_945_close"])
    lines = [f"{len(scored)} scored (9:45->close).",
             f"avg {avg:+.2f}%  W1(>=+5%) hits: {w1}",
             f"best: {best['symbol']} {best['ret_945_close']:+.1f}%"]
    if rand:
        rw1 = sum(1 for p in rand if p["ret_945_close"] >= 5.0)
        lines.append(f"random basket W1: {rw1}/{len(rand)}")
    msg = "\n".join(lines)
    ok = notify(msg, title=f"Lottery {date} scored", priority=3, tags=["bar_chart"])
    print(msg)
    print("ntfy push sent" if ok else "ntfy NOT sent (NTFY_TOPIC unset?)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
