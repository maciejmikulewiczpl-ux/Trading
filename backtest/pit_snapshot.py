"""Print TODAY's qualifying high-vol expansion names (the live watchlist adds).

Rule (validated in pit_expand.py, no-ETF arm): symbol qualifies if
  (a) in the CURRENT month's PIT top-100 by trailing-63d dollar volume,
  (b) latest 20d realized vol >= 1.4%,
  (c) a single name (not ETF / leveraged / crypto proxy),
  (d) not already in the hand watchlist.
Refresh monthly-to-quarterly (membership churns ~18%/mo).

Run:  .venv/Scripts/python.exe backtest/pit_snapshot.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.universe_scan import UNIVERSE  # noqa: E402
from backtest.pit_expand import BLOCK, ETFS  # noqa: E402

WINDOW = 730
FLOOR = 1.4


def main() -> int:
    members = pickle.load(open(ROOT / "backtest" / f".pit_members_{WINDOW}d.pkl", "rb"))["members"]
    closes = pickle.load(open(ROOT / "backtest" / f".pit_daily_{WINDOW}d.pkl", "rb"))["close"]
    latest_month = max(members)
    top = set(members[latest_month])
    rv = (closes.pct_change().rolling(20).std() * 100).iloc[-1]
    hand = set(UNIVERSE)
    adds = sorted(
        s for s in top
        if s not in hand and s not in BLOCK and s not in ETFS
        and s in rv.index and not pd.isna(rv[s]) and float(rv[s]) >= FLOOR
    )
    print(f"PIT month: {latest_month} | vol as of: {closes.index[-1].date()}")
    print(f"QUALIFYING ADDS ({len(adds)}):")
    for s in adds:
        print(f"  {s:<6} vol {float(rv[s]):.2f}%")
    print('\npython list:\n' + ", ".join(f'"{s}"' for s in adds))
    return 0


if __name__ == "__main__":
    sys.exit(main())
