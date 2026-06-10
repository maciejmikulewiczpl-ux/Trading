"""Operational gate for the universe expansion: does a ~150-name watchlist fit the
live runner's 10s poll cycle?

Replicates live/paper_orb.py fetch_today_bars EXACTLY (single multi-symbol IEX
minute-bar request, session open -> now) and times it for the current 100-name HAND
list vs the expanded list (HAND + the floor-1.4% high-vol adds from pit_expand).
Run during/after a session so the response is full-size (worst case ~390min x 150
symbols ~ 58k rows). Also dumps the concrete add list for the ship-spec.

PASS bar: expanded fetch comfortably inside the cycle (median <5s; the runner also
does exits/heartbeat work in the same loop).

Run:  .venv/Scripts/python.exe backtest/bench_poll_capacity.py
"""
from __future__ import annotations

import os
import pickle
import statistics
import sys
import time as clock
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.run_orb import load_env  # noqa: E402
from backtest.compare_selection import _tday  # noqa: E402
from backtest.compare_or_range_realcost import or_pct  # noqa: E402
from backtest.universe_scan import UNIVERSE  # noqa: E402

from alpaca.data.enums import DataFeed  # noqa: E402
from alpaca.data.historical import StockHistoricalDataClient  # noqa: E402
from alpaca.data.requests import StockBarsRequest  # noqa: E402
from alpaca.data.timeframe import TimeFrame  # noqa: E402

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
WINDOW = 730
FLOOR = 1.4
BLOCK = {"TQQQ", "SQQQ", "SOXL", "SOXS", "TZA", "TNA", "SPXL", "SPXS", "UPRO",
         "UVXY", "SVXY", "TMF", "TMV", "YINN", "FNGU", "BOIL", "UCO",
         "MSTR", "IBIT", "ETHA", "BITO", "BMNR", "CRCL", "CRWV", "MARA", "RIOT"}
REPS = 3


def added_names() -> list[str]:
    """Recompute the floor-1.4% add set from cache (same rule as pit_expand)."""
    import bisect
    blob = pickle.load(open(ROOT / "backtest" / f".pit_trailtrades_{WINDOW}d.pkl", "rb"))
    members = pickle.load(open(ROOT / "backtest" / f".pit_members_{WINDOW}d.pkl", "rb"))["members"]
    closes = pickle.load(open(ROOT / "backtest" / f".pit_daily_{WINDOW}d.pkl", "rb"))["close"]
    month_map = {(pd.Timestamp(k).year, pd.Timestamp(k).month): set(v) for k, v in members.items()}
    rv = closes.pct_change().rolling(20).std() * 100
    rv.index = [d.date() for d in rv.index]
    rv_idx = list(rv.index)
    hand = set(UNIVERSE)
    out = set()
    for syms in blob.values():
        for t in syms:
            s = t.symbol
            if s in hand or s in BLOCK or s in out or or_pct(t) > 0.5:
                continue
            d = _tday(t)
            if (d.year, d.month) not in month_map or s not in month_map[(d.year, d.month)]:
                continue
            if s not in rv.columns:
                continue
            i = bisect.bisect_left(rv_idx, d)
            if i > 0:
                v = rv[s].iloc[i - 1]
                if not pd.isna(v) and float(v) >= FLOOR:
                    out.add(s)
    return sorted(out)


def fetch_once(dc, symbols, day) -> tuple[float, int]:
    start_et = datetime.combine(day, time(9, 30), tzinfo=ET)
    end_et = datetime.now(ET)
    t0 = clock.perf_counter()
    req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame.Minute,
                           start=start_et.astimezone(UTC), end=end_et.astimezone(UTC),
                           feed=DataFeed.IEX)
    df = dc.get_stock_bars(req).df
    dt = clock.perf_counter() - t0
    return dt, len(df)


def main() -> int:
    load_env()
    dc = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    adds = added_names()
    expanded = sorted(set(UNIVERSE) | set(adds))
    print(f"HAND {len(UNIVERSE)} names | adds {len(adds)} | expanded {len(expanded)}")
    print(f"ADD LIST: {', '.join(adds)}")

    # use today if a weekday session has bars; else walk back to the last weekday
    day = datetime.now(ET).date()
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    print(f"\nTiming full-session fetch for {day} (runner-identical request), {REPS} reps:")
    for label, syms in (("HAND-100", sorted(UNIVERSE)), ("EXPANDED-150", expanded)):
        times, rows = [], 0
        for _ in range(REPS):
            dt, n = fetch_once(dc, syms, day)
            times.append(dt)
            rows = max(rows, n)
        print(f"  {label:<14} rows={rows:>7,}  times: " +
              ", ".join(f"{t:.2f}s" for t in times) +
              f"   median {statistics.median(times):.2f}s")
    print("\nPASS bar: EXPANDED median < 5s (10s poll cycle minus exits/heartbeat headroom).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
