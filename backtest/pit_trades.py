"""Point-in-time universe test, step 3: stream minute bars + generate trailing trades.

Reads the union from pit_universe.py, fetches 1-min IEX bars ONE SMALL CHUNK at a
time (memory-safe on this 8GB box — never holds all names at once, which is what
OOM'd the tier-2 build), runs the live close-confirm ORB entry + trail-1R exit per
symbol, and caches ONLY the resulting trades. Fetches union ∪ hand-picked (~210
names) so step 4 (pit_compare.py) can compare both universes from one identical pass
without reloading the 905MB hand-picked minute cache.

Incremental + resumable: after each chunk it rewrites .pit_trailtrades_730d.pkl
({symbol: [Trade]}); a re-run skips symbols already present.

Run (long, background — minute fetch for ~210 names):
    .venv/Scripts/python.exe backtest/pit_trades.py
"""
from __future__ import annotations

import os
import pickle
import socket
import sys
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

# Belt-and-suspenders: cap any single network op so a dead socket (e.g. the laptop
# sleeping mid-fetch) raises instead of hanging forever. A timed-out chunk is caught
# per-chunk and retried on the next resumable run.
socket.setdefaulttimeout(120)

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from strategies.orb import Params  # noqa: E402
from backtest.run_orb import get_trading_days, load_env, run_backtest, to_et  # noqa: E402
from backtest.compare_exits import bucket, reexit, POLICIES, EOD  # noqa: E402
from backtest.universe_scan import UNIVERSE  # noqa: E402

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

WINDOW = 730
FETCH_CHUNK = 6                # names per request — small, memory-safe
STARTING_EQUITY = 100_000.0
PARAMS = Params(or_minutes=15, target_r=2.0, risk_per_trade=100.0,
                max_position_pct=0.25, max_position_dollars=10_000.0,
                no_entry_after_time=time(11, 30))
TRADES_CACHE = ROOT / "backtest" / f".pit_trailtrades_{WINDOW}d.pkl"


def main() -> int:
    load_env()
    key, sec = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    if not key or not sec:
        print("ERROR: API keys missing", file=sys.stderr)
        return 1
    dc = StockHistoricalDataClient(key, sec)
    tc = TradingClient(key, sec, paper=True)

    members_blob = pickle.load(open(ROOT / "backtest" / f".pit_members_{WINDOW}d.pkl", "rb"))
    union = members_blob["union"]
    targets = sorted(set(union) | set(UNIVERSE))
    print(f"Target names: {len(targets)} (union {len(union)} + hand {len(UNIVERSE)})")

    end = datetime.now(tz=ET)
    start = end - timedelta(days=WINDOW)
    trading_days = get_trading_days(tc, start, end)
    eod_ns = {d: pd.Timestamp.combine(d, EOD).tz_localize(ET).value for d in trading_days}

    done = {}
    if TRADES_CACHE.exists():
        done = pickle.load(open(TRADES_CACHE, "rb"))
        print(f"Resuming: {len(done)} symbols already cached.")
    todo = [s for s in targets if s not in done]
    print(f"To fetch: {len(todo)} symbols, chunk {FETCH_CHUNK}.\n")

    for i in range(0, len(todo), FETCH_CHUNK):
        grp = todo[i:i + FETCH_CHUNK]
        print(f"  {i + 1}-{i + len(grp)} of {len(todo)}: {','.join(grp)} ...", flush=True)
        try:
            req = StockBarsRequest(symbol_or_symbols=grp, timeframe=TimeFrame.Minute,
                                   start=start.astimezone(UTC), end=end.astimezone(UTC),
                                   feed=DataFeed.IEX)
            raw = dc.get_stock_bars(req).df
        except Exception as e:
            print(f"    fetch failed ({e}); marking empty, will retry next run", flush=True)
            continue
        if raw.empty:
            for s in grp:
                done[s] = []
            pickle.dump(done, open(TRADES_CACHE, "wb"))
            continue
        bars = to_et(raw)
        present = set(bars.index.get_level_values(0).unique())
        for s in grp:
            if s not in present:
                done[s] = []
                continue
            mini = bars.loc[[s]]
            base, _ = run_backtest(mini, trading_days, [s], PARAMS, STARTING_EQUITY)
            if not base:
                done[s] = []
                continue
            bk = bucket(mini, [s])
            trail = reexit(base, bk, POLICIES["trail_1R"], eod_ns)
            done[s] = trail
            del mini, bk
        del raw, bars
        pickle.dump(done, open(TRADES_CACHE, "wb"))

    total = sum(len(v) for v in done.values())
    nonempty = sum(1 for v in done.values() if v)
    print(f"\nDONE. {nonempty}/{len(done)} symbols produced trades; {total:,} trailing trades total.")
    print(f"Cached -> {TRADES_CACHE.name}. Next: pit_compare.py for the verdict.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
