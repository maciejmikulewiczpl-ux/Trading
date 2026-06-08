"""Dig #2 (memory-frugal): build tier-2 trailing trades in BATCHES (7.9GB-RAM safe).

A combined ~250-name backtest won't fit in this machine's RAM, so we never hold the full
universe. For each small batch of tier-2 names: fetch its minute+daily bars, run the ORB
backtest, re-sim the trailing exit, apply the trend filter, keep only the resulting TRADES
(tiny), and discard the bars. Accumulate to .tier2_trail_{w}d.pkl. compare_tightOR_universe
then merges these with tier-1 trades at the trade level (no big bars) and prices tiered slip.

Run AFTER the param sweep finishes (don't run two memory-heavy jobs at once):
    .venv/Scripts/python.exe backtest/build_tier2_trades.py
"""
from __future__ import annotations

import gc
import os
import pickle
import sys
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from strategies.orb import Params  # noqa: E402
from backtest.run_orb import run_backtest, load_env, STARTING_EQUITY  # noqa: E402
from backtest.universe_scan import UNIVERSE, fetch_chunked  # noqa: E402
from backtest.compare_norefill_trend import fetch_daily_closes, trend_eligibility, apply_filter, DAILY_BUFFER_DAYS  # noqa: E402
from backtest.compare_exits import bucket, reexit, POLICIES, EOD  # noqa: E402
from backtest.fetch_universe_expanded import EXPANSION  # noqa: E402

import pandas as pd  # noqa: E402
from alpaca.data.historical import StockHistoricalDataClient  # noqa: E402

ET = ZoneInfo("America/New_York")
WINDOWS = [730, 180]
BATCH = 30
PARAMS = Params(or_minutes=15, target_r=2.0, risk_per_trade=100.0, max_position_pct=0.25,
                max_position_dollars=10_000.0, no_entry_after_time=time(11, 30))


def main():
    load_env()
    dc = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    new = [s for s in EXPANSION if s not in set(UNIVERSE)]

    for w in WINDOWS:
        out_pkl = ROOT / "backtest" / f".tier2_trail_{w}d.pkl"
        if out_pkl.exists():
            print(f"{w}d: {out_pkl.name} exists, skipping.")
            continue
        base = pickle.load(open(ROOT / "backtest" / f".bars_cache_univ_{w}d.pkl", "rb"))
        days = base["days"]
        spy_daily = pickle.load(open(ROOT / "backtest" / f".bars_cache_daily_{w}d.pkl", "rb"))[["SPY"]]
        del base; gc.collect()
        start = datetime.combine(min(days), time(9, 0), ET)
        end = datetime.combine(max(days), time(16, 0), ET) + timedelta(days=1)
        dstart = datetime.combine(min(days), time(0, 0), ET) - timedelta(days=DAILY_BUFFER_DAYS)
        dend = datetime.combine(max(days), time(0, 0), ET) + timedelta(days=1)

        all_trail = []
        for i in range(0, len(new), BATCH):
            grp = new[i:i + BATCH]
            print(f"{w}d batch {i//BATCH + 1}: names {i+1}-{i+len(grp)} of {len(new)} — fetch...", flush=True)
            bars = fetch_chunked(dc, grp, start, end)
            if bars.empty:
                continue
            pres = sorted(bars.index.get_level_values(0).unique())
            closes = fetch_daily_closes(grp, dstart, dend).join(spy_daily, how="outer")
            raw, _ = run_backtest(bars, days, pres, PARAMS, STARTING_EQUITY)
            bk = bucket(bars, pres)
            tz = bars.index.get_level_values(1).tz
            eod_ns = {d: pd.Timestamp.combine(d, EOD).tz_localize(tz).value for d in days}
            elig = trend_eligibility(closes, pres, days)
            trail = apply_filter([t for t in reexit(raw, bk, POLICIES["trail_1R"], eod_ns)
                                  if t.side == "long"], elig)
            all_trail.extend(trail)
            print(f"   -> {len(trail)} trend-filtered trailing trades (cum {len(all_trail)})", flush=True)
            del bars, bk, raw, trail, closes, elig; gc.collect()

        pickle.dump(all_trail, open(out_pkl, "wb"))
        names = sorted({t.symbol for t in all_trail})
        print(f"{w}d: saved {len(all_trail)} tier-2 trailing trades across {len(names)} names -> {out_pkl.name}")
    print("Done. Next: compare_tightOR_universe.py (trade-level merge).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
