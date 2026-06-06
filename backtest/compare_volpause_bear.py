"""Stress-test the 'down-mornings are fine for ORB longs' finding through 2022's BEAR.

compare_volpause.py found (on 2024-2026, a mostly-calm/bull sample) that a PROACTIVE
same-day pause (sit out when SPY is down by 09:45) BACKFIRES — down-mornings flagged
MORE-profitable days because the longs that break out up against a falling tape are
relative-strength survivors. The honest caveat: that sample had no deep sustained
bear, where correlations go to 1 and relative strength fails.

This re-runs the SAME comparison on the full 2022 bear market (SPX ~-25%). The
decisive number: avg $/day on same-day-flagged (down-morning) days. If it's clearly
NEGATIVE in 2022 (vs +$57 in 2024-26), the finding FLIPS — a proactive/crisis gate
would be warranted after all. If it stays positive/neutral, the finding holds even
in a bear. Same trailing config, net of costs, with an OOS split inside 2022.

Run (fetches 2022 universe bars on first run, a few min; then cached):
    .venv/Scripts/python.exe backtest/compare_volpause_bear.py
"""
from __future__ import annotations

import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetCalendarRequest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from strategies.orb import Params  # noqa: E402
from backtest.run_orb import run_backtest, to_et, load_env, STARTING_EQUITY  # noqa: E402
from backtest.universe_scan import UNIVERSE, fetch_chunked  # noqa: E402
from backtest.universe_portfolio import portfolio  # noqa: E402
from backtest.compare_norefill_trend import trend_eligibility, apply_filter  # noqa: E402
from backtest.compare_exits import bucket, reexit, POLICIES, EOD  # noqa: E402
from backtest.compare_volpause import (  # noqa: E402
    prior_vol_flags, sameday_flags, series, three, HEAD, prow, RISK, CAP,
)
from datetime import time as dtime  # noqa: E402

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
CACHE = ROOT / "backtest" / ".bars_cache_univ_2022.pkl"


def load_2022():
    if CACHE.exists():
        print(f"loading cached 2022 bars from {CACHE.name} ...")
        d = pickle.load(open(CACHE, "rb"))
        return d["bars"], d["days"], d["closes"]
    load_env()
    key, sec = os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"]
    dc = StockHistoricalDataClient(key, sec)
    tc = TradingClient(key, sec, paper=True)
    tstart = datetime(2022, 1, 1, tzinfo=ET)
    tend = datetime(2022, 12, 31, tzinfo=ET)
    days = [c.date for c in tc.get_calendar(GetCalendarRequest(start=tstart.date(), end=tend.date()))]
    print(f"fetching 2022 minute bars ({len(days)} sessions, {len(UNIVERSE)} names)...")
    bars = to_et(fetch_chunked(dc, UNIVERSE, tstart, tend))
    print("fetching daily closes 2021-2022 (trend filter + vol regime)...")
    dstart = datetime(2021, 1, 1, tzinfo=ET)
    syms = sorted(set(UNIVERSE) | {"SPY"})
    frames = []
    for i in range(0, len(syms), 20):
        grp = syms[i:i + 20]
        df = dc.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=grp, timeframe=TimeFrame.Day,
            start=dstart.astimezone(UTC), end=tend.astimezone(UTC),
            feed=DataFeed.IEX, adjustment=Adjustment.ALL)).df
        if not df.empty:
            frames.append(df["close"].unstack(level=0))
    closes = pd.concat(frames, axis=1)
    closes.index = pd.to_datetime(closes.index).tz_convert(ET).normalize().tz_localize(None)
    closes = closes.sort_index()
    pickle.dump({"bars": bars, "days": days, "closes": closes}, open(CACHE, "wb"))
    return bars, days, closes


def main():
    all_bars, days, closes = load_2022()
    present = sorted(all_bars.index.get_level_values(0).unique())
    mid = sorted(days)[len(days) // 2]
    params = Params(or_minutes=15, target_r=2.0, risk_per_trade=100.0, max_position_pct=0.25,
                    max_position_dollars=10_000.0, no_entry_after_time=dtime(11, 30))
    trades, _ = run_backtest(all_bars, days, present, params, STARTING_EQUITY)

    elig = trend_eligibility(closes, present, days)
    buckets = bucket(all_bars, present)
    tz = all_bars.index.get_level_values(1).tz
    eod_ns = {d: pd.Timestamp.combine(d, EOD).tz_localize(tz).value for d in days}
    trail = apply_filter(reexit(trades, buckets, POLICIES["trail_1R"], eod_ns), elig)
    taken = portfolio(trail, CAP)

    prior = prior_vol_flags(closes, days)
    same = sameday_flags(all_bars, days)
    one = {d: 1.0 for d in days}
    full = series(taken, days, one)
    def avg_on(flag, val):
        sel = [full[d] for d in sorted(days) if flag[d] == val]
        return (sum(sel) / len(sel)) if sel else 0.0

    print(f"\n=== 2022 BEAR: {len(present)} names, {len(days)} sessions, OOS split {mid} ===")
    spy = closes["SPY"].dropna()
    yr = spy[(spy.index >= pd.Timestamp(2022, 1, 1)) & (spy.index <= pd.Timestamp(2022, 12, 31))]
    print(f"  SPY 2022: {yr.iloc[0]:.0f} -> {yr.iloc[-1]:.0f}  ({(yr.iloc[-1]/yr.iloc[0]-1)*100:+.1f}% on the year)")
    print(f"  PRIOR-vol gate: flags {sum(prior.values())} days | avg $/day flagged {avg_on(prior,True):+.0f} vs calm {avg_on(prior,False):+.0f}")
    print(f"  SAME-day gate : flags {sum(same.values())} days | avg $/day flagged {avg_on(same,True):+.0f} vs calm {avg_on(same,False):+.0f}")
    print("  >>> If SAME-day flagged avg is NEGATIVE, the proactive-pause finding FLIPS in a bear. <<<")
    print(HEAD); print("-" * len(HEAD))
    prow("normal (no filter)", *three(taken, days, mid, one))
    prow("prior_vol pause", *three(taken, days, mid, {d: (0.0 if prior[d] else 1.0) for d in days}))
    prow("prior_vol half", *three(taken, days, mid, {d: (0.5 if prior[d] else 1.0) for d in days}))
    prow("sameday pause", *three(taken, days, mid, {d: (0.0 if same[d] else 1.0) for d in days}))
    prow("sameday half", *three(taken, days, mid, {d: (0.5 if same[d] else 1.0) for d in days}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
