"""Multi-timeframe trend-filter A/B for ORB longs (the strongest untested idea).

Within-day filters (RVOL, volume expansion, gap) all failed or didn't survive
OOS. The category we hadn't tested: bring HIGHER-TIMEFRAME context (daily trend)
into the intraday breakout decision. Two factors with strong academic basis:

  - Daily trend alignment: take the long only if the stock's close on the prior
    trading day was above its 50d or 200d SMA (already in a daily uptrend).
  - Cross-sectional relative strength: take it only if the stock's trailing
    20-day return beat SPY's (it's outperforming the index).

Both pull information from OUTSIDE the breakout bar -- which is precisely what
within-day filters lacked. The combination ("ORB on strong trend-aligned names")
is the multi-timeframe momentum construct that academic / professional momentum
strategies use, and is what makes the trend-following bedrock work.

Method: post-hoc filter on the broad-universe trade list (same baseline as
compare_volexp.py: 11:30 ET cutoff). Daily bars from Alpaca (TimeFrame.Day,
adjustment=ALL). Signal computed AS-OF the prior trading day's close -- no
lookahead. OOS split (first / second half) for survivors.

Run:
    .venv\\Scripts\\python.exe backtest\\compare_trend_filter.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from strategies.orb import Params  # noqa: E402
from backtest.run_orb import (  # noqa: E402
    LOOKBACK_DAYS, STARTING_EQUITY, get_trading_days, load_env, run_backtest, to_et,
)
from backtest.universe_scan import UNIVERSE, fetch_chunked  # noqa: E402

ET = ZoneInfo("America/New_York")

# SMA windows we want available on the prior-day close.
SMA_LONG = 200
SMA_SHORT = 50
RET_LOOKBACK = 20


def fetch_daily(syms, start, end, key, sec):
    """Daily adjusted closes from Alpaca for the universe + SPY (chunked)."""
    dc = StockHistoricalDataClient(key, sec)
    all_syms = sorted(set(syms) | {"SPY"})
    frames = []
    for i in range(0, len(all_syms), 15):
        grp = all_syms[i:i + 15]
        print(f"  daily {i + 1}-{i + len(grp)} of {len(all_syms)} ...", flush=True)
        req = StockBarsRequest(
            symbol_or_symbols=grp, timeframe=TimeFrame.Day,
            start=start, end=end, feed=DataFeed.IEX, adjustment=Adjustment.ALL,
        )
        df = dc.get_stock_bars(req).df
        if not df.empty:
            frames.append(df)
    raw = pd.concat(frames) if frames else pd.DataFrame()
    if raw.empty:
        return raw
    closes = raw["close"].unstack(level=0)
    # Naive ET date index (midnight, no tz) so it lines up with naive
    # pd.Timestamp(t.entry_time.date()) lookups below.
    closes.index = (pd.to_datetime(closes.index)
                    .tz_convert("America/New_York").normalize().tz_localize(None))
    return closes.sort_index()


def attach_factors(trades, daily):
    """For each trade, look up prior-day close + SMAs + 20d returns. Returns
    dict id(trade) -> {above_50, above_200, rel_strong} or omits trades with
    insufficient daily history (drops them from the comparable base)."""
    sma50 = daily.rolling(SMA_SHORT).mean()
    sma200 = daily.rolling(SMA_LONG).mean()
    ret20 = daily.pct_change(RET_LOOKBACK)
    out = {}
    for t in trades:
        sym = t.symbol
        if sym not in daily.columns or "SPY" not in daily.columns:
            continue
        # Last daily close STRICTLY before the trade's entry day (no lookahead).
        entry_day = pd.Timestamp(t.entry_time.date())
        try:
            sub = daily[sym].loc[:entry_day - pd.Timedelta(days=1)].dropna()
            if sub.empty:
                continue
            prior = sub.index[-1]
            prior_close = float(sub.iloc[-1])
            s50 = float(sma50[sym].loc[prior])
            s200 = float(sma200[sym].loc[prior])
            r20 = float(ret20[sym].loc[prior])
            spy_r20 = float(ret20["SPY"].loc[prior])
        except Exception:
            continue
        if any(pd.isna(x) for x in (s50, s200, r20, spy_r20)):
            continue
        out[id(t)] = {
            "above_50": prior_close > s50,
            "above_200": prior_close > s200,
            "rel_strong": r20 > spy_r20,
        }
    return out


def stats(trades):
    if not trades:
        return {"n": 0}
    df = pd.DataFrame([{"r": t.pnl_r, "d": t.pnl_dollars,
                        "reason": t.exit_reason, "ex": t.exit_time}
                       for t in trades])
    sdf = df.sort_values("ex")
    eq = STARTING_EQUITY + sdf["d"].cumsum()
    dd = (eq - eq.cummax()).min()
    return {
        "n": len(df),
        "win": (df["r"] > 0).mean() * 100,
        "avg_r": df["r"].mean(),
        "sum_r": df["r"].sum(),
        "pnl": df["d"].sum(),
        "dd": dd,
        "tgt": int((df["reason"] == "target").sum()),
        "stop": int((df["reason"] == "stop").sum()),
        "eod": int((df["reason"] == "eod").sum()),
    }


def print_row(name, s):
    if s["n"] == 0:
        print(f"  {name:<32}  (no trades)")
        return
    tse = f"{s['tgt']}/{s['stop']}/{s['eod']}"
    print(f"  {name:<32}{s['n']:>5}{s['win']:>6.1f}%{s['avg_r']:>+9.4f}"
          f"{s['sum_r']:>+8.1f}{tse:>14}${s['pnl']:>+10,.0f}${s['dd']:>+11,.0f}")


def main() -> int:
    load_env()
    key, sec = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    if not (key and sec):
        print("ERROR: API keys missing", file=sys.stderr)
        return 1

    print(f"Universe: {len(UNIVERSE)} names")
    print("Fetching intraday minute bars for the broad universe...")
    dc = StockHistoricalDataClient(key, sec)
    tc = TradingClient(key, sec, paper=True)
    end = datetime.now(tz=ET)
    start = end - timedelta(days=LOOKBACK_DAYS)
    trading_days = get_trading_days(tc, start, end)
    raw = fetch_chunked(dc, UNIVERSE, start, end)
    if raw.empty:
        print("ERROR: no intraday bars", file=sys.stderr)
        return 1
    intraday = to_et(raw)
    present = sorted(intraday.index.get_level_values(0).unique())
    print(f"Intraday: {len(intraday):,} rows x {len(present)} symbols; "
          f"{len(trading_days)} sessions")
    print("Running baseline (11:30 cutoff) on the broad universe...")
    base = Params(or_minutes=15, target_r=2.0, risk_per_trade=100.0,
                  max_position_pct=0.25, max_position_dollars=10_000.0,
                  no_entry_after_time=time(11, 30))
    all_trades, _ = run_backtest(intraday, trading_days, present, base, STARTING_EQUITY)
    print(f"Baseline broad-universe trades: {len(all_trades)}\n")

    # Daily bars need ~SMA_LONG sessions of warmup before backtest start.
    intraday_start = min(trading_days)
    daily_start = intraday_start - timedelta(days=int(SMA_LONG * 1.6))
    daily_end = max(trading_days) + timedelta(days=1)
    print(f"Fetching daily bars {daily_start} -> {daily_end}...")
    daily = fetch_daily(UNIVERSE, daily_start, daily_end, key, sec)
    print(f"Daily frame: {len(daily)} rows x {daily.shape[1]} symbols\n")

    fmap = attach_factors(all_trades, daily)
    eligible = [t for t in all_trades if id(t) in fmap]
    print(f"Eligible (factors computable): {len(eligible)} of {len(all_trades)}\n")

    configs = [
        ("baseline (all eligible)", lambda t: True),
        ("above 50d SMA", lambda t: fmap[id(t)]["above_50"]),
        ("above 200d SMA", lambda t: fmap[id(t)]["above_200"]),
        ("rel-strong (20d vs SPY)", lambda t: fmap[id(t)]["rel_strong"]),
        ("above 200d AND rel-strong", lambda t: fmap[id(t)]["above_200"] and fmap[id(t)]["rel_strong"]),
        ("above 50d AND rel-strong", lambda t: fmap[id(t)]["above_50"] and fmap[id(t)]["rel_strong"]),
    ]

    print(f"  {'config':<32}{'n':>5}{'win%':>7}{'avg_R':>9}"
          f"{'sumR':>8}{'tgt/stop/eod':>14}{'PnL$':>11}{'maxDD$':>12}")
    print("-" * 100)
    kept_map = {}
    for name, pred in configs:
        kept = [t for t in eligible if pred(t)]
        kept_map[name] = kept
        print_row(name, stats(kept))

    # OOS split on every config
    days_sorted = sorted(set(t.exit_time.date() for t in eligible))
    mid = days_sorted[len(days_sorted) // 2]
    print(f"\nOOS SPLIT at {mid}:")
    print(f"  {'config':<32}{'h1_n':>6}{'h1_avg_R':>11}{'h1_PnL':>11}"
          f"{'h2_n':>6}{'h2_avg_R':>11}{'h2_PnL':>11}")
    print("-" * 100)
    for name, kept in kept_map.items():
        h1 = [t for t in kept if t.exit_time.date() < mid]
        h2 = [t for t in kept if t.exit_time.date() >= mid]
        s1, s2 = stats(h1), stats(h2)
        if s1["n"] == 0 or s2["n"] == 0:
            print(f"  {name:<32}  (insufficient sample one side)"); continue
        print(f"  {name:<32}{s1['n']:>6}{s1['avg_r']:>+11.4f}${s1['pnl']:>+10,.0f}"
              f"{s2['n']:>6}{s2['avg_r']:>+11.4f}${s2['pnl']:>+10,.0f}")

    print("\nGATE: ship only if a config beats baseline avg_R AND stays positive in BOTH halves.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
