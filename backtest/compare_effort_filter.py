"""Wyckoff "effort vs result" filter for ORB long breakouts.

Tests the suggestion that the breakout bar's PRICE-EFFORT (range) relative to
VOLUME (effort) discriminates real demand from institutional churning:

  - "Good breakout": wide bar range + high volume  (effort -> directional result)
  - "Bad breakout" (churning): narrow range + high volume  (effort absorbed)

Both metrics are normalized per-day-per-symbol against the OR-window (09:30-09:45)
bars so they're cross-comparable. Tests four quadrants vs baseline, plus the
single "range/volume ratio" terciles.

Method: post-hoc filter on the broad-universe trade list (baseline = 11:30
cutoff, no trend filter so we see standalone value of the effort signal).
OOS split.

Run:
    .venv\\Scripts\\python.exe backtest\\compare_effort_filter.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from strategies.orb import Params  # noqa: E402
from backtest.run_orb import (  # noqa: E402
    LOOKBACK_DAYS, STARTING_EQUITY, get_trading_days, load_env, run_backtest, to_et,
)
from backtest.universe_scan import UNIVERSE, fetch_chunked  # noqa: E402

ET = ZoneInfo("America/New_York")
OR_END = time(9, 45)


def get_metrics(all_bars, t):
    """Return {range_norm, vol_norm, ratio_norm} for the breakout bar that
    triggered trade t, normalized against the OR-window medians for that
    symbol on that day. None if data is missing."""
    try:
        sb = all_bars.xs(t.symbol, level=0)
    except KeyError:
        return None
    tt = sb.index.time
    rth = sb[(tt >= time(9, 30)) & (tt < time(16, 0))]
    day = rth[rth.index.date == t.entry_time.date()]
    before = day[day.index < t.entry_time]
    if before.empty:
        return None
    bar = before.iloc[-1]
    or_bars = day[day.index.time < OR_END]
    if or_bars.empty:
        return None
    bar_range = float(bar["high"] - bar["low"])
    bar_vol = float(bar["volume"])
    or_med_range = float((or_bars["high"] - or_bars["low"]).median())
    or_med_vol = float(or_bars["volume"].median())
    if or_med_range <= 0 or or_med_vol <= 0 or bar_vol <= 0:
        return None
    range_norm = bar_range / or_med_range
    vol_norm = bar_vol / or_med_vol
    # range/volume ratio normalized: high = wide-per-share (clean), low = churning
    ratio_norm = range_norm / vol_norm
    return {"range_norm": range_norm, "vol_norm": vol_norm, "ratio_norm": ratio_norm}


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
    }


def row(name, s):
    if s["n"] == 0:
        print(f"  {name:<38}  (no trades)"); return
    print(f"  {name:<38}{s['n']:>5}{s['win']:>6.1f}%{s['avg_r']:>+9.4f}"
          f"{s['sum_r']:>+8.1f}${s['pnl']:>+10,.0f}${s['dd']:>+11,.0f}")


def main() -> int:
    load_env()
    key, sec = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    if not (key and sec):
        print("ERROR: API keys missing", file=sys.stderr); return 1

    print(f"Universe: {len(UNIVERSE)} names")
    print("Fetching intraday bars (broad universe)...")
    dc = StockHistoricalDataClient(key, sec)
    tc = TradingClient(key, sec, paper=True)
    end = datetime.now(tz=ET)
    start = end - timedelta(days=LOOKBACK_DAYS)
    trading_days = get_trading_days(tc, start, end)
    raw = fetch_chunked(dc, UNIVERSE, start, end)
    if raw.empty:
        print("ERROR: no bars"); return 1
    intraday = to_et(raw)
    present = sorted(intraday.index.get_level_values(0).unique())
    print(f"Intraday: {len(intraday):,} rows x {len(present)} symbols, "
          f"{len(trading_days)} sessions")
    base = Params(or_minutes=15, target_r=2.0, risk_per_trade=100.0,
                  max_position_pct=0.25, max_position_dollars=10_000.0,
                  no_entry_after_time=time(11, 30))
    all_trades, _ = run_backtest(intraday, trading_days, present, base, STARTING_EQUITY)
    print(f"Baseline trades: {len(all_trades)}\n")

    metrics = {}
    for t in all_trades:
        m = get_metrics(intraday, t)
        if m is not None:
            metrics[id(t)] = m
    eligible = [t for t in all_trades if id(t) in metrics]
    print(f"Eligible (metrics computable): {len(eligible)}")

    # Distribution sanity check
    ranges = pd.Series([metrics[id(t)]["range_norm"] for t in eligible])
    vols = pd.Series([metrics[id(t)]["vol_norm"] for t in eligible])
    print(f"range_norm: median {ranges.median():.2f}, p25 {ranges.quantile(0.25):.2f}, "
          f"p75 {ranges.quantile(0.75):.2f}")
    print(f"vol_norm:   median {vols.median():.2f}, p25 {vols.quantile(0.25):.2f}, "
          f"p75 {vols.quantile(0.75):.2f}\n")
    # Quadrant cut at the medians
    r_med, v_med = ranges.median(), vols.median()

    def q1(t):  # wide + high vol  (LLM's GOOD breakout)
        m = metrics[id(t)]; return m["range_norm"] > r_med and m["vol_norm"] > v_med
    def q2(t):  # wide + low vol
        m = metrics[id(t)]; return m["range_norm"] > r_med and m["vol_norm"] <= v_med
    def q3(t):  # narrow + high vol  (LLM's CHURNING -- should underperform)
        m = metrics[id(t)]; return m["range_norm"] <= r_med and m["vol_norm"] > v_med
    def q4(t):  # narrow + low vol
        m = metrics[id(t)]; return m["range_norm"] <= r_med and m["vol_norm"] <= v_med

    # Single-metric: range/volume ratio terciles
    ratios = pd.Series([metrics[id(t)]["ratio_norm"] for t in eligible])
    rt33, rt67 = ratios.quantile(0.333), ratios.quantile(0.667)

    def ratio_top(t):  return metrics[id(t)]["ratio_norm"] >= rt67
    def ratio_bot(t):  return metrics[id(t)]["ratio_norm"] <= rt33
    def not_q3(t):     return not q3(t)  # reject only the LLM's "churning"

    configs = [
        ("baseline (all eligible)", lambda t: True),
        ("Q1 wide+high_vol (LLM GOOD)", q1),
        ("Q2 wide+low_vol", q2),
        ("Q3 narrow+high_vol (LLM CHURN)", q3),
        ("Q4 narrow+low_vol", q4),
        ("reject Q3 only (LLM filter)", not_q3),
        ("range/vol top tercile", ratio_top),
        ("range/vol bottom tercile", ratio_bot),
    ]

    print(f"  {'config':<38}{'n':>5}{'win%':>7}{'avg_R':>9}"
          f"{'sumR':>8}{'PnL$':>11}{'maxDD$':>12}")
    print("-" * 92)
    kept_map = {}
    for name, pred in configs:
        kept = [t for t in eligible if pred(t)]
        kept_map[name] = kept
        row(name, stats(kept))

    # OOS split
    days_sorted = sorted(set(t.exit_time.date() for t in eligible))
    mid = days_sorted[len(days_sorted) // 2]
    print(f"\nOOS SPLIT at {mid}:")
    print(f"  {'config':<38}{'h1_n':>6}{'h1_avg_R':>11}{'h2_n':>6}{'h2_avg_R':>11}")
    print("-" * 78)
    for name, kept in kept_map.items():
        h1 = [t for t in kept if t.exit_time.date() < mid]
        h2 = [t for t in kept if t.exit_time.date() >= mid]
        s1, s2 = stats(h1), stats(h2)
        if s1["n"] == 0 or s2["n"] == 0:
            print(f"  {name:<38}  (insufficient)"); continue
        print(f"  {name:<38}{s1['n']:>6}{s1['avg_r']:>+11.4f}"
              f"{s2['n']:>6}{s2['avg_r']:>+11.4f}")

    print("\nLLM hypothesis is supported only if Q1 (wide+high_vol) clearly beats")
    print("baseline AND Q3 (narrow+high_vol) clearly underperforms, in BOTH halves.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
