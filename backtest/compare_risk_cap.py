"""Should we raise MAX_RISK_PER_SHARE above $10?

Today's session (2026-06-01) had ~6 trend-eligible breakouts blocked by the
$10 per-share risk cap (AMAT, CRWD, GS, AMD, AVGO, IBM — most just slightly
over at $13-15/share). The cap was originally set when the watchlist was 5
mega-caps; with the broad 100-name universe including more $300-$500 names,
$10 may be leaving real high-quality trades on the table.

Tests: broad 100-name + 11:30 cutoff + trend filter (above 200d SMA AND
rel-strong vs SPY — the LIVE config), varying MAX_RISK_PER_SHARE in
{10, 12, 15, 20, no cap}. Reports n / win% / avg_R / sum_R / PnL / max DD
plus an OOS split.

Method: post-hoc filter on the trade list — a cap is equivalent to dropping
trades whose (entry - stop) exceeds the cap, because the live cap simply
rejects entry when risk_per_share is too large.

Run:
    .venv\\Scripts\\python.exe backtest\\compare_risk_cap.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

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
SMA_LONG = 200
RET_LOOKBACK = 20
CAP_VALUES = [10.0, 12.0, 15.0, 20.0, None]


def fetch_daily(syms, start, end, key, sec):
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
    closes.index = (pd.to_datetime(closes.index)
                    .tz_convert("America/New_York").normalize().tz_localize(None))
    return closes.sort_index()


def trend_eligible(trades, daily):
    """Set of id(t) for trades passing LIVE trend filter (above 200d SMA AND
    20d return > SPY's 20d return), measured at the prior session's close."""
    sma200 = daily.rolling(SMA_LONG).mean()
    ret20 = daily.pct_change(RET_LOOKBACK)
    out = set()
    for t in trades:
        sym = t.symbol
        if sym not in daily.columns or "SPY" not in daily.columns:
            continue
        entry_day = pd.Timestamp(t.entry_time.date())
        try:
            sub = daily[sym].loc[:entry_day - pd.Timedelta(days=1)].dropna()
            if sub.empty:
                continue
            prior = sub.index[-1]
            prior_close = float(sub.iloc[-1])
            s200 = float(sma200[sym].loc[prior])
            r20 = float(ret20[sym].loc[prior])
            spy_r20 = float(ret20["SPY"].loc[prior])
        except Exception:
            continue
        if any(pd.isna(x) for x in (s200, r20, spy_r20)):
            continue
        if prior_close > s200 and r20 > spy_r20:
            out.add(id(t))
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


def cap_label(cap):
    return "no cap" if cap is None else f"<= ${cap:.0f}/sh"


def print_row(name, s):
    if s["n"] == 0:
        print(f"  {name:<20}  (no trades)")
        return
    tse = f"{s['tgt']}/{s['stop']}/{s['eod']}"
    print(f"  {name:<20}{s['n']:>6}{s['win']:>7.1f}%{s['avg_r']:>+9.4f}"
          f"{s['sum_r']:>+9.1f}{tse:>13}${s['pnl']:>+11,.0f}${s['dd']:>+11,.0f}")


def main() -> int:
    load_env()
    key, sec = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    if not (key and sec):
        print("ERROR: API keys missing", file=sys.stderr)
        return 1

    print(f"Universe: {len(UNIVERSE)} names")
    print("Fetching intraday minute bars for broad universe...")
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
    print("Running baseline backtest (11:30 cutoff)...")
    base = Params(or_minutes=15, target_r=2.0, risk_per_trade=100.0,
                  max_position_pct=0.25, max_position_dollars=10_000.0,
                  no_entry_after_time=time(11, 30))
    all_trades, _ = run_backtest(intraday, trading_days, present, base, STARTING_EQUITY)
    print(f"Baseline trades: {len(all_trades)}\n")

    intraday_start = min(trading_days)
    daily_start = intraday_start - timedelta(days=int(SMA_LONG * 1.6))
    daily_end = max(trading_days) + timedelta(days=1)
    print(f"Fetching daily bars {daily_start} -> {daily_end}...")
    daily = fetch_daily(UNIVERSE, daily_start, daily_end, key, sec)
    print(f"Daily frame: {len(daily)} rows x {daily.shape[1]} symbols\n")

    elig = trend_eligible(all_trades, daily)
    trend_trades = [t for t in all_trades if id(t) in elig]
    print(f"Trend-filtered trades (above 200d + rel-strong): {len(trend_trades)} of {len(all_trades)}\n")

    # Distribution of risk_per_share among the trend-eligible set.
    rps = pd.Series([abs(t.entry_price - t.stop_price) for t in trend_trades])
    print("Distribution of risk_per_share (trend-eligible trades):")
    print(f"  median ${rps.median():.2f}  p25 ${rps.quantile(0.25):.2f}  "
          f"p75 ${rps.quantile(0.75):.2f}  p90 ${rps.quantile(0.90):.2f}  "
          f"max ${rps.max():.2f}")
    for c in [10, 12, 15, 20, 30, 50]:
        keep = int((rps <= c).sum())
        drop = int((rps > c).sum())
        print(f"  cap ${c:>3}/sh: keep {keep:>4}  drop {drop:>4}  "
              f"({100 * keep / len(rps):.1f}% kept)")
    print()

    print(f"  {'cap':<20}{'n':>6}{'win%':>8}{'avg_R':>9}"
          f"{'sumR':>9}{'tgt/stop/eod':>13}{'PnL$':>12}{'maxDD$':>12}")
    print("-" * 100)
    kept_map = {}
    for cap in CAP_VALUES:
        if cap is None:
            kept = trend_trades
        else:
            kept = [t for t in trend_trades if abs(t.entry_price - t.stop_price) <= cap]
        kept_map[cap_label(cap)] = kept
        print_row(cap_label(cap), stats(kept))

    # OOS split at midpoint of trade days
    days_sorted = sorted(set(t.exit_time.date() for t in trend_trades))
    if not days_sorted:
        print("No trades to OOS-split.")
        return 0
    mid = days_sorted[len(days_sorted) // 2]
    print(f"\nOOS SPLIT at {mid}:")
    print(f"  {'cap':<20}{'h1_n':>6}{'h1_avg_R':>11}{'h1_PnL':>11}"
          f"{'h2_n':>6}{'h2_avg_R':>11}{'h2_PnL':>11}")
    print("-" * 100)
    for name, kept in kept_map.items():
        h1 = [t for t in kept if t.exit_time.date() < mid]
        h2 = [t for t in kept if t.exit_time.date() >= mid]
        s1, s2 = stats(h1), stats(h2)
        if s1["n"] == 0 or s2["n"] == 0:
            print(f"  {name:<20}  (insufficient sample one side)"); continue
        print(f"  {name:<20}{s1['n']:>6}{s1['avg_r']:>+11.4f}${s1['pnl']:>+10,.0f}"
              f"{s2['n']:>6}{s2['avg_r']:>+11.4f}${s2['pnl']:>+10,.0f}")

    print("\nGATE: raise the cap only if a higher cap beats $10 on avg_R AND")
    print("stays positive in BOTH halves. If looser caps only help via PnL$ (more")
    print("trades at lower avg_R), that's just adding noise -- keep $10.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
