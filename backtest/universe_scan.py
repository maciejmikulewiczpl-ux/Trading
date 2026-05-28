"""Does the ORB edge survive on a broad liquid universe, or is it an artifact
of the 5-name watchlist?

Runs the CURRENT shipped strategy (long-only, 15-min OR, 2R target, 11:30 ET
entry cutoff) across ~55 large/liquid US names over the 180-day window.
Reports per-signal expectancy (avg_R — capital-agnostic, the cleanest edge
measure), an OOS first/second-half split, and a per-name breakdown so we can
see which tickers actually carry ORB edge.

Bias notes:
- Universe is a FIXED list of names that were liquid and listed throughout the
  window (no "today's top movers" — that would be lookahead/survivorship bias).
- avg_R and sum(pnl_r) are capital-agnostic; total $ PnL assumes every signal
  is taken (needs far more than $100k of buying power across 55 names) so it is
  an UPPER BOUND, reported only for color. The edge verdict rests on avg_R.

Run (data fetch takes a few minutes):
    .venv/Scripts/python.exe backtest/universe_scan.py
"""
from __future__ import annotations

import sys
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from strategies.orb import Params, Trade  # noqa: E402
from backtest.run_orb import (  # noqa: E402
    LOOKBACK_DAYS, STARTING_EQUITY, get_trading_days, load_env, run_backtest, to_et,
)

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

# Fixed, liquid, listed-throughout universe (no survivorship/lookahead bias).
UNIVERSE = [
    # broad ETFs
    "SPY", "QQQ", "IWM", "DIA",
    # mega/large-cap tech & semis
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "ORCL",
    "AMD", "NFLX", "ADBE", "CRM", "INTC", "CSCO", "QCOM", "TXN", "MU",
    # financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "V", "MA", "AXP",
    # healthcare
    "UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV",
    # consumer
    "WMT", "HD", "COST", "NKE", "MCD", "SBUX", "DIS", "KO", "PEP",
    # energy / industrial
    "XOM", "CVX", "CAT", "BA", "GE",
    # higher-beta / high-volume
    "PLTR", "COIN", "UBER", "BABA",
]


def fetch_chunked(client, symbols, start, end, chunk=12):
    frames = []
    for i in range(0, len(symbols), chunk):
        grp = symbols[i:i + chunk]
        print(f"  bars {i + 1}-{i + len(grp)} of {len(symbols)} ...", flush=True)
        req = StockBarsRequest(
            symbol_or_symbols=grp, timeframe=TimeFrame.Minute,
            start=start.astimezone(UTC), end=end.astimezone(UTC), feed=DataFeed.IEX,
        )
        df = client.get_stock_bars(req).df
        if not df.empty:
            frames.append(df)
    return pd.concat(frames) if frames else pd.DataFrame()


def stats(trades: list[Trade]) -> dict:
    if not trades:
        return {"n": 0}
    df = pd.DataFrame([{"r": t.pnl_r, "d": t.pnl_dollars,
                        "reason": t.exit_reason} for t in trades])
    return {
        "n": len(df),
        "win": (df["r"] > 0).mean() * 100,
        "avg_r": df["r"].mean(),
        "sum_r": df["r"].sum(),
        "pnl": df["d"].sum(),
        "tgt": int((df["reason"] == "target").sum()),
        "stop": int((df["reason"] == "stop").sum()),
        "eod": int((df["reason"] == "eod").sum()),
    }


def main() -> int:
    load_env()
    import os
    key, sec = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    if not key or not sec:
        print("ERROR: API keys missing", file=sys.stderr)
        return 1
    dc = StockHistoricalDataClient(key, sec)
    tc = TradingClient(key, sec, paper=True)

    end = datetime.now(tz=ET)
    start = end - timedelta(days=LOOKBACK_DAYS)
    print(f"Universe: {len(UNIVERSE)} names, window {start.date()} -> {end.date()}")
    trading_days = get_trading_days(tc, start, end)
    print(f"Sessions: {len(trading_days)}. Fetching bars (chunked)...")
    raw = fetch_chunked(dc, UNIVERSE, start, end)
    if raw.empty:
        print("ERROR: no bars", file=sys.stderr)
        return 1
    all_bars = to_et(raw)
    present = sorted(all_bars.index.get_level_values(0).unique())
    print(f"Bars: {len(all_bars):,} rows across {len(present)} symbols")
    missing = set(UNIVERSE) - set(present)
    if missing:
        print(f"  (no data for: {', '.join(sorted(missing))})")

    params = Params(or_minutes=15, target_r=2.0, risk_per_trade=100.0,
                    max_position_pct=0.25, max_position_dollars=10_000.0,
                    no_entry_after_time=time(11, 30))

    trades, _ = run_backtest(all_bars, trading_days, present, params, STARTING_EQUITY)

    # ---- headline ----
    s = stats(trades)
    print("\n" + "=" * 64)
    print(f"UNIVERSE EDGE ({len(present)} names)")
    print("=" * 64)
    print(f"  Trades        : {s['n']}")
    print(f"  Win rate      : {s['win']:.1f}%")
    print(f"  Avg R         : {s['avg_r']:+.4f}   <-- the edge metric")
    print(f"  Sum R         : {s['sum_r']:+.1f}")
    print(f"  Exit mix      : {s['tgt']} target / {s['stop']} stop / {s['eod']} eod")
    print(f"  PnL (uncapped): ${s['pnl']:+,.0f}  (upper bound; needs >$100k BP)")

    # ---- OOS split ----
    days_sorted = sorted(trading_days)
    mid = days_sorted[len(days_sorted) // 2]
    first = [t for t in trades if t.date.date() < mid]
    second = [t for t in trades if t.date.date() >= mid]
    s1, s2 = stats(first), stats(second)
    print(f"\nOOS split at {mid}:")
    print(f"  first half : n={s1['n']:>4}  win {s1['win']:.1f}%  avg_R {s1['avg_r']:+.4f}  sumR {s1['sum_r']:+.1f}")
    print(f"  second half: n={s2['n']:>4}  win {s2['win']:.1f}%  avg_R {s2['avg_r']:+.4f}  sumR {s2['sum_r']:+.1f}")
    if s1["n"] and s2["n"]:
        verdict = ("ROBUST (positive both halves)" if s1["avg_r"] > 0 and s2["avg_r"] > 0
                   else "REGIME-ONLY (positive one half)" if s1["avg_r"] > 0 or s2["avg_r"] > 0
                   else "NO EDGE (negative both)")
        print(f"  verdict: {verdict}")

    # ---- per-name ----
    by_sym = {}
    for t in trades:
        by_sym.setdefault(t.symbol, []).append(t)
    rows = [(sym, stats(ts)) for sym, ts in by_sym.items()]
    rows.sort(key=lambda r: r[1]["sum_r"], reverse=True)
    print(f"\nPer-name (sorted by sumR):")
    print(f"  {'sym':<7}{'n':>4}{'win%':>7}{'avg_R':>9}{'sumR':>8}")
    for sym, st in rows:
        print(f"  {sym:<7}{st['n']:>4}{st['win']:>6.1f}%{st['avg_r']:>+9.4f}{st['sum_r']:>+8.1f}")

    # 5-name baseline comparison
    base5 = {"SPY", "QQQ", "AAPL", "NVDA", "TSLA"}
    b5 = stats([t for t in trades if t.symbol in base5])
    rest = stats([t for t in trades if t.symbol not in base5])
    print(f"\nOriginal 5 vs the other {len(present) - 5}:")
    if b5["n"]:
        print(f"  orig 5  : n={b5['n']:>4}  win {b5['win']:.1f}%  avg_R {b5['avg_r']:+.4f}  sumR {b5['sum_r']:+.1f}")
    if rest["n"]:
        print(f"  the rest: n={rest['n']:>4}  win {rest['win']:.1f}%  avg_R {rest['avg_r']:+.4f}  sumR {rest['sum_r']:+.1f}")

    # ---- PERSISTENCE: does first-half per-name edge predict second-half? ----
    # The decider for select-vs-broad. Build per-name avg_R in each half, then
    # (a) correlate them, (b) run a TRUE OOS selection: pick the names that were
    # above-median in the first half, measure their SECOND-half avg_R.
    print("\n" + "=" * 64)
    print("PERSISTENCE: first-half per-name edge -> second-half edge")
    print("=" * 64)
    h1, h2 = {}, {}
    for t in trades:
        (h1 if t.date.date() < mid else h2).setdefault(t.symbol, []).append(t)
    common = sorted(set(h1) & set(h2))
    rec = []
    for sym in common:
        a1 = pd.Series([x.pnl_r for x in h1[sym]]).mean()
        a2 = pd.Series([x.pnl_r for x in h2[sym]]).mean()
        rec.append((sym, a1, a2, len(h1[sym]), len(h2[sym])))
    pdf = pd.DataFrame(rec, columns=["sym", "avg_r_h1", "avg_r_h2", "n1", "n2"])
    if len(pdf) >= 5:
        pear = pdf["avg_r_h1"].corr(pdf["avg_r_h2"], method="pearson")
        # Spearman = Pearson of ranks (avoids a scipy dependency).
        spear = pdf["avg_r_h1"].rank().corr(pdf["avg_r_h2"].rank(), method="pearson")
        print(f"  names in both halves: {len(pdf)}")
        print(f"  corr(avg_R h1, avg_R h2): pearson {pear:+.3f}, spearman {spear:+.3f}")
        med1 = pdf["avg_r_h1"].median()
        winners_h1 = pdf[pdf["avg_r_h1"] > med1]
        losers_h1 = pdf[pdf["avg_r_h1"] <= med1]
        print(f"  H1 above-median names ({len(winners_h1)}): "
              f"their H2 avg_R = {winners_h1['avg_r_h2'].mean():+.4f}")
        print(f"  H1 below-median names ({len(losers_h1)}): "
              f"their H2 avg_R = {losers_h1['avg_r_h2'].mean():+.4f}")
        spread = winners_h1["avg_r_h2"].mean() - losers_h1["avg_r_h2"].mean()
        print(f"  selection spread in H2: {spread:+.4f}")
        if spear > 0.2 and spread > 0.02:
            print("  -> SELECTION HAS SIGNAL: first-half winners keep winning. "
                  "A curated watchlist beats equal-weight.")
        elif spear < 0.1 and abs(spread) < 0.02:
            print("  -> NO PERSISTENCE: ranking is noise. Trade BROAD equal-weight; "
                  "don't cherry-pick names.")
        else:
            print("  -> WEAK/AMBIGUOUS: mild signal at best. Lean broad, be skeptical "
                  "of any 'best names' list.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
