"""Profit hunt (Fable B): does DAILY-timeframe compression stack on tight-OR?

Tight-OR works because intraday compression precedes expansion. Same logic one
timeframe up: a tight OR that sets up AFTER a compressed daily bar (narrow range /
NR7 / inside day) or while consolidating near highs should be an even cleaner coiled
spring. This stratifies the tight-OR trailing trades by the PRIOR day's daily context
(all known before the session open — lookahead-free) and checks whether any context
sub-filter beats the tight-OR baseline avg_R.

Daily context for a trade on day d (from bars strictly before d):
  gap%       : (open_d - close_{d-1}) / close_{d-1}
  NR7        : range_{d-1} is the narrowest of the trailing 7 daily ranges
  inside_day : day d-1 is an inside day (high<=prev high AND low>=prev low)
  near_high  : close_{d-1} >= 0.97 * 20-day high

R-space (avg_R), capital-agnostic. Fetches daily OHLC for the hand symbols (cheap;
run AFTER pit_trades.py so it doesn't share the API with the minute fetch).

Run:
    .venv/Scripts/python.exe backtest/compare_daily_context.py
"""
from __future__ import annotations

import os
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.compare_selection import _tday  # noqa: E402
from backtest.compare_or_range_realcost import or_pct  # noqa: E402
from backtest.run_orb import load_env  # noqa: E402
from backtest.universe_scan import UNIVERSE  # noqa: E402

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
WINDOW = 730
OR_THR = 0.5
DAILY_CACHE = ROOT / "backtest" / f".pit_dailyohlc_{WINDOW}d.pkl"


def fetch_daily_ohlc(symbols):
    if DAILY_CACHE.exists():
        return pickle.load(open(DAILY_CACHE, "rb"))
    load_env()
    dc = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    end = datetime.now(tz=ET)
    start = end - timedelta(days=WINDOW + 40)     # +buffer for trailing windows
    frames = []
    for i in range(0, len(symbols), 100):
        grp = symbols[i:i + 100]
        print(f"  daily {i + 1}-{i + len(grp)} of {len(symbols)} ...", flush=True)
        req = StockBarsRequest(symbol_or_symbols=grp, timeframe=TimeFrame.Day,
                               start=start.astimezone(UTC), end=end.astimezone(UTC), feed=DataFeed.IEX)
        df = dc.get_stock_bars(req).df
        if not df.empty:
            frames.append(df[["open", "high", "low", "close"]])
    daily = pd.concat(frames)
    out = {}
    for sym in symbols:
        if sym in daily.index.get_level_values(0):
            sb = daily.xs(sym, level=0).copy()
            sb.index = pd.to_datetime(sb.index).tz_convert(ET).date
            out[sym] = sb[~sb.index.duplicated(keep="last")].sort_index()
    pickle.dump(out, open(DAILY_CACHE, "wb"))
    return out


def context(sb: pd.DataFrame, d):
    """Daily context for a trade on date d, from bars strictly before d (+ d's open)."""
    idx = list(sb.index)
    if d not in sb.index:
        return None
    pos = idx.index(d)
    if pos < 8:
        return None
    prev = sb.iloc[pos - 1]
    prev2 = sb.iloc[pos - 2]
    open_d = float(sb.iloc[pos]["open"])
    rng = (sb["high"] - sb["low"])
    r_prev = float(rng.iloc[pos - 1])
    nr7 = r_prev <= rng.iloc[pos - 7:pos].min() + 1e-9
    inside = (prev["high"] <= prev2["high"]) and (prev["low"] >= prev2["low"])
    hi20 = float(sb["high"].iloc[max(0, pos - 21):pos].max())
    near_high = float(prev["close"]) >= 0.97 * hi20
    gap = (open_d - float(prev["close"])) / float(prev["close"]) * 100
    return {"nr7": bool(nr7), "inside": bool(inside), "near_high": bool(near_high), "gap": gap}


def stats(rows):
    if not rows:
        return None
    n = len(rows)
    return {"n": n, "win": 100 * sum(1 for r in rows if r > 0) / n,
            "avg_r": sum(rows) / n, "sum_r": sum(rows)}


def line(label, st, base):
    if st is None or st["n"] < 15:
        print(f"  {label:<24}{(st['n'] if st else 0):>6}    (too few)")
        return
    delta = f"{st['avg_r'] - base:+.4f}" if base is not None else ""
    print(f"  {label:<24}{st['n']:>6}{st['win']:>7.1f}%{st['avg_r']:>+9.4f}{st['sum_r']:>+9.1f}   {delta:>9}")


def main() -> int:
    blob = pickle.load(open(ROOT / "backtest" / f".pit_trailtrades_{WINDOW}d.pkl", "rb"))
    hand = set(UNIVERSE)
    trades = [t for syms in blob.values() for t in syms
              if t.symbol in hand and or_pct(t) <= OR_THR]
    syms = sorted({t.symbol for t in trades})
    daily = fetch_daily_ohlc(syms)

    ctx = {}
    for t in trades:
        sb = daily.get(t.symbol)
        ctx[id(t)] = context(sb, _tday(t)) if sb is not None else None

    base = stats([t.pnl_r for t in trades])
    print(f"\n{'='*72}\nDAILY-CONTEXT STRATIFICATION — tight-OR<={OR_THR}% trailing, hand universe")
    print(f"{len(trades)} trades; {sum(1 for t in trades if ctx[id(t)])} with daily context")
    print(f"{'='*72}")
    print(f"  {'stratum':<24}{'n':>6}{'win%':>7}{'avg_R':>9}{'sum_R':>9}   {'vs base':>9}")
    print("  " + "-" * 66)
    line("ALL (baseline)", base, None)

    def sub(pred):
        return stats([t.pnl_r for t in trades if ctx[id(t)] and pred(ctx[id(t)])])

    line("NR7 prior day", sub(lambda c: c["nr7"]), base["avg_r"])
    line("inside prior day", sub(lambda c: c["inside"]), base["avg_r"])
    line("NR7 OR inside", sub(lambda c: c["nr7"] or c["inside"]), base["avg_r"])
    line("near 20d high", sub(lambda c: c["near_high"]), base["avg_r"])
    line("compressed & near-high", sub(lambda c: (c["nr7"] or c["inside"]) and c["near_high"]), base["avg_r"])
    line("gap up >0.5%", sub(lambda c: c["gap"] > 0.5), base["avg_r"])
    line("gap flat -0.5..0.5%", sub(lambda c: -0.5 <= c["gap"] <= 0.5), base["avg_r"])
    line("gap down <-0.5%", sub(lambda c: c["gap"] < -0.5), base["avg_r"])
    print("\nRead: a context stratum that beats baseline avg_R with decent n is a stackable")
    print("filter (a daily coiled-spring on top of the intraday one). Near-baseline = no daily edge.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
