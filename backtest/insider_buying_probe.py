"""Backtest probe: do open-market INSIDER PURCHASES (Form 4, via edgartools) predict
forward MARKET-ADJUSTED returns? A fresher (files within ~2 days) cousin of the 45-day-
lagged 13F accumulation signal — a candidate for the swing/accumulation toolkit, NOT the
intraday bots.

Design (lookahead-free):
  - For each name, pull Form 4s over ~12mo, parse market_trades, keep OPEN-MARKET BUYS
    (Code 'P'); record (filing_date, $value).
  - For each buy event on day d: forward return d -> d+10 and d -> d+20 trading days,
    MARKET-ADJUSTED (minus SPY's same-window return) to strip out market drift.
  - Compare to a RANDOM-date baseline (same names/horizons) — the buys must beat random.

Honest: insider BUYS are rarer in mega-caps (execs mostly sell), so the universe is tilted
to financials/energy/mid-cap value where open-market buys actually happen. Caches events.

Run (Form 4 parsing is slow):  .venv/Scripts/python.exe backtest/insider_buying_probe.py
"""
from __future__ import annotations

import pickle
import random
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.run_orb import load_env  # noqa: E402

UTC = ZoneInfo("UTC")
CACHE = ROOT / "backtest" / ".insider_buys_cache.pkl"
START = datetime(2025, 6, 1, tzinfo=UTC)        # ~12 months
HORIZONS = [10, 20]                              # trading-day forward windows
MAX_F4_PER_NAME = 60
# tilted to where open-market insider buys actually happen (not mega-caps)
UNIVERSE = ["PNC", "TFC", "COF", "ALLY", "SOFI", "HOOD", "KEY", "CFG",
            "OXY", "KMI", "OKE", "MPC", "VLO", "EOG",
            "F", "GM", "KR", "DG", "DLTR", "INTC", "HPQ", "WDC", "CVS", "BMY", "PFE", "NUE", "FCX"]


def fetch_buys(symbols) -> dict:
    """{sym: [(date, usd_value), ...]} open-market insider purchases (Form 4 Code 'P')."""
    if CACHE.exists():
        print(f"using cached insider buys: {CACHE.name}")
        return pickle.load(open(CACHE, "rb"))
    from edgar import set_identity, Company
    set_identity("news-edge research maciej.mikulewicz@gmail.com")
    out = {}
    for i, sym in enumerate(symbols, 1):
        buys = []
        try:
            fs = [f for f in Company(sym).get_filings(form="4")
                  if f.filing_date and pd.Timestamp(f.filing_date) >= pd.Timestamp(START.date())]
            for filing in fs[:MAX_F4_PER_NAME]:
                try:
                    mt = filing.obj().market_trades
                    if mt is None or mt.empty:
                        continue
                    for _, t in mt.iterrows():
                        if str(t.get("Code")) == "P":            # open-market purchase
                            sh = float(t.get("Shares") or 0); px = float(t.get("Price") or 0)
                            buys.append((pd.Timestamp(filing.filing_date), sh * px))
                except Exception:
                    continue
        except Exception as e:
            print(f"  {sym}: ERR {str(e)[:50]}")
        out[sym] = buys
        print(f"  {i}/{len(symbols)} {sym}: {len(buys)} insider buys", flush=True)
    pickle.dump(out, open(CACHE, "wb"))
    return out


def daily_closes(symbols):
    import os
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed, Adjustment
    dc = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    req = StockBarsRequest(symbol_or_symbols=symbols + ["SPY"], timeframe=TimeFrame.Day,
                           start=START - timedelta(days=5), end=datetime.now(UTC),
                           feed=DataFeed.IEX, adjustment=Adjustment.ALL)
    df = dc.get_stock_bars(req).df
    out = {}
    for s in symbols + ["SPY"]:
        try:
            c = df.xs(s, level=0)["close"]
            c.index = pd.DatetimeIndex([t.date() for t in c.index])
            out[s] = c.sort_index()
        except KeyError:
            pass
    return out


def fwd_adj_return(close, spy, d, h):
    """Market-adjusted return from the trading day on/after d, over h trading days."""
    idx = close.index
    pos = idx.searchsorted(pd.Timestamp(d).normalize())
    if pos + h >= len(idx):
        return None
    d0, d1 = idx[pos], idx[pos + h]
    try:
        r = close.loc[d0:d1].iloc[-1] / close.loc[d0] - 1.0
        sp0 = spy.asof(d0); sp1 = spy.asof(d1)
        rs = (sp1 / sp0 - 1.0) if sp0 and sp1 else 0.0
        return (r - rs) * 100
    except Exception:
        return None


def main() -> int:
    load_env()
    print(f"insider-buying probe (edgartools Form 4): {len(UNIVERSE)} names since {START.date()}\n")
    buys = fetch_buys(UNIVERSE)
    closes = daily_closes(UNIVERSE)
    spy = closes.get("SPY")
    n_events = sum(len(v) for v in buys.values())
    print(f"\n{n_events} open-market insider buys across {sum(1 for v in buys.values() if v)} names")

    rng = random.Random(42)
    for h in HORIZONS:
        ev, base = [], []
        for sym, blist in buys.items():
            c = closes.get(sym)
            if c is None or len(c) < h + 5:
                continue
            for (d, val) in blist:
                r = fwd_adj_return(c, spy, d, h)
                if r is not None:
                    ev.append(r)
            # random-date baseline: same count of random dates for this name
            for _ in range(max(len(blist), 1)):
                rd = c.index[rng.randrange(0, max(1, len(c) - h - 1))]
                rb = fwd_adj_return(c, spy, rd, h)
                if rb is not None:
                    base.append(rb)
        def stat(xs):
            if not xs:
                return (0, 0.0, 0.0)
            return (len(xs), sum(xs) / len(xs), sum(1 for x in xs if x > 0) / len(xs) * 100)
        ne, me, we = stat(ev)
        nb, mb, wb = stat(base)
        print(f"\n=== +{h} trading days (market-adjusted) ===")
        print(f"  after insider BUY : n={ne:>4}  avg {me:>+6.2f}%  win {we:>4.0f}%")
        print(f"  random baseline   : n={nb:>4}  avg {mb:>+6.2f}%  win {wb:>4.0f}%")
        print(f"  EDGE (buy - random): {me - mb:>+6.2f}%")
    print("\nReal signal = insider-buy forward return clearly beats the random baseline,")
    print("market-adjusted, at both horizons. Small/negative edge -> insider buys (in this")
    print("liquid universe) don't add a tradeable signal beyond market drift.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
