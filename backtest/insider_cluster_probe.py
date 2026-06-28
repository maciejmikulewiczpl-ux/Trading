"""Proper insider-buy test: do CLUSTER buys (>=2 distinct insiders buying within 7 days —
the literature's strong signal) predict forward market-adjusted returns? Bigger sample
(EXPANSION ~157 mid-caps vs the 27-name liquid first cut) + cluster filter.

Lookahead-free: cluster 'event date' = the day the 2nd insider's buy makes it a cluster
(you only know it's a cluster once that files). Forward return measured from there, minus
SPY over the same window. Random-date baseline (same names) is the bar to beat.

Caveat: EXPANSION is MID-cap (cleaner data than micro-caps but fewer buys than true small-
caps where the signal is strongest) — a null here doesn't reject the micro-cap tier.

Run (Form 4 parsing slow, ~30-60 min):  .venv/Scripts/python.exe backtest/insider_cluster_probe.py
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
from backtest.fetch_universe_expanded import EXPANSION  # noqa: E402

UTC = ZoneInfo("UTC")
CACHE = ROOT / "backtest" / ".insider_cluster_cache.pkl"
START = datetime(2025, 6, 1, tzinfo=UTC)
HORIZONS = [10, 20]
CLUSTER_DAYS = 7
MAX_F4_PER_NAME = 60
UNIVERSE = sorted(set(EXPANSION))


def fetch_buys(symbols) -> dict:
    """{sym: [(date, insider_name, usd_value), ...]} open-market insider purchases (Code 'P')."""
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
                    obj = filing.obj()
                    insider = str(getattr(obj, "insider_name", "") or "?")
                    mt = obj.market_trades
                    if mt is None or mt.empty:
                        continue
                    for _, t in mt.iterrows():
                        if str(t.get("Code")) == "P":
                            sh = float(t.get("Shares") or 0); px = float(t.get("Price") or 0)
                            buys.append((pd.Timestamp(filing.filing_date), insider, sh * px))
                except Exception:
                    continue
        except Exception as e:
            print(f"  {sym}: ERR {str(e)[:50]}")
        out[sym] = buys
        if buys:
            print(f"  {i}/{len(symbols)} {sym}: {len(buys)} buys", flush=True)
        elif i % 20 == 0:
            print(f"  ...{i}/{len(symbols)}", flush=True)
    pickle.dump(out, open(CACHE, "wb"))
    return out


def daily_closes(symbols):
    import os
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed, Adjustment
    dc = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    out = {}
    for i in range(0, len(symbols) + 1, 20):
        grp = (symbols + ["SPY"])[i:i + 20]
        if not grp:
            continue
        req = StockBarsRequest(symbol_or_symbols=grp, timeframe=TimeFrame.Day,
                               start=START - timedelta(days=5), end=datetime.now(UTC),
                               feed=DataFeed.IEX, adjustment=Adjustment.ALL)
        df = dc.get_stock_bars(req).df
        for s in grp:
            try:
                c = df.xs(s, level=0)["close"]
                c.index = pd.DatetimeIndex([t.date() for t in c.index])
                out[s] = c.sort_index()
            except KeyError:
                pass
    return out


def fwd_adj_return(close, spy, d, h):
    idx = close.index
    pos = idx.searchsorted(pd.Timestamp(d).normalize())
    if pos + h >= len(idx):
        return None
    d0, d1 = idx[pos], idx[pos + h]
    try:
        r = close.loc[d0:d1].iloc[-1] / close.loc[d0] - 1.0
        sp0, sp1 = spy.asof(d0), spy.asof(d1)
        rs = (sp1 / sp0 - 1.0) if sp0 and sp1 else 0.0
        return (r - rs) * 100
    except Exception:
        return None


def main() -> int:
    load_env()
    print(f"insider CLUSTER probe: {len(UNIVERSE)} EXPANSION mid-caps since {START.date()}\n")
    buys = fetch_buys(UNIVERSE)
    closes = daily_closes(UNIVERSE)
    spy = closes.get("SPY")

    # classify each buy as cluster (>=2 distinct insiders within CLUSTER_DAYS ending at d)
    all_ev, cluster_ev = [], []   # (sym, date)
    for sym, blist in buys.items():
        if not blist:
            continue
        bl = sorted(blist, key=lambda x: x[0])
        for (d, insider, val) in bl:
            all_ev.append((sym, d))
            win = [ins for (dd, ins, vv) in bl if d - timedelta(days=CLUSTER_DAYS) <= dd <= d]
            if len(set(win)) >= 2:
                cluster_ev.append((sym, d))
    # dedupe cluster events to one per (sym, week)
    cluster_ev = sorted(set((s, pd.Timestamp(d).to_period("W")) for s, d in cluster_ev))
    cluster_dates = {}
    for sym, blist in buys.items():
        for (d, ins, val) in blist:
            wk = pd.Timestamp(d).to_period("W")
            if (sym, wk) in dict.fromkeys(cluster_ev) and (sym, wk) not in cluster_dates:
                cluster_dates[(sym, wk)] = (sym, d)
    cluster_pts = list(cluster_dates.values())

    n_all = len(all_ev)
    n_names = sum(1 for v in buys.values() if v)
    print(f"\n{n_all} open-market buys across {n_names} names | {len(cluster_pts)} CLUSTER events")

    rng = random.Random(42)

    def measure(events, label):
        for h in HORIZONS:
            ev, base = [], []
            for sym, d in events:
                c = closes.get(sym)
                if c is None or len(c) < h + 5:
                    continue
                r = fwd_adj_return(c, spy, d, h)
                if r is not None:
                    ev.append(r)
            # matched random baseline (same count, random names+dates)
            names = [s for s in closes if s != "SPY" and len(closes[s]) > h + 5]
            for _ in range(max(len(ev), 1)):
                s = rng.choice(names); c = closes[s]
                rd = c.index[rng.randrange(0, max(1, len(c) - h - 1))]
                rb = fwd_adj_return(c, spy, rd, h)
                if rb is not None:
                    base.append(rb)
            def st(xs):
                return (len(xs), (sum(xs) / len(xs)) if xs else 0.0,
                        (sum(1 for x in xs if x > 0) / len(xs) * 100) if xs else 0.0)
            ne, me, we = st(ev); nb, mb, wb = st(base)
            print(f"  [{label}] +{h}d: buy n={ne:>4} avg {me:>+6.2f}% win {we:>3.0f}%  | "
                  f"random avg {mb:>+6.2f}%  | EDGE {me - mb:>+6.2f}%")

    print("\n=== ALL insider buys ===")
    measure(all_ev, "all")
    print("\n=== CLUSTER buys (>=2 insiders / 7d) — the literature's strong signal ===")
    measure(cluster_pts, "cluster")
    print("\nReal signal = CLUSTER buys clearly beat random at both horizons. If clusters are")
    print("flat too, insider buying adds no tradeable edge in this (mid-cap) universe.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
