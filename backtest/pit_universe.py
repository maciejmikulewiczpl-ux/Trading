"""Point-in-time universe construction (survivorship/selection-bias test, step 1+2).

Fable's review (2026-06-09) #1 hole: the ~100-name watchlist was HAND-PICKED in 2026
knowing what's liquid/winning today -> the tight-OR backtest may be survivorship-biased.
This replaces hand-picking with a MECHANICAL point-in-time rule:

  - Candidate pool = the full active US-equity list from Alpaca (get_all_assets),
    NOT a curated set.
  - Fetch DAILY bars (cheap) for all of them over the window.
  - For each MONTH, eligible universe = the TOP_N symbols by mean daily dollar
    volume over the trailing RANK_DAYS trading days STRICTLY BEFORE that month
    (no lookahead). A trade on date D counts only if its symbol is top-N for D's month.

This script does pool + daily fetch + ranking, and CACHES:
  .pit_daily_{w}d.pkl     -> {"dvol": dollar-vol pivot, "close": close pivot, "days":[...]}
  .pit_members_{w}d.pkl   -> {"members": {month_start_date: [syms]}, "union": [syms]}

Then it prints the union size + churn so we can size the (heavy) minute-bar fetch
in pit_trades.py before committing to it.

Residual bias (documented): get_all_assets lists only CURRENTLY-active names, so truly
delisted/dead tickers are absent — a survivorship floor we can't remove without a paid
PIT constituent dataset. But swapping hand-picking for a dollar-volume rule kills the
larger bias. Dollar volume uses the IEX feed (partial tape) — understated but rank-
consistent with what the strategy actually trades on.

Run:
    .venv/Scripts/python.exe backtest/pit_universe.py
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
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, AssetStatus
from alpaca.trading.requests import GetAssetsRequest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.run_orb import get_trading_days, load_env  # noqa: E402

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

WINDOW = 730
TOP_N = 100
RANK_DAYS = 63                 # trailing trading days for the dollar-volume rank
MIN_COVER = 40                 # need >= this many bars in the trailing window to rank
DAILY_CHUNK = 200
KEEP_EXCHANGES = {"NYSE", "NASDAQ", "ARCA", "AMEX", "BATS", "NYSEARCA"}


def clients():
    load_env()
    key, sec = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    if not key or not sec:
        print("ERROR: API keys missing", file=sys.stderr)
        sys.exit(1)
    return (StockHistoricalDataClient(key, sec),
            TradingClient(key, sec, paper=True))


def candidate_pool(tc) -> list[str]:
    assets = tc.get_all_assets(GetAssetsRequest(
        asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE))
    syms = []
    for a in assets:
        if not a.tradable:
            continue
        exch = getattr(a.exchange, "value", str(a.exchange))
        if exch not in KEEP_EXCHANGES:
            continue
        sym = a.symbol
        if not sym.isalpha():          # drop warrants/units/odd classes (dots, digits)
            continue
        syms.append(sym)
    return sorted(set(syms))


def fetch_daily(dc, symbols, start, end) -> pd.DataFrame:
    frames = []
    for i in range(0, len(symbols), DAILY_CHUNK):
        grp = symbols[i:i + DAILY_CHUNK]
        print(f"  daily {i + 1}-{i + len(grp)} of {len(symbols)} ...", flush=True)
        try:
            req = StockBarsRequest(
                symbol_or_symbols=grp, timeframe=TimeFrame.Day,
                start=start.astimezone(UTC), end=end.astimezone(UTC), feed=DataFeed.IEX)
            df = dc.get_stock_bars(req).df
            if not df.empty:
                frames.append(df[["close", "volume"]])
        except Exception as e:
            print(f"    chunk failed ({e}); skipping", flush=True)
    return pd.concat(frames) if frames else pd.DataFrame()


def build_pivots(daily: pd.DataFrame):
    """Return (dvol_pivot, close_pivot) indexed by date, columns by symbol."""
    d = daily.copy()
    d["dvol"] = d["close"] * d["volume"]
    d = d.reset_index()
    d["date"] = pd.to_datetime(d["timestamp"]).dt.tz_convert(ET).dt.date
    dvol = d.pivot_table(index="date", values="dvol", columns="symbol", aggfunc="sum")
    close = d.pivot_table(index="date", values="close", columns="symbol", aggfunc="last")
    dvol.index = pd.to_datetime(dvol.index)
    close.index = pd.to_datetime(close.index)
    return dvol.sort_index(), close.sort_index()


def monthly_members(dvol: pd.DataFrame, trading_days) -> dict:
    """{first-trading-day-of-month: [top-N symbols by trailing RANK_DAYS mean dvol]}."""
    tdays = pd.to_datetime(sorted(trading_days))
    # first trading day of each calendar month present in the window
    month_starts = {}
    for d in tdays:
        key = (d.year, d.month)
        if key not in month_starts:
            month_starts[key] = d
    members = {}
    idx = dvol.index
    for (_, _), mstart in sorted(month_starts.items()):
        prior = idx[idx < mstart]
        if len(prior) < MIN_COVER:
            continue
        win = dvol.loc[prior[-RANK_DAYS:]]
        cover = win.count()                       # bars present per symbol
        mean_dvol = win.mean()
        elig = mean_dvol[cover >= MIN_COVER].dropna()
        top = elig.sort_values(ascending=False).head(TOP_N)
        members[mstart.date()] = list(top.index)
    return members


def main() -> int:
    dc, tc = clients()
    end = datetime.now(tz=ET)
    start = end - timedelta(days=WINDOW)
    trading_days = get_trading_days(tc, start, end)
    print(f"Window {start.date()} -> {end.date()} | {len(trading_days)} sessions")

    daily_cache = ROOT / "backtest" / f".pit_daily_{WINDOW}d.pkl"
    if daily_cache.exists():
        print(f"Loading cached daily bars {daily_cache.name}")
        blob = pickle.load(open(daily_cache, "rb"))
        dvol, close = blob["dvol"], blob["close"]
    else:
        pool = candidate_pool(tc)
        print(f"Candidate pool: {len(pool)} active US-equity symbols. Fetching daily bars...")
        daily = fetch_daily(dc, pool, start, end)
        if daily.empty:
            print("ERROR: no daily bars", file=sys.stderr)
            return 1
        print(f"Daily rows: {len(daily):,}. Building pivots...")
        dvol, close = build_pivots(daily)
        pickle.dump({"dvol": dvol, "close": close, "days": trading_days},
                    open(daily_cache, "wb"))
        print(f"Cached -> {daily_cache.name}  ({len(dvol.columns)} symbols with data)")

    members = monthly_members(dvol, trading_days)
    union = sorted({s for syms in members.values() for s in syms})
    pickle.dump({"members": members, "union": union, "top_n": TOP_N,
                 "rank_days": RANK_DAYS, "window": WINDOW},
                open(ROOT / "backtest" / f".pit_members_{WINDOW}d.pkl", "wb"))

    # churn / overlap reporting
    from backtest.universe_scan import UNIVERSE
    hand = set(UNIVERSE)
    union_set = set(union)
    months = sorted(members)
    sizes = [len(members[m]) for m in months]
    # average month-over-month turnover
    turns = []
    for a, b in zip(months, months[1:]):
        sa, sb = set(members[a]), set(members[b])
        turns.append(len(sa ^ sb) / max(len(sa), 1))
    print("\n" + "=" * 70)
    print("POINT-IN-TIME UNIVERSE (mechanical top-100 by trailing dollar volume)")
    print("=" * 70)
    print(f"  months ranked        : {len(months)}  ({months[0]} -> {months[-1]})")
    print(f"  per-month size        : {min(sizes)}-{max(sizes)} (target {TOP_N})")
    print(f"  UNION over all months : {len(union)} distinct symbols  <-- minute-fetch size")
    print(f"  avg monthly turnover  : {100 * sum(turns) / len(turns):.1f}% of names swap/month")
    print(f"  hand-picked watchlist : {len(hand)} names")
    print(f"  overlap (hand & PIT)  : {len(hand & union_set)} of {len(hand)} hand names are ever PIT-eligible")
    only_hand = sorted(hand - union_set)
    print(f"  hand names NEVER PIT-eligible ({len(only_hand)}): {', '.join(only_hand) if only_hand else '—'}")
    # a few PIT names that are NOT hand-picked (the bias we were missing)
    extra = [s for s in union if s not in hand]
    print(f"  PIT names NOT hand-picked: {len(extra)}  e.g. {', '.join(extra[:30])}")
    print(f"\nCached members -> .pit_members_{WINDOW}d.pkl. Next: pit_trades.py streams minute")
    print(f"bars for the {len(union)}-name union (per-symbol, no OOM) and runs ORB+trailing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
