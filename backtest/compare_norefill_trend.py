"""Does the no-refill churn fix still beat refill once the LIVE trend filter is on?

compare_selection.py / validate_selection.py found that capping TOTAL DAILY
ENTRIES (no-refill) beats the live cap-8 concurrent-with-refill mechanism — but
those backtests were long-only WITHOUT the live 200d trend filter, which already
removes weak longs. This closes that caveat: it replays the trend filter exactly
as live (per session, using daily closes strictly before that day: close > 200d
SMA AND 20d return > SPY's 20d return) and compares refill vs no-refill with the
filter ON.

The actual current live behaviour = refill + filter. The candidate = no-refill +
filter. PASS BAR: no-refill+filter beats refill+filter on PnL in the full window
AND both OOS halves, for BOTH the 180d and 730d windows. Reference rows (filter
OFF) show how much the filter itself already does.

Caches (all gitignored via the .bars_cache_* rule):
  .bars_cache_univ_{w}d.pkl    minute bars (from compare_selection.py; required)
  .bars_cache_trades_{w}d.pkl  the broad-universe ORB signals (built once)
  .bars_cache_daily_{w}d.pkl   daily adj closes for the trend filter (fetched once)

Run (needs the minute caches; fetches daily bars on first run):
    .venv/Scripts/python.exe backtest/compare_norefill_trend.py
"""
from __future__ import annotations

import os
import pickle
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from strategies.orb import Params  # noqa: E402
from backtest.run_orb import STARTING_EQUITY, load_env, run_backtest  # noqa: E402
from backtest.universe_portfolio import perf, portfolio  # noqa: E402
from backtest.compare_selection import daily_cap, _tday  # noqa: E402

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

WINDOWS = [180, 730]
CAP = 8
SMA_DAYS = 200       # live TREND_SMA_DAYS
RET_DAYS = 20        # live TREND_RET_DAYS
DAILY_BUFFER_DAYS = 360   # calendar days of daily history before window start (>=200 sessions)
PARAMS = Params(or_minutes=15, target_r=2.0, risk_per_trade=100.0,
                max_position_pct=0.25, max_position_dollars=10_000.0,
                no_entry_after_time=time(11, 30))


def bars_cache(w):
    return ROOT / "backtest" / f".bars_cache_univ_{w}d.pkl"


def trades_cache(w):
    return ROOT / "backtest" / f".bars_cache_trades_{w}d.pkl"


def daily_cache(w):
    return ROOT / "backtest" / f".bars_cache_daily_{w}d.pkl"


def fetch_daily_closes(symbols, start, end) -> pd.DataFrame:
    """Adjusted daily closes (wide: index=naive ET date, cols=symbols), chunked."""
    load_env()
    key, sec = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    if not key or not sec:
        raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY not set in .env")
    dc = StockHistoricalDataClient(key, sec)
    frames = []
    for i in range(0, len(symbols), 20):
        grp = symbols[i:i + 20]
        print(f"  daily bars {i + 1}-{i + len(grp)} of {len(symbols)} ...", flush=True)
        req = StockBarsRequest(
            symbol_or_symbols=grp, timeframe=TimeFrame.Day,
            start=start.astimezone(UTC), end=end.astimezone(UTC),
            feed=DataFeed.IEX, adjustment=Adjustment.ALL,
        )
        df = dc.get_stock_bars(req).df
        if not df.empty:
            frames.append(df["close"].unstack(level=0))
    closes = pd.concat(frames, axis=1) if frames else pd.DataFrame()
    closes.index = pd.to_datetime(closes.index).tz_convert(ET).normalize().tz_localize(None)
    return closes.sort_index()


def trend_eligibility(closes: pd.DataFrame, present, trading_days) -> dict:
    """{session_date: set(eligible symbols)}; matches live compute_trend_eligibility.

    Eligible = (prior close > 200d SMA) AND (20d return > SPY's 20d return), using
    only daily closes strictly before the session. None for a day = fail-open
    (every name eligible), exactly as live when < 200 prior days exist.
    """
    sma = closes.rolling(SMA_DAYS).mean()
    ret = closes.pct_change(RET_DAYS)
    out = {}
    for d in trading_days:
        dts = pd.Timestamp(d)
        prior = closes.index[closes.index < dts]
        if len(prior) < SMA_DAYS:
            out[d] = None
            continue
        last = prior[-1]
        pc, s, r = closes.loc[last], sma.loc[last], ret.loc[last]
        spyr = r.get("SPY")
        if spyr is None or pd.isna(spyr):
            out[d] = None
            continue
        elig = set()
        for sym in present:
            a, b, c = pc.get(sym), s.get(sym), r.get(sym)
            if any(pd.isna(x) for x in (a, b, c)):
                continue
            if a > b and c > spyr:
                elig.add(sym)
        out[d] = elig
    return out


def apply_filter(trades, elig: dict):
    """Keep a (long) trade iff its symbol is eligible that day (None = keep all)."""
    out = []
    for t in trades:
        e = elig.get(_tday(t))
        if e is None or t.symbol in e:
            out.append(t)
    return out


def perf_on(taken, days):
    dset = set(days)
    return perf([t for t in taken if _tday(t) in dset], days)


def three(taken, days, mid):
    d1 = [d for d in days if d < mid]
    d2 = [d for d in days if d >= mid]
    return perf_on(taken, days), perf_on(taken, d1), perf_on(taken, d2)


HEAD = (f"{'config':<28}{'PnL$':>11}{'Sharpe':>8}   "
        f"{'PnL h1':>9}{'Sh h1':>7}   {'PnL h2':>9}{'Sh h2':>7}")


def prow(label, full, h1, h2):
    def c(s, k, fmt):
        return format(s[k], fmt) if s.get("n", 0) else "—"
    print(f"{label:<28}{c(full,'pnl','>+11,.0f')}{c(full,'sharpe','>8.2f')}   "
          f"{c(h1,'pnl','>+9,.0f')}{c(h1,'sharpe','>7.2f')}   "
          f"{c(h2,'pnl','>+9,.0f')}{c(h2,'sharpe','>7.2f')}")


def run_window(w) -> dict:
    if not bars_cache(w).exists():
        print(f"  ! no minute cache {bars_cache(w).name} — run compare_selection.py for {w}d.")
        return {}
    with open(bars_cache(w), "rb") as f:
        d = pickle.load(f)
    all_bars, trading_days = d["bars"], d["days"]
    present = sorted(all_bars.index.get_level_values(0).unique())
    mid = sorted(trading_days)[len(trading_days) // 2]

    # trades (cached)
    if trades_cache(w).exists():
        with open(trades_cache(w), "rb") as f:
            all_trades = pickle.load(f)
    else:
        all_trades, _ = run_backtest(all_bars, trading_days, present, PARAMS, STARTING_EQUITY)
        with open(trades_cache(w), "wb") as f:
            pickle.dump(all_trades, f)

    # daily closes for the trend filter (cached)
    if daily_cache(w).exists():
        with open(daily_cache(w), "rb") as f:
            closes = pickle.load(f)
    else:
        start = datetime.combine(min(trading_days), time(0, 0), ET) - timedelta(days=DAILY_BUFFER_DAYS)
        end = datetime.combine(max(trading_days), time(0, 0), ET) + timedelta(days=1)
        print(f"  fetching daily bars for {len(present)} names + SPY ({w}d)...")
        closes = fetch_daily_closes(sorted(set(present) | {"SPY"}), start, end)
        with open(daily_cache(w), "wb") as f:
            pickle.dump(closes, f)

    elig = trend_eligibility(closes, present, trading_days)
    n_elig = [len(v) for v in elig.values() if v is not None]
    filt = apply_filter(all_trades, elig)

    refill_off = three(portfolio(all_trades, CAP), trading_days, mid)
    norefill_off = three(daily_cap(all_trades, CAP, lambda dd, t: t.entry_time, False), trading_days, mid)
    refill_on = three(portfolio(filt, CAP), trading_days, mid)             # = ACTUAL LIVE
    norefill_on = three(daily_cap(filt, CAP, lambda dd, t: t.entry_time, False), trading_days, mid)  # candidate

    print(f"\n=== {w}d: {len(present)} names, {len(trading_days)} sessions, OOS split {mid} ===")
    if n_elig:
        print(f"    trend filter: {np.mean(n_elig):.0f}/{len(present)} names eligible on an "
              f"avg day; {len(all_trades)} signals -> {len(filt)} after filter")
    print(HEAD)
    print("-" * len(HEAD))
    prow("refill, filter OFF", *refill_off)
    prow("no-refill, filter OFF", *norefill_off)
    prow("refill + FILTER (LIVE)", *refill_on)
    prow("no-refill + FILTER (cand.)", *norefill_on)
    return {"live": refill_on, "cand": norefill_on}


def main() -> int:
    results = {w: run_window(w) for w in WINDOWS}

    print("\n" + "=" * 64)
    print("PASS BAR: no-refill+filter must beat refill+filter (LIVE) on PnL")
    print("in full + both OOS halves, both windows.")
    print("=" * 64)
    all_pass = True
    for w in WINDOWS:
        r = results.get(w)
        if not r:
            print(f"  {w}d: SKIPPED"); all_pass = False; continue
        checks = []
        for i in range(3):
            cand, live = r["cand"][i], r["live"][i]
            checks.append(cand.get("n", 0) > 0 and cand["pnl"] > live.get("pnl", 0))
        ok = all(checks)
        all_pass &= ok
        seg = "  ".join(f"{s}:{'ok' if c else 'X'}" for s, c in zip(("full", "h1", "h2"), checks))
        print(f"  {w}d: {'PASS' if ok else 'FAIL'}   [{seg}]")
    print("-" * 64)
    if all_pass:
        print("VERDICT: PASS — the no-refill churn fix survives the live trend filter.")
        print("         Worth shipping to the runner behind a config flag (paper-first).")
    else:
        print("VERDICT: FAIL — once the trend filter is on, no-refill does not robustly")
        print("         beat refill. The filter already captures the churn benefit; leave")
        print("         the live mechanism as-is.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
