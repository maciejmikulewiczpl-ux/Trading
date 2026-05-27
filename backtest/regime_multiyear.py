"""Multi-year validation of regime-gated shorts (2021-present).

The 180-day study could only see ONE bear->bull transition. IEX minute history
on this tier reaches back to ~2021-01, which spans 2021 bull, the 2022 BEAR
market, 2023 recovery, 2024-25 bull — several real regimes. This re-tests the
regime gate (shorts only when SPY's prior close < its N-day SMA, ex-TSLA, flip)
against long-only across all of them, broken out by calendar year.

Bars are fetched once and cached to a local pickle (re-runs are instant).
A regime gate is trustworthy if it HELPS in down years (esp. 2022) without
hurting up years.

Run:
    uv run --with pip-system-certs python backtest/regime_multiyear.py
"""
from __future__ import annotations

import pickle
import sys
from datetime import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from strategies.orb import simulate_session  # noqa: E402
from backtest.run_orb import STARTING_EQUITY, WATCHLIST, load_all_bars  # noqa: E402
from backtest.eval_index_short import summarize  # noqa: E402
from backtest.regime_short import params, spy_bearish_by_day, SHORT_SYMS  # noqa: E402

LOOKBACK_DAYS = 2000           # ~5.5y; IEX starts ~2021-01 so older just empties
CACHE = ROOT / "backtest" / f".bars_cache_{LOOKBACK_DAYS}d.pkl"
SMA_WINDOWS = [10, 20, 50]


def load_cached():
    if CACHE.exists():
        print(f"Loading cached bars from {CACHE.name} ...")
        with open(CACHE, "rb") as f:
            d = pickle.load(f)
        return d["bars"], d["days"]
    print("No cache — fetching multi-year bars from Alpaca (one-time, slow)...")
    bars, days = load_all_bars(verbose=True, lookback_days=LOOKBACK_DAYS)
    with open(CACHE, "wb") as f:
        pickle.dump({"bars": bars, "days": days}, f)
    print(f"Cached to {CACHE.name}")
    return bars, days


def bucket(all_bars):
    """{symbol: {date: RTH day-bars}} — pre-split once so per-day sims are O(1)."""
    syms = set(all_bars.index.get_level_values(0).unique())
    out = {}
    for sym in WATCHLIST:
        if sym not in syms:
            continue
        sb = all_bars.xs(sym, level=0)
        t = sb.index.time
        sb = sb[(t >= time(9, 30)) & (t < time(16, 0))]
        out[sym] = {d: g for d, g in sb.groupby(sb.index.date)}
    return out


def run(buckets, days, pick_params):
    trades = []
    for day in days:
        for sym, by_date in buckets.items():
            db = by_date.get(day)
            if db is None or db.empty:
                continue
            trades.extend(simulate_session(db, sym, STARTING_EQUITY, pick_params(day, sym)))
    return trades


def long_only(buckets, days):
    p = {s: params(True, False) for s in WATCHLIST}
    return run(buckets, days, lambda d, s: p[s])


def static_short(buckets, days):
    p = {s: params(True, s in SHORT_SYMS, 1) for s in WATCHLIST}
    return run(buckets, days, lambda d, s: p[s])


def regime_gated(buckets, days, bearish):
    lo = {s: params(True, False) for s in WATCHLIST}
    ws = {s: params(True, s in SHORT_SYMS, 1) for s in WATCHLIST}
    return run(buckets, days, lambda d, s: ws[s] if bearish.get(d, False) else lo[s])


def by_year(trades):
    out: dict[int, list] = {}
    for t in trades:
        out.setdefault(t.date.year, []).append(t)
    return out


def line(label, s):
    if s.get("n", 0) == 0:
        print(f"{label:<30}  (no trades)")
        return
    print(f"{label:<30} {s['n']:>5} {s['n_long']:>5}/{s['n_short']:<5} "
          f"{s['win_rate']:>5.1f}% {s['avg_r']:>+7.3f} "
          f"${s['total_pnl']:>+11,.0f} ${s['max_dd']:>+10,.0f} short=${s['short_pnl']:>+9,.0f}")


HDR = (f"{'config':<30} {'n':>5} {'L/S':>11} {'win%':>6} {'avg_R':>7} "
       f"{'total PnL':>12} {'max DD':>11} {'short pnl':>15}")


def main() -> int:
    all_bars, days = load_cached()
    yrs = sorted({d.year for d in days})
    print(f"Universe: {WATCHLIST}   short set (ex-TSLA): {sorted(SHORT_SYMS)}")
    print(f"Sessions: {len(days)}  ({days[0]} -> {days[-1]})  years {yrs}\n")

    buckets = bucket(all_bars)
    regimes = {w: spy_bearish_by_day(all_bars, w) for w in SMA_WINDOWS}

    # Full-period overall.
    print("===== FULL PERIOD =====")
    print(HDR)
    print("-" * len(HDR))
    lo_tr = long_only(buckets, days)
    line("long only (baseline)", summarize(lo_tr))
    line("static short ex-TSLA +flip", summarize(static_short(buckets, days)))
    gated = {w: regime_gated(buckets, days, regimes[w]) for w in SMA_WINDOWS}
    for w in SMA_WINDOWS:
        line(f"regime SMA{w} (bear-day shorts)", summarize(gated[w]))
    print()

    # Per-year: long-only vs regime SMA20 (representative), with bear-day counts.
    w = 20
    print(f"===== PER YEAR: long-only vs regime SMA{w} =====")
    print(f"{'year':<6} {'bear days':>10} {'long PnL':>12} {'long DD':>11} "
          f"{'regime PnL':>12} {'regime DD':>11} {'short pnl':>11} {'dPnL':>11}")
    print("-" * 90)
    lo_y, rg_y = by_year(lo_tr), by_year(gated[w])
    for y in yrs:
        ld = [d for d in days if d.year == y]
        nbear = sum(1 for d in ld if regimes[w].get(d, False))
        ls = summarize(lo_y.get(y, []))
        rs = summarize(rg_y.get(y, []))
        if ls.get("n", 0) == 0:
            continue
        print(f"{y:<6} {nbear:>4}/{len(ld):<5} ${ls['total_pnl']:>+10,.0f} ${ls['max_dd']:>+9,.0f} "
              f"${rs['total_pnl']:>+10,.0f} ${rs['max_dd']:>+9,.0f} "
              f"${rs['short_pnl']:>+9,.0f} ${rs['total_pnl']-ls['total_pnl']:>+9,.0f}")

    print("\nPASS if regime gating lifts down years (esp. 2022) and ~matches long-only in up years.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
