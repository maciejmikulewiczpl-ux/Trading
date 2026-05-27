"""Harden the regime signal for gated shorts (multi-year, cached bars).

The plain single-SMA gate (regime_multiyear.py) helped in the 2022 bear but was
fragile: only SMA20 beat long-only (SMA10/50 didn't), and it gave back money in
the 2023 recovery by shorting one-day dips. This tests signals designed to be
less whippy and less knife-edged on the exact window:

  - confirm-N : SPY below SMA for N CONSECUTIVE days (kills one-day dips)
  - slope     : SPY below SMA AND the SMA itself falling (real downtrend)
  - dual      : SPY below BOTH a fast and a slow SMA
  - confirm+slope : both conditions

Robustness is the bar: a good signal beats long-only across a RANGE of its
parameter, and lifts down years (2022) without bleeding recoveries (2023).

Decision uses only prior-day info (shift by 1) — no lookahead. Run:
    uv run --with pip-system-certs python backtest/regime_hardened.py
"""
from __future__ import annotations

import sys
from datetime import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.eval_index_short import summarize  # noqa: E402
from backtest.regime_multiyear import (  # noqa: E402
    bucket, by_year, load_cached, long_only, regime_gated,
)


def spy_daily_close(all_bars) -> pd.Series:
    spy = all_bars.xs("SPY", level=0)
    t = spy.index.time
    rth = spy[(t >= time(9, 30)) & (t < time(16, 0))]
    s = rth.groupby(rth.index.date).last()["close"]
    s.index = pd.Index(list(s.index))  # plain date index
    return s


def _decision(asof_bool: pd.Series) -> dict:
    """Boolean 'bearish as of this close' -> {date: bearish for NEXT session}."""
    nxt = asof_bool.shift(1)
    return {d: bool(v) for d, v in nxt.items() if pd.notna(v)}


def sig_plain(close, w):
    return _decision(close < close.rolling(w).mean())


def sig_confirm(close, w, n):
    below = close < close.rolling(w).mean()
    return _decision(below.rolling(n).sum() == n)


def sig_slope(close, w, lag=5):
    sma = close.rolling(w).mean()
    return _decision((close < sma) & (sma < sma.shift(lag)))


def sig_dual(close, fast, slow):
    return _decision((close < close.rolling(fast).mean()) &
                     (close < close.rolling(slow).mean()))


def sig_confirm_slope(close, w, n, lag=5):
    sma = close.rolling(w).mean()
    cond = (close < sma) & (sma < sma.shift(lag))
    return _decision(cond.rolling(n).sum() == n)


def n_bear(days, dec):
    return sum(1 for d in days if dec.get(d, False))


def line(label, s, base_pnl=None):
    if s.get("n", 0) == 0:
        print(f"{label:<30}  (no trades)"); return
    d = f"  dPnL ${s['total_pnl']-base_pnl:>+8,.0f}" if base_pnl is not None else ""
    print(f"{label:<30} {s['n']:>5} {s['n_long']:>5}/{s['n_short']:<5} "
          f"{s['win_rate']:>5.1f}% {s['avg_r']:>+6.3f} "
          f"${s['total_pnl']:>+10,.0f} ${s['max_dd']:>+9,.0f} "
          f"short=${s['short_pnl']:>+8,.0f}{d}")


HDR = (f"{'config':<30} {'n':>5} {'L/S':>11} {'win%':>6} {'avg_R':>6} "
       f"{'total PnL':>11} {'max DD':>10} {'short pnl':>14}")


def main() -> int:
    all_bars, days = load_cached()
    yrs = sorted({d.year for d in days})
    buckets = bucket(all_bars)
    close = spy_daily_close(all_bars)

    lo = long_only(buckets, days)
    lo_s = summarize(lo)
    base = lo_s["total_pnl"]
    print(f"Sessions: {len(days)} ({days[0]} -> {days[-1]})   years {yrs}")
    print(f"Long-only baseline: PnL ${base:+,.0f}  DD ${lo_s['max_dd']:+,.0f}\n")

    # --- Candidate hardened signals (full period) ---
    candidates = {
        "plain SMA20 (ref)":      sig_plain(close, 20),
        "confirm-3 SMA20":        sig_confirm(close, 20, 3),
        "confirm-5 SMA20":        sig_confirm(close, 20, 5),
        "slope SMA20 (lag5)":     sig_slope(close, 20, 5),
        "dual 20&50":             sig_dual(close, 20, 50),
        "confirm-3 + slope SMA20": sig_confirm_slope(close, 20, 3, 5),
        "confirm-5 + slope SMA50": sig_confirm_slope(close, 50, 5, 10),
    }
    print("===== FULL PERIOD: hardened signals vs long-only =====")
    print(HDR); print("-" * len(HDR))
    line("long only (baseline)", lo_s)
    gated = {}
    for name, dec in candidates.items():
        gated[name] = regime_gated(buckets, days, dec)
        line(name, summarize(gated[name]), base)
    print()

    # --- Robustness: sweep confirm-N on SMA20; all should beat long-only ---
    print("===== ROBUSTNESS: confirm-N sweep on SMA20 (want ALL >= baseline) =====")
    print(HDR); print("-" * len(HDR))
    line("long only (baseline)", lo_s)
    for n in [1, 2, 3, 4, 5, 7]:
        dec = sig_confirm(close, 20, n)
        line(f"confirm-{n} SMA20", summarize(regime_gated(buckets, days, dec)), base)
    print()

    # --- Robustness: confirm-3 across SMA windows (was the fragile axis) ---
    print("===== ROBUSTNESS: confirm-3 across SMA windows (plain was knife-edged here) =====")
    print(HDR); print("-" * len(HDR))
    line("long only (baseline)", lo_s)
    for w in [10, 15, 20, 30, 50]:
        dec = sig_confirm(close, w, 3)
        line(f"confirm-3 SMA{w}", summarize(regime_gated(buckets, days, dec)), base)
    print()

    # --- Per-year: long vs best hardened (confirm-3 SMA20) focus 2022/2023 ---
    best_name = "confirm-3 SMA20"
    best = regime_gated(buckets, days, candidates.get(best_name) or sig_confirm(close, 20, 3))
    print(f"===== PER YEAR: long-only vs {best_name} =====")
    print(f"{'year':<6} {'bear days':>10} {'long PnL':>11} {'long DD':>10} "
          f"{'reg PnL':>11} {'reg DD':>10} {'short':>10} {'dPnL':>10}")
    print("-" * 84)
    lo_y, bt_y = by_year(lo), by_year(best)
    dec = sig_confirm(close, 20, 3)
    for y in yrs:
        ld = [d for d in days if d.year == y]
        ls, rs = summarize(lo_y.get(y, [])), summarize(bt_y.get(y, []))
        if ls.get("n", 0) == 0:
            continue
        print(f"{y:<6} {n_bear(ld, dec):>4}/{len(ld):<5} "
              f"${ls['total_pnl']:>+9,.0f} ${ls['max_dd']:>+8,.0f} "
              f"${rs['total_pnl']:>+9,.0f} ${rs['max_dd']:>+8,.0f} "
              f"${rs['short_pnl']:>+8,.0f} ${rs['total_pnl']-ls['total_pnl']:>+8,.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
