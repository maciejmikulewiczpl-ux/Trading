"""Pyramiding the fat tail (Fable E): add 0.5x at +2R, trail covers both lots.

The trailing exit lives on rare big winners. Adding a half-size lot once a trade reaches
+2R (both lots trailing 1R below the HWM, exiting together) fattens the right tail. KEY:
the add does NOT move the stop (the trail tracks price HWM, not position size), so the
EXIT price is identical to the base trail-1R sim — the pyramid P&L is fully determined by
each trade's exit_R and whether it reached +2R. So:
    base_R    = exit_R
    pyramid_R = exit_R + (0.5 * (exit_R - 2)  if HWM reached +2R  else 0)
The add helps trend-day runners (exit_R >> 2) and slightly hurts +2R-touchers that pull
back (exit_R between 1 and 2 after tagging 2R). R-space, capital-agnostic; cap-16 portfolio.

Re-sims trail-1R over cached minute bars (reuses compare_exits infra) to get exit_R AND
the reached-2R flag accurately (incl. EOD exits where HWM isn't recoverable from exit_R).

Run:
    .venv/Scripts/python.exe backtest/compare_pyramid.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.universe_portfolio import portfolio  # noqa: E402
from backtest.compare_selection import _tday  # noqa: E402
from backtest.compare_norefill_trend import trend_eligibility, apply_filter  # noqa: E402
from backtest.compare_exits import load, bucket, POLICIES, EOD  # noqa: E402
from backtest.compare_or_range_realcost import or_pct  # noqa: E402

WINDOWS = [730, 180]
CAP = 16
RISK = 50.0
OR_THR = 0.5
ADD_AT_R = 2.0
ADD_SIZE = 0.5


def sim_with_hwm(day, start, entry, init_stop, eod_ns):
    """Trail-1R exit; returns (exit_R, reached_add_R) — whether HWM tagged +ADD_AT_R."""
    ns, hi, lo, cl, idx = day["ns"], day["hi"], day["lo"], day["cl"], day["idx"]
    n = len(ns)
    risk = entry - init_stop
    if risk <= 0:
        return None
    hwm, stop = entry, init_stop
    reached = False
    for i in range(start, n):
        if ns[i] >= eod_ns:
            return (cl[i] - entry) / risk, reached
        s2 = hwm - 1.0 * risk
        if s2 > stop:
            stop = s2
        if lo[i] <= stop:
            return (stop - entry) / risk, reached
        if hi[i] > hwm:
            hwm = hi[i]
        if (hwm - entry) / risk >= ADD_AT_R:
            reached = True
    return (cl[n - 1] - entry) / risk, reached


def run_window(w):
    all_bars, days, present, trades, closes = load(w)
    elig = trend_eligibility(closes, present, days)
    buckets = bucket(all_bars, present)
    tz = all_bars.index.get_level_values(1).tz
    eod_ns = {d: pd.Timestamp.combine(d, EOD).tz_localize(tz).value for d in days}
    mid = sorted(days)[len(days) // 2]

    rows = []
    for t in trades:
        if t.side != "long" or or_pct(t) > OR_THR:
            continue
        day = buckets.get(t.symbol, {}).get(_tday(t))
        if day is None:
            continue
        start = int(day["idx"].searchsorted(t.entry_time, side="left"))
        if start >= len(day["ns"]):
            continue
        res = sim_with_hwm(day, start, t.entry_price, t.stop_price, eod_ns[_tday(t)])
        if res is None:
            continue
        exit_r, reached = res
        from dataclasses import replace
        rows.append(replace(t, pnl_r=exit_r, pnl_dollars=exit_r * RISK,
                            exit_reason=("add" if reached else "noadd")))

    kept = apply_filter(rows, elig)
    taken = portfolio(kept, CAP)

    def agg(get_r):
        by = {}
        for t in taken:
            by[_tday(t)] = by.get(_tday(t), 0.0) + get_r(t)
        s = pd.Series(by).reindex(sorted(days), fill_value=0.0) * RISK
        eq = s.cumsum()
        dd = (eq - eq.cummax()).min()
        mu, sd = (s / RISK).mean(), (s / RISK).std()
        sharpe = mu / sd * (252 ** 0.5) if sd else float("nan")
        h1 = s[[d for d in s.index if d < mid]].sum()
        h2 = s[[d for d in s.index if d >= mid]].sum()
        return {"pnl": s.sum(), "sharpe": sharpe, "dd": dd, "h1": h1, "h2": h2,
                "avgR": sum(get_r(t) for t in taken) / len(taken)}

    base = agg(lambda t: t.pnl_r)
    pyr = agg(lambda t: t.pnl_r + (ADD_SIZE * (t.pnl_r - ADD_AT_R) if t.exit_reason == "add" else 0.0))
    n_add = sum(1 for t in taken if t.exit_reason == "add")

    print(f"\n=== {w}d: {len(taken)} tight-OR trades, {n_add} reached +{ADD_AT_R:.0f}R "
          f"({100*n_add/len(taken):.0f}%), OOS {mid} ===")
    print(f"  {'policy':<16}{'avgR':>8}{'PnL$':>10}{'Sharpe':>8}{'maxDD$':>9}{'h1':>8}{'h2':>8}")
    print(f"  {'base trail-1R':<16}{base['avgR']:>+8.3f}{base['pnl']:>+10,.0f}{base['sharpe']:>8.2f}"
          f"{base['dd']:>9,.0f}{base['h1']:>+8,.0f}{base['h2']:>+8,.0f}")
    print(f"  {'+ pyramid 0.5x':<16}{pyr['avgR']:>+8.3f}{pyr['pnl']:>+10,.0f}{pyr['sharpe']:>8.2f}"
          f"{pyr['dd']:>9,.0f}{pyr['h1']:>+8,.0f}{pyr['h2']:>+8,.0f}")


def main():
    for w in WINDOWS:
        run_window(w)
    print("\nNote: R-space, ZERO extra cost modeled for the add — the add is a 2nd market entry that")
    print("pays slippage too, so a positive result here must clear ~0.04R/add net before it's real.")
    print("Pyramid wins only if it lifts avgR AND Sharpe without worse drawdown, both windows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
