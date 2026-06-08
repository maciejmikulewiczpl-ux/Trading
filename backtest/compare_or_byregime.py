"""Does a LONGER opening range on high-vol days beat the fixed 15-min OR?

Idea: on a chaotic morning the first 15 min is noise, so OR_high/OR_low are
unreliable and breakouts whipsaw. A longer OR (30/45 min) on high-vol days lets
the volatility settle -> more reliable levels, fewer false breakouts. Tradeoffs:
later entries (less runway to EOD), wider stops (smaller size), fewer trades.

Tests a REGIME-CONDITIONAL OR: 15 min on calm days, {30,45} min on high-vol days
(SPY 20d realized vol > 126d median), vs the fixed 15-min baseline. Fixed-2R
exits, net of ~0.042R cost, cap 16, $50, half-risk dial on high-vol days (as live).
Key diagnostic: avgR on HIGH-VOL days at each OR length — does longer OR lift the
expectancy of the volatile days, or just trade fewer/later for the same poor edge?

Run (re-runs the backtest at each OR length, ~10-15 min):
    .venv/Scripts/python.exe backtest/compare_or_byregime.py
"""
from __future__ import annotations

import sys
from datetime import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from strategies.orb import Params  # noqa: E402
from backtest.run_orb import run_backtest, STARTING_EQUITY  # noqa: E402
from backtest.universe_portfolio import portfolio  # noqa: E402
from backtest.compare_norefill_trend import trend_eligibility, apply_filter  # noqa: E402
from backtest.compare_exits import load  # noqa: E402
from backtest.compare_volpause import prior_vol_flags, series, perf, RISK, CAP  # noqa: E402

WINDOW = "730"
ORS = [15, 30, 45]
COST = 0.042


def trades_for_or(all_bars, days, present, cached15, or_min):
    if or_min == 15:
        return cached15
    p = Params(or_minutes=or_min, target_r=2.0, risk_per_trade=100.0, max_position_pct=0.25,
               max_position_dollars=10_000.0, no_entry_after_time=time(11, 30))
    t, _ = run_backtest(all_bars, days, present, p, STARTING_EQUITY)
    return t


def main():
    all_bars, days, present, cached15, closes = load(int(WINDOW))
    elig = trend_eligibility(closes, present, days)
    hv = prior_vol_flags(closes, days)
    mid = sorted(days)[len(days) // 2]
    from backtest.compare_selection import _tday

    taken_by_or = {}
    for o in ORS:
        raw = trades_for_or(all_bars, days, present, cached15, o)
        taken_by_or[o] = portfolio(apply_filter(raw, elig), CAP)
        # diagnostic: avgR (net) on high-vol vs calm days at this OR length
        def avgr(flagval):
            sel = [t.pnl_r - COST for t in taken_by_or[o] if hv[_tday(t)] == flagval]
            return (sum(sel) / len(sel)) if sel else 0.0
        nhv = sum(1 for t in taken_by_or[o] if hv[_tday(t)])
        print(f"OR={o:>2}min: {len(taken_by_or[o])} trades ({nhv} on hi-vol days) | "
              f"net avgR  hi-vol {avgr(True):+.3f}  calm {avgr(False):+.3f}")

    print()
    # conditional strategies: calm days use OR15, hi-vol days use OR{30,45}
    def cond(or_hi):
        keep = [t for t in taken_by_or[15] if not hv[_tday(t)]]
        keep += [t for t in taken_by_or[or_hi] if hv[_tday(t)]]
        return keep
    mult = {d: (0.5 if hv[d] else 1.0) for d in days}   # live half-risk dial

    def report(label, taken):
        s = perf(series(taken, days, mult))
        d1 = [d for d in days if d < mid]
        d2 = [d for d in days if d >= mid]
        s1 = perf(series([t for t in taken if _tday(t) < mid], d1, mult))
        s2 = perf(series([t for t in taken if _tday(t) >= mid], d2, mult))
        print(f"{label:<28}{s['pnl']:>+10,.0f}{s['sharpe']:>8.2f}{s['maxdd']:>10,.0f}   "
              f"h1 {s1['sharpe']:>5.2f}  h2 {s2['sharpe']:>5.2f}")

    print(f"=== {WINDOW}d: conditional OR (half-risk dial, fixed-2R, net of cost) ===")
    print(f"{'config':<28}{'PnL$':>10}{'Sharpe':>8}{'maxDD$':>10}")
    report("baseline OR15 (shipped)", taken_by_or[15])
    report("calm OR15 + hi-vol OR30", cond(30))
    report("calm OR15 + hi-vol OR45", cond(45))
    print("\nReads: the diagnostic (net avgR on hi-vol days) is the core test — if longer OR")
    print("LIFTS hi-vol-day avgR, the idea works; if not, it just trades fewer/later for the")
    print("same poor edge. Conditional config must beat baseline OR15 on Sharpe + both halves.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
