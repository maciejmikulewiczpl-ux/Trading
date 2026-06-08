"""Which EXIT best captures the tight-OR runners? (trail-1R was tuned on the old pop.)

Tight-OR trades have tiny initial risk and are the big runners, so the exit is where
their asymmetry is captured or lost. trail-1R was chosen on the FULL breakout population
(compare_exits.py). On the tight-OR subset a different exit may win: a WIDER trail
(1.5R/2R) gives a runner more room before stopping out; a PARTIAL banks half at +1R and
lets the rest run. This compares all policies on tight-OR (<=0.5%) in honest cap-aware $.

Correct $ accounting: pnl_$ = pnl_r * shares * risk_per_share (works for partials, whose
pnl_r is blended), minus legs*cents*shares (partials pay 3 legs: 1 entry + 2 exits; the
rest pay 2). cents calibrated on all-trades median = 0.042R (conservative). Both windows
+ OOS. A new exit ships only if it beats trail-1R on Sharpe AND maxDD in BOTH windows.

Run (loads minute bars once/window, re-sims each policy):
    .venv/Scripts/python.exe backtest/compare_tightOR_exit.py
"""
from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.universe_portfolio import portfolio  # noqa: E402
from backtest.compare_selection import _tday  # noqa: E402
from backtest.compare_norefill_trend import trend_eligibility, apply_filter  # noqa: E402
from backtest.compare_exits import load, bucket, reexit, POLICIES, EOD  # noqa: E402
from backtest.compare_volpause import prior_vol_flags, perf, RISK, CAP  # noqa: E402
from backtest.compare_or_range_realcost import or_pct, risk_ps  # noqa: E402

import pandas as pd  # noqa: E402

WINDOWS = [730, 180]
TIGHT = 0.5
TARGET_MEDIAN_R = 0.042
NOTIONAL_CAP = 10_000.0
ORDER = ["fixed_2R (live)", "trail_1R", "trail_1.5R", "trail_2R", "partial_2R", "partial_trail1R"]


def tight(trades):
    return [t for t in trades if or_pct(t) <= TIGHT]


def dser(taken, days, mult, cents, legs):
    by = {}
    for t in taken:
        rps = risk_ps(t)
        target = RISK * mult.get(_tday(t), 1.0)
        shares = min(math.floor(target / rps), math.floor(NOTIONAL_CAP / t.entry_price))
        if shares <= 0:
            continue
        pnl = t.pnl_r * shares * rps - legs * cents * shares
        by[_tday(t)] = by.get(_tday(t), 0.0) + pnl
    return pd.Series({d: by.get(d, 0.0) for d in sorted(days)})


HEAD = f"{'exit policy':<20}{'trades':>7}{'PnL$':>9}{'Sharpe':>8}{'maxDD$':>9}   {'h1$':>8}{'h2$':>8}{'avg$/tr':>8}"


def run_window(w):
    all_bars, days, present, trades, closes = load(w)
    elig = trend_eligibility(closes, present, days)
    buckets = bucket(all_bars, present)
    tz = all_bars.index.get_level_values(1).tz
    eod_ns = {d: pd.Timestamp.combine(d, EOD).tz_localize(tz).value for d in days}
    mid = sorted(days)[len(days) // 2]
    prior = prior_vol_flags(closes, days)
    half = {d: (0.5 if prior[d] else 1.0) for d in days}

    def trail_for(name):
        return apply_filter([t for t in reexit(trades, buckets, POLICIES[name], eod_ns)
                             if t.side == "long"], elig)

    cents = TARGET_MEDIAN_R * statistics.median(risk_ps(t) for t in trail_for("trail_1R")) / 2.0
    print(f"\n========== {w}d  (tight-OR<= {TIGHT}%, real cap-aware $, OOS {mid}) ==========")
    print(HEAD); print("  " + "-" * (len(HEAD) - 2))
    for name in ORDER:
        legs = 3 if "partial" in name else 2
        taken = portfolio(tight(trail_for(name)), CAP)
        s = dser(taken, days, half, cents, legs)
        h1 = s[[d for d in s.index if d < mid]].sum()
        h2 = s[[d for d in s.index if d >= mid]].sum()
        f = perf(s)
        avgtr = s.sum() / len(taken) if taken else 0.0
        star = "  <- live" if name == "trail_1R" else ""
        print(f"  {name:<20}{len(taken):>7}{f['pnl']:>+9,.0f}{f['sharpe']:>8.2f}{f['maxdd']:>9,.0f}"
              f"   {h1:>+8,.0f}{h2:>+8,.0f}{avgtr:>+8.1f}{star}")


def main():
    for w in WINDOWS:
        run_window(w)
    print("\nReads: a new exit ships only if it beats trail_1R on Sharpe AND maxDD in BOTH windows.")
    print("Wider trail capturing more = tight-OR runners want room. Partial winning = bank-half")
    print("smooths the variance. If trail_1R stays best, the live exit already fits tight-OR.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
