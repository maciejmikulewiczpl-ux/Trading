"""DECISIVE test: does cutting to tight-OR help the LIVE (trailing) config, at REAL cost?

The fixed-2R tests (compare_or_range_filter / _realcost) showed: wide-OR breakouts are
robust losers, and the tight-OR (<0.5%) edge is real but slippage-fragile on fixed-2R.
But the live bot runs the TRAILING exit, where winners run to large R — so a fixed-cents
slippage is a SMALLER fraction of realized R, and tight-OR may survive where it was
marginal. This runs the trailing exit (the shipped one), applies the max-OR filter at
the portfolio level, and charges the honest cents-based cost (calibrated so the median
trade pays ~0.042R), both windows + OOS halves + slippage sweep.

A ship-worthy result = the OR filter lifts Sharpe and cuts maxDD vs no-filter, in BOTH
windows, BOTH halves, and SURVIVES the 1.5x slippage stress. Otherwise we cut only the
wide tail (or nothing). Heavy: loads minute bars + re-simulates trailing exits.

Run:
    .venv/Scripts/python.exe backtest/compare_or_range_trail.py
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.universe_portfolio import portfolio  # noqa: E402
from backtest.compare_norefill_trend import trend_eligibility, apply_filter  # noqa: E402
from backtest.compare_exits import load, bucket, reexit, POLICIES, EOD  # noqa: E402
from backtest.compare_volpause import prior_vol_flags, RISK, CAP  # noqa: E402
from backtest.compare_or_range_realcost import or_pct, risk_ps, three_rc, THRESHOLDS  # noqa: E402

import pandas as pd  # noqa: E402

WINDOWS = [730, 180]
TARGET_MEDIAN_R = 0.042
SLIP_MULT = [1.0, 1.5]

HEAD = f"{'config':<22}{'trades':>7}{'PnL$':>10}{'Sharpe':>8}{'maxDD$':>10}   {'h1 Sh':>6}{'h2 Sh':>6}{'h2 PnL':>9}"


def prow(label, n, f, h1, h2):
    print(f"  {label:<22}{n:>7}{f['pnl']:>+10,.0f}{f['sharpe']:>8.2f}{f['maxdd']:>10,.0f}   "
          f"{h1['sharpe']:>6.2f}{h2['sharpe']:>6.2f}{h2['pnl']:>+9,.0f}")


def run_window(w):
    all_bars, days, present, trades, closes = load(w)
    elig = trend_eligibility(closes, present, days)
    buckets = bucket(all_bars, present)
    tz = all_bars.index.get_level_values(1).tz
    eod_ns = {d: pd.Timestamp.combine(d, EOD).tz_localize(tz).value for d in days}
    trail = apply_filter(reexit(trades, buckets, POLICIES["trail_1R"], eod_ns), elig)
    trail = [t for t in trail if t.side == "long"]

    mid = sorted(days)[len(days) // 2]
    prior = prior_vol_flags(closes, days)
    mult = {d: (0.5 if prior[d] else 1.0) for d in days}

    med_risk = statistics.median(risk_ps(t) for t in trail)
    base_cents = TARGET_MEDIAN_R * med_risk / 2.0
    print(f"\n=== {w}d TRAILING: {len(days)} sessions, OOS {mid} | median risk/share ${med_risk:.2f} "
          f"-> slip ${base_cents:.3f}/share (median = {TARGET_MEDIAN_R:.3f}R) ===")

    for sm in SLIP_MULT:
        cents = base_cents * sm
        tight = [min(2.0 * cents / risk_ps(t), 0.40) for t in trail if or_pct(t) <= 0.5]
        med_t = statistics.median(tight) if tight else float("nan")
        print(f"\n  slip ${cents:.3f}/share ({sm:.1f}x) | median tight-OR(<0.5%) cost {med_t:.3f}R")
        print("  " + HEAD); print("  " + "-" * len(HEAD))
        for thr in THRESHOLDS:
            kept = trail if thr is None else [t for t in trail if or_pct(t) <= thr]
            taken = portfolio(kept, CAP)
            label = "no filter (baseline)" if thr is None else f"max OR <= {thr:.1f}%"
            prow(label, len(taken), *three_rc(taken, days, mid, mult, cents))


def main():
    for w in WINDOWS:
        run_window(w)
    print("\nReads: ship the OR filter only if it lifts Sharpe + cuts maxDD vs baseline in BOTH")
    print("windows AND survives 1.5x slippage. On TRAILING, tight-OR winners run far, so the")
    print("fixed-cents cost should bite less than it did on fixed-2R — this is the honest verdict.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
