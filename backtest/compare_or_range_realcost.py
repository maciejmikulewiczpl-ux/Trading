"""The make-or-break test for the tight-OR finding: does it survive REALISTIC slippage?

compare_or_range_filter.py found that cutting to tight opening ranges (<0.5% of
price) ~triples Sharpe in both windows. But tight OR = tiny risk-per-share, and a
market order slips a roughly CONSTANT number of cents/share — so in R-units the
slippage BLOWS UP on exactly those tight-OR trades. A flat 0.042R cost (used in the
filter test) hides this. This re-prices every trade with a cents-based cost that
scales 1/risk_per_share, the honest model for the tight-OR question.

Calibration (no free lunch / no thumb on the scale): pick the per-share cent cost so
the MEDIAN trade still pays ~0.042R (our measured round-trip baseline). Then every
trade pays its OWN R-cost = 2*cents / risk_per_share — tight-OR trades pay much more,
wide-OR much less. We sweep the cents level (calibrated, 1.5x, 2x) for robustness.

If the <=0.5% cut STILL beats baseline on Sharpe+drawdown in both windows after this,
the tight-OR edge is real and tradeable. If it collapses, it was the slippage trap
and we cut the WIDE tail only (the part that's a robust loser at any cost). Cheap:
cached trades, no minute bars.

Run:
    .venv/Scripts/python.exe backtest/compare_or_range_realcost.py
"""
from __future__ import annotations

import pickle
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.universe_portfolio import portfolio  # noqa: E402
from backtest.compare_selection import _tday  # noqa: E402
from backtest.compare_norefill_trend import trend_eligibility, apply_filter  # noqa: E402
from backtest.compare_volpause import prior_vol_flags, perf, RISK, CAP  # noqa: E402

import pandas as pd  # noqa: E402

WINDOWS = [730, 180]
THRESHOLDS = [None, 0.8, 0.5]
TARGET_MEDIAN_R = 0.042      # calibrate cents so the median trade pays this round-trip
COST_CAP_R = 0.40            # don't let a degenerate tiny-risk trade pay an absurd R
SLIP_MULT = [1.0, 1.5, 2.0]  # robustness sweep on the cents level


def or_pct(t):
    return (t.or_high - t.or_low) / t.entry_price * 100 if t.entry_price else 0.0


def risk_ps(t):
    return max(t.entry_price - t.stop_price, 1e-6)


def load_cached(w):
    trades = pickle.load(open(ROOT / "backtest" / f".bars_cache_trades_{w}d.pkl", "rb"))
    closes = pickle.load(open(ROOT / "backtest" / f".bars_cache_daily_{w}d.pkl", "rb"))
    present = sorted({t.symbol for t in trades})
    days = sorted({_tday(t) for t in trades})
    trades = apply_filter(trades, trend_eligibility(closes, present, days))
    trades = [t for t in trades if t.side == "long"]
    return trades, closes, days


def series_rc(taken, days, mult, cents):
    """Daily P&L in $ using a per-trade cents-based cost = 2*cents/risk_per_share (R), capped."""
    by = {}
    for t in taken:
        cost_r = min(2.0 * cents / risk_ps(t), COST_CAP_R)
        by[_tday(t)] = by.get(_tday(t), 0.0) + (t.pnl_r - cost_r)
    idx = sorted(days)
    s = pd.Series({d: by.get(d, 0.0) for d in idx})
    m = pd.Series({d: mult.get(d, 1.0) for d in idx})
    return s * RISK * m


def three_rc(taken, days, mid, mult, cents):
    d1 = [d for d in days if d < mid]
    d2 = [d for d in days if d >= mid]
    return (perf(series_rc(taken, days, mult, cents)),
            perf(series_rc(taken, d1, mult, cents)),
            perf(series_rc(taken, d2, mult, cents)))


HEAD = f"{'config':<22}{'trades':>7}{'PnL$':>10}{'Sharpe':>8}{'maxDD$':>10}   {'h1 Sh':>6}{'h2 Sh':>6}{'h2 PnL':>9}"


def prow(label, n, f, h1, h2):
    print(f"{label:<22}{n:>7}{f['pnl']:>+10,.0f}{f['sharpe']:>8.2f}{f['maxdd']:>10,.0f}   "
          f"{h1['sharpe']:>6.2f}{h2['sharpe']:>6.2f}{h2['pnl']:>+9,.0f}")


def run_window(w):
    trades, closes, days = load_cached(w)
    mid = sorted(days)[len(days) // 2]
    prior = prior_vol_flags(closes, days)
    mult = {d: (0.5 if prior[d] else 1.0) for d in days}

    med_risk = statistics.median(risk_ps(t) for t in trades)
    base_cents = TARGET_MEDIAN_R * med_risk / 2.0     # cents s.t. median trade pays TARGET_MEDIAN_R
    print(f"\n=== {w}d: {len(days)} sessions, OOS {mid} | median risk/share ${med_risk:.2f} -> "
          f"calibrated slip ${base_cents:.3f}/share (median trade = {TARGET_MEDIAN_R:.3f}R) ===")

    for sm in SLIP_MULT:
        cents = base_cents * sm
        # show what tight-OR trades actually pay at this cents level
        costs = [min(2.0 * cents / risk_ps(t), COST_CAP_R) for t in trades]
        tight = [min(2.0 * cents / risk_ps(t), COST_CAP_R) for t in trades if or_pct(t) <= 0.5]
        med_c = statistics.median(costs)
        med_t = statistics.median(tight) if tight else float("nan")
        print(f"\n  slip ${cents:.3f}/share ({sm:.1f}x) | median cost {med_c:.3f}R | "
              f"median tight-OR(<0.5%) cost {med_t:.3f}R")
        print("  " + HEAD); print("  " + "-" * len(HEAD))
        for thr in THRESHOLDS:
            kept = trades if thr is None else [t for t in trades if or_pct(t) <= thr]
            taken = portfolio(kept, CAP)
            label = "no filter (baseline)" if thr is None else f"max OR <= {thr:.1f}%"
            f, h1, h2 = three_rc(taken, days, mid, mult, cents)
            print("  ", end="")
            prow(label, len(taken), f, h1, h2)


def main():
    for w in WINDOWS:
        run_window(w)
    print("\nReads: under a cents-based cost, tight-OR trades pay a MUCH bigger R-haircut. If the")
    print("<=0.5% cut still beats baseline on Sharpe+maxDD in both windows here, the edge is real.")
    print("If its advantage collapses as slip rises, it was the slippage trap -> cut only the wide tail.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
