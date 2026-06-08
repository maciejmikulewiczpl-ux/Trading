"""Robustness battery for the tight-OR / trailing win — before we ship it.

compare_or_range_trail.py showed OR<=0.5% on the trailing exit is a major win that
survives 1.5x slippage. Three ways it could still be a fluke, tested here:
  (1) THRESHOLD SMOOTHNESS: is 0.5% a smooth optimum or a lucky spike? Sweep
      0.3/0.4/0.5/0.6/0.7/0.8/none. A real effect is monotone-ish, not a cliff.
  (2) 2x SLIPPAGE: does it survive double the calibrated cost (paranoid fills)?
  (3) (separate script) 2022 BEAR + cap-aware.
Trailing exit, vol-dial, cap 16, $50, both windows + OOS halves, cents-based cost.

Run (loads minute bars + re-sims trailing exits, a few min):
    .venv/Scripts/python.exe backtest/compare_or_range_robust.py
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
from backtest.compare_or_range_realcost import or_pct, risk_ps, three_rc  # noqa: E402

import pandas as pd  # noqa: E402

WINDOWS = [730, 180]
THRESHOLDS = [None, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
TARGET_MEDIAN_R = 0.042
SLIP_MULT = [1.0, 2.0]

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
    trail = [t for t in apply_filter(reexit(trades, buckets, POLICIES["trail_1R"], eod_ns), elig)
             if t.side == "long"]
    mid = sorted(days)[len(days) // 2]
    prior = prior_vol_flags(closes, days)
    mult = {d: (0.5 if prior[d] else 1.0) for d in days}
    base_cents = TARGET_MEDIAN_R * statistics.median(risk_ps(t) for t in trail) / 2.0

    print(f"\n=== {w}d TRAILING: {len(days)} sessions, OOS {mid} | calibrated slip ${base_cents:.3f}/share ===")
    for sm in SLIP_MULT:
        cents = base_cents * sm
        print(f"\n  slip {sm:.1f}x (${cents:.3f}/share)")
        print("  " + HEAD); print("  " + "-" * len(HEAD))
        for thr in THRESHOLDS:
            kept = trail if thr is None else [t for t in trail if or_pct(t) <= thr]
            taken = portfolio(kept, CAP)
            label = "no filter (baseline)" if thr is None else f"max OR <= {thr:.1f}%"
            prow(label, len(taken), *three_rc(taken, days, mid, mult, cents))


def main():
    for w in WINDOWS:
        run_window(w)
    print("\nReads: a REAL effect is a smooth gradient (tighter -> better, no single-bucket spike)")
    print("and the chosen threshold stays clearly best at 2x slip in BOTH windows. A spike at one")
    print("threshold that vanishes nearby, or that needs the optimistic cost, = overfit -> don't ship.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
