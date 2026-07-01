"""tightOR_robustness_grid.py -- is the 0.5%-of-price OR cut a PLATEAU or a CHERRY-PICK? (review #7)

The tight-OR win ships at OR<=0.5%. A single lucky threshold is a red flag; a robust edge is a
PLATEAU -- neighbors (0.3-0.7) should ALL beat baseline, changing smoothly, no cliff at 0.5. This
sweeps the fine grid on the cap-aware real-$ lens (the sizing truth), both windows + OOS halves,
1x + 1.5x slip. Same calibrated cents (market constant), same $10k cap, same vol-dial.

Reads as ROBUST iff: every threshold in the grid beats the no-filter baseline on Sharpe, the metric
moves monotonically-ish (no single-point spike at 0.5), and both OOS halves stay non-negative around
the shipped value. Reads as CHERRY-PICK iff 0.5 spikes while 0.4/0.6 collapse.

Run (loads minute bars + re-sims trailing; ~few min):
    .venv/Scripts/python.exe backtest/tightOR_robustness_grid.py
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.compare_exits import load, bucket, reexit, POLICIES, EOD  # noqa: E402
from backtest.compare_norefill_trend import trend_eligibility, apply_filter  # noqa: E402
from backtest.compare_or_range_capaware import dollar_series, TARGET_MEDIAN_R  # noqa: E402
from backtest.compare_or_range_realcost import or_pct, risk_ps  # noqa: E402
from backtest.compare_volpause import prior_vol_flags, perf, RISK, CAP  # noqa: E402
from backtest.universe_portfolio import portfolio  # noqa: E402

import pandas as pd  # noqa: E402

WINDOWS = [730, 180]
GRID = [None, 0.3, 0.4, 0.5, 0.6, 0.7]   # None = no filter (baseline)
SLIP_MULT = [1.0, 1.5]

HEAD = f"{'OR cut':<16}{'trades':>7}{'PnL$':>10}{'Sharpe':>8}{'maxDD$':>9}   {'h1 PnL':>9}{'h2 PnL':>9}"


def run_window(w: int):
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

    print(f"\n=== {w}d cap-aware real-$, OOS {mid} ===")
    for sm in SLIP_MULT:
        cents = base_cents * sm
        print(f"\n  slip {sm:.1f}x (${cents:.3f}/share)")
        print("  " + HEAD); print("  " + "-" * len(HEAD))
        for thr in GRID:
            kept = trail if thr is None else [t for t in trail if or_pct(t) <= thr]
            taken = portfolio(kept, CAP)
            s, _ = dollar_series(taken, days, mult, cents)
            f = perf(s)
            h1 = s[[d for d in s.index if d < mid]].sum()
            h2 = s[[d for d in s.index if d >= mid]].sum()
            label = "no filter" if thr is None else f"OR <= {thr:.1f}%"
            star = "  <<shipped" if thr == 0.5 else ""
            print(f"  {label:<16}{len(taken):>7}{f['pnl']:>+10,.0f}{f['sharpe']:>8.2f}"
                  f"{f['maxdd']:>9,.0f}   {h1:>+9,.0f}{h2:>+9,.0f}{star}")


def main():
    print("Tight-OR threshold robustness: plateau (robust) vs single-point spike (cherry-pick)?")
    for w in WINDOWS:
        run_window(w)
    print("\nRead: if every grid threshold beats no-filter on Sharpe and the curve is smooth around")
    print("0.5 (no cliff, both halves OK), the cut is a PLATEAU = robust. If 0.5 spikes while 0.4/0.6")
    print("collapse, it's a cherry-pick and the shipped value is overfit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
