"""Robustness sweep for the diversified dual-momentum winner from run_trend.py.

run_trend.py used canonical defaults (12-month lookback, top-3). Before trusting
it, confirm the edge is a plateau, not a knife-edge: sweep lookback x top-K and
add a blended-lookback variant (averaging 3/6/12-month momentum ranks — a
standard way to avoid single-window luck). If most of the grid shows similar
Sharpe / drawdown, the strategy is robust. If only 12m/top-3 is good and
neighbours collapse, it was overfit.

  .venv-openbb\\Scripts\\python.exe backtest\\trend_robustness.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtest.run_trend import (  # noqa: E402
    CASH, EVAL_START, RISK, evaluate, load_monthly, w_div_momentum,
)


def w_div_momentum_blend(monthly, lookbacks=(3, 6, 12), topk=3):
    """Rank risk assets by the AVERAGE of their momentum across several lookbacks;
    hold the top-K that beat cash on that blended score, else cash."""
    moms = {lb: monthly / monthly.shift(lb) - 1 for lb in lookbacks}
    blend = sum(moms.values()) / len(lookbacks)
    w = pd.DataFrame(0.0, index=monthly.index, columns=monthly.columns)
    for t in monthly.index:
        mc = blend.loc[t, CASH]
        if np.isnan(mc):
            continue
        scores = {a: blend.loc[t, a] for a in RISK if not np.isnan(blend.loc[t, a])}
        eligible = {a: s for a, s in scores.items() if s > mc}
        top = sorted(eligible, key=eligible.get, reverse=True)[:topk]
        if top:
            for a in top:
                w.loc[t, a] = 1.0 / topk
            w.loc[t, CASH] = 1.0 - len(top) / topk
        else:
            w.loc[t, CASH] = 1.0
    return w


def main():
    print(f"Loading monthly data (yfinance)... metrics from {EVAL_START}")
    monthly = load_monthly()
    monthly_ret = monthly.pct_change()
    rf = monthly_ret[CASH]

    lookbacks = [3, 6, 9, 12]
    topks = [2, 3, 4]

    print("\nDiversified momentum sweep — each cell = CAGR% / Sharpe / maxDD%")
    print(f"{'lookback':>10} | " + " | ".join(f"top-{k:<14}" for k in topks))
    print("-" * 70)
    grid = {}
    for lb in lookbacks:
        cells = []
        for k in topks:
            m = evaluate(w_div_momentum(monthly, lb=lb, topk=k), monthly_ret, rf)
            grid[(lb, k)] = m
            cells.append(f"{m['cagr']:>5.1f}/{m['sharpe']:.2f}/{m['maxdd']:>5.0f}")
        print(f"{lb:>8}mo | " + " | ".join(f"{c:<17}" for c in cells))

    # Blended-lookback variants
    print("\nBlended-lookback (avg of 3/6/12mo) variants:")
    print(f"  {'config':<22}{'CAGR%':>7}{'Sharpe':>8}{'maxDD%':>8}{'2008':>8}{'2022':>8}")
    for k in topks:
        m = evaluate(w_div_momentum_blend(monthly, (3, 6, 12), topk=k), monthly_ret, rf)
        print(f"  blend top-{k:<13}{m['cagr']:>7.2f}{m['sharpe']:>8.2f}"
              f"{m['maxdd']:>8.1f}{m['y2008']:>+8.1f}{m['y2022']:>+8.1f}")

    # Robustness summary
    sharpes = [m["sharpe"] for m in grid.values()]
    dds = [m["maxdd"] for m in grid.values()]
    print(f"\nGrid Sharpe: min {min(sharpes):.2f}, median {np.median(sharpes):.2f}, "
          f"max {max(sharpes):.2f}")
    print(f"Grid maxDD%: best {max(dds):.0f}, median {np.median(dds):.0f}, "
          f"worst {min(dds):.0f}")
    print("Robust if Sharpe stays ~0.5-0.7 and maxDD ~-20 to -30% across the grid")
    print("(i.e. 12m/top-3 is on a plateau, not a lone spike).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
