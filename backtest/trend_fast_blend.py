"""Should the dual-momentum blend include a faster (1-month) lookback?

Tests several blend variants vs the shipped 3/6/12 blend over both the full
2007-2026 window AND the AI-craze era (2023+) separately, so we can see whether
any faster-signal advantage is robust or just a regime artifact.

Variants (all top-3, equal-weight, beat-cash filter — same engine as run_trend):
  (12,)       slowest single lookback
  (6,)        medium single
  (3,)        fastest single
  (1,)        very fast single (1-month momentum)
  (3,6,12)    SHIPPED baseline
  (1,3,6,12)  add 1mo to baseline blend
  (1,3,6)     drop 12mo, add 1mo (faster blend)
  (1,3)       very fast blend
  (6,12)      slower blend for contrast

For each: CAGR / Sharpe / maxDD on (full window) vs (pre-AI 2007-2022) vs
(AI-era 2023-now). Robust = positive Sharpe in both sub-periods.

  .venv-openbb\\Scripts\\python.exe backtest\\trend_fast_blend.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtest.run_trend import (  # noqa: E402
    CASH, EVAL_START, RISK, evaluate, load_monthly,
)
from backtest.trend_robustness import w_div_momentum_blend  # noqa: E402

AI_ERA_START = "2023-01-01"
PRE_AI_END = "2022-12-31"

VARIANTS = [
    ("12mo single",       (12,)),
    ("6mo single",        (6,)),
    ("3mo single",        (3,)),
    ("1mo single",        (1,)),
    ("3/6/12 blend (LIVE)", (3, 6, 12)),
    ("1/3/6/12 blend",    (1, 3, 6, 12)),
    ("1/3/6 blend",       (1, 3, 6)),
    ("1/3 blend",         (1, 3)),
    ("6/12 blend",        (6, 12)),
]


def slice_metrics(weights, monthly_ret, rf, start, end):
    """Evaluate the weights/strategy over [start, end] slice of months."""
    aligned = weights.shift(1).reindex(monthly_ret.index).fillna(0.0)
    strat = (aligned * monthly_ret.reindex(columns=aligned.columns).fillna(0.0)).sum(axis=1)
    strat = strat.loc[start:end].dropna()
    if strat.empty:
        return None
    eq = (1 + strat).cumprod()
    n = len(strat)
    cagr = eq.iloc[-1] ** (12 / n) - 1
    vol = strat.std() * np.sqrt(12)
    rf_m = rf.reindex(strat.index).fillna(0.0)
    sharpe = (strat - rf_m).mean() / strat.std() * np.sqrt(12) if strat.std() > 0 else np.nan
    dd = (eq / eq.cummax() - 1).min()
    return {
        "cagr": cagr * 100, "vol": vol * 100, "sharpe": sharpe, "maxdd": dd * 100,
        "n_months": n,
    }


def print_block(title, rows):
    print(f"\n{title}")
    print(f"  {'variant':<24}{'CAGR%':>7}{'vol%':>7}{'Sharpe':>8}{'maxDD%':>9}{'months':>8}")
    print("  " + "-" * 65)
    for name, m in rows:
        if m is None:
            print(f"  {name:<24}  (no data)"); continue
        print(f"  {name:<24}{m['cagr']:>7.2f}{m['vol']:>7.2f}"
              f"{m['sharpe']:>8.2f}{m['maxdd']:>+9.1f}{m['n_months']:>8d}")


def main():
    print(f"Loading monthly data (yfinance)... metrics start {EVAL_START}")
    monthly = load_monthly()
    monthly_ret = monthly.pct_change()
    rf = monthly_ret[CASH]
    print(f"Months: {len(monthly)}  | last: {monthly.index[-1].date()}")

    # Precompute weights per variant once
    var_weights = {name: w_div_momentum_blend(monthly, lookbacks=lb, topk=3)
                   for name, lb in VARIANTS}

    # Three slices
    full_rows = [(n, slice_metrics(var_weights[n], monthly_ret, rf, EVAL_START, "2099"))
                 for n, _ in VARIANTS]
    pre_rows = [(n, slice_metrics(var_weights[n], monthly_ret, rf, EVAL_START, PRE_AI_END))
                for n, _ in VARIANTS]
    ai_rows = [(n, slice_metrics(var_weights[n], monthly_ret, rf, AI_ERA_START, "2099"))
               for n, _ in VARIANTS]

    print_block("FULL WINDOW (2007-07 onward):", full_rows)
    print_block(f"PRE-AI era (2007-07 to {PRE_AI_END}):", pre_rows)
    print_block(f"AI era ({AI_ERA_START} onward):", ai_rows)

    print()
    print("Robustness reading: a faster blend is a true improvement only if it")
    print("beats 3/6/12 in BOTH pre-AI and AI eras. Beating only in AI era = the")
    print("'fast signal advantage' is just regime luck / overfit to the AI bubble.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
