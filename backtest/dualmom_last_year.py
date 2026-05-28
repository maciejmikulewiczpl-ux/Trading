"""What would $50k in the dual-momentum strategy have made over the last year?

Computes the strategy's trailing-12-month return (and 2025 calendar year, and
the full-period average) for the recommended configs, applied to a $50k stake.

IMPORTANT: a single year is a tiny, high-variance sample and this is in-sample.
The strategy is built for multi-year/crisis robustness, not single-year
prediction. One year tells you almost nothing about next year.

  .venv-openbb\\Scripts\\python.exe backtest\\dualmom_last_year.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtest.run_trend import CASH, load_monthly, w_div_momentum  # noqa: E402
from backtest.trend_robustness import w_div_momentum_blend  # noqa: E402

STAKE = 50_000.0


def strat_returns(weights, monthly_ret):
    aligned = weights.shift(1).reindex(monthly_ret.index).fillna(0.0)
    r = (aligned * monthly_ret.reindex(columns=aligned.columns).fillna(0.0)).sum(axis=1)
    return r.dropna()


def report(name, r):
    last12 = r.iloc[-12:]
    ret12 = (1 + last12).prod() - 1
    cal2025 = (1 + r[r.index.year == 2025]).prod() - 1
    n = len(r)
    cagr = (1 + r).prod() ** (12 / n) - 1
    win = ", ".join(f"{x:+.1%}" for x in last12)
    print(f"\n{name}")
    print(f"  trailing 12 months: {ret12:+.2%}  ->  ${STAKE:,.0f} would be "
          f"${STAKE*(1+ret12):,.0f}  (profit ${STAKE*ret12:+,.0f})")
    print(f"  calendar 2025     : {cal2025:+.2%}  (profit ${STAKE*cal2025:+,.0f})")
    print(f"  full-period CAGR  : {cagr:+.2%}/yr avg  (on ${STAKE:,.0f} = "
          f"${STAKE*cagr:+,.0f}/yr typical)")
    print(f"  monthly path (last 12): {win}")


def main():
    monthly = load_monthly()
    monthly_ret = monthly.pct_change()
    print(f"Data through {monthly.index[-1].date()}  (stake ${STAKE:,.0f})")
    report("Blended-lookback (3/6/12) top-3  [recommended]",
           strat_returns(w_div_momentum_blend(monthly, (3, 6, 12), topk=3), monthly_ret))
    report("Canonical 12-month top-3",
           strat_returns(w_div_momentum(monthly, lb=12, topk=3), monthly_ret))
    # SPY for reference
    spy = strat_returns(pd.DataFrame({"SPY": 1.0}, index=monthly.index)
                        .reindex(columns=monthly.columns).fillna(0.0), monthly_ret)
    report("SPY buy & hold [reference]", spy)
    print("\nCAVEAT: one year is high-variance and in-sample. The full-period CAGR")
    print("is the honest 'typical year' anchor; any single year can be far above or")
    print("below it. Live results degrade further for costs/slippage/regime.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
