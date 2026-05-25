"""Side-by-side comparison of three position-sizing schemes for ORB.

Pulls bars once, runs the same strategy logic with different Params, prints a table.
Purpose: illustrate that 'risk per trade' and 'max position size' are independent
levers that often pull in opposite directions on a momentum strategy like ORB.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from strategies.orb import Params, Trade  # noqa: E402
from backtest.run_orb import (  # noqa: E402
    STARTING_EQUITY,
    WATCHLIST,
    load_all_bars,
    run_backtest,
)


CONFIGS = [
    ("baseline           ", Params(risk_per_trade=100.0, max_position_pct=0.25, max_position_dollars=None)),
    ("user proposed      ", Params(risk_per_trade=500.0, max_position_pct=0.25, max_position_dollars=5_000.0)),
    ("just scale risk 5x ", Params(risk_per_trade=500.0, max_position_pct=0.25, max_position_dollars=None)),
]


def summarize(trades: list[Trade], final_equity: float) -> dict:
    df = pd.DataFrame([{
        "pnl_dollars": t.pnl_dollars,
        "pnl_r": t.pnl_r,
        "notional": t.shares * t.entry_price,
        "risk_dollars": t.risk_dollars,
        "shares": t.shares,
        "exit_time": t.exit_time,
    } for t in trades])
    if df.empty:
        return {"n": 0}
    df_sorted = df.sort_values("exit_time")
    eq_curve = STARTING_EQUITY + df_sorted["pnl_dollars"].cumsum()
    dd = (eq_curve - eq_curve.cummax()).min()
    return {
        "n": len(df),
        "win_rate": (df["pnl_r"] > 0).mean() * 100,
        "avg_r": df["pnl_r"].mean(),
        "avg_notional": df["notional"].mean(),
        "avg_risk_dollars": df["risk_dollars"].mean(),
        "total_pnl": df["pnl_dollars"].sum(),
        "max_dd": dd,
        "final_equity": final_equity,
        "return_pct": (final_equity / STARTING_EQUITY - 1) * 100,
    }


def main() -> int:
    print(f"Universe : {WATCHLIST}")
    try:
        all_bars, trading_days = load_all_bars()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print()

    results = []
    for label, params in CONFIGS:
        trades, final_eq = run_backtest(all_bars, trading_days, WATCHLIST, params, STARTING_EQUITY)
        s = summarize(trades, final_eq)
        s["label"] = label
        s["params"] = params
        results.append(s)

    header = f"{'config':<22} {'risk$':>7} {'cap$':>10} {'n':>4} {'win%':>6} {'avg_R':>7} {'avg notional':>13} {'avg $risk':>10} {'total PnL':>12} {'max DD':>11} {'return%':>8}"
    print(header)
    print("-" * len(header))
    for s in results:
        p = s["params"]
        cap_str = f"${p.max_position_dollars:,.0f}" if p.max_position_dollars else f"{p.max_position_pct:.0%}eq"
        print(
            f"{s['label']:<22} "
            f"${p.risk_per_trade:>6.0f} "
            f"{cap_str:>10} "
            f"{s['n']:>4} "
            f"{s['win_rate']:>5.1f}% "
            f"{s['avg_r']:>+7.2f} "
            f"${s['avg_notional']:>12,.0f} "
            f"${s['avg_risk_dollars']:>9.2f} "
            f"${s['total_pnl']:>+11,.2f} "
            f"${s['max_dd']:>+10,.2f} "
            f"{s['return_pct']:>+7.2f}%"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
