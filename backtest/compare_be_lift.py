"""A/B comparison of breakeven-stop lift policies for ORB.

Runs the simulator three ways on the same 180-day bar set:
  - baseline:        no BE-lift (move_stop_to_be_at_r = None)
  - BE at +0.5R:     lift stop to entry once unrealized >= 0.5R
  - BE at +1.0R:     lift stop to entry once unrealized >= 1.0R

Purpose: validate Tier-1 #1 of plans/put-yourself-as-an-majestic-cupcake.md
before shipping the live behavior change.

Run:
    .venv/Scripts/python.exe backtest/compare_be_lift.py
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


# Match the live PARAMS in live/paper_orb.py for everything EXCEPT the BE knob.
def _params(be_at_r):
    return Params(
        or_minutes=15,
        target_r=2.0,
        risk_per_trade=100.0,
        max_position_pct=0.25,
        max_position_dollars=10_000.0,
        move_stop_to_be_at_r=be_at_r,
    )


CONFIGS = [
    ("baseline   (BE off)  ", _params(None)),
    ("BE-lift at +0.5R     ", _params(0.5)),
    ("BE-lift at +1.0R     ", _params(1.0)),
]


def summarize(trades: list[Trade], final_equity: float) -> dict:
    if not trades:
        return {"n": 0, "label_pad": 0}
    df = pd.DataFrame([{
        "pnl_dollars": t.pnl_dollars,
        "pnl_r": t.pnl_r,
        "exit_time": t.exit_time,
        "exit_reason": t.exit_reason,
    } for t in trades])
    df_sorted = df.sort_values("exit_time")
    eq_curve = STARTING_EQUITY + df_sorted["pnl_dollars"].cumsum()
    dd = (eq_curve - eq_curve.cummax()).min()
    return {
        "n": len(df),
        "win_rate": (df["pnl_r"] > 0).mean() * 100,
        "avg_r": df["pnl_r"].mean(),
        "median_r": df["pnl_r"].median(),
        "total_pnl": df["pnl_dollars"].sum(),
        "max_dd": dd,
        "final_equity": final_equity,
        "return_pct": (final_equity / STARTING_EQUITY - 1) * 100,
        # Exit-reason breakdown
        "n_target": int((df["exit_reason"] == "target").sum()),
        "n_stop":   int((df["exit_reason"] == "stop").sum()),
        "n_eod":    int((df["exit_reason"] == "eod").sum()),
        "be_scratch_r": float(df[(df["exit_reason"] == "stop") & (df["pnl_r"].abs() < 0.05)]["pnl_r"].count()),
    }


def main() -> int:
    print(f"Universe: {WATCHLIST}")
    print(f"Starting equity: ${STARTING_EQUITY:,.0f}")
    try:
        all_bars, trading_days = load_all_bars()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"Sessions: {len(trading_days)}")
    print()

    rows = []
    for label, params in CONFIGS:
        trades, final_eq = run_backtest(
            all_bars, trading_days, WATCHLIST, params, STARTING_EQUITY
        )
        s = summarize(trades, final_eq)
        s["label"] = label
        s["params"] = params
        rows.append(s)

    header = (f"{'config':<22} {'n':>4} {'win%':>6} {'avg_R':>7} {'med_R':>7} "
              f"{'tgt/stop/eod':>13} {'BEscr':>5} {'total PnL':>13} "
              f"{'max DD':>11} {'return%':>8}")
    print(header)
    print("-" * len(header))
    for s in rows:
        target_stop_eod = f"{s['n_target']}/{s['n_stop']}/{s['n_eod']}"
        print(
            f"{s['label']:<22} "
            f"{s['n']:>4} "
            f"{s['win_rate']:>5.1f}% "
            f"{s['avg_r']:>+7.3f} "
            f"{s['median_r']:>+7.3f} "
            f"{target_stop_eod:>13} "
            f"{int(s['be_scratch_r']):>5} "
            f"${s['total_pnl']:>+12,.2f} "
            f"${s['max_dd']:>+10,.2f} "
            f"{s['return_pct']:>+7.2f}%"
        )

    print()
    # Delta vs baseline
    baseline = rows[0]
    print(f"Delta vs '{baseline['label'].strip()}':")
    for s in rows[1:]:
        d_pnl = s["total_pnl"] - baseline["total_pnl"]
        d_pnl_pct = (d_pnl / abs(baseline["total_pnl"]) * 100) if baseline["total_pnl"] else float("nan")
        d_dd  = s["max_dd"] - baseline["max_dd"]
        d_avg_r = s["avg_r"] - baseline["avg_r"]
        print(f"  {s['label'].strip()}: "
              f"PnL {d_pnl:+,.2f} ({d_pnl_pct:+.1f}%)  "
              f"max-DD {d_dd:+,.2f}  "
              f"avg_R {d_avg_r:+.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
