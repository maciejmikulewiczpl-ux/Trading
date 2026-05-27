"""A/B comparison of time-of-day entry cutoffs for ORB.

Sweeps several "no new entries after X" cutoffs on the same 180-day bar set:
  - baseline:   no cutoff (entries valid all the way to eod_flat 15:55 ET)
  - 10:30 ET:   entries only in the first hour after open
  - 11:00 ET:   first 90 min
  - 11:30 ET:   first 2h
  - 12:00 ET:   first 2.5h
  - 13:00 ET:   first 3.5h

Validates Tier-1 #3 of plans/put-yourself-as-an-majestic-cupcake.md
before shipping the live behavior change. Cutoff only affects NEW
entries; existing trades still ride to eod_flat as usual.

Run:
    .venv/Scripts/python.exe backtest/compare_cutoff.py
"""
from __future__ import annotations

import sys
from datetime import time
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


def _params(cutoff):
    return Params(
        or_minutes=15,
        target_r=2.0,
        risk_per_trade=100.0,
        max_position_pct=0.25,
        max_position_dollars=10_000.0,
        no_entry_after_time=cutoff,
    )


CONFIGS = [
    ("baseline (no cutoff)", _params(None)),
    ("cutoff 10:30 ET     ", _params(time(10, 30))),
    ("cutoff 11:00 ET     ", _params(time(11, 0))),
    ("cutoff 11:30 ET     ", _params(time(11, 30))),
    ("cutoff 12:00 ET     ", _params(time(12, 0))),
    ("cutoff 13:00 ET     ", _params(time(13, 0))),
]


def summarize(trades: list[Trade], final_equity: float) -> dict:
    if not trades:
        return {"n": 0}
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
        "total_pnl": df["pnl_dollars"].sum(),
        "max_dd": dd,
        "final_equity": final_equity,
        "return_pct": (final_equity / STARTING_EQUITY - 1) * 100,
        "n_target": int((df["exit_reason"] == "target").sum()),
        "n_stop":   int((df["exit_reason"] == "stop").sum()),
        "n_eod":    int((df["exit_reason"] == "eod").sum()),
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

    header = (f"{'config':<22} {'n':>4} {'win%':>6} {'avg_R':>8} "
              f"{'tgt/stop/eod':>13} {'total PnL':>13} {'max DD':>11} "
              f"{'return%':>8}  {'avg PnL/trade':>14}")
    print(header)
    print("-" * len(header))
    for s in rows:
        if s.get("n", 0) == 0:
            print(f"{s['label']:<22}  (no trades)")
            continue
        avg_per_trade = s['total_pnl'] / s['n']
        target_stop_eod = f"{s['n_target']}/{s['n_stop']}/{s['n_eod']}"
        print(
            f"{s['label']:<22} "
            f"{s['n']:>4} "
            f"{s['win_rate']:>5.1f}% "
            f"{s['avg_r']:>+8.4f} "
            f"{target_stop_eod:>13} "
            f"${s['total_pnl']:>+12,.2f} "
            f"${s['max_dd']:>+10,.2f} "
            f"{s['return_pct']:>+7.2f}%  "
            f"${avg_per_trade:>+13,.2f}"
        )

    print()
    baseline = rows[0]
    print(f"Delta vs '{baseline['label'].strip()}':")
    for s in rows[1:]:
        d_pnl = s["total_pnl"] - baseline["total_pnl"]
        d_pnl_pct = (d_pnl / abs(baseline["total_pnl"]) * 100) if baseline["total_pnl"] else float("nan")
        d_n = s["n"] - baseline["n"]
        d_dd = s["max_dd"] - baseline["max_dd"]
        d_avg = (s["avg_r"] - baseline["avg_r"]) if s.get("n") else float("nan")
        print(f"  {s['label'].strip():<20}: "
              f"trades {d_n:+d}  PnL {d_pnl:+,.2f} ({d_pnl_pct:+.1f}%)  "
              f"max-DD {d_dd:+,.2f}  avg_R {d_avg:+.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
