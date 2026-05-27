"""A/B comparison of the short-side ORB addition on the same 180-day bar set.

Four configs, identical bars/sizing/cutoff, only the direction policy varies:
  - long only (baseline) : current production behavior (close > OR_high).
  - short only           : mirror entries on close < OR_low.
  - long + short         : first breakout in either direction wins, one shot/day.
  - long + short + flip  : as above, but a stop-out may flip into the opposite
                           direction once (max_flips=1).

Decision gate for Phase 1: keep shorts / keep the flip only if they improve
risk-adjusted return (total PnL and max-DD) on this window. Nothing here touches
live trading — see plans/yes-the-script-should-nifty-kite.md.

Run:
    .venv/Scripts/python.exe backtest/compare_short.py
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


def _params(enable_long: bool, enable_short: bool, max_flips: int = 0):
    # Mirrors live PARAMS (target_r=2.0, $100 risk, $10k cap, 11:30 ET cutoff)
    # so the comparison reflects the configuration we would actually ship.
    return Params(
        or_minutes=15,
        target_r=2.0,
        risk_per_trade=100.0,
        max_position_pct=0.25,
        max_position_dollars=10_000.0,
        no_entry_after_time=time(11, 30),
        enable_long=enable_long,
        enable_short=enable_short,
        max_flips=max_flips,
    )


CONFIGS = [
    ("long only (baseline) ", _params(True, False)),
    ("short only           ", _params(False, True)),
    ("long + short         ", _params(True, True)),
    ("long + short + flip  ", _params(True, True, max_flips=1)),
]


def summarize(trades: list[Trade], final_equity: float) -> dict:
    if not trades:
        return {"n": 0}
    df = pd.DataFrame([{
        "side": t.side,
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
        "n_long": int((df["side"] == "long").sum()),
        "n_short": int((df["side"] == "short").sum()),
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
        rows.append(s)

    header = (f"{'config':<22} {'n':>4} {'L/S':>8} {'win%':>6} {'avg_R':>8} "
              f"{'tgt/stop/eod':>13} {'total PnL':>13} {'max DD':>11} "
              f"{'return%':>8}")
    print(header)
    print("-" * len(header))
    for s in rows:
        if s.get("n", 0) == 0:
            print(f"{s['label']:<22}  (no trades)")
            continue
        ls = f"{s['n_long']}/{s['n_short']}"
        target_stop_eod = f"{s['n_target']}/{s['n_stop']}/{s['n_eod']}"
        print(
            f"{s['label']:<22} "
            f"{s['n']:>4} "
            f"{ls:>8} "
            f"{s['win_rate']:>5.1f}% "
            f"{s['avg_r']:>+8.4f} "
            f"{target_stop_eod:>13} "
            f"${s['total_pnl']:>+12,.2f} "
            f"${s['max_dd']:>+10,.2f} "
            f"{s['return_pct']:>+7.2f}%"
        )

    print()
    baseline = rows[0]
    print(f"Delta vs '{baseline['label'].strip()}':")
    for s in rows[1:]:
        if s.get("n", 0) == 0:
            print(f"  {s['label'].strip():<20}: (no trades)")
            continue
        d_pnl = s["total_pnl"] - baseline["total_pnl"]
        d_pnl_pct = (d_pnl / abs(baseline["total_pnl"]) * 100) if baseline["total_pnl"] else float("nan")
        d_n = s["n"] - baseline["n"]
        d_dd = s["max_dd"] - baseline["max_dd"]
        d_avg = s["avg_r"] - baseline["avg_r"]
        print(f"  {s['label'].strip():<20}: "
              f"trades {d_n:+d}  PnL {d_pnl:+,.2f} ({d_pnl_pct:+.1f}%)  "
              f"max-DD {d_dd:+,.2f}  avg_R {d_avg:+.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
