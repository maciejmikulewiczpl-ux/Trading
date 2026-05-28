"""Out-of-sample robustness check for the volume-expansion filter.

compare_volexp.py found that gating ORB longs on "breakout-bar volume >=
mean(prior 5 bars)" roughly doubles avg_R and cuts drawdown ~28% on the full
180-day window. Before shipping, confirm the edge isn't concentrated in one
regime: split the sessions into a first half and a second half (contiguous in
time) and compare baseline vs the filter in EACH half independently.

A robust filter should improve avg_R (and not wreck PnL) in both halves. If it
only helps in one, it's likely a regime artifact / overfit.

Run:
    .venv/Scripts/python.exe backtest/compare_volexp_oos.py
"""
from __future__ import annotations

import sys
from datetime import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from strategies.orb import Params  # noqa: E402
from backtest.run_orb import (  # noqa: E402
    STARTING_EQUITY,
    WATCHLIST,
    load_all_bars,
    run_backtest,
)
from backtest.compare_volexp import expansion_ratio, summarize  # noqa: E402

LOOKBACK_N = 5
THRESHOLD = 1.0


def report_half(name, trades, all_bars):
    base = summarize(trades, f"{name} baseline")
    kept = [t for t in trades
            if (r := expansion_ratio(all_bars, t, LOOKBACK_N, "mean")) is not None and r >= THRESHOLD]
    filt = summarize(kept, f"{name} exp>=1.0")
    print(f"\n--- {name} ({base['n']} trades) ---")
    for s in (base, filt):
        if s["n"] == 0:
            print(f"  {s['label']:<24}  (no trades)")
            continue
        print(f"  {s['label']:<24} n={s['n']:>3}  win {s['win_rate']:>4.1f}%  "
              f"avg_R {s['avg_r']:>+.4f}  PnL ${s['total_pnl']:>+9,.2f}  "
              f"maxDD ${s['max_dd']:>+9,.2f}")
    if base["n"] and filt["n"]:
        print(f"  delta: avg_R {filt['avg_r'] - base['avg_r']:+.4f}  "
              f"PnL {filt['total_pnl'] - base['total_pnl']:+,.2f}  "
              f"maxDD {filt['max_dd'] - base['max_dd']:+,.2f}  "
              f"trades {filt['n'] - base['n']:+d}")
    return base, filt


def main() -> int:
    print(f"Universe: {WATCHLIST}   filter: breakout vol >= mean(prior {LOOKBACK_N}) x {THRESHOLD}")
    try:
        all_bars, trading_days = load_all_bars()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    params = Params(
        or_minutes=15, target_r=2.0,
        risk_per_trade=100.0, max_position_pct=0.25,
        max_position_dollars=10_000.0,
        no_entry_after_time=time(11, 30),
    )
    all_trades, _ = run_backtest(all_bars, trading_days, WATCHLIST,
                                 params, STARTING_EQUITY)

    # Split sessions in half by date (contiguous halves).
    days_sorted = sorted(trading_days)
    mid = days_sorted[len(days_sorted) // 2]
    print(f"Split at {mid}: first half < {mid} <= second half")

    first = [t for t in all_trades if t.entry_time.date() < mid]
    second = [t for t in all_trades if t.entry_time.date() >= mid]

    b1, f1 = report_half("FIRST HALF ", first, all_bars)
    b2, f2 = report_half("SECOND HALF", second, all_bars)

    print("\n=== verdict ===")
    d1 = (f1["avg_r"] - b1["avg_r"]) if (b1["n"] and f1["n"]) else None
    d2 = (f2["avg_r"] - b2["avg_r"]) if (b2["n"] and f2["n"]) else None
    print(f"avg_R improvement: first half {d1:+.4f}, second half {d2:+.4f}"
          if d1 is not None and d2 is not None else "insufficient data")
    if d1 is not None and d2 is not None:
        if d1 > 0 and d2 > 0:
            print("ROBUST: filter improves avg_R in BOTH halves.")
        elif d1 > 0 or d2 > 0:
            print("MIXED: helps in only one half — likely regime-dependent, treat with caution.")
        else:
            print("FAILS: no avg_R improvement in either half.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
