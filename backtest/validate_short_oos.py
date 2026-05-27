"""Out-of-sample (train/test) validation of the short-side ORB addition.

Config F (long all + short all-except-TSLA + 1 flip) doubled PnL and cut
drawdown on the FULL window — but F was chosen using that same full window, so
this checks whether the short edge survives on held-out data.

Split: first 2/3 of sessions = train, last 1/3 = test (sweep_orb.py convention).
Corrected exit semantics throughout (entries gated 11:30, exits ride to 15:55).

Two questions:
  1. Does my hand-picked F hold on the pristine test third?
  2. CLEAN test: pick the short set from TRAIN per-symbol PnL only (short the
     names that are profitable in-sample, exclude losers), then apply that rule
     to test. This emulates the real workflow with no test-period peeking.

Configs compared on BOTH periods:
  - long only (baseline)
  - F: long all + short ex-TSLA + flip      (hand-picked on full window)
  - T: long all + short {train-positive}+flip (set derived from train only)

Run:
    uv run --with pip-system-certs python backtest/validate_short_oos.py
"""
from __future__ import annotations

import sys
from datetime import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from strategies.orb import Params  # noqa: E402
from backtest.run_orb import (  # noqa: E402
    STARTING_EQUITY,
    WATCHLIST,
    load_all_bars,
)
from backtest.eval_index_short import run_per_symbol, summarize  # noqa: E402

CUTOFF = time(11, 30)
TRAIN_FRACTION = 2 / 3


def params(enable_long: bool, enable_short: bool, max_flips: int = 0) -> Params:
    return Params(
        or_minutes=15, target_r=2.0,
        risk_per_trade=100.0, max_position_pct=0.25,
        max_position_dollars=10_000.0, no_entry_after_time=CUTOFF,
        enable_long=enable_long, enable_short=enable_short, max_flips=max_flips,
    )


def long_only_cfg():
    return {s: params(True, False) for s in WATCHLIST}


def mixed_cfg(short_syms, flips=0):
    return {s: params(True, s in short_syms, flips) for s in WATCHLIST}


def short_only_cfg():
    return {s: params(False, True) for s in WATCHLIST}


def per_symbol_short_pnl(all_bars, days) -> dict[str, float]:
    """Short-only PnL by symbol over `days` (for train-period selection)."""
    trades = run_per_symbol(all_bars, days, short_only_cfg())
    out = {s: 0.0 for s in WATCHLIST}
    for t in trades:
        out[t.symbol] += t.pnl_dollars
    return out


def line(label, s):
    if s.get("n", 0) == 0:
        print(f"{label:<30}  (no trades)")
        return
    print(f"{label:<30} {s['n']:>4} {s['n_long']:>4}/{s['n_short']:<4} "
          f"{s['win_rate']:>5.1f}% {s['avg_r']:>+7.4f} "
          f"${s['total_pnl']:>+10,.0f} ${s['max_dd']:>+9,.0f} "
          f"short=${s['short_pnl']:>+8,.0f}")


HDR = (f"{'config':<30} {'n':>4} {'L/S':>9} {'win%':>6} {'avg_R':>7} "
       f"{'total PnL':>11} {'max DD':>10} {'short pnl':>14}")


def main() -> int:
    print(f"Universe: {WATCHLIST}")
    try:
        all_bars, trading_days = load_all_bars()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    n_train = int(len(trading_days) * TRAIN_FRACTION)
    train_days, test_days = trading_days[:n_train], trading_days[n_train:]
    print(f"Train: {train_days[0]} -> {train_days[-1]}  ({len(train_days)} sessions)")
    print(f"Test : {test_days[0]} -> {test_days[-1]}  ({len(test_days)} sessions)\n")

    # Train-derived short set: names with POSITIVE short-only PnL in-sample.
    train_short_pnl = per_symbol_short_pnl(all_bars, train_days)
    train_set = {s for s, v in train_short_pnl.items() if v > 0}
    print("Per-symbol SHORT-only PnL on TRAIN (selection basis):")
    for s in WATCHLIST:
        mark = "  <- short" if s in train_set else ""
        print(f"   {s:<5} ${train_short_pnl[s]:>+9,.0f}{mark}")
    print(f"Train-derived short set (T): {sorted(train_set)}\n")

    ex_tsla = {"SPY", "QQQ", "NVDA", "AAPL"}
    cfgs = [
        ("long only (baseline)", long_only_cfg()),
        ("F: long + short ex-TSLA +flip", mixed_cfg(ex_tsla, flips=1)),
        ("T: long + short {train+} +flip", mixed_cfg(train_set, flips=1)),
    ]

    for period_name, days in [("TRAIN", train_days), ("TEST (held-out)", test_days)]:
        print(f"===== {period_name} =====")
        print(HDR)
        print("-" * len(HDR))
        for label, pbs in cfgs:
            line(label, summarize(run_per_symbol(all_bars, days, pbs)))
        print()

    print("Read: shorts pass OOS if F and T beat the baseline on PnL/DD in the "
          "TEST block too — not just train. A big train>>test gap = overfit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
