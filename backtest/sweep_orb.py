"""Parameter sweep for ORB with train/test split.

Goal: demonstrate the difference between in-sample (optimized) performance and
out-of-sample (held-out) performance. The gap is overfitting.

Grid:
  - or_minutes        : 5, 15, 30
  - target_r          : 0.75, 1.0, 1.5, 2.0, 2.5, 3.0
  - move_stop_to_be_at_r : None, 0.5, 1.0
  -> 54 configs

Sizing held constant (so we isolate strategy edge, not sizing scale):
  - risk_per_trade   : $100
  - max_position_dollars : $10,000

Split:
  - train : first ~2/3 of trading sessions
  - test  : last ~1/3 (the optimizer never sees this)
"""
from __future__ import annotations

import sys
from itertools import product
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


OR_MINUTES = [5, 15, 30]
TARGET_R = [0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
BE_AT_R = [None, 0.5, 1.0]

TRAIN_FRACTION = 2 / 3
RISK = 100.0
MAX_POS_DOLLARS = 10_000.0


def summarize(trades: list[Trade], final_equity: float) -> dict:
    if not trades:
        return {"n": 0, "win_rate": 0.0, "avg_r": 0.0, "total_pnl": 0.0,
                "max_dd": 0.0, "final_equity": final_equity, "return_pct": 0.0}
    df = pd.DataFrame([{"pnl_dollars": t.pnl_dollars, "pnl_r": t.pnl_r,
                        "exit_time": t.exit_time} for t in trades])
    df = df.sort_values("exit_time")
    eq_curve = STARTING_EQUITY + df["pnl_dollars"].cumsum()
    dd = (eq_curve - eq_curve.cummax()).min()
    return {
        "n": len(df),
        "win_rate": (df["pnl_r"] > 0).mean() * 100,
        "avg_r": df["pnl_r"].mean(),
        "total_pnl": df["pnl_dollars"].sum(),
        "max_dd": float(dd) if pd.notna(dd) else 0.0,
        "final_equity": final_equity,
        "return_pct": (final_equity / STARTING_EQUITY - 1) * 100,
    }


def fmt_be(x):
    return "off" if x is None else f"+{x}R"


def main() -> int:
    print(f"Universe : {WATCHLIST}")
    try:
        all_bars, trading_days = load_all_bars()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    n_train = int(len(trading_days) * TRAIN_FRACTION)
    train_days = trading_days[:n_train]
    test_days = trading_days[n_train:]
    print(f"Train    : {train_days[0]} -> {train_days[-1]}  ({len(train_days)} sessions)")
    print(f"Test     : {test_days[0]} -> {test_days[-1]}  ({len(test_days)} sessions)")
    print()

    configs = list(product(OR_MINUTES, TARGET_R, BE_AT_R))
    print(f"Sweeping {len(configs)} parameter combinations on train period...")
    print()

    results = []
    for orm, tgt, be in configs:
        p = Params(
            or_minutes=orm,
            target_r=tgt,
            move_stop_to_be_at_r=be,
            risk_per_trade=RISK,
            max_position_dollars=MAX_POS_DOLLARS,
        )
        trades, final_eq = run_backtest(all_bars, train_days, WATCHLIST, p, STARTING_EQUITY)
        s = summarize(trades, final_eq)
        s.update({"or_minutes": orm, "target_r": tgt, "be_at_r": be, "params": p})
        results.append(s)

    results.sort(key=lambda r: r["total_pnl"], reverse=True)

    print("Top 10 configs by TRAIN total PnL:")
    print(f"{'rank':>4}  {'OR':>3} {'tgt_R':>6} {'BE':>5}  {'n':>4} {'win%':>6} {'avg_R':>7} {'pnl $':>11} {'max DD $':>11}")
    for i, r in enumerate(results[:10], 1):
        print(f"{i:>4}  {r['or_minutes']:>3} {r['target_r']:>6.2f} {fmt_be(r['be_at_r']):>5}  "
              f"{r['n']:>4} {r['win_rate']:>5.1f}% {r['avg_r']:>+7.2f} "
              f"${r['total_pnl']:>+10,.2f} ${r['max_dd']:>+10,.2f}")

    best = results[0]
    p_best = best["params"]

    print()
    print(f"Best train config: OR={best['or_minutes']}min  target={best['target_r']}R  BE={fmt_be(best['be_at_r'])}")
    print("Applying it to the held-out TEST period...")
    print()

    # Apply the winner to test data — pristine, optimizer never saw it.
    train_trades, _ = run_backtest(all_bars, train_days, WATCHLIST, p_best, STARTING_EQUITY)
    test_trades, test_final = run_backtest(all_bars, test_days, WATCHLIST, p_best, STARTING_EQUITY)
    s_train = summarize(train_trades, STARTING_EQUITY + sum(t.pnl_dollars for t in train_trades))
    s_test = summarize(test_trades, test_final)

    print(f"{'period':>8}  {'sessions':>9}  {'n':>4}  {'win%':>6}  {'avg_R':>7}  {'pnl $':>11}  {'max DD $':>11}  {'return %':>9}")
    print(f"{'train':>8}  {len(train_days):>9}  {s_train['n']:>4}  {s_train['win_rate']:>5.1f}%  {s_train['avg_r']:>+7.2f}  "
          f"${s_train['total_pnl']:>+10,.2f}  ${s_train['max_dd']:>+10,.2f}  {s_train['return_pct']:>+8.2f}%")
    print(f"{'test':>8}  {len(test_days):>9}  {s_test['n']:>4}  {s_test['win_rate']:>5.1f}%  {s_test['avg_r']:>+7.2f}  "
          f"${s_test['total_pnl']:>+10,.2f}  ${s_test['max_dd']:>+10,.2f}  {s_test['return_pct']:>+8.2f}%")

    # For comparison: what would the original 2R / 15min / no-BE baseline give on test?
    p_baseline = Params(
        or_minutes=15, target_r=2.0, move_stop_to_be_at_r=None,
        risk_per_trade=RISK, max_position_dollars=MAX_POS_DOLLARS,
    )
    base_test_trades, base_test_final = run_backtest(all_bars, test_days, WATCHLIST, p_baseline, STARTING_EQUITY)
    s_base = summarize(base_test_trades, base_test_final)
    print(f"{'(base)':>8}  {len(test_days):>9}  {s_base['n']:>4}  {s_base['win_rate']:>5.1f}%  {s_base['avg_r']:>+7.2f}  "
          f"${s_base['total_pnl']:>+10,.2f}  ${s_base['max_dd']:>+10,.2f}  {s_base['return_pct']:>+8.2f}%")
    print("  ('base' = original 15min / 2R / no-BE, applied to test period for comparison)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
