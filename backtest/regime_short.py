"""Regime-gated shorts: enable shorts ONLY when the market is in a downtrend.

OOS validation (validate_short_oos.py) showed static shorts are a bearish-regime
bet — they made money in the bearish train half and lost in the bull test half.
The economically sensible fix: only short when the market regime is bearish, and
otherwise run the (excellent) long-only book.

Regime signal (no lookahead): SPY is "bearish" for session day D if SPY's close
on D-1 is below its N-day SMA as of D-1. Shorts are enabled on bearish days only;
every other day is long-only. Longs always run.

A good gate should BEAT long-only in the bearish TRAIN block (capture short gains)
and roughly MATCH long-only in the bull TEST block (shorts mostly disabled, so we
stop bleeding). We test several SMA windows on both blocks.

Run:
    uv run --with pip-system-certs python backtest/regime_short.py
"""
from __future__ import annotations

import sys
from datetime import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from strategies.orb import Params, simulate_session  # noqa: E402
from backtest.run_orb import (  # noqa: E402
    STARTING_EQUITY,
    WATCHLIST,
    load_all_bars,
)
from backtest.eval_index_short import summarize  # noqa: E402

CUTOFF = time(11, 30)
TRAIN_FRACTION = 2 / 3
SHORT_SYMS = {"SPY", "QQQ", "NVDA", "AAPL"}  # ex-TSLA (squeeze-prone)
SMA_WINDOWS = [10, 20, 50]


def params(enable_long: bool, enable_short: bool, max_flips: int = 0) -> Params:
    return Params(
        or_minutes=15, target_r=2.0,
        risk_per_trade=100.0, max_position_pct=0.25,
        max_position_dollars=10_000.0, no_entry_after_time=CUTOFF,
        enable_long=enable_long, enable_short=enable_short, max_flips=max_flips,
    )


def spy_bearish_by_day(all_bars: pd.DataFrame, window: int) -> dict:
    """{date -> bool}: True if SPY close on the PRIOR session was below its
    N-day SMA as of that prior session (decision uses only pre-open info)."""
    spy = all_bars.xs("SPY", level=0)
    t = spy.index.time
    rth = spy[(t >= time(9, 30)) & (t < time(16, 0))]
    daily_close = rth.groupby(rth.index.date).last()["close"]
    sma = daily_close.rolling(window).mean()
    bearish_asof = daily_close < sma           # regime as of each day's close
    bearish_today = bearish_asof.shift(1)      # use prior day's regime for today
    return {d: bool(v) for d, v in bearish_today.items() if pd.notna(v)}


def run_static(all_bars, days, short_syms, flips):
    pbs = {s: params(True, s in short_syms, flips) for s in WATCHLIST}
    return _run(all_bars, days, lambda d, s: pbs[s])


def run_regime_gated(all_bars, days, short_syms, flips, bearish):
    long_only = {s: params(True, False) for s in WATCHLIST}
    with_short = {s: params(True, s in short_syms, flips) for s in WATCHLIST}
    def pick(day, sym):
        return with_short[sym] if bearish.get(day, False) else long_only[sym]
    return _run(all_bars, days, pick)


def _run(all_bars, days, pick_params):
    symbols_in_data = set(all_bars.index.get_level_values(0).unique())
    trades = []
    for day in days:
        for sym in WATCHLIST:
            if sym not in symbols_in_data:
                continue
            sb = all_bars.xs(sym, level=0)
            db = sb[sb.index.date == day]
            if db.empty:
                continue
            t = db.index.time
            db = db[(t >= time(9, 30)) & (t < time(16, 0))]
            if db.empty:
                continue
            trades.extend(simulate_session(db, sym, STARTING_EQUITY, pick_params(day, sym)))
    return trades


def line(label, s):
    if s.get("n", 0) == 0:
        print(f"{label:<34}  (no trades)")
        return
    print(f"{label:<34} {s['n']:>4} {s['n_long']:>4}/{s['n_short']:<4} "
          f"{s['win_rate']:>5.1f}% {s['avg_r']:>+7.4f} "
          f"${s['total_pnl']:>+10,.0f} ${s['max_dd']:>+9,.0f} "
          f"short=${s['short_pnl']:>+8,.0f}")


HDR = (f"{'config':<34} {'n':>4} {'L/S':>9} {'win%':>6} {'avg_R':>7} "
       f"{'total PnL':>11} {'max DD':>10} {'short pnl':>14}")


def main() -> int:
    print(f"Universe: {WATCHLIST}   short set (ex-TSLA): {sorted(SHORT_SYMS)}")
    try:
        all_bars, trading_days = load_all_bars()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    n_train = int(len(trading_days) * TRAIN_FRACTION)
    train_days, test_days = trading_days[:n_train], trading_days[n_train:]
    print(f"Train: {train_days[0]} -> {train_days[-1]}  ({len(train_days)} sessions)")
    print(f"Test : {test_days[0]} -> {test_days[-1]}  ({len(test_days)} sessions)\n")

    blocks = [("TRAIN", train_days), ("TEST (held-out)", test_days)]

    # Baseline + static-F once per block; regime-gated per SMA window.
    regimes = {w: spy_bearish_by_day(all_bars, w) for w in SMA_WINDOWS}
    for w in SMA_WINDOWS:
        bd = regimes[w]
        n_bear_tr = sum(1 for d in train_days if bd.get(d, False))
        n_bear_te = sum(1 for d in test_days if bd.get(d, False))
        print(f"SMA{w}: bearish days  train {n_bear_tr}/{len(train_days)}   "
              f"test {n_bear_te}/{len(test_days)}")
    print()

    for name, days in blocks:
        print(f"===== {name} =====")
        print(HDR)
        print("-" * len(HDR))
        line("long only (baseline)",
             summarize(run_static(all_bars, days, set(), 0)))
        line("static: short ex-TSLA +flip",
             summarize(run_static(all_bars, days, SHORT_SYMS, 1)))
        for w in SMA_WINDOWS:
            line(f"regime SMA{w}: short on bear days",
                 summarize(run_regime_gated(all_bars, days, SHORT_SYMS, 1, regimes[w])))
        print()

    print("PASS if a regime config BEATS long-only on TRAIN and roughly MATCHES "
          "it on TEST (shorts off in the bull block).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
