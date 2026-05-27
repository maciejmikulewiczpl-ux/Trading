"""Evaluate adding SHORTS ONLY on index-like names to the long book.

Phase-1 exploration showed the naive universe-wide short loses money, driven by
single-name squeezes (TSLA/AAPL). QQQ (and mildly NVDA) showed a real short
edge. This script tests, with the CORRECTED exit semantics (entries gated at
11:30, exits ride to 15:55 like live):

  A. long-only baseline (all 5)            - current production
  B. long all + short {QQQ,NVDA}           - first breakout wins on index names
  C. long all + short {QQQ,NVDA} + 1 flip  - flip allowed on the index names
  D. long all + short {QQQ} only           - the single strongest short name

Per-symbol direction: shorts are enabled only for SHORT_SYMBOLS; every other
name stays long-only. Nothing here touches live.

Run:
    uv run --with pip-system-certs python backtest/eval_index_short.py
"""
from __future__ import annotations

import sys
from datetime import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from strategies.orb import Params, Trade, simulate_session  # noqa: E402
from backtest.run_orb import (  # noqa: E402
    STARTING_EQUITY,
    WATCHLIST,
    load_all_bars,
)

CUTOFF = time(11, 30)


def _p(enable_short: bool, max_flips: int = 0) -> Params:
    return Params(
        or_minutes=15, target_r=2.0,
        risk_per_trade=100.0, max_position_pct=0.25,
        max_position_dollars=10_000.0, no_entry_after_time=CUTOFF,
        enable_long=True, enable_short=enable_short, max_flips=max_flips,
    )


def run_per_symbol(all_bars, trading_days, params_by_symbol: dict[str, Params]) -> list[Trade]:
    """Like run_orb.run_backtest but with a per-symbol Params (so shorts can be
    enabled on some names only). Equity for sizing is held at STARTING_EQUITY —
    sizing here is dominated by the $10k notional / $100 risk caps, so intraday
    compounding is immaterial to the comparison."""
    symbols_in_data = set(all_bars.index.get_level_values(0).unique())
    trades: list[Trade] = []
    for session_date in trading_days:
        for symbol, params in params_by_symbol.items():
            if symbol not in symbols_in_data:
                continue
            sym_bars = all_bars.xs(symbol, level=0)
            day_bars = sym_bars[sym_bars.index.date == session_date]
            if day_bars.empty:
                continue
            t = day_bars.index.time
            day_bars = day_bars[(t >= time(9, 30)) & (t < time(16, 0))]
            if day_bars.empty:
                continue
            trades.extend(simulate_session(day_bars, symbol, STARTING_EQUITY, params))
    return trades


def summarize(trades: list[Trade]) -> dict:
    if not trades:
        return {"n": 0}
    df = pd.DataFrame([{
        "side": t.side, "pnl_dollars": t.pnl_dollars, "pnl_r": t.pnl_r,
        "exit_time": t.exit_time, "exit_reason": t.exit_reason,
    } for t in trades])
    eq = STARTING_EQUITY + df.sort_values("exit_time")["pnl_dollars"].cumsum()
    return {
        "n": len(df),
        "n_long": int((df["side"] == "long").sum()),
        "n_short": int((df["side"] == "short").sum()),
        "win_rate": (df["pnl_r"] > 0).mean() * 100,
        "avg_r": df["pnl_r"].mean(),
        "total_pnl": df["pnl_dollars"].sum(),
        "max_dd": (eq - eq.cummax()).min(),
        "short_pnl": df[df["side"] == "short"]["pnl_dollars"].sum(),
    }


def prow(label: str, s: dict, base: dict | None = None) -> None:
    if s.get("n", 0) == 0:
        print(f"{label:<34}  (no trades)")
        return
    ls = f"{s['n_long']}/{s['n_short']}"
    d = ""
    if base is not None:
        dp = s["total_pnl"] - base["total_pnl"]
        dd = s["max_dd"] - base["max_dd"]
        d = f"  | dPnL {dp:+,.0f}  dDD {dd:+,.0f}"
    print(f"{label:<34} {s['n']:>4} {ls:>9} {s['win_rate']:>5.1f}% {s['avg_r']:>+7.4f} "
          f"${s['total_pnl']:>+10,.0f} ${s['max_dd']:>+9,.0f} "
          f"short=${s['short_pnl']:>+8,.0f}{d}")


def main() -> int:
    print(f"Universe: {WATCHLIST}   (corrected exits: entries<=11:30, exits ride to 15:55)")
    try:
        all_bars, trading_days = load_all_bars()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"Sessions: {len(trading_days)}\n")

    long_only = {s: _p(enable_short=False) for s in WATCHLIST}

    def with_shorts(short_syms, flips=0):
        return {s: _p(enable_short=(s in short_syms), max_flips=flips) for s in WATCHLIST}

    ex_tsla = {"SPY", "QQQ", "NVDA", "AAPL"}  # short all but the squeeze-prone single name
    configs = [
        ("A. long only (baseline)", long_only),
        ("B. long all + short QQQ,NVDA", with_shorts({"QQQ", "NVDA"})),
        ("C. long all + short QQQ,NVDA +flip", with_shorts({"QQQ", "NVDA"}, flips=1)),
        ("D. long all + short QQQ only", with_shorts({"QQQ"})),
        ("E. long all + short ex-TSLA", with_shorts(ex_tsla)),
        ("F. long all + short ex-TSLA +flip", with_shorts(ex_tsla, flips=1)),
        ("G. long all + short ALL +flip", with_shorts(set(WATCHLIST), flips=1)),
    ]

    print(f"{'config':<34} {'n':>4} {'L/S':>9} {'win%':>6} {'avg_R':>7} "
          f"{'total PnL':>11} {'max DD':>10} {'short pnl':>13}")
    print("-" * 104)
    base_s = None
    for label, pbs in configs:
        s = summarize(run_per_symbol(all_bars, trading_days, pbs))
        prow(label, s, base_s)
        if base_s is None:
            base_s = s
    return 0


if __name__ == "__main__":
    sys.exit(main())
