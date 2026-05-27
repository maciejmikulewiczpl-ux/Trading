"""Hunt for a CONDITIONAL short edge in ORB.

The naive symmetric short loses money (see backtest/compare_short.py). This
script loads the 180-day bars once and sweeps the levers most likely to isolate
a profitable short subset, all SHORT-ONLY (enable_long=False), all with the
shipped 11:30 ET cutoff unless noted:

  A. Per-symbol breakdown      - is the loss universal or one bad name?
  B. target_r sweep            - shorts may mean-revert -> tighter target wins?
  C. stop_offset_pct sweep     - buffer above OR_high to dodge squeeze sweeps.
  D. Gap-down regime filter    - take the short only when the symbol ALREADY
                                 gapped down (continuation), post-hoc by date.

Nothing here touches live. Run:
    uv run --with pip-system-certs python backtest/explore_short.py
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

CUTOFF = time(11, 30)


def short_params(target_r: float = 2.0, stop_offset_pct: float = 0.0) -> Params:
    return Params(
        or_minutes=15,
        target_r=target_r,
        risk_per_trade=100.0,
        max_position_pct=0.25,
        max_position_dollars=10_000.0,
        no_entry_after_time=CUTOFF,
        stop_offset_pct=stop_offset_pct,
        enable_long=False,
        enable_short=True,
    )


def compute_gaps_all(all_bars: pd.DataFrame, symbols) -> dict:
    """{(symbol, date) -> open-vs-prior-close gap %} for each symbol."""
    out: dict = {}
    for sym in symbols:
        if sym not in set(all_bars.index.get_level_values(0).unique()):
            continue
        sb = all_bars.xs(sym, level=0)
        t = sb.index.time
        rth = sb[(t >= time(9, 30)) & (t < time(16, 0))]
        by_date = rth.groupby(rth.index.date)
        firsts = by_date.first()["open"]
        lasts = by_date.last()["close"]
        gap = (firsts - lasts.shift(1)) / lasts.shift(1) * 100
        for d, g in gap.items():
            out[(sym, d)] = g
    return out


def _tdate(t: Trade):
    return t.date.date() if hasattr(t.date, "date") else t.date


def summarize(trades: list[Trade]) -> dict:
    if not trades:
        return {"n": 0}
    df = pd.DataFrame([{
        "pnl_dollars": t.pnl_dollars, "pnl_r": t.pnl_r,
        "exit_time": t.exit_time, "exit_reason": t.exit_reason,
    } for t in trades])
    df_sorted = df.sort_values("exit_time")
    eq = STARTING_EQUITY + df_sorted["pnl_dollars"].cumsum()
    return {
        "n": len(df),
        "win_rate": (df["pnl_r"] > 0).mean() * 100,
        "avg_r": df["pnl_r"].mean(),
        "total_pnl": df["pnl_dollars"].sum(),
        "max_dd": (eq - eq.cummax()).min(),
        "n_target": int((df["exit_reason"] == "target").sum()),
        "n_stop": int((df["exit_reason"] == "stop").sum()),
        "n_eod": int((df["exit_reason"] == "eod").sum()),
    }


def prow(label: str, s: dict) -> None:
    if s.get("n", 0) == 0:
        print(f"{label:<26}  (no trades)")
        return
    tse = f"{s['n_target']}/{s['n_stop']}/{s['n_eod']}"
    print(f"{label:<26} {s['n']:>4} {s['win_rate']:>5.1f}% {s['avg_r']:>+8.4f} "
          f"{tse:>13} ${s['total_pnl']:>+11,.2f} ${s['max_dd']:>+10,.2f}")


HEADER = (f"{'config':<26} {'n':>4} {'win%':>6} {'avg_R':>8} "
          f"{'tgt/stop/eod':>13} {'total PnL':>12} {'max DD':>11}")


def main() -> int:
    print(f"Universe: {WATCHLIST}   (SHORT-ONLY, 11:30 ET cutoff)")
    try:
        all_bars, trading_days = load_all_bars()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"Sessions: {len(trading_days)}\n")

    gaps = compute_gaps_all(all_bars, WATCHLIST)

    # Baseline short-only trade list (target_r=2.0, no offset).
    base_trades, _ = run_backtest(all_bars, trading_days, WATCHLIST,
                                  short_params(), STARTING_EQUITY)

    # --- A. Per-symbol breakdown ---
    print("=== A. Per-symbol (short-only baseline, target_r=2.0) ===")
    print(HEADER)
    print("-" * len(HEADER))
    by_sym: dict[str, list[Trade]] = {}
    for t in base_trades:
        by_sym.setdefault(t.symbol, []).append(t)
    for sym in WATCHLIST:
        prow(sym, summarize(by_sym.get(sym, [])))
    prow("ALL", summarize(base_trades))

    # --- B. target_r sweep ---
    print("\n=== B. target_r sweep (short-only) ===")
    print(HEADER)
    print("-" * len(HEADER))
    for tr in [1.0, 1.5, 2.0, 3.0]:
        trades, _ = run_backtest(all_bars, trading_days, WATCHLIST,
                                 short_params(target_r=tr), STARTING_EQUITY)
        prow(f"target_r={tr}", summarize(trades))

    # --- C. stop_offset sweep ---
    print("\n=== C. stop_offset_pct sweep (short-only, target_r=2.0) ===")
    print(HEADER)
    print("-" * len(HEADER))
    for so in [0.0, 0.05, 0.10, 0.20]:
        trades, _ = run_backtest(all_bars, trading_days, WATCHLIST,
                                 short_params(stop_offset_pct=so), STARTING_EQUITY)
        prow(f"stop_offset={so:.2f}", summarize(trades))

    # --- D. Gap regime filter (post-hoc on baseline short trades) ---
    # Hypothesis: short continuation works when the name already gapped DOWN.
    print("\n=== D. Gap regime filter (short-only baseline, by symbol's own gap) ===")
    print(HEADER)
    print("-" * len(HEADER))
    filters = [
        ("all (no filter)", lambda g: True),
        ("gap <= -0.25% (down)", lambda g: g <= -0.25),
        ("gap <= -0.50% (down)", lambda g: g <= -0.50),
        ("gap <= -1.00% (down)", lambda g: g <= -1.00),
        ("gap >= +0.50% (up/fade)", lambda g: g >= 0.50),
    ]
    for label, cond in filters:
        kept = [t for t in base_trades
                if (gk := gaps.get((t.symbol, _tdate(t)))) is not None and cond(gk)]
        prow(label, summarize(kept))

    return 0


if __name__ == "__main__":
    sys.exit(main())
