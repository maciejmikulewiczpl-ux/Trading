"""A/B comparison of pre-market gap filters for ORB.

Skip the whole trading day if SPY opens with a gap larger than a threshold
(today's 09:30 open vs yesterday's 15:59 close, in percent). Gaps capture
the "already moved overnight" regime where ORB tends to chase tops.

Sweeps thresholds on the same 180-day bar set:
  - baseline: no filter
  - skip if |gap| > 1.5%
  - skip if |gap| > 1.0%
  - skip if |gap| > 0.75%
  - skip if |gap| > 0.50%

All configs include the already-shipped 11:30 ET cutoff.

Validates Tier-1 #5 of plans/put-yourself-as-an-majestic-cupcake.md.

Run:
    .venv/Scripts/python.exe backtest/compare_regime.py
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

# Reference symbol for measuring market-wide pre-market gap.
GAP_REF_SYMBOL = "SPY"

# Thresholds to test (absolute pct). 0 means no filter.
THRESHOLDS = [None, 1.5, 1.0, 0.75, 0.50]


def compute_daily_gaps(all_bars: pd.DataFrame, ref_symbol: str) -> pd.Series:
    """For each session day with prior-day data, compute the open-vs-prior-close
    gap (percent) for the reference symbol.

    Returns a Series indexed by session date (date object) with float gap_pct.
    """
    sym_bars = all_bars.xs(ref_symbol, level=0)
    # RTH only: 09:30 (incl) .. 16:00 (excl)
    t = sym_bars.index.time
    rth = sym_bars[(t >= time(9, 30)) & (t < time(16, 0))]
    # Group by date; need each day's first open and last close.
    by_date = rth.groupby(rth.index.date)
    firsts = by_date.first()["open"]   # first 09:30+ open per day
    lasts = by_date.last()["close"]    # last close per day
    # Prior-day close: shift lasts by one trading day index.
    prev_close = lasts.shift(1)
    gap = (firsts - prev_close) / prev_close * 100
    return gap


def summarize(trades: list[Trade], session_count: int) -> dict:
    if not trades:
        return {"n": 0, "n_sessions_traded": 0}
    df = pd.DataFrame([{
        "pnl_dollars": t.pnl_dollars,
        "pnl_r": t.pnl_r,
        "exit_time": t.exit_time,
        "exit_reason": t.exit_reason,
        "date": t.date.date() if hasattr(t.date, "date") else t.date,
    } for t in trades])
    df_sorted = df.sort_values("exit_time")
    eq_curve = STARTING_EQUITY + df_sorted["pnl_dollars"].cumsum()
    dd = (eq_curve - eq_curve.cummax()).min()
    return {
        "n": len(df),
        "n_sessions_traded": df["date"].nunique(),
        "n_sessions_skipped": session_count - df["date"].nunique(),
        "win_rate": (df["pnl_r"] > 0).mean() * 100,
        "avg_r": df["pnl_r"].mean(),
        "total_pnl": df["pnl_dollars"].sum(),
        "max_dd": dd,
        "return_pct": (df["pnl_dollars"].sum() / STARTING_EQUITY) * 100,
        "n_target": int((df["exit_reason"] == "target").sum()),
        "n_stop":   int((df["exit_reason"] == "stop").sum()),
        "n_eod":    int((df["exit_reason"] == "eod").sum()),
    }


def main() -> int:
    print(f"Universe: {WATCHLIST}")
    print(f"Starting equity: ${STARTING_EQUITY:,.0f}")
    print("All configs include the 11:30 ET no-entry cutoff (shipped).")
    print(f"Gap reference symbol: {GAP_REF_SYMBOL}")
    try:
        all_bars, trading_days = load_all_bars()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"Sessions: {len(trading_days)}")

    gaps = compute_daily_gaps(all_bars, GAP_REF_SYMBOL)
    print(f"Gap stats: median |gap| = {gaps.abs().median():.2f}%, "
          f"p75 = {gaps.abs().quantile(0.75):.2f}%, "
          f"max = {gaps.abs().max():.2f}%")
    print()

    # Baseline trade list with shipped 11:30 cutoff
    base_params = Params(
        or_minutes=15, target_r=2.0,
        risk_per_trade=100.0, max_position_pct=0.25,
        max_position_dollars=10_000.0,
        no_entry_after_time=time(11, 30),
    )
    all_trades, _ = run_backtest(
        all_bars, trading_days, WATCHLIST, base_params, STARTING_EQUITY
    )

    rows = []
    for threshold in THRESHOLDS:
        if threshold is None:
            label = "baseline (no gap filter)    "
            skip_dates = set()
        else:
            label = f"skip if |gap| > {threshold:.2f}%       "
            skip_dates = set(gaps[gaps.abs() > threshold].index)
        # Post-hoc filter: drop trades on skipped dates.
        kept = [t for t in all_trades
                if (t.date.date() if hasattr(t.date, "date") else t.date) not in skip_dates]
        s = summarize(kept, len(trading_days))
        s["label"] = label
        s["skipped_dates"] = len(skip_dates)
        rows.append(s)

    header = (f"{'config':<32} {'n':>4} {'sk':>3} {'win%':>6} {'avg_R':>8} "
              f"{'tgt/stop/eod':>13} {'total PnL':>13} {'max DD':>11} {'return%':>8}")
    print(header)
    print("-" * len(header))
    for s in rows:
        if s.get("n", 0) == 0:
            print(f"{s['label']:<32}  (no trades)")
            continue
        target_stop_eod = f"{s['n_target']}/{s['n_stop']}/{s['n_eod']}"
        print(
            f"{s['label']:<32} "
            f"{s['n']:>4} "
            f"{s['skipped_dates']:>3} "
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
        d_pnl = s["total_pnl"] - baseline["total_pnl"]
        d_pnl_pct = (d_pnl / abs(baseline["total_pnl"]) * 100) if baseline["total_pnl"] else float("nan")
        d_n = s["n"] - baseline["n"]
        d_dd = s["max_dd"] - baseline["max_dd"]
        d_avg = (s["avg_r"] - baseline["avg_r"]) if s.get("n") else float("nan")
        print(f"  {s['label'].strip():<28}: "
              f"sessions skipped: {s['skipped_dates']:>2}  "
              f"trades {d_n:+d}  PnL {d_pnl:+,.2f} ({d_pnl_pct:+.1f}%)  "
              f"max-DD {d_dd:+,.2f}  avg_R {d_avg:+.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
