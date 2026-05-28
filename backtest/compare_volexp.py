"""A/B comparison of LOCAL volume-expansion breakout confirmation for ORB longs.

Distinct from compare_rvol.py: that compared the breakout bar to a 20-day
volume profile (a long-term baseline) and FAILED. This tests the rule from
the r/Daytrading ORB+volume writeup, which is a LOCAL comparison:

  "The confirming volume bar does not need to be massive, but it must be
   larger than the recent volume bars preceding the breakout. If volume
   doesn't support the move, don't enter — lack of volume signals the
   breakout will fail."

So the signal is volume EXPANSION at the moment of breakout relative to the
immediately preceding bars (quiet consolidation -> surge on the break), NOT
volume vs a 20-session norm.

Metric per trade: expansion = breakout_bar_volume / mean(volume of the N bars
immediately before the breakout bar). A trade survives if expansion >= k.

Configs sweep N (3, 5 preceding bars) and k (1.0, 1.25, 1.5), plus a strict
"larger than the MAX of the prior 5" variant (the literal "larger than recent
bars" reading). All on top of the shipped 11:30 ET cutoff.

Run:
    .venv/Scripts/python.exe backtest/compare_volexp.py
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


def expansion_ratio(all_bars: pd.DataFrame, t: Trade, n: int, agg: str):
    """breakout_bar_volume / agg(volume of the n bars immediately before it).

    agg is 'mean' or 'max'. Returns None if data is unavailable. The breakout
    bar is the last RTH bar strictly before the entry bar.
    """
    try:
        sb = all_bars.xs(t.symbol, level=0)
    except KeyError:
        return None
    tt = sb.index.time
    rth = sb[(tt >= time(9, 30)) & (tt < time(16, 0))]
    day = rth[rth.index.date == t.entry_time.date()]
    before_entry = day[day.index < t.entry_time]
    if len(before_entry) < n + 1:
        return None
    breakout_vol = float(before_entry.iloc[-1]["volume"])
    prior = before_entry.iloc[-(n + 1):-1]["volume"].astype(float)
    if prior.empty:
        return None
    base = prior.max() if agg == "max" else prior.mean()
    if base is None or pd.isna(base) or base <= 0:
        return None
    return breakout_vol / float(base)


def summarize(trades, label):
    if not trades:
        return {"n": 0, "label": label}
    df = pd.DataFrame([{
        "pnl_dollars": t.pnl_dollars, "pnl_r": t.pnl_r,
        "exit_time": t.exit_time, "exit_reason": t.exit_reason,
    } for t in trades])
    df_sorted = df.sort_values("exit_time")
    eq = STARTING_EQUITY + df_sorted["pnl_dollars"].cumsum()
    dd = (eq - eq.cummax()).min()
    return {
        "label": label, "n": len(df),
        "win_rate": (df["pnl_r"] > 0).mean() * 100,
        "avg_r": df["pnl_r"].mean(),
        "total_pnl": df["pnl_dollars"].sum(),
        "max_dd": dd,
        "return_pct": df["pnl_dollars"].sum() / STARTING_EQUITY * 100,
        "n_target": int((df["exit_reason"] == "target").sum()),
        "n_stop": int((df["exit_reason"] == "stop").sum()),
        "n_eod": int((df["exit_reason"] == "eod").sum()),
    }


# (label, N preceding bars, agg, threshold k). None threshold = baseline.
CONFIGS = [
    ("baseline (all trades)      ", 5, "mean", None),
    ("exp vs mean(3) >= 1.0      ", 3, "mean", 1.0),
    ("exp vs mean(5) >= 1.0      ", 5, "mean", 1.0),
    ("exp vs mean(5) >= 1.25     ", 5, "mean", 1.25),
    ("exp vs mean(5) >= 1.5      ", 5, "mean", 1.5),
    ("exp vs MAX(5)  >= 1.0      ", 5, "max", 1.0),
]


def main() -> int:
    print(f"Universe: {WATCHLIST}")
    print(f"Starting equity: ${STARTING_EQUITY:,.0f}")
    print("Volume EXPANSION = breakout bar vol / agg(prior N bars). Local, not 20-day.")
    print("All configs include the shipped 11:30 ET cutoff.")
    try:
        all_bars, trading_days = load_all_bars()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"Sessions: {len(trading_days)}")

    base_params = Params(
        or_minutes=15, target_r=2.0,
        risk_per_trade=100.0, max_position_pct=0.25,
        max_position_dollars=10_000.0,
        no_entry_after_time=time(11, 30),
    )
    all_trades, _ = run_backtest(all_bars, trading_days, WATCHLIST,
                                 base_params, STARTING_EQUITY)

    # Precompute ratios per (N, agg) to avoid recomputation.
    ratio_cache = {}
    def ratio(t, n, agg):
        key = (id(t), n, agg)
        if key not in ratio_cache:
            ratio_cache[key] = expansion_ratio(all_bars, t, n, agg)
        return ratio_cache[key]

    # Eligibility: a trade must have a computable ratio for ALL (N,agg) used,
    # so every config is scored on the same trade set.
    needed = {(n, agg) for _, n, agg, _ in CONFIGS}
    def eligible(t):
        return all(ratio(t, n, agg) is not None for (n, agg) in needed)
    elig = [t for t in all_trades if eligible(t)]
    dropped = len(all_trades) - len(elig)
    print(f"Eligible trades: {len(elig)} (dropped {dropped} for missing prior bars)")
    sample = pd.Series([ratio(t, 5, "mean") for t in elig])
    print(f"exp-vs-mean(5) distribution: median {sample.median():.2f}, "
          f"p25 {sample.quantile(0.25):.2f}, p75 {sample.quantile(0.75):.2f}, "
          f"max {sample.max():.2f}")
    print()

    rows = []
    for label, n, agg, thr in CONFIGS:
        if thr is None:
            kept = elig
        else:
            kept = [t for t in elig if ratio(t, n, agg) >= thr]
        rows.append(summarize(kept, label))

    header = (f"{'config':<28} {'n':>4} {'win%':>6} {'avg_R':>8} "
              f"{'tgt/stop/eod':>13} {'total PnL':>13} {'max DD':>11} {'ret%':>7}")
    print(header)
    print("-" * len(header))
    for s in rows:
        if s["n"] == 0:
            print(f"{s['label']:<28}  (no trades)")
            continue
        tse = f"{s['n_target']}/{s['n_stop']}/{s['n_eod']}"
        print(f"{s['label']:<28} {s['n']:>4} {s['win_rate']:>5.1f}% "
              f"{s['avg_r']:>+8.4f} {tse:>13} ${s['total_pnl']:>+12,.2f} "
              f"${s['max_dd']:>+10,.2f} {s['return_pct']:>+6.2f}%")

    print()
    base = rows[0]
    print(f"Delta vs '{base['label'].strip()}':")
    for s in rows[1:]:
        if s["n"] == 0:
            continue
        d_pnl = s["total_pnl"] - base["total_pnl"]
        d_pct = d_pnl / abs(base["total_pnl"]) * 100 if base["total_pnl"] else float("nan")
        print(f"  {s['label'].strip():<22}: trades {s['n'] - base['n']:+d}  "
              f"PnL {d_pnl:+,.2f} ({d_pct:+.1f}%)  max-DD {s['max_dd'] - base['max_dd']:+,.2f}  "
              f"avg_R {s['avg_r'] - base['avg_r']:+.4f}  win% {s['win_rate'] - base['win_rate']:+.1f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
