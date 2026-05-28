"""A/B comparison of relative-volume (RVOL) breakout confirmation for ORB longs.

Hypothesis: a breakout that fires on heavy volume (institutional participation)
follows through more reliably than one on thin volume. Test it by gating
entries on the breakout bar's RVOL.

RVOL definition (lookahead-free):
  For the 1-min bar that triggered the breakout (close > OR_high), RVOL =
  that bar's volume / the average volume for THAT minute-of-day over the
  trailing 20 sessions strictly before the trade date. RVOL = 1.0 means
  "normal volume for this time of day"; 2.0 means "twice normal".

Method: post-hoc filter on the baseline trade list (with the shipped 11:30 ET
cutoff). Sweeps RVOL thresholds; a trade survives if its breakout-bar RVOL
>= threshold. Trades in the first ~20 sessions (no profile yet) are excluded
from ALL configs so the comparison is apples-to-apples.

Validates Tier-2 #6 of plans/put-yourself-as-an-majestic-cupcake.md.

Run:
    .venv/Scripts/python.exe backtest/compare_rvol.py
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

PROFILE_WINDOW = 20          # trailing sessions for the volume profile
PROFILE_MIN_PERIODS = 10     # need at least this many prior sessions
THRESHOLDS = [None, 1.0, 1.25, 1.5, 2.0]


def build_volume_profiles(all_bars: pd.DataFrame, watchlist: list[str]) -> dict:
    """Per symbol, a DataFrame indexed by date, columns = minute-of-day ('HH:MM'),
    values = trailing-{PROFILE_WINDOW}-session mean volume at that minute,
    shifted by 1 session so date D uses only sessions strictly before D.
    """
    profiles = {}
    symbols_in_data = set(all_bars.index.get_level_values(0).unique())
    for sym in watchlist:
        if sym not in symbols_in_data:
            continue
        sb = all_bars.xs(sym, level=0)
        t = sb.index.time
        rth = sb[(t >= time(9, 30)) & (t < time(16, 0))].copy()
        if rth.empty:
            continue
        rth["date"] = rth.index.date
        rth["minute"] = [ts.strftime("%H:%M") for ts in rth.index]
        pivot = rth.pivot_table(index="date", columns="minute",
                                values="volume", aggfunc="sum")
        trailing = pivot.shift(1).rolling(PROFILE_WINDOW,
                                          min_periods=PROFILE_MIN_PERIODS).mean()
        profiles[sym] = trailing
    return profiles


def breakout_bar_for_trade(all_bars: pd.DataFrame, t: Trade):
    """Return (volume, 'HH:MM') of the bar that triggered the breakout — the
    last available bar strictly before the entry bar on the trade's symbol/day.
    """
    try:
        sb = all_bars.xs(t.symbol, level=0)
    except KeyError:
        return None
    entry_ts = t.entry_time
    day_bars = sb[sb.index.date == entry_ts.date()]
    before_entry = day_bars[day_bars.index < entry_ts]
    if before_entry.empty:
        return None
    bar = before_entry.iloc[-1]
    return float(bar["volume"]), before_entry.index[-1].strftime("%H:%M")


def compute_rvol(all_bars: pd.DataFrame, profiles: dict, t: Trade):
    """RVOL of the trade's breakout bar, or None if no profile/data."""
    prof = profiles.get(t.symbol)
    if prof is None:
        return None
    bb = breakout_bar_for_trade(all_bars, t)
    if bb is None:
        return None
    vol, minute = bb
    trade_date = t.entry_time.date()
    try:
        avg = prof.loc[trade_date, minute]
    except KeyError:
        return None
    if avg is None or pd.isna(avg) or avg <= 0:
        return None
    return vol / float(avg)


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


def main() -> int:
    print(f"Universe: {WATCHLIST}")
    print(f"Starting equity: ${STARTING_EQUITY:,.0f}")
    print(f"RVOL profile: trailing {PROFILE_WINDOW} sessions, per minute-of-day, lookahead-free.")
    print("All configs include the shipped 11:30 ET cutoff.")
    try:
        all_bars, trading_days = load_all_bars()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"Sessions: {len(trading_days)}")

    profiles = build_volume_profiles(all_bars, WATCHLIST)

    base_params = Params(
        or_minutes=15, target_r=2.0,
        risk_per_trade=100.0, max_position_pct=0.25,
        max_position_dollars=10_000.0,
        no_entry_after_time=time(11, 30),
    )
    all_trades, _ = run_backtest(all_bars, trading_days, WATCHLIST,
                                 base_params, STARTING_EQUITY)

    # Attach RVOL; drop trades with no profile (early sessions) from ALL configs.
    rvols = {id(t): compute_rvol(all_bars, profiles, t) for t in all_trades}
    eligible = [t for t in all_trades if rvols[id(t)] is not None]
    dropped = len(all_trades) - len(eligible)
    rv_series = pd.Series([rvols[id(t)] for t in eligible])
    print(f"Trades with RVOL: {len(eligible)} (dropped {dropped} early/no-profile)")
    print(f"RVOL distribution: median {rv_series.median():.2f}, "
          f"p25 {rv_series.quantile(0.25):.2f}, p75 {rv_series.quantile(0.75):.2f}, "
          f"max {rv_series.max():.2f}")
    print()

    rows = []
    for thr in THRESHOLDS:
        if thr is None:
            label = "baseline (all eligible)   "
            kept = eligible
        else:
            label = f"RVOL >= {thr:.2f}            "
            kept = [t for t in eligible if rvols[id(t)] >= thr]
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
        print(f"  {s['label'].strip():<18}: trades {s['n'] - base['n']:+d}  "
              f"PnL {d_pnl:+,.2f} ({d_pct:+.1f}%)  max-DD {s['max_dd'] - base['max_dd']:+,.2f}  "
              f"avg_R {s['avg_r'] - base['avg_r']:+.4f}  win% {s['win_rate'] - base['win_rate']:+.1f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
