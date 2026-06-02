"""A/B: EOD time-stop exit  vs.  hold overnight, sell at next session's open.

Hypothesis (user, 2026-06-02): ORB winners that survive to the 15:55 ET time-
stop may carry momentum overnight, so selling at the NEXT session's open could
capture gap-continuation instead of giving it back into the close.

What this tests:
  - baseline : current shipped behavior — every trade that reaches eod_flat
               exits at that bar's close (exit_reason == "eod").
  - overnight: identical trades and identical share sizes, but each "eod" exit
               is REPRICED to the next trading day's 09:30 open for that symbol.
               Intraday stop/target exits are untouched (they already closed the
               position before the bell).

Method note — this is a deliberately clean isolation: same entries, same sizes,
only the EOD exit price changes. It does NOT recompute position sizing off the
altered equity curve (a second-order effect), so the delta reflects purely the
overnight hold, not sizing feedback.

Risk note — holding overnight removes the stop while the market is closed. The
output therefore breaks out the carried subset and its tail (best/worst single
overnight move) so the gap-DOWN risk is visible, not just the average.

VERDICT (2026-06-02, 180-day window, 5-name long-only, 11:30 cutoff): REJECTED
from every angle. Baseline EOD-close +$1,408 / -$1,713 maxDD. Carrying overnight
loses vs baseline in all three slices:
  - all eod exits   : +$106   (-92% PnL, ~2x drawdown)
  - winners only    : +$735   (-48% PnL, +55% drawdown)
  - losers only     : +$779   (-45% PnL) -- BUT this is a TRAP: it shows the
                      highest win rate (54%) because a few red trades bounce
                      green, while the subset is 24 help / 34 hurt, -$629 net,
                      -$10.84/trade (the worst per-trade effect). Win-rate
                      mirage masking negative expectancy.
The overnight gap is directionless noise regardless of the day's result:
winners don't continue (66 help / 63 hurt) and losers don't bounce
(24 help / 34 hurt). No edge either way -- carrying just adds coin-flip variance
with the stop switched off. Keep the 15:55 flatten. Don't re-litigate without a
fundamentally different premise (e.g. an overnight-specific signal, not "the day
went well").

Run:
    .venv/Scripts/python.exe backtest/compare_overnight.py
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

# Representative of current shipped live behavior: 15-min OR, 2R target, $100
# risk, 25% / $10k position cap, and the shipped 11:30 ET no-entry cutoff.
PARAMS = Params(
    or_minutes=15,
    target_r=2.0,
    risk_per_trade=100.0,
    max_position_pct=0.25,
    max_position_dollars=10_000.0,
    no_entry_after_time=time(11, 30),
)


def build_next_open_lookup(all_bars: pd.DataFrame, trading_days):
    """(symbol, session_date) -> open price of that symbol's NEXT-session 09:30 bar.

    Returns a dict keyed by (symbol, date) giving the next trading day's first
    RTH (>= 09:30) open. Missing when the symbol has no data next session or the
    session is the last in the window (nothing to carry into).
    """
    day_list = sorted(set(trading_days))
    next_day = {d: day_list[i + 1] for i, d in enumerate(day_list[:-1])}

    # Per-symbol: date -> first RTH open.
    first_open: dict[tuple[str, object], float] = {}
    symbols = all_bars.index.get_level_values(0).unique()
    for sym in symbols:
        sb = all_bars.xs(sym, level=0)
        rth = sb[[t >= time(9, 30) for t in sb.index.time]]
        if rth.empty:
            continue
        # first bar per calendar date
        for d, grp in rth.groupby(rth.index.date):
            first_open[(sym, d)] = float(grp.iloc[0]["open"])

    out: dict[tuple[str, object], float] = {}
    for (sym, d), _ in first_open.items():
        nd = next_day.get(d)
        if nd is not None and (sym, nd) in first_open:
            out[(sym, d)] = first_open[(sym, nd)]
    return out


def _pnl(side: str, entry: float, exit_px: float, stop: float, shares: int):
    if side == "long":
        return (exit_px - entry) * shares, (exit_px - entry) / (entry - stop)
    return (entry - exit_px) * shares, (entry - exit_px) / (stop - entry)


def reprice_overnight(trades: list[Trade], next_open) -> pd.DataFrame:
    """Per-trade frame with baseline vs two overnight scenarios.

    on_*  : carry EVERY 'eod' exit overnight (blanket).
    won_* : carry ONLY 'eod' exits that were in profit at 15:55 (winners-only) —
            momentum should continue best on positions already working.
    """
    rows = []
    for t in trades:
        base_pnl, base_r = t.pnl_dollars, t.pnl_r
        eod = t.exit_reason == "eod"
        key = (t.symbol, t.date.date())
        has_next = key in next_open

        # Blanket overnight: every eod exit carried.
        carried = eod and has_next
        on_exit = next_open[key] if carried else t.exit_price
        on_pnl, on_r = (_pnl(t.side, t.entry_price, on_exit, t.stop_price, t.shares)
                        if carried else (base_pnl, base_r))

        # Winners-only: carry an eod exit ONLY if it was green at the close.
        carried_w = carried and base_pnl > 0
        won_pnl, won_r = (on_pnl, on_r) if carried_w else (base_pnl, base_r)

        # Losers-only: carry an eod exit ONLY if it was red at the close (does a
        # beaten-down position bounce overnight, or keep bleeding?).
        carried_l = carried and base_pnl < 0
        los_pnl, los_r = (on_pnl, on_r) if carried_l else (base_pnl, base_r)

        rows.append({
            "symbol": t.symbol,
            "date": t.date.date(),
            "side": t.side,
            "entry_time": t.entry_time,
            "exit_reason": t.exit_reason,
            "shares": t.shares,
            "entry_price": t.entry_price,
            "eod_exit": t.exit_price,
            "base_pnl": base_pnl,
            "base_r": base_r,
            "carried": carried,
            "carried_w": carried_w,
            "carried_l": carried_l,
            "on_exit": on_exit,
            "on_pnl": on_pnl,
            "on_r": on_r,
            "won_pnl": won_pnl,
            "won_r": won_r,
            "los_pnl": los_pnl,
            "los_r": los_r,
            "overnight_delta": on_pnl - base_pnl,
            "won_delta": won_pnl - base_pnl,
            "los_delta": los_pnl - base_pnl,
        })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, pnl_col: str, r_col: str) -> dict:
    s = df.sort_values("entry_time")  # same order both scenarios -> fair DD compare
    eq = STARTING_EQUITY + s[pnl_col].cumsum()
    dd = (eq - eq.cummax()).min()
    return {
        "n": len(df),
        "win_rate": (df[r_col] > 0).mean() * 100,
        "avg_r": df[r_col].mean(),
        "total_pnl": df[pnl_col].sum(),
        "max_dd": dd,
        "return_pct": (df[pnl_col].sum() / STARTING_EQUITY) * 100,
    }


def main() -> int:
    print(f"Universe: {WATCHLIST}")
    print(f"Starting equity: ${STARTING_EQUITY:,.0f}")
    print(f"Params  : OR={PARAMS.or_minutes}m target={PARAMS.target_r}R "
          f"risk=${PARAMS.risk_per_trade:.0f} cap=${PARAMS.max_position_dollars:,.0f} "
          f"cutoff={PARAMS.no_entry_after_time}")
    try:
        all_bars, trading_days = load_all_bars()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"Sessions: {len(trading_days)}")
    print()

    trades, _ = run_backtest(all_bars, trading_days, WATCHLIST, PARAMS, STARTING_EQUITY)
    if not trades:
        print("No trades generated.")
        return 0

    next_open = build_next_open_lookup(all_bars, trading_days)
    df = reprice_overnight(trades, next_open)

    base = summarize(df, "base_pnl", "base_r")
    over = summarize(df, "on_pnl", "on_r")
    won = summarize(df, "won_pnl", "won_r")
    los = summarize(df, "los_pnl", "los_r")

    hdr = f"{'scenario':<26}{'n':>5}{'win%':>7}{'avg_R':>9}{'total PnL':>14}{'max DD':>13}{'return%':>9}"
    print(hdr)
    print("-" * len(hdr))
    for label, s in (("baseline (EOD close)", base),
                     ("overnight (all eod)", over),
                     ("overnight (winners only)", won),
                     ("overnight (losers only)", los)):
        print(f"{label:<26}{s['n']:>5}{s['win_rate']:>6.1f}%{s['avg_r']:>+9.4f}"
              f"${s['total_pnl']:>+13,.2f}${s['max_dd']:>+12,.2f}{s['return_pct']:>+8.2f}%")

    print()
    for label, s in (("overnight all", over), ("overnight winners-only", won),
                     ("overnight losers-only", los)):
        d_pnl = s["total_pnl"] - base["total_pnl"]
        d_pnl_pct = (d_pnl / abs(base["total_pnl"]) * 100) if base["total_pnl"] else float("nan")
        print(f"Delta {label:<24} vs baseline: PnL {d_pnl:+,.2f} ({d_pnl_pct:+.1f}%)  "
              f"max-DD {s['max_dd'] - base['max_dd']:+,.2f}  "
              f"avg_R {s['avg_r'] - base['avg_r']:+.4f}")

    # ---- the carried subsets: where the overnight effect actually lives ----
    n_eod = int((df["exit_reason"] == "eod").sum())
    n_eod_green = int(((df["exit_reason"] == "eod") & (df["base_pnl"] > 0)).sum())
    print()
    print(f"EOD exits total: {n_eod}  (of which green at 15:55: {n_eod_green})")

    def _subset(mask, delta_col, name):
        sub = df[mask].copy()
        print()
        print(f"[{name}] carried overnight: {len(sub)}")
        if not len(sub):
            return
        wins = (sub[delta_col] > 0).sum()
        loss = (sub[delta_col] < 0).sum()
        flat = len(sub) - wins - loss
        print(f"  overnight move helped: {wins}   hurt: {loss}   flat: {flat}")
        print(f"  total overnight delta: ${sub[delta_col].sum():+,.2f}   "
              f"avg/carried: ${sub[delta_col].mean():+,.2f}")
        best = sub.loc[sub[delta_col].idxmax()]
        worst = sub.loc[sub[delta_col].idxmin()]
        print(f"  best  overnight: {best['symbol']} {best['date']} ${best[delta_col]:+,.2f} "
              f"(eod ${best['eod_exit']:.2f} -> open ${best['on_exit']:.2f})")
        print(f"  worst overnight: {worst['symbol']} {worst['date']} ${worst[delta_col]:+,.2f} "
              f"(eod ${worst['eod_exit']:.2f} -> open ${worst['on_exit']:.2f})")
        big_adverse = sub[sub[delta_col] < -PARAMS.risk_per_trade]
        print(f"  carried trades whose overnight move alone lost > 1R "
              f"(${PARAMS.risk_per_trade:.0f}): {len(big_adverse)}")

    _subset(df["carried"], "overnight_delta", "all eod")
    _subset(df["carried_w"], "won_delta", "winners only")
    _subset(df["carried_l"], "los_delta", "losers only")

    return 0


if __name__ == "__main__":
    sys.exit(main())
