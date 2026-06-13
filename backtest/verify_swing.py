"""verify_swing.py -- Fable's audit of the swing-engine results (2026-06-12).

Two suspected flaws in the executed backtest:

  FLAW A (accounting): run_swing's daily_pnl DOUBLE-COUNTS every trade --
  close-to-close MTM is added on each open day AND the full entry->exit PnL is
  added again on exit day. Sharpe 3.03 / maxDD / crisis Sharpes were computed
  on that corrupted series.
    Check: sum(daily_pnl) vs sum(trade pnl) -- equal if correct, ~2x if double.
    Fix: rebuild the daily series from trades + price marks (entry day:
    close-entry; mid days: close-close; exit day: exit-prevclose-cost). The
    rebuilt series sums EXACTLY to trade-level PnL by construction.

  FLAW B (survivorship): the candidate pool = top-500 by MEAN DOLLAR VOLUME
  2024-2026 (pit_daily_730d). The 2016 sim therefore trades only names known
  to be 2024's giants -- future-winner selection, exactly what the spec warned
  against. Unfixable for 2016-2023 without PIT constituent data.
    Measurement: per-trade expectancy + corrected Sharpe on the CLEAN window
    (entries >= 2024-07-01, where the pool is approximately point-in-time)
    vs the full window. The clean window is the only defensible number.

Also reports actual capital deployment (notional in use) so "how much can we
make" can be answered honestly.

Run:  .venv-openbb\\Scripts\\python.exe backtest\\verify_swing.py
"""
from __future__ import annotations

import math
import pickle
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.run_swing import run_simulation, CACHE, COST_RT_PCT  # noqa: E402

CLEAN_START = date(2024, 7, 1)
CRISES = {"2020 crash": (date(2020, 2, 1), date(2020, 4, 30)),
          "2022 bear": (date(2022, 1, 1), date(2022, 12, 31)),
          "2025-04 tariff": (date(2025, 4, 1), date(2025, 4, 30))}


def sharpe(s: pd.Series) -> float:
    if len(s) < 10 or s.std() == 0:
        return 0.0
    return float(s.mean() / s.std() * math.sqrt(252))


def rebuild_daily(trades, data) -> pd.Series:
    """Correct daily PnL: per trade -- entry day close-entry, mid days dClose,
    exit day exit-prevclose-cost. Sums exactly to sum(trade.pnl_net)."""
    by: dict[date, float] = {}
    for t in trades:
        df = data["symbols"][t.symbol]
        e_ts, x_ts = pd.Timestamp(t.entry_date), pd.Timestamp(t.exit_date)
        marks = df.loc[(df.index >= e_ts) & (df.index <= x_ts), "Close"]
        days = list(marks.index)
        sh = t.shares
        if len(days) <= 1:
            # degenerate: everything lands on exit day
            by[t.exit_date] = by.get(t.exit_date, 0.0) + t.pnl_net
            continue
        # entry day: close - entry_price
        d0 = days[0].date()
        by[d0] = by.get(d0, 0.0) + (float(marks.iloc[0]) - t.entry_price) * sh
        # mid days: close-to-close (up to the day BEFORE exit)
        for i in range(1, len(days) - 1):
            d = days[i].date()
            by[d] = by.get(d, 0.0) + (float(marks.iloc[i]) - float(marks.iloc[i - 1])) * sh
        # exit day: exit price vs previous close, minus cost
        dx = days[-1].date()
        by[dx] = by.get(dx, 0.0) + (t.exit_price - float(marks.iloc[-2])) * sh - t.cost
    ser = pd.Series(by).sort_index()
    # reindex over all SPY trading days in the span (zeros when flat)
    spy_days = [d.date() for d in data["spy"].index
                if ser.index[0] <= d.date() <= ser.index[-1]]
    return ser.reindex(spy_days, fill_value=0.0)


def report(label, trades, daily):
    pnl = sum(t.pnl_net for t in trades)
    wins = [t for t in trades if t.pnl_net > 0]
    print(f"\n  --- {label} ---")
    print(f"  trades {len(trades)} | win {100*len(wins)/max(len(trades),1):.0f}% | "
          f"net PnL ${pnl:+,.0f}")
    cum = daily.cumsum()
    print(f"  CORRECTED Sharpe {sharpe(daily):.2f} | maxDD ${float((cum-cum.cummax()).min()):,.0f}")
    for name, (d0, d1) in CRISES.items():
        w = daily[(daily.index >= d0) & (daily.index <= d1)]
        print(f"  {name}: ${w.sum():+,.0f}")


def main() -> None:
    data = pickle.load(open(CACHE, "rb"))
    print("Running V1 simulation (as executed)...")
    trades, daily_buggy = run_simulation(data, variant="V1")

    # ---------- FLAW A: double-count check ----------
    trade_pnl = sum(t.pnl_net for t in trades)
    print("\n=== FLAW A: accounting ===")
    print(f"  sum(trade PnL):          ${trade_pnl:+,.0f}")
    print(f"  sum(buggy daily series): ${daily_buggy.sum():+,.0f}")
    print(f"  ratio: {daily_buggy.sum()/trade_pnl:.2f}x  (1.00 = correct, ~2 = double-counted)")

    daily = rebuild_daily(trades, data)
    print(f"  rebuilt daily sum:       ${daily.sum():+,.0f}  (must equal trade PnL)")
    print(f"  buggy   Sharpe: {sharpe(daily_buggy):.2f}")
    print(f"  CORRECT Sharpe: {sharpe(daily):.2f}")
    report("FULL WINDOW (still survivorship-contaminated)", trades, daily)

    # ---------- FLAW B: clean-window measurement ----------
    print("\n=== FLAW B: survivorship ===")
    clean = [t for t in trades if t.entry_date >= CLEAN_START]
    dirty = [t for t in trades if t.entry_date < CLEAN_START]
    exp_c = sum(t.pnl_net for t in clean) / max(len(clean), 1)
    exp_d = sum(t.pnl_net for t in dirty) / max(len(dirty), 1)
    print(f"  pre-2024-07 (contaminated): {len(dirty)} trades, ${exp_d:+.2f}/trade")
    print(f"  2024-07+   (approx clean):  {len(clean)} trades, ${exp_c:+.2f}/trade")
    daily_clean = daily[daily.index >= CLEAN_START]
    report("CLEAN WINDOW 2024-07 -> now", clean, daily_clean)
    yrs = (daily_clean.index[-1] - daily_clean.index[0]).days / 365.25
    print(f"  clean-window annualized PnL: ${sum(t.pnl_net for t in clean)/yrs:+,.0f}/yr "
          f"over {yrs:.1f}y")

    # ---------- capital usage ----------
    print("\n=== capital deployment (full window) ===")
    notionals = [t.entry_price * t.shares for t in trades]
    print(f"  per-position notional: median ${pd.Series(notionals).median():,.0f}  "
          f"mean ${pd.Series(notionals).mean():,.0f}  max ${max(notionals):,.0f}")
    # avg concurrent: approximate from holds
    open_days = sum(t.hold_days for t in trades)
    span_days = (trades[-1].exit_date - trades[0].entry_date).days
    print(f"  avg concurrent positions: {open_days/max(span_days,1):.1f} (cap 12)")
    print(f"  avg deployed notional: ${open_days/max(span_days,1)*pd.Series(notionals).mean():,.0f}")


if __name__ == "__main__":
    main()
