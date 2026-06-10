"""Hole-check (Fable #6): are scheduled-event days a quiet drain on tight-OR ORB?

ORB on FOMC / OPEX / earnings mornings is classically chop-then-violence. The vol-dial
is REACTIVE (yesterday's realized vol) and can't see a KNOWN-in-advance event. This
stratifies the tight-OR trailing trades by event tag and asks whether any event class
has materially worse expectancy than normal days — in which case excluding it (or sizing
down) is free edge.

Tags (lookahead-free — all known the night before):
  OPEX     : monthly options expiry = the 3rd Friday of each month.
  FOMC     : Fed decision days (hardcoded 2024-2026 schedule) + the morning after.
  earnings : symbol within +/-1 calendar day of an edate in pead_events.csv.

R-space (avg_R / sum_R per stratum) — capital-agnostic, no slippage model needed to
see a hole. Uses the trailing-trades cache from pit_trades.py (hand universe = the
shipped watchlist). Cheap: no fetch, no minute bars.

Run (after pit_trades.py):
    .venv/Scripts/python.exe backtest/compare_event_days.py
"""
from __future__ import annotations

import pickle
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.compare_selection import _tday  # noqa: E402
from backtest.compare_or_range_realcost import or_pct  # noqa: E402
from backtest.universe_scan import UNIVERSE  # noqa: E402

import pandas as pd  # noqa: E402

WINDOW = 730
OR_THR = 0.5

# Fed decision days (scheduled). 2026 are the published projected dates.
FOMC = {
    date(2024, 6, 12), date(2024, 7, 31), date(2024, 9, 18), date(2024, 11, 7), date(2024, 12, 18),
    date(2025, 1, 29), date(2025, 3, 19), date(2025, 5, 7), date(2025, 6, 18),
    date(2025, 7, 30), date(2025, 9, 17), date(2025, 10, 29), date(2025, 12, 10),
    date(2026, 1, 28), date(2026, 3, 18), date(2026, 4, 29), date(2026, 6, 17),
}


def opex_days(days):
    """3rd Friday of each (year, month) present in the trade days."""
    out = set()
    for d in days:
        first = date(d.year, d.month, 1)
        # weekday(): Mon=0..Sun=6; Friday=4
        first_fri = first + timedelta(days=(4 - first.weekday()) % 7)
        out.add(first_fri + timedelta(days=14))   # 3rd Friday
    return out


def earnings_dates():
    df = pd.read_csv(ROOT / "backtest" / "pead_events.csv", usecols=["symbol", "edate"])
    df["edate"] = pd.to_datetime(df["edate"]).dt.date
    by = {}
    for sym, ed in zip(df["symbol"], df["edate"]):
        by.setdefault(sym, set()).add(ed)
    return by


def stats(rows):
    if not rows:
        return None
    n = len(rows)
    wins = sum(1 for r in rows if r > 0)
    s = sum(rows)
    return {"n": n, "win": 100 * wins / n, "avg_r": s / n, "sum_r": s}


def line(label, st, base):
    if st is None:
        print(f"  {label:<22}    (none)")
        return
    delta = f"{st['avg_r'] - base:+.4f}" if base is not None else ""
    print(f"  {label:<22}{st['n']:>6}{st['win']:>7.1f}%{st['avg_r']:>+9.4f}"
          f"{st['sum_r']:>+9.1f}   {delta:>9}")


def main() -> int:
    blob = pickle.load(open(ROOT / "backtest" / f".pit_trailtrades_{WINDOW}d.pkl", "rb"))
    hand = set(UNIVERSE)
    trades = [t for syms in blob.values() for t in syms
              if t.symbol in hand and or_pct(t) <= OR_THR]
    days = {_tday(t) for t in trades}
    opex = opex_days(days)
    earn = earnings_dates()

    def tag(t):
        d = _tday(t)
        tags = []
        if d in opex:
            tags.append("opex")
        if d in FOMC or (d - timedelta(days=1)) in FOMC:
            tags.append("fomc")
        eds = earn.get(t.symbol, set())
        if any(abs((d - ed).days) <= 1 for ed in eds):
            tags.append("earn")
        return tags

    base = stats([t.pnl_r for t in trades])
    print(f"\n{'='*70}\nEVENT-DAY STRATIFICATION — tight-OR<={OR_THR}% trailing, hand universe")
    print(f"{len(trades)} trades over {len(days)} sessions")
    print(f"{'='*70}")
    print(f"  {'stratum':<22}{'n':>6}{'win%':>7}{'avg_R':>9}{'sum_R':>9}   {'vs base':>9}")
    print("  " + "-" * 64)
    line("ALL (baseline)", base, None)
    for name in ("opex", "fomc", "earn"):
        sub = [t.pnl_r for t in trades if name in tag(t)]
        line(f"{name} days", stats(sub), base["avg_r"])
    # any-event vs clean
    anyev = [t.pnl_r for t in trades if tag(t)]
    clean = [t.pnl_r for t in trades if not tag(t)]
    line("ANY event", stats(anyev), base["avg_r"])
    line("CLEAN (no event)", stats(clean), base["avg_r"])
    print("\nRead: a stratum with materially negative avg_R (and enough n) is a hole —")
    print("excluding it or sizing down is free edge. A near-baseline stratum = no event effect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
