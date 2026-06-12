"""Earnings-day "stocks in play" flag: does the catalyst rescue wide-OR breakouts?

MOTIVATION (strategy-landscape survey 2026-06-12): Zarattini/Barbon/Aziz find ORB
works best restricted to "stocks in play" (unusual-activity names). Our RVOL gate
test already REJECTED the volume flavor of that selector. The earnings-calendar
flavor is untested — and it can't just overlay the live config: earnings mornings
have WIDE opening ranges, which tight-OR<=0.5% mostly excludes by construction
(only 30/6338 tight-OR trades sit near earnings). So the only way an earnings flag
ADDS anything is if the catalyst rescues the wide-OR bucket we currently discard
(wide-OR non-event trades are net-negative — compare_or_range_filter).

PRE-REGISTERED DESIGN (R-space, longs, written before running):
  Trades: .pit_trailtrades_730d.pkl (trailing exits, no OR filter applied yet),
          restricted to symbols in BOTH the shipped watchlist and pead_events.csv
          coverage, trade days <= last covered edate. OOS = window split in half.
  in-play(sym, d): d == event entry_day (entry_day is lookahead-free: AMC -> next
          session, BMO -> same session; built by the PEAD dig).
  Strata: A tight-OR (<=0.5%) not in-play   [shipped-config baseline]
          B tight-OR in-play
          C wide-OR  (>0.5%)  not in-play   [the discarded bucket — known negative]
          D wide-OR  in-play                [THE CANDIDATE]
          E = D split by overnight gap direction (descriptive only)

GATE (to even consider portfolio wiring): stratum D avg_R >= stratum A avg_R,
with n_D >= 100, positive in BOTH OOS halves. Anything less = no earnings edge in
our framework; close the question. (Wide-OR trades are low-slippage, so R-space
is fair here.)

Run:
    .venv/Scripts/python.exe backtest/compare_earnings_flag.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.compare_or_range_realcost import or_pct  # noqa: E402
from backtest.universe_scan import UNIVERSE  # noqa: E402

import pandas as pd  # noqa: E402

OR_THR = 0.5


def stats(rows):
    if not rows:
        return None
    n = len(rows)
    wins = sum(1 for r in rows if r > 0)
    return {"n": n, "win": 100 * wins / n, "avg_r": sum(rows) / n, "sum_r": sum(rows)}


def line(label, st):
    if st is None or st["n"] == 0:
        print(f"  {label:<34}    (none)")
        return
    print(f"  {label:<34}{st['n']:>7}{st['win']:>7.1f}%{st['avg_r']:>+9.4f}{st['sum_r']:>+10.1f}")


def main() -> int:
    ev = pd.read_csv(ROOT / "backtest" / "pead_events.csv")
    ev["entry_day"] = pd.to_datetime(ev["entry_day"]).dt.date
    covered_syms = set(ev["symbol"]) & set(UNIVERSE)
    last_cov = ev["entry_day"].max()
    inplay = {}      # (sym, day) -> gap_pct
    for sym, d, gap in zip(ev["symbol"], ev["entry_day"], ev["gap_pct"]):
        if sym in covered_syms:
            inplay[(sym, d)] = gap

    blob = pickle.load(open(ROOT / "backtest" / ".pit_trailtrades_730d.pkl", "rb"))
    trades = [t for syms in blob.values() for t in syms
              if t.symbol in covered_syms and t.side == "long"
              and t.entry_time.date() <= last_cov]
    days = sorted({t.entry_time.date() for t in trades})
    mid = days[len(days) // 2]

    def strat(t):
        tight = or_pct(t) <= OR_THR
        ip = (t.symbol, t.entry_time.date()) in inplay
        return ("tight" if tight else "wide", ip)

    print(f"\n{'='*78}")
    print(f"EARNINGS IN-PLAY x OR-WIDTH — trailing longs, {len(covered_syms)} covered "
          f"symbols, {days[0]} -> {days[-1]}")
    print(f"{len(trades)} trades | in-play events usable: "
          f"{len({k for k in inplay if k[1] >= days[0]})}")
    print(f"{'='*78}")

    for half, dlo, dhi in (("FULL", days[0], days[-1]),
                           ("H1", days[0], mid),
                           ("H2", mid, days[-1])):
        sub = [t for t in trades if dlo <= t.entry_time.date() < dhi] if half != "FULL" \
            else trades
        print(f"\n  [{half}]  {'stratum':<32}{'n':>7}{'win%':>8}{'avg_R':>9}{'sum_R':>10}")
        print("  " + "-" * 68)
        for label, w, ip in (("A tight-OR, not in play (shipped)", "tight", False),
                             ("B tight-OR, EARNINGS in play", "tight", True),
                             ("C wide-OR, not in play (discard)", "wide", False),
                             ("D wide-OR, EARNINGS in play", "wide", True)):
            line(label, stats([t.pnl_r for t in sub if strat(t) == (w, ip)]))
        # E: candidate D split by gap direction (descriptive)
        if half == "FULL":
            for gl, cond in (("   D & gap UP", lambda g: g > 0),
                             ("   D & gap DOWN", lambda g: g <= 0)):
                rows = [t.pnl_r for t in sub
                        if strat(t) == ("wide", True)
                        and cond(inplay[(t.symbol, t.entry_time.date())])]
                line(gl, stats(rows))

    print(f"\nGATE: D.avg_R >= A.avg_R with n_D >= 100 and D positive in both halves.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
