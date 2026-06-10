"""Point-in-time universe test, step 4: the verdict.

Compares the tight-OR / trailing edge on TWO universes from one identical trade pool
(built by pit_trades.py), with identical cap-aware real-$ accounting:

  HAND : the curated ~100-name watchlist (any day a name is in it).
  PIT  : point-in-time eligible — a trade counts only if its symbol was in that
         month's mechanical top-100 by trailing dollar volume (no hindsight).

Both restricted to the SAME sessions (months that have a PIT ranking, 2024-09+), same
trend filter, same tight-OR<=0.5% cut, same cap-16 portfolio, same cents-slippage
(calibrated ONCE on the HAND set so it reproduces the ~+$6.7k/1.80 anchor and PIT is
penalized correctly for any cheaper/tighter names it adds).

READ: if PIT tight-OR stays clearly positive with comparable Sharpe + small maxDD,
the edge is NOT just survivorship — it survives a mechanically-chosen universe. If PIT
craters vs HAND, the hand-picking was load-bearing and every $ projection must be
discounted.

Run (after pit_trades.py finishes):
    .venv/Scripts/python.exe backtest/pit_compare.py
"""
from __future__ import annotations

import math
import pickle
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.universe_portfolio import portfolio  # noqa: E402
from backtest.compare_selection import _tday  # noqa: E402
from backtest.compare_norefill_trend import trend_eligibility, apply_filter  # noqa: E402
from backtest.compare_volpause import prior_vol_flags, perf, RISK, CAP  # noqa: E402
from backtest.compare_or_range_realcost import or_pct, risk_ps  # noqa: E402
from backtest.universe_scan import UNIVERSE  # noqa: E402

import pandas as pd  # noqa: E402

WINDOW = 730
OR_THR = 0.5
TARGET_MEDIAN_R = 0.042
NOTIONAL_CAP = 10_000.0
SLIP_MULT = [1.0, 1.5]


def dollar_series(taken, days, days_mult, cents):
    by, tot = {}, 0.0
    for t in taken:
        rps = risk_ps(t)
        shares = min(math.floor(RISK * days_mult.get(_tday(t), 1.0) / rps),
                     math.floor(NOTIONAL_CAP / t.entry_price))
        if shares <= 0:
            continue
        pnl = (t.exit_price - t.entry_price) * shares - 2.0 * cents * shares
        by[_tday(t)] = by.get(_tday(t), 0.0) + pnl
        tot += pnl
    s = pd.Series({d: by.get(d, 0.0) for d in sorted(days)})
    return s, (tot / len(taken) if taken else 0.0)


HEAD = (f"{'universe / filter':<28}{'trades':>7}{'PnL$':>10}{'Sharpe':>8}{'maxDD$':>9}"
        f"   {'h1 PnL':>9}{'h2 PnL':>9}{'avg$/tr':>8}")


def row(label, arm, days, mid, days_mult, cents):
    taken = portfolio(arm, CAP)
    s, avgtr = dollar_series(taken, days, days_mult, cents)
    f = perf(s)
    h1 = s[[d for d in s.index if d < mid]].sum()
    h2 = s[[d for d in s.index if d >= mid]].sum()
    print(f"  {label:<28}{len(taken):>7}{f['pnl']:>+10,.0f}{f['sharpe']:>8.2f}"
          f"{f['maxdd']:>9,.0f}   {h1:>+9,.0f}{h2:>+9,.0f}{avgtr:>+8.1f}")


def main() -> int:
    blob = pickle.load(open(ROOT / "backtest" / f".pit_trailtrades_{WINDOW}d.pkl", "rb"))
    members_blob = pickle.load(open(ROOT / "backtest" / f".pit_members_{WINDOW}d.pkl", "rb"))
    members = members_blob["members"]
    daily = pickle.load(open(ROOT / "backtest" / f".pit_daily_{WINDOW}d.pkl", "rb"))
    closes_all = daily["close"]

    # month -> eligible set (keyed by calendar (year, month))
    month_map = {(pd.Timestamp(k).year, pd.Timestamp(k).month): set(v)
                 for k, v in members.items()}

    # all trailing trades in the pool, restricted to PIT-ranked months
    all_tr = [t for syms in blob.values() for t in syms
              if (_tday(t).year, _tday(t).month) in month_map]
    days = sorted({_tday(t) for t in all_tr})
    mid = days[len(days) // 2]
    present = sorted({t.symbol for t in all_tr})

    # trend filter + vol-dial from the PIT daily closes (subset for speed), need SPY
    need = sorted(set(present) | set(UNIVERSE) | {"SPY"})
    closes = closes_all[[c for c in need if c in closes_all.columns]]
    elig = trend_eligibility(closes, present, days)
    prior = prior_vol_flags(closes, days)
    days_mult = {d: (0.5 if prior.get(d) else 1.0) for d in days}

    # Instrument-class poisons a sensible trader excludes A PRIORI (no hindsight):
    # leveraged/inverse ETFs + crypto proxies. ORB whipsaws on these; the mechanical
    # top-100 grabs them but a human (the HAND list) never would. Excluding them
    # disentangles "curation = instrument-selection skill" from "curation = survivorship".
    BLOCK = {"TQQQ", "SQQQ", "SOXL", "SOXS", "TZA", "TNA", "SPXL", "SPXS", "UPRO",
             "UVXY", "SVXY", "TMF", "TMV", "YINN", "FNGU", "BOIL", "UCO",
             "MSTR", "IBIT", "ETHA", "BITO", "BMNR", "CRCL", "CRWV", "MARA", "RIOT"}

    def hand_keep(t):
        return t.symbol in set(UNIVERSE)

    def pit_keep(t):
        return t.symbol in month_map[(_tday(t).year, _tday(t).month)]

    hand = apply_filter([t for t in all_tr if hand_keep(t)], elig)
    pit = apply_filter([t for t in all_tr if pit_keep(t)], elig)
    pit_clean = [t for t in pit if t.symbol not in BLOCK]
    hand_tight = [t for t in hand if or_pct(t) <= OR_THR]
    pit_tight = [t for t in pit if or_pct(t) <= OR_THR]
    pit_clean_tight = [t for t in pit_clean if or_pct(t) <= OR_THR]
    pit_all = pit  # PIT, no tight-OR cut (baseline)
    n_block = len({t.symbol for t in pit if t.symbol in BLOCK})

    base_cents = TARGET_MEDIAN_R * statistics.median(risk_ps(t) for t in hand_tight) / 2.0

    print(f"\n{'='*80}")
    print(f"PIT UNIVERSE TEST — {len(days)} sessions ({days[0]} -> {days[-1]}), OOS {mid}")
    print(f"pool {len(present)} names | HAND {len(hand)} trades, PIT {len(pit)} trades "
          f"(tight-OR: HAND {len(hand_tight)}, PIT {len(pit_tight)})")
    print(f"{'='*80}")
    for sm in SLIP_MULT:
        cents = base_cents * sm
        print(f"\n  --- slip {sm:.1f}x (${cents:.3f}/share) ---")
        print("  " + HEAD); print("  " + "-" * len(HEAD))
        row("HAND  tight-OR<=0.5%", hand_tight, days, mid, days_mult, cents)
        row("PIT   tight-OR<=0.5%", pit_tight, days, mid, days_mult, cents)
        row("PIT ex-lev/crypto", pit_clean_tight, days, mid, days_mult, cents)
        row("PIT   no filter (base)", pit_all, days, mid, days_mult, cents)

    print("\nVERDICT: PIT tight-OR ~matching HAND (positive, comparable Sharpe, small maxDD)")
    print("=> the edge is NOT mainly survivorship; it survives a mechanical universe.")
    print("PIT cratering vs HAND => the curation was load-bearing; discount all $ projections.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
