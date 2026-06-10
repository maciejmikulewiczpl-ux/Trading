"""Does a REALIZED-VOLATILITY FLOOR improve the live (HAND) tight-OR config?

The PIT vol split showed the tight-OR edge lives in HIGH-vol names; low-vol names form
tight ORs that don't expand and bleed. This tests the actionable follow-up ON THE LIVE
WATCHLIST: filter HAND tight-OR trades by the symbol's trailing-20d realized vol (as-of,
lookahead-free) and see whether a floor lifts Sharpe / PnL / drawdown vs trading the whole
list — and WHICH hand names are the low-vol dead weight.

HONEST on overfitting: choosing a threshold on the same window is in-sample. So (a) sweep
several floors and demand a SMOOTH, monotone-ish response (not a single spike), (b) check
BOTH OOS halves, (c) prefer a round threshold. The PIT split already gave independent
evidence (high>low on a DIFFERENT universe), so this is confirmatory, not a blind search.

Cheap: cached trailing trades + cached daily closes.
Run:
    .venv/Scripts/python.exe backtest/pit_volfloor.py
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
FLOORS = [None, 0.8, 1.0, 1.2, 1.4]    # realized-vol floor in % (None = no floor)


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


HEAD = (f"{'config':<22}{'trades':>7}{'PnL$':>10}{'Sharpe':>8}{'maxDD$':>9}"
        f"   {'h1 PnL':>9}{'h2 PnL':>9}{'avg$/tr':>8}")


def row(label, arm, days, mid, days_mult, cents):
    taken = portfolio(arm, CAP)
    s, avgtr = dollar_series(taken, days, days_mult, cents)
    f = perf(s)
    h1 = s[[d for d in s.index if d < mid]].sum()
    h2 = s[[d for d in s.index if d >= mid]].sum()
    print(f"  {label:<22}{len(taken):>7}{f['pnl']:>+10,.0f}{f['sharpe']:>8.2f}"
          f"{f['maxdd']:>9,.0f}   {h1:>+9,.0f}{h2:>+9,.0f}{avgtr:>+8.1f}")


def main() -> int:
    blob = pickle.load(open(ROOT / "backtest" / f".pit_trailtrades_{WINDOW}d.pkl", "rb"))
    daily = pickle.load(open(ROOT / "backtest" / f".pit_daily_{WINDOW}d.pkl", "rb"))
    closes_all = daily["close"]

    all_tr = [t for syms in blob.values() for t in syms if t.symbol in set(UNIVERSE)]
    days = sorted({_tday(t) for t in all_tr})
    mid = days[len(days) // 2]
    present = sorted({t.symbol for t in all_tr})

    need = sorted(set(present) | {"SPY"})
    closes = closes_all[[c for c in need if c in closes_all.columns]]
    elig = trend_eligibility(closes, present, days)
    prior = prior_vol_flags(closes, days)
    days_mult = {d: (0.5 if prior.get(d) else 1.0) for d in days}

    rv = closes.pct_change().rolling(20).std() * 100   # percent
    rv.index = [d.date() for d in rv.index]

    def vol_asof(sym, d):
        if sym not in rv.columns:
            return None
        prior_idx = [x for x in rv.index if x < d]
        if not prior_idx:
            return None
        v = rv[sym].loc[prior_idx[-1]]
        return None if pd.isna(v) else float(v)

    hand = [t for t in apply_filter(all_tr, elig) if or_pct(t) <= OR_THR]
    vol = {id(t): vol_asof(t.symbol, _tday(t)) for t in hand}
    hand = [t for t in hand if vol[id(t)] is not None]   # need a vol to floor on

    base_cents = TARGET_MEDIAN_R * statistics.median(risk_ps(t) for t in hand) / 2.0

    print(f"\n{'='*80}\nREALIZED-VOL FLOOR on the LIVE (HAND) tight-OR config — {len(days)} sessions, OOS {mid}")
    print(f"{'='*80}")
    for sm in SLIP_MULT:
        cents = base_cents * sm
        print(f"\n  --- slip {sm:.1f}x (${cents:.3f}/share) ---")
        print("  " + HEAD); print("  " + "-" * len(HEAD))
        for fl in FLOORS:
            arm = hand if fl is None else [t for t in hand if vol[id(t)] >= fl]
            label = "no floor (live)" if fl is None else f"vol >= {fl:.1f}%"
            row(label, arm, days, mid, days_mult, cents)

    # which hand names are the low-vol dead weight: per-name median as-of vol + contribution
    by_sym = {}
    for t in hand:
        by_sym.setdefault(t.symbol, []).append((vol[id(t)], t.pnl_r))
    rows = []
    for sym, lst in by_sym.items():
        rows.append((sym, statistics.median(v for v, _ in lst), len(lst), sum(r for _, r in lst)))
    rows.sort(key=lambda r: r[1])   # lowest vol first
    print(f"\n  Lowest-vol HAND names (median as-of 20d vol, n tight-OR trades, sumR):")
    print(f"  {'sym':<7}{'vol%':>7}{'n':>6}{'sumR':>9}")
    for sym, v, n, sr in rows[:18]:
        print(f"  {sym:<7}{v:>7.2f}{n:>6}{sr:>+9.1f}")
    print("\nRead: a SMOOTH lift in Sharpe/PnL + smaller maxDD as the floor rises, holding in BOTH")
    print("halves, = a real, shippable universe filter (drop the low-vol dead weight). The per-name")
    print("list shows exactly which names a floor removes (expect the 4 ETFs + staples/utilities).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
