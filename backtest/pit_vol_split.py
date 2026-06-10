"""PIT survivorship test, decisive refinement: is the surviving edge just LOW VOLATILITY?

The PIT test showed tight-OR halves on a mechanical top-100 universe vs the curated
HAND list, and excluding leveraged/crypto junk did NOT close the gap. Remaining question:
is the curation's value a hindsight-free, reproducible principle ("trade STABLE names")
or genuine name-selection survivorship?

Decisive split: rank each PIT tight-OR trade by its symbol's TRAILING 20d realized vol
(known before the session — no lookahead), split at the median, and compare low-vol vs
high-vol PIT against HAND. If LOW-VOL PIT recovers toward HAND, the edge is reproducible
with a mechanical volatility filter (not survivorship). If even low-vol PIT lags HAND,
name-selection survivorship dominates -> discount the $ projections hard.

Cheap: cached trailing trades + cached daily closes.
Run:
    .venv/Scripts/python.exe backtest/pit_vol_split.py
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


HEAD = (f"{'set':<24}{'trades':>7}{'PnL$':>10}{'Sharpe':>8}{'maxDD$':>9}"
        f"   {'h1 PnL':>9}{'h2 PnL':>9}{'avg$/tr':>8}")


def row(label, arm, days, mid, days_mult, cents):
    taken = portfolio(arm, CAP)
    s, avgtr = dollar_series(taken, days, days_mult, cents)
    f = perf(s)
    h1 = s[[d for d in s.index if d < mid]].sum()
    h2 = s[[d for d in s.index if d >= mid]].sum()
    print(f"  {label:<24}{len(taken):>7}{f['pnl']:>+10,.0f}{f['sharpe']:>8.2f}"
          f"{f['maxdd']:>9,.0f}   {h1:>+9,.0f}{h2:>+9,.0f}{avgtr:>+8.1f}")


def main() -> int:
    blob = pickle.load(open(ROOT / "backtest" / f".pit_trailtrades_{WINDOW}d.pkl", "rb"))
    members_blob = pickle.load(open(ROOT / "backtest" / f".pit_members_{WINDOW}d.pkl", "rb"))
    members = members_blob["members"]
    daily = pickle.load(open(ROOT / "backtest" / f".pit_daily_{WINDOW}d.pkl", "rb"))
    closes_all = daily["close"]

    month_map = {(pd.Timestamp(k).year, pd.Timestamp(k).month): set(v) for k, v in members.items()}
    all_tr = [t for syms in blob.values() for t in syms
              if (_tday(t).year, _tday(t).month) in month_map]
    days = sorted({_tday(t) for t in all_tr})
    mid = days[len(days) // 2]
    present = sorted({t.symbol for t in all_tr})

    need = sorted(set(present) | set(UNIVERSE) | {"SPY"})
    closes = closes_all[[c for c in need if c in closes_all.columns]]
    elig = trend_eligibility(closes, present, days)
    prior = prior_vol_flags(closes, days)
    days_mult = {d: (0.5 if prior.get(d) else 1.0) for d in days}

    # trailing 20d realized vol per symbol, as-of the trade date (no lookahead)
    rv = closes.pct_change().rolling(20).std()
    rv.index = [d.date() for d in rv.index]

    def vol_asof(sym, d):
        if sym not in rv.columns:
            return None
        ser = rv[sym]
        prior_idx = [x for x in ser.index if x < d]
        if not prior_idx:
            return None
        v = ser.loc[prior_idx[-1]]
        return None if pd.isna(v) else float(v)

    hand = [t for t in apply_filter([t for t in all_tr if t.symbol in set(UNIVERSE)], elig)
            if or_pct(t) <= OR_THR]
    pit = [t for t in apply_filter(
        [t for t in all_tr if t.symbol in month_map[(_tday(t).year, _tday(t).month)]], elig)
        if or_pct(t) <= OR_THR]

    pit_v = [(t, vol_asof(t.symbol, _tday(t))) for t in pit]
    pit_v = [(t, v) for t, v in pit_v if v is not None]
    med = statistics.median(v for _, v in pit_v)
    pit_lo = [t for t, v in pit_v if v <= med]
    pit_hi = [t for t, v in pit_v if v > med]
    # also: HAND's own median realized vol, for context on how stable the curated names are
    hand_vols = [v for v in (vol_asof(t.symbol, _tday(t)) for t in hand) if v is not None]

    base_cents = TARGET_MEDIAN_R * statistics.median(risk_ps(t) for t in hand) / 2.0

    print(f"\n{'='*80}")
    print(f"PIT VOLATILITY SPLIT — {len(days)} sessions, OOS {mid}")
    print(f"PIT median 20d realized vol {med*100:.2f}% | HAND median {statistics.median(hand_vols)*100:.2f}% "
          f"(curated names ARE more stable: {'yes' if statistics.median(hand_vols) < med else 'no'})")
    print(f"{'='*80}")
    for sm in SLIP_MULT:
        cents = base_cents * sm
        print(f"\n  --- slip {sm:.1f}x (${cents:.3f}/share) ---")
        print("  " + HEAD); print("  " + "-" * len(HEAD))
        row("HAND tight-OR", hand, days, mid, days_mult, cents)
        row("PIT low-vol half", pit_lo, days, mid, days_mult, cents)
        row("PIT high-vol half", pit_hi, days, mid, days_mult, cents)
    print("\nRead: if PIT LOW-VOL ~recovers toward HAND, the edge is reproducible with a mechanical")
    print("volatility filter (NOT survivorship). If low-vol PIT still lags HAND badly, name-selection")
    print("survivorship dominates -> discount the headline $ and rethink the scale-up.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
