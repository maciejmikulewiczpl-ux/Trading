"""Sizing tilts dig (Fable, 2026-06-10): size UP on validated favourable strata.

Two strata showed materially higher avg_R on the tight-OR trailing set:
  FOMC days   : +0.78 vs +0.27 baseline (compare_event_days, n=368) — and SCHEDULED.
  gap-up >0.5%: +0.45 vs +0.27 (compare_daily_context, n=986) — known at the bell.
Filters were rejected (they'd cut too many trades); this tests the right tool: a RISK
MULTIPLIER on those trades only, leaving everything else unchanged.

PRE-REGISTERED GATE: a tilt passes only if PnL up AND Sharpe >= baseline (tilts add
variance — they must pay for it) AND maxDD <= 1.3x baseline, at both slip levels, with
BOTH OOS halves not degraded. NOTE the $10k notional cap may damp tilts (it binds on
tight-OR names) — reported honestly via the 'binding%' column.

Run:  .venv/Scripts/python.exe backtest/compare_sizing_tilts.py
"""
from __future__ import annotations

import math
import pickle
import statistics
import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.universe_portfolio import portfolio  # noqa: E402
from backtest.compare_selection import _tday  # noqa: E402
from backtest.compare_norefill_trend import trend_eligibility, apply_filter  # noqa: E402
from backtest.compare_volpause import prior_vol_flags, perf, RISK, CAP  # noqa: E402
from backtest.compare_or_range_realcost import or_pct, risk_ps  # noqa: E402
from backtest.compare_event_days import FOMC  # noqa: E402
from backtest.universe_scan import UNIVERSE  # noqa: E402

import pandas as pd  # noqa: E402

WINDOW = 730
OR_THR = 0.5
TARGET_MEDIAN_R = 0.042
NOTIONAL_CAP = 10_000.0
SLIP_MULT = [1.0, 1.5]
GAP_THR = 0.5     # % gap-up for the per-name tilt


def main() -> int:
    blob = pickle.load(open(ROOT / "backtest" / f".pit_trailtrades_{WINDOW}d.pkl", "rb"))
    daily = pickle.load(open(ROOT / "backtest" / f".pit_daily_{WINDOW}d.pkl", "rb"))
    closes_all = daily["close"]
    ohlc = pickle.load(open(ROOT / "backtest" / f".pit_dailyohlc_{WINDOW}d.pkl", "rb"))

    all_tr = [t for syms in blob.values() for t in syms if t.symbol in set(UNIVERSE)]
    days = sorted({_tday(t) for t in all_tr})
    mid = days[len(days) // 2]
    present = sorted({t.symbol for t in all_tr})
    closes = closes_all[[c for c in sorted(set(present) | {"SPY"}) if c in closes_all.columns]]
    elig = trend_eligibility(closes, present, days)
    prior = prior_vol_flags(closes, days)
    vmult = {d: (0.5 if prior.get(d) else 1.0) for d in days}

    trades = [t for t in apply_filter(all_tr, elig) if or_pct(t) <= OR_THR]
    taken = portfolio(trades, CAP)
    cents0 = TARGET_MEDIAN_R * statistics.median(risk_ps(t) for t in trades) / 2.0

    # tags (lookahead-free): FOMC day (or morning after); gap-up at the bell
    def is_fomc(d):
        return d in FOMC or (d - timedelta(days=1)) in FOMC

    gapcache: dict = {}

    def gap_pct(sym, d):
        key = (sym, d)
        if key in gapcache:
            return gapcache[key]
        g = None
        sb = ohlc.get(sym)
        if sb is not None and d in sb.index:
            idx = list(sb.index)
            i = idx.index(d)
            if i > 0:
                pc = float(sb["close"].iloc[i - 1])
                g = (float(sb["open"].iloc[i]) / pc - 1.0) * 100 if pc else None
        gapcache[key] = g
        return g

    def series(tilt_fomc=1.0, tilt_gap=1.0, cents=cents0):
        by, n_bind, n_tilted = {}, 0, 0
        for t in taken:
            d = _tday(t)
            tilt = 1.0
            if tilt_fomc != 1.0 and is_fomc(d):
                tilt *= tilt_fomc
            if tilt_gap != 1.0:
                g = gap_pct(t.symbol, d)
                if g is not None and g > GAP_THR:
                    tilt *= tilt_gap
            if tilt != 1.0:
                n_tilted += 1
            rps = risk_ps(t)
            want = math.floor(tilt * RISK * vmult.get(d, 1.0) / rps)
            cap_sh = math.floor(NOTIONAL_CAP / t.entry_price)
            sh = min(want, cap_sh)
            if tilt > 1.0 and want > cap_sh:
                n_bind += 1
            if sh <= 0:
                continue
            by[d] = by.get(d, 0.0) + (t.exit_price - t.entry_price) * sh - 2 * cents * sh
        s = pd.Series({d: by.get(d, 0.0) for d in sorted(days)})
        return s, n_tilted, n_bind

    HEAD = (f"{'arm':<26}{'tilted':>7}{'bind%':>6}{'PnL$':>10}{'Sharpe':>8}{'maxDD$':>9}"
            f"   {'h1 PnL':>9}{'h2 PnL':>9}")
    ARMS = [
        ("baseline (live)", dict()),
        ("FOMC x1.5", dict(tilt_fomc=1.5)),
        ("FOMC x2.0", dict(tilt_fomc=2.0)),
        (f"gap-up>{GAP_THR}% x1.5", dict(tilt_gap=1.5)),
        ("both x1.5", dict(tilt_fomc=1.5, tilt_gap=1.5)),
    ]
    print(f"\n{'='*86}\nSIZING TILTS — tight-OR trailing, HAND universe | {len(days)} sessions, OOS {mid}")
    print(f"GATE: PnL up AND Sharpe >= baseline AND maxDD <= 1.3x, both slips, both halves not degraded")
    print(f"{'='*86}")
    results = {}
    for sm in SLIP_MULT:
        cents = cents0 * sm
        print(f"\n  --- slip {sm:.1f}x (${cents:.3f}/share) ---")
        print("  " + HEAD); print("  " + "-" * len(HEAD))
        for label, kw in ARMS:
            s, n_t, n_b = series(cents=cents, **kw)
            f = perf(s)
            h1 = s[[d for d in s.index if d < mid]].sum()
            h2 = s[[d for d in s.index if d >= mid]].sum()
            bindpct = (100 * n_b / n_t) if n_t else 0
            print(f"  {label:<26}{n_t:>7}{bindpct:>5.0f}%{f['pnl']:>+10,.0f}{f['sharpe']:>8.2f}"
                  f"{f['maxdd']:>9,.0f}   {h1:>+9,.0f}{h2:>+9,.0f}")
            results[(label, sm)] = (f, h1, h2)

    print(f"\n  --- GATE ---")
    for label, _ in ARMS[1:]:
        ok = True
        for sm in SLIP_MULT:
            fb, h1b, h2b = results[("baseline (live)", sm)]
            fe, h1e, h2e = results[(label, sm)]
            ok &= (fe["pnl"] > fb["pnl"] and fe["sharpe"] >= fb["sharpe"]
                   and abs(fe["maxdd"]) <= 1.3 * abs(fb["maxdd"])
                   and h1e >= h1b * 0.95 and h2e >= h2b * 0.95)
        print(f"    {label}: {'PASS' if ok else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
