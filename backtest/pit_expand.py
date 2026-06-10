"""Universe EXPANSION dig (Fable, 2026-06-10): add mechanical HIGH-VOL liquid names to HAND.

The PIT vol split showed the mechanical top-100's HIGH-vol half BEATS the hand list
(Sharpe 1.75 vs 1.57, similar n). pit_volfloor tested SUBTRACTING low-vol names from HAND
(no win). This tests the unexplored direction: ADDING the point-in-time high-vol names the
hand list misses (CVNA/DASH/APP/VRT/...) -> more tight-OR trades/day at flat-or-better
Sharpe = more dollars with NO leverage.

Arms (same 442 PIT-ranked sessions, same trend filter / tight-OR<=0.5% / cap-16 / $50 risk /
$10k notional / vol-dial / cents slippage calibrated on HAND):
  HAND                : the live watchlist (baseline; reproduces the known anchor).
  HAND + hivol>=F%    : hand trades PLUS trades on names that are (a) PIT top-100 by dollar
                        volume THAT month (lookahead-free), (b) as-of 20d realized vol >= F
                        (lookahead-free), (c) not in the a-priori junk BLOCK (lev/inverse
                        ETFs + crypto proxies). F swept 1.0/1.2/1.4 — demand a SMOOTH response.

PRE-REGISTERED GATE (set before running): an expansion arm passes only if PnL >= +15% vs
HAND AND Sharpe >= HAND-0.15 AND maxDD <= 1.5x HAND, at BOTH slip levels, h2 not degraded.

Run:  .venv/Scripts/python.exe backtest/pit_expand.py
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
VOL_FLOORS = [1.0, 1.2, 1.4]      # % — as-of 20d realized vol for ADDED names
BLOCK = {"TQQQ", "SQQQ", "SOXL", "SOXS", "TZA", "TNA", "SPXL", "SPXS", "UPRO",
         "UVXY", "SVXY", "TMF", "TMV", "YINN", "FNGU", "BOIL", "UCO",
         "MSTR", "IBIT", "ETHA", "BITO", "BMNR", "CRCL", "CRWV", "MARA", "RIOT"}
# ETFs in the candidate union (for the single-names-only arm — poll-capacity trim;
# ETFs were the dead-weight class in the HAND vol analysis, single names carry the
# earnings/catalyst vol the breakout edge feeds on)
ETFS = {"EEM", "EFA", "EWY", "EWZ", "FXI", "GDX", "GLD", "IEFA", "IVV", "KRE", "KWEB",
        "RSP", "SLV", "SMH", "SOXX", "XBI", "XLE", "XLF", "XLI", "XLK", "XLU", "XLV",
        "AGG", "TLT", "IEF", "LQD", "HYG", "JNK", "EMB", "VCIT", "VCLT", "USHY",
        "SPY", "QQQ", "IWM", "DIA", "VOO", "IGV"}


def dollar_series(taken, days, mult, cents):
    by = {}
    for t in taken:
        rps = risk_ps(t)
        sh = min(math.floor(RISK * mult.get(_tday(t), 1.0) / rps), math.floor(NOTIONAL_CAP / t.entry_price))
        if sh <= 0:
            continue
        by[_tday(t)] = by.get(_tday(t), 0.0) + (t.exit_price - t.entry_price) * sh - 2 * cents * sh
    return pd.Series({d: by.get(d, 0.0) for d in sorted(days)})


HEAD = (f"{'arm':<22}{'trades':>7}{'tr/day':>7}{'PnL$':>10}{'Sharpe':>8}{'maxDD$':>9}"
        f"   {'h1 PnL':>9}{'h2 PnL':>9}")


def row(label, arm, days, mid, mult, cents):
    taken = portfolio(arm, CAP)
    s = dollar_series(taken, days, mult, cents)
    f = perf(s)
    h1 = s[[d for d in s.index if d < mid]].sum()
    h2 = s[[d for d in s.index if d >= mid]].sum()
    print(f"  {label:<22}{len(taken):>7}{len(taken)/len(days):>7.1f}{f['pnl']:>+10,.0f}"
          f"{f['sharpe']:>8.2f}{f['maxdd']:>9,.0f}   {h1:>+9,.0f}{h2:>+9,.0f}")
    return f, h2


def main() -> int:
    blob = pickle.load(open(ROOT / "backtest" / f".pit_trailtrades_{WINDOW}d.pkl", "rb"))
    members = pickle.load(open(ROOT / "backtest" / f".pit_members_{WINDOW}d.pkl", "rb"))["members"]
    daily = pickle.load(open(ROOT / "backtest" / f".pit_daily_{WINDOW}d.pkl", "rb"))
    closes_all = daily["close"]
    month_map = {(pd.Timestamp(k).year, pd.Timestamp(k).month): set(v) for k, v in members.items()}

    all_tr = [t for syms in blob.values() for t in syms
              if (_tday(t).year, _tday(t).month) in month_map]
    days = sorted({_tday(t) for t in all_tr})
    mid = days[len(days) // 2]
    present = sorted({t.symbol for t in all_tr})
    closes = closes_all[[c for c in sorted(set(present) | {"SPY"}) if c in closes_all.columns]]
    elig = trend_eligibility(closes, present, days)
    prior = prior_vol_flags(closes, days)
    mult = {d: (0.5 if prior.get(d) else 1.0) for d in days}

    rv = closes.pct_change().rolling(20).std() * 100
    rv.index = [d.date() for d in rv.index]
    rv_idx = list(rv.index)

    volcache: dict = {}

    def vol_asof(sym, d):
        key = (sym, d)
        if key in volcache:
            return volcache[key]
        v = None
        if sym in rv.columns:
            import bisect
            i = bisect.bisect_left(rv_idx, d)
            if i > 0:
                x = rv[sym].iloc[i - 1]
                v = None if pd.isna(x) else float(x)
        volcache[key] = v
        return v

    hand_set = set(UNIVERSE)
    pool = [t for t in apply_filter(all_tr, elig) if or_pct(t) <= OR_THR]
    hand = [t for t in pool if t.symbol in hand_set]
    base_cents = TARGET_MEDIAN_R * statistics.median(risk_ps(t) for t in hand) / 2.0

    def expansion(floor, no_etf=False):
        out = list(hand)
        added_syms = set()
        for t in pool:
            if t.symbol in hand_set or t.symbol in BLOCK:
                continue
            if no_etf and t.symbol in ETFS:
                continue
            d = _tday(t)
            if t.symbol not in month_map[(d.year, d.month)]:
                continue
            v = vol_asof(t.symbol, d)
            if v is not None and v >= floor:
                out.append(t)
                added_syms.add(t.symbol)
        return out, added_syms

    print(f"\n{'='*84}\nUNIVERSE EXPANSION — HAND + mechanical PIT high-vol adds | {len(days)} sessions, OOS {mid}")
    print(f"GATE (pre-registered): PnL>=+15% vs HAND, Sharpe>=HAND-0.15, maxDD<=1.5x, both slips, h2 ok")
    print(f"{'='*84}")
    arms = [("HAND (live)", hand, set())]
    for fl in VOL_FLOORS:
        exp, added = expansion(fl)
        arms.append((f"+hivol >= {fl:.1f}%", exp, added))
        print(f"  +hivol>={fl:.1f}%: adds {len(added)} names, {len(exp)-len(hand)} trades "
              f"(e.g. {', '.join(sorted(added)[:10])})")
    exp_ne, added_ne = expansion(1.4, no_etf=True)
    arms.append(("+hivol1.4 noETF", exp_ne, added_ne))
    print(f"  +hivol1.4 noETF: adds {len(added_ne)} single names, {len(exp_ne)-len(hand)} trades")

    results = {}
    for sm in SLIP_MULT:
        cents = base_cents * sm
        print(f"\n  --- slip {sm:.1f}x (${cents:.3f}/share) ---")
        print("  " + HEAD); print("  " + "-" * len(HEAD))
        for label, arm, _ in arms:
            f, h2 = row(label, arm, days, mid, mult, cents)
            results[(label, sm)] = (f, h2)

    # gate evaluation
    print(f"\n  --- GATE ---")
    for label in [f"+hivol >= {fl:.1f}%" for fl in VOL_FLOORS] + ["+hivol1.4 noETF"]:
        ok = True
        for sm in SLIP_MULT:
            fb, h2b = results[("HAND (live)", sm)]
            fe, h2e = results[(label, sm)]
            ok &= (fe["pnl"] >= 1.15 * fb["pnl"] and fe["sharpe"] >= fb["sharpe"] - 0.15
                   and abs(fe["maxdd"]) <= 1.5 * abs(fb["maxdd"]) and h2e >= 0.8 * h2b)
        print(f"    {label}: {'PASS' if ok else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
