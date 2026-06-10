"""Shorts revisit (Fable spec, 2026-06-10): tight-OR + trailing SHORTS under the regime gate.

Shorts were validated only under the OLD fixed-2R bracket, then staged OFF. This tests them
under the CURRENT edge stack (tight-OR <=0.5% + trailing 1R) — the hedge for the bull-only,
front-loaded long edge. PRE-REGISTERED GATE (memory shorts_revisit_spec; fixed before run):
  (1) short sleeve net-positive in 2022 bear, standalone Sharpe > 0.5
  (2) short sleeve PnL >= -$500 over 2024-26 (730d) bull (regime gate should keep it ~idle)
  (3) adding shorts IMPROVES the combined 2022 drawdown vs long-only
All three or shorts stay OFF. One shot; no tuning after results.

Mirror sim (lookahead-free): OR 9:30-9:45; first close beyond OR-low(short)/OR-high(long) in
9:45-11:30 -> enter next-bar open; tight-OR (OR range/entry <= 0.5%); trailing 1R (short: stop
trails DOWN to lwm+1R; long: up to hwm-1R); same-bar stop-first; EOD 15:55 flat. Regime gate:
SPY close < 20d SMA for 3 consecutive prior sessions (lookahead-free). Shorts EXEMPT from the
vol-dial (they want turbulent days). Costs: cents (median trade 0.042R) x1.5 short surcharge +
borrow 1.5%/yr prorated; slip sweep 1.0x/1.5x. $50 risk, $10k cap, cap-16 pool.

Memory: 2022 needs all 100 names (long baseline); 730d needs only the 4 short names. Processed
sequentially with del+gc (8GB box).

Run:  .venv/Scripts/python.exe backtest/compare_shorts.py
"""
from __future__ import annotations

import bisect
import gc
import math
import pickle
import statistics
import sys
from datetime import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.universe_portfolio import portfolio  # noqa: E402
from backtest.compare_selection import _tday  # noqa: E402
from backtest.compare_norefill_trend import trend_eligibility, apply_filter  # noqa: E402
from backtest.compare_volpause import perf  # noqa: E402

OR_S, OR_E, CUT, EOD, RTH_E = time(9, 30), time(9, 45), time(11, 30), time(15, 55), time(16, 0)
SHORT_UNIV = ["SPY", "QQQ", "NVDA", "AAPL"]
RISK, CAP, NOTIONAL = 50.0, 16, 10_000.0
TARGET_MEDIAN_R, TRAIL = 0.042, 1.0
SHORT_SURCHARGE, BORROW_YR = 1.5, 0.015
TIGHT_MAX = 0.5


class T:
    __slots__ = ("symbol", "date", "side", "entry_time", "exit_time",
                 "entry_price", "stop_price", "exit_price", "pnl_r")

    def __init__(self, **k):
        for a, v in k.items():
            setattr(self, a, v)


def build_arrays(bars, names):
    out = {}
    for sym in names:
        if sym not in bars.index.get_level_values(0):
            continue
        sb = bars.xs(sym, level=0)
        t = sb.index.time
        m = (t >= OR_S) & (t < RTH_E)
        sb = sb[m]
        d = {}
        for day, g in sb.groupby(sb.index.date):
            d[day] = (g["open"].to_numpy(float), g["high"].to_numpy(float),
                      g["low"].to_numpy(float), g["close"].to_numpy(float),
                      g.index.time, g.index)
        out[sym] = d
    return out


def sim(side, day_arr):
    o, h, l, c, tt, idx = day_arr
    n = len(c)
    orm = [(OR_S <= x < OR_E) for x in tt]
    if not any(orm):
        return None
    or_high = max(h[i] for i in range(n) if orm[i])
    or_low = min(l[i] for i in range(n) if orm[i])
    bo = None
    for i in range(n):
        if tt[i] < OR_E:
            continue
        if tt[i] > CUT:
            break
        if (side == "short" and c[i] < or_low) or (side == "long" and c[i] > or_high):
            bo = i
            break
    if bo is None or bo + 1 >= n:
        return None
    entry = o[bo + 1]
    rng = or_high - or_low
    if entry <= 0 or rng / entry * 100 > TIGHT_MAX:
        return None
    stop = or_high if side == "short" else or_low
    risk = (stop - entry) if side == "short" else (entry - stop)
    if risk <= 0:
        return None
    ext = None
    if side == "short":
        lwm, cur = entry, stop
        for j in range(bo + 1, n):
            if tt[j] >= EOD:
                ext = (c[j], j); break
            s2 = lwm + TRAIL * risk
            if s2 < cur:
                cur = s2
            if h[j] >= cur:
                ext = (cur, j); break
            if l[j] < lwm:
                lwm = l[j]
    else:
        hwm, cur = entry, stop
        for j in range(bo + 1, n):
            if tt[j] >= EOD:
                ext = (c[j], j); break
            s2 = hwm - TRAIL * risk
            if s2 > cur:
                cur = s2
            if l[j] <= cur:
                ext = (cur, j); break
            if h[j] > hwm:
                hwm = h[j]
    if ext is None:
        ext = (c[-1], n - 1)
    exitp, ei = ext
    pnl_r = ((entry - exitp) if side == "short" else (exitp - entry)) / risk
    return T(symbol=None, date=None, side=side, entry_time=idx[bo + 1], exit_time=idx[ei],
             entry_price=float(entry), stop_price=float(stop), exit_price=float(exitp),
             pnl_r=float(pnl_r))


def gen(arrays, names, side, days_set):
    out = []
    for sym in names:
        for day, da in arrays.get(sym, {}).items():
            if day not in days_set:
                continue
            tr = sim(side, da)
            if tr is not None:
                tr.symbol = sym
                tr.date = pd.Timestamp(day)
                out.append(tr)
    return out


def regime_on(closes, days):
    spy = closes["SPY"].dropna().sort_index()
    below = (spy < spy.rolling(20).mean()).to_numpy()
    cdates = [pd.Timestamp(x).date() for x in spy.index]
    out = {}
    for D in days:
        p = bisect.bisect_left(cdates, D)
        out[D] = bool(p >= 3 and below[p - 1] and below[p - 2] and below[p - 3])
    return out


def risk_ps(t):
    return max(abs(t.entry_price - t.stop_price), 1e-6)


def daily_dollars(taken, days, cents_by_side):
    by = {}
    for t in taken:
        rps = risk_ps(t)
        sh = min(math.floor(RISK / rps), math.floor(NOTIONAL / t.entry_price))
        if sh <= 0:
            continue
        cents = cents_by_side[t.side]
        if t.side == "short":
            gross = (t.entry_price - t.exit_price) * sh
            borrow = sh * t.entry_price * (BORROW_YR / 252)
        else:
            gross = (t.exit_price - t.entry_price) * sh
            borrow = 0.0
        by[_tday(t)] = by.get(_tday(t), 0.0) + gross - 2 * cents * sh - borrow
    return pd.Series({d: by.get(d, 0.0) for d in sorted(days)})


def cents_for(trades, surcharge=1.0):
    if not trades:
        return 0.0
    return TARGET_MEDIAN_R * statistics.median(risk_ps(t) for t in trades) / 2.0 * surcharge


def run_2022():
    blob = pickle.load(open(ROOT / "backtest" / ".bars_cache_univ_2022.pkl", "rb"))
    bars, days, closes = blob["bars"], blob["days"], blob["closes"]
    present = sorted(bars.index.get_level_values(0).unique())
    arrays = build_arrays(bars, present)
    del bars, blob
    gc.collect()
    days_set = set(days)
    elig = trend_eligibility(closes, present, days)
    reg = regime_on(closes, days)
    n_reg = sum(reg.values())

    longs = [t for t in apply_filter(gen(arrays, present, "long", days_set), elig)]
    shorts_all = gen(arrays, SHORT_UNIV, "short", days_set)
    shorts = [t for t in shorts_all if reg.get(_tday(t))]   # regime-gated
    del arrays
    gc.collect()

    lc = {"long": 0.0, "short": 0.0}
    return dict(days=days, longs=longs, shorts=shorts, n_reg=n_reg, tot=len(days),
                lc=lc, elig=elig)


def run_730():
    blob = pickle.load(open(ROOT / "backtest" / ".bars_cache_univ_730d.pkl", "rb"))
    bars, days = blob["bars"], blob["days"]
    closes = pickle.load(open(ROOT / "backtest" / ".bars_cache_daily_730d.pkl", "rb"))
    arrays = build_arrays(bars, SHORT_UNIV)
    del bars, blob
    gc.collect()
    days_set = set(days)
    reg = regime_on(closes, days)
    shorts = [t for t in gen(arrays, SHORT_UNIV, "short", days_set) if reg.get(_tday(t))]
    del arrays
    gc.collect()
    return dict(days=days, shorts=shorts, n_reg=sum(reg.values()), tot=len(days))


def show(label, series, days):
    f = perf(series)
    print(f"  {label:<26} n_days_active={int((series!=0).sum()):>4}  "
          f"PnL ${f['pnl']:>+8,.0f}  Sharpe {f['sharpe']:>6.2f}  maxDD ${f['maxdd']:>+8,.0f}")
    return f


def main():
    print("=" * 78)
    print("SHORTS REVISIT — tight-OR + trailing, regime-gated | gate pre-registered")
    print("=" * 78)

    # ---- 2022 bear ----
    r22 = run_2022()
    print(f"\n### 2022 BEAR ({r22['tot']} sessions, {r22['n_reg']} regime-ON) ###")
    print(f"  long trades: {len(r22['longs'])}  short trades (regime-gated): {len(r22['shorts'])}")
    for sm in (1.0, 1.5):
        cents = {"long": cents_for(r22["longs"]) * sm,
                 "short": cents_for(r22["shorts"], SHORT_SURCHARGE) * sm}
        print(f"\n  --- slip {sm:.1f}x (long ${cents['long']:.3f} / short ${cents['short']:.3f}) ---")
        long_taken = portfolio(r22["longs"], CAP)
        short_taken = portfolio(r22["shorts"], CAP)
        comb_taken = portfolio(r22["longs"] + r22["shorts"], CAP)
        ls = daily_dollars(long_taken, r22["days"], cents)
        ss = daily_dollars(short_taken, r22["days"], cents)
        cs = daily_dollars(comb_taken, r22["days"], cents)
        fL = show("long-only", ls, r22["days"])
        fS = show("SHORT sleeve (standalone)", ss, r22["days"])
        fC = show("long + short combined", cs, r22["days"])
        if sm == 1.0:
            g1 = (fS["sharpe"] > 0.5 and fS["pnl"] > 0)
            g3 = (abs(fC["maxdd"]) < abs(fL["maxdd"]))
            print(f"    GATE(1) short net-positive 2022 & Sharpe>0.5: {'PASS' if g1 else 'FAIL'} "
                  f"(PnL ${fS['pnl']:+,.0f}, Sharpe {fS['sharpe']:.2f})")
            print(f"    GATE(3) combined maxDD better than long-only: {'PASS' if g3 else 'FAIL'} "
                  f"(${fC['maxdd']:+,.0f} vs ${fL['maxdd']:+,.0f})")
    del r22
    gc.collect()

    # ---- 2024-26 bull (short sleeve only) ----
    r73 = run_730()
    print(f"\n### 2024-26 BULL 730d ({r73['tot']} sessions, {r73['n_reg']} regime-ON) ###")
    print(f"  short trades (regime-gated): {len(r73['shorts'])}")
    cents = {"long": 0.0, "short": cents_for(r73["shorts"], SHORT_SURCHARGE)}
    ss = daily_dollars(portfolio(r73["shorts"], CAP), r73["days"], cents)
    fS73 = show("SHORT sleeve (standalone)", ss, r73["days"])
    g2 = fS73["pnl"] >= -500
    print(f"    GATE(2) short PnL >= -$500 in bull: {'PASS' if g2 else 'FAIL'} (${fS73['pnl']:+,.0f})")

    print("\n" + "=" * 78)
    print("VERDICT: all three gates PASS -> shorts are a real hedge; proceed to live-build")
    print("decision. Any FAIL -> shorts stay OFF (this run does not tune to force a pass).")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
