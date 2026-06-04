"""Exit-management study: can trailing / partial exits beat the fixed 2R target?

The strategy hard-caps every winner at 2R — so trend days that could run 4-8R get
cut, while losers run full distance to the stop. This tests whether *letting
winners run* helps, by holding the ENTRIES fixed (the same breakouts the strategy
already takes) and re-simulating only the EXIT over the cached minute bars. Pure
A/B on exit logic; touches no live code.

Policies (long-only, matching the live config):
  fixed_2R       : baseline — target = entry + 2R, stop = OR low. (reproduces live)
  trail_1R/1.5R/2R: no fixed target; trail the stop N*R below the high-water mark.
  partial_2R     : take half at +1R, move stop to breakeven, runner targets 2R.
  partial_trail1R: take half at +1R, breakeven, runner trails 1R below HWM (no cap).

Reported per policy: avg_R (the pure exit edge, capital-agnostic) plus the capped
portfolio (cap 16, $50/trade — the just-shipped config) Sharpe / drawdown / PnL,
over both windows + OOS halves. Same-bar ties resolve stop-first (conservative);
trailing is lookahead-free (stop uses the HWM through the prior bar).

Run (uses caches from compare_selection.py / compare_norefill_trend.py):
    .venv/Scripts/python.exe backtest/compare_exits.py
"""
from __future__ import annotations

import pickle
import sys
from dataclasses import replace
from datetime import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.universe_portfolio import portfolio  # noqa: E402
from backtest.compare_selection import _tday  # noqa: E402
from backtest.compare_norefill_trend import trend_eligibility, apply_filter  # noqa: E402

WINDOWS = [730, 180]
CAP = 16
RISK = 50.0
EOD = time(15, 55)
RTH_S, RTH_E = time(9, 30), time(16, 0)

# policy -> dict(target_R, trail_R, partial)
POLICIES = {
    "fixed_2R (live)":  dict(target_R=2.0, trail_R=None, partial=False),
    "trail_1R":         dict(target_R=None, trail_R=1.0, partial=False),
    "trail_1.5R":       dict(target_R=None, trail_R=1.5, partial=False),
    "trail_2R":         dict(target_R=None, trail_R=2.0, partial=False),
    "partial_2R":       dict(target_R=2.0, trail_R=None, partial=True),
    "partial_trail1R":  dict(target_R=None, trail_R=1.0, partial=True),
}


def load(w):
    bars = pickle.load(open(ROOT / "backtest" / f".bars_cache_univ_{w}d.pkl", "rb"))
    trades = pickle.load(open(ROOT / "backtest" / f".bars_cache_trades_{w}d.pkl", "rb"))
    closes = pickle.load(open(ROOT / "backtest" / f".bars_cache_daily_{w}d.pkl", "rb"))
    all_bars, days = bars["bars"], bars["days"]
    present = sorted(all_bars.index.get_level_values(0).unique())
    return all_bars, days, present, trades, closes


def bucket(all_bars, present):
    """{symbol: {date: {idx, ns, hi, lo, cl}}} as numpy arrays — fast exit re-sim
    (a plain loop over numpy scalars, ~50x faster than DataFrame.iterrows)."""
    out = {}
    for sym in present:
        sb = all_bars.xs(sym, level=0)
        t = sb.index.time
        sb = sb[(t >= RTH_S) & (t < RTH_E)]
        d = {}
        for day, g in sb.groupby(sb.index.date):
            d[day] = {"idx": g.index, "ns": g.index.asi8,
                      "hi": g["high"].to_numpy(float),
                      "lo": g["low"].to_numpy(float),
                      "cl": g["close"].to_numpy(float)}
        out[sym] = d
    return out


def sim_long_exit(day, start, entry, init_stop, eod_ns, mode):
    """(exit_ts, exit_px, pnl_r) for a long under `mode`, scanning numpy arrays
    from position `start`. pnl_r is blended across halves for partial policies."""
    ns, hi, lo, cl, idx = day["ns"], day["hi"], day["lo"], day["cl"], day["idx"]
    n = len(ns)
    risk = entry - init_stop
    if risk <= 0:
        return None
    trailR = mode["trail_R"]
    target = entry + mode["target_R"] * risk if mode["target_R"] else None
    one_r = entry + risk
    part = mode["partial"]
    hwm, stop, partial = entry, init_stop, False
    for i in range(start, n):
        if ns[i] >= eod_ns:
            r = (cl[i] - entry) / risk
            return idx[i], cl[i], (0.5 + 0.5 * r) if partial else r
        if trailR:                                 # trail from prior-bar HWM (no lookahead)
            s2 = hwm - trailR * risk
            if s2 > stop:
                stop = s2
        if lo[i] <= stop:                          # stop first (conservative)
            r = (stop - entry) / risk
            return idx[i], stop, (0.5 + 0.5 * r) if partial else r
        if part and not partial and hi[i] >= one_r:
            partial = True                          # bank half at +1R, lift runner stop to BE
            if entry > stop:
                stop = entry
        if target is not None and hi[i] >= target:
            r = (target - entry) / risk
            return idx[i], target, (0.5 + 0.5 * r) if partial else r
        if hi[i] > hwm:
            hwm = hi[i]
    r = (cl[n - 1] - entry) / risk
    return idx[n - 1], cl[n - 1], (0.5 + 0.5 * r) if partial else r


def reexit(trades, buckets, mode, eod_ns_by_date):
    """New trade list with the same entries but `mode` exits (R-space, $50/trade)."""
    out = []
    for t in trades:
        if t.side != "long":
            continue
        day = buckets.get(t.symbol, {}).get(_tday(t))
        if day is None:
            continue
        start = int(day["idx"].searchsorted(t.entry_time, side="left"))
        if start >= len(day["ns"]):
            continue
        res = sim_long_exit(day, start, t.entry_price, t.stop_price,
                            eod_ns_by_date[_tday(t)], mode)
        if res is None:
            continue
        ex_ts, ex_px, pr = res
        out.append(replace(t, exit_time=ex_ts, exit_price=float(ex_px),
                           pnl_r=pr, pnl_dollars=pr * RISK))
    return out


def stats(taken, days):
    by = {}
    for t in taken:
        by[_tday(t)] = by.get(_tday(t), 0.0) + t.pnl_r
    s = pd.Series(by).reindex(sorted(days), fill_value=0.0) if by else pd.Series(0.0, index=sorted(days))
    dollar = s * RISK
    eq = dollar.cumsum()
    dd = (eq - eq.cummax()).min() if len(eq) else 0.0
    mu, sd = s.mean(), s.std()
    sharpe = (mu / sd * (252 ** 0.5)) if sd and sd > 0 else float("nan")
    wins = sum(1 for t in taken if t.pnl_r > 0)
    return {"n": len(taken), "avgR": (s.sum() / len(taken) if taken else 0.0),
            "sumR": s.sum(), "pnl": dollar.sum(), "maxdd": dd, "sharpe": sharpe,
            "win": (100 * wins / len(taken) if taken else 0.0)}


def three(taken, days, mid):
    d1 = [d for d in days if d < mid]
    d2 = [d for d in days if d >= mid]
    return (stats(taken, days),
            stats([t for t in taken if _tday(t) < mid], d1),
            stats([t for t in taken if _tday(t) >= mid], d2))


HEAD = (f"{'policy':<18}{'trades':>7}{'avgR':>8}{'win%':>7}{'PnL$':>10}{'Sharpe':>8}{'maxDD$':>10}"
        f"   {'h1 Sh':>6}{'h2 Sh':>6}{'h2 PnL':>8}")


def prow(label, f, h1, h2):
    def c(s, k, fmt):
        return format(s[k], fmt) if s.get("n", 0) else "—"
    print(f"{label:<18}{f['n']:>7}{c(f,'avgR','>+8.3f')}{c(f,'win','>6.1f')}%"
          f"{c(f,'pnl','>+10,.0f')}{c(f,'sharpe','>8.2f')}{c(f,'maxdd','>10,.0f')}   "
          f"{c(h1,'sharpe','>6.2f')}{c(h2,'sharpe','>6.2f')}{c(h2,'pnl','>+8,.0f')}")


def run_window(w):
    all_bars, days, present, trades, closes = load(w)
    mid = sorted(days)[len(days) // 2]
    elig = trend_eligibility(closes, present, days)
    buckets = bucket(all_bars, present)
    tz = all_bars.index.get_level_values(1).tz
    eod_ns_by_date = {d: pd.Timestamp.combine(d, EOD).tz_localize(tz).value for d in days}
    print(f"\n=== {w}d: {len(days)} sessions, OOS split {mid}  (cap {CAP}, ${RISK:.0f}/trade) ===")
    print(HEAD)
    print("-" * len(HEAD))
    for name, mode in POLICIES.items():
        taken = portfolio(apply_filter(reexit(trades, buckets, mode, eod_ns_by_date), elig), CAP)
        f, h1, h2 = three(taken, days, mid)
        prow(name, f, h1, h2)


def main():
    for w in WINDOWS:
        run_window(w)
    print("\navgR = pure exit edge per trade (capital-agnostic). A trailing/partial policy")
    print("earns its keep only if it lifts avgR AND Sharpe vs fixed_2R, in both windows +")
    print("both OOS halves. Watch the win% drop trailing brings — bigger winners, fewer of them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
