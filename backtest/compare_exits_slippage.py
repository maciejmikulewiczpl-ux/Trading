"""Slippage-aware confirmation of the trailing-stop exit, + 1R vs 1.5R pick.

compare_exits.py modeled exits at the exact trail level. Real stop/trailing/EOD
fills are MARKET orders that slip against you; only limit exits (the fixed-2R
target, the 1R partial) fill at price. This re-prices every MARKET exit with a
slippage haircut of S*risk_per_share and sweeps S = 0, 0.05, 0.10, 0.20 R.

Because trailing exits are ALL market fills (no protective limit target), this
penalizes trailing MORE than the fixed-2R baseline (which banks many winners at
a no-slip limit) — a deliberately conservative, fair test.

PRE-REGISTERED at a realistic S=0.10R: trail must still beat fixed_2R on avgR AND
Sharpe in both windows + both OOS halves to ship. 1.5R is the default pick (more
robust to choppy fills); 1R reported alongside.

Run (uses caches + helpers from compare_exits.py):
    .venv/Scripts/python.exe backtest/compare_exits_slippage.py
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.universe_portfolio import portfolio  # noqa: E402
from backtest.compare_selection import _tday  # noqa: E402
from backtest.compare_norefill_trend import trend_eligibility, apply_filter  # noqa: E402
from backtest.compare_exits import (  # noqa: E402
    load, bucket, three, HEAD, prow, CAP, RISK, EOD,
)

WINDOWS = [730, 180]
SLIPS = [0.0, 0.05, 0.10, 0.20]     # R-units of slippage per MARKET exit
PRE_REG_S = 0.10
POLICIES = {
    "fixed_2R (live)": dict(target_R=2.0, trail_R=None, partial=False),
    "trail_1R":        dict(target_R=None, trail_R=1.0, partial=False),
    "trail_1.5R":      dict(target_R=None, trail_R=1.5, partial=False),
}


def sim_long_exit_slip(day, start, entry, init_stop, eod_ns, mode, S):
    """Like compare_exits.sim_long_exit but MARKET exits (stop/trail/EOD) lose
    S*risk; LIMIT exits (target, the +1R partial half) fill clean."""
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

    def blend(run_r):                       # partial half banks +1R at a clean limit
        return (0.5 + 0.5 * run_r) if partial else run_r

    for i in range(start, n):
        if ns[i] >= eod_ns:                 # EOD = market
            return idx[i], cl[i], blend((cl[i] - entry) / risk - S)
        if trailR:
            s2 = hwm - trailR * risk
            if s2 > stop:
                stop = s2
        if lo[i] <= stop:                   # stop / trailing stop = market
            return idx[i], stop, blend((stop - entry) / risk - S)
        if part and not partial and hi[i] >= one_r:
            partial = True
            if entry > stop:
                stop = entry
        if target is not None and hi[i] >= target:   # take-profit = limit, no slip
            return idx[i], target, blend((target - entry) / risk)
        if hi[i] > hwm:
            hwm = hi[i]
    return idx[n - 1], cl[n - 1], blend((cl[n - 1] - entry) / risk - S)   # EOD market


def reexit_slip(trades, buckets, mode, eod_map, S):
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
        res = sim_long_exit_slip(day, start, t.entry_price, t.stop_price,
                                 eod_map[_tday(t)], mode, S)
        if res is None:
            continue
        ex_ts, ex_px, pr = res
        out.append(replace(t, exit_time=ex_ts, exit_price=float(ex_px),
                           pnl_r=pr, pnl_dollars=pr * RISK))
    return out


def main():
    verdict = {}
    for w in WINDOWS:
        all_bars, days, present, trades, closes = load(w)
        mid = sorted(days)[len(days) // 2]
        elig = trend_eligibility(closes, present, days)
        buckets = bucket(all_bars, present)
        tz = all_bars.index.get_level_values(1).tz
        eod_map = {d: pd.Timestamp.combine(d, EOD).tz_localize(tz).value for d in days}
        print(f"\n=== {w}d: {len(days)} sessions, OOS split {mid}  (cap {CAP}, ${RISK:.0f}/trade) ===")
        for S in SLIPS:
            print(f"\n  slippage {S:.2f}R per market exit")
            print("  " + HEAD)
            print("  " + "-" * len(HEAD))
            res = {}
            for name, mode in POLICIES.items():
                taken = portfolio(apply_filter(reexit_slip(trades, buckets, mode, eod_map, S), elig), CAP)
                f, h1, h2 = three(taken, days, mid)
                res[name] = (f, h1, h2)
                print("  ", end="")
                prow(name, f, h1, h2)
            if abs(S - PRE_REG_S) < 1e-9:
                verdict[w] = res

    # ---- pre-registered verdict at S=0.10 ----
    print("\n" + "=" * 64)
    print(f"PRE-REGISTERED VERDICT @ {PRE_REG_S:.2f}R slippage")
    print("trail must beat fixed_2R on avgR AND Sharpe in full + both OOS halves, both windows")
    print("=" * 64)
    for cand in ("trail_1.5R", "trail_1R"):
        ok_all = True
        line = []
        for w in WINDOWS:
            base, tr = verdict[w]["fixed_2R (live)"], verdict[w][cand]
            segs = []
            for i, nm in enumerate(("full", "h1", "h2")):
                good = (tr[i]["avgR"] > base[i]["avgR"]) and (tr[i]["sharpe"] > base[i]["sharpe"])
                segs.append(f"{nm}:{'ok' if good else 'X'}")
                ok_all &= good
            line.append(f"{w}d [{'  '.join(segs)}]")
        print(f"  {cand:<11} {'PASS' if ok_all else 'FAIL'}   " + "   ".join(line))
    return 0


if __name__ == "__main__":
    sys.exit(main())
