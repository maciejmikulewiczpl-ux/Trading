"""Scale-up frontier: minimum account size for ~$1k/month on the validated edge.

The user's goal is $1k/mo with minimal investment. Per tightOR_finding, profit on
this edge scales with the NOTIONAL CAP (tiny risk/share -> huge share counts ->
the $10k cap binds), NOT risk_per_trade. This maps the whole frontier from cached
data instead of guessing:

  caps   : $10k (live), $15k, $20k, $25k, $35k per position
  stacks : plain        = shipped config (tight-OR<=0.5%, trail-1R, trend filter,
                          vol-dial half, $50 risk)
           +reenter     = + one same-direction re-entry after a stop-out on a
                          fresh close > OR-high before 11:30 (validated 2026-06-10,
                          compare_reentry_retest.py; double costs charged)
           +reenter+pyr = + pyramid 0.5x at +2R under the same trail (validated
                          mild-positive, compare_pyramid.py; extra round trip
                          charged; pyramid notional counted in exposure)

For each (cap, stack) x (1.0x, 1.5x slippage) x (730d, 180d): PnL/month, Sharpe,
maxDD, peak concurrent open NOTIONAL (minute-resolution walk of actual entry/exit
times), p95 of daily peak notional, peak concurrent open RISK, and REQUIRED
EQUITY = max(peak_notional / 4x Reg-T intraday margin, $25k PDT floor).

PRE-REGISTERED RECOMMENDATION RULE (before seeing results): recommend the
smallest required-equity cell with 730d PnL >= $1,000/mo at 1.0x slip AND
>= $600/mo at 1.5x slip AND 730d maxDD <= 3% of required equity AND 180d PnL
positive at both slips. Caveats that always apply: backtest cents-slippage is
calibrated to CURRENT size — real slippage grows with size, so any ramp must be
gradual with measured fills (the known biggest risk, tightOR_finding).

Run:
    .venv/Scripts/python.exe backtest/scaleup_frontier.py
"""
from __future__ import annotations

import math
import statistics
import sys
from datetime import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.compare_selection import _tday  # noqa: E402
from backtest.compare_norefill_trend import trend_eligibility, apply_filter  # noqa: E402
from backtest.compare_exits import load, bucket as bucket_hlc, reexit, POLICIES, EOD  # noqa: E402
from backtest.compare_volpause import prior_vol_flags, perf, RISK  # noqa: E402
from backtest.compare_or_range_realcost import or_pct, risk_ps  # noqa: E402
from backtest.compare_reentry_retest import bucket_full  # noqa: E402

WINDOWS = [730, 180]
OR_THR = 0.5
TARGET_MEDIAN_R = 0.042
SLIP_MULT = [1.0, 1.5]
TRAIL_R = 1.0
CAPS = [10_000.0, 15_000.0, 20_000.0, 25_000.0, 35_000.0]
MARGIN = 4.0            # Reg-T intraday buying power
PDT_FLOOR = 25_000.0    # pattern-day-trader minimum equity
MONTHS = {730: 24.0, 180: 6.0}


def trail_sim_full(day, start, entry, init_stop):
    """Trail-1R; returns (exit_i, exit_px, hwm, pyr_i) where pyr_i is the first
    bar index at which HWM crossed entry+2R (None if never)."""
    hi, lo, cl = day["hi"], day["lo"], day["cl"]
    n = len(cl)
    risk = entry - init_stop
    if risk <= 0:
        return None
    two_r = entry + 2.0 * risk
    hwm, stop, pyr_i = entry, init_stop, None
    for i in range(start, n):
        s2 = hwm - TRAIL_R * risk
        if s2 > stop:
            stop = s2
        if lo[i] <= stop:
            return i, stop, hwm, pyr_i
        if hi[i] > hwm:
            hwm = hi[i]
            if pyr_i is None and hwm >= two_r:
                pyr_i = i
    return n - 1, cl[n - 1], hwm, pyr_i


def build_legs(t, day, start, reenter):
    """Leg list [(entry_i, exit_i, entry_px, exit_px, rps, pyr_i)] for a trade."""
    res = trail_sim_full(day, start, t.entry_price, t.stop_price)
    if res is None:
        return []
    exit_i, ex_px, hwm, pyr_i = res
    rps = t.entry_price - t.stop_price
    legs = [(start, exit_i, t.entry_price, ex_px, rps, pyr_i)]
    if reenter and ex_px <= t.stop_price + 1e-9 and exit_i < len(day["cl"]) - 1:
        end = min(day["cut_i"], len(day["cl"]))
        for j in range(exit_i + 1, end):
            if day["cl"][j] > t.or_high:
                if j + 1 < len(day["cl"]):
                    e2 = day["op"][j + 1]
                    if e2 > t.or_low:
                        r2 = trail_sim_full(day, j + 1, e2, t.or_low)
                        if r2 is not None:
                            legs.append((j + 1, r2[0], e2, r2[1], e2 - t.or_low, r2[3]))
                break
    return legs


def run_window(w):
    all_bars, days, present, trades, closes = load(w)
    elig = trend_eligibility(closes, present, days)
    b_hlc = bucket_hlc(all_bars, present)
    b_full = bucket_full(all_bars, present)
    tz = all_bars.index.get_level_values(1).tz
    eod_ns = {d: pd.Timestamp.combine(d, EOD).tz_localize(tz).value for d in days}
    trail = [t for t in apply_filter(reexit(trades, b_hlc, POLICIES["trail_1R"], eod_ns), elig)
             if t.side == "long"]
    tight = [t for t in trail if or_pct(t) <= OR_THR]
    mid = sorted(days)[len(days) // 2]
    prior = prior_vol_flags(closes, days)
    days_mult = {d: (0.5 if prior[d] else 1.0) for d in days}
    base_cents = TARGET_MEDIAN_R * statistics.median(risk_ps(t) for t in tight) / 2.0

    # precompute legs once per stack flag (independent of cap/slip)
    legs_plain, legs_re = {}, {}
    starts = {}
    for t in tight:
        day = b_full.get(t.symbol, {}).get(_tday(t))
        if day is None:
            continue
        start = int(day["idx"].searchsorted(t.entry_time, side="left"))
        if start >= len(day["cl"]):
            continue
        starts[id(t)] = (day, start)
        legs_plain[id(t)] = build_legs(t, day, start, reenter=False)
        legs_re[id(t)] = build_legs(t, day, start, reenter=True)

    stacks = {"plain": (legs_plain, False),
              "+reenter": (legs_re, False),
              "+reenter+pyr": (legs_re, True)}

    print(f"\n=== {w}d: {len(days)} sessions ({len(tight)} tight-OR trades), "
          f"{MONTHS[w]:.0f} months ===")
    head = (f"{'cap':>7} {'stack':<13}{'PnL/mo':>8}{'Sharpe':>7}{'maxDD$':>8}"
            f"{'pkNot$':>9}{'p95Not$':>9}{'pkRisk$':>8}{'reqEq$':>9}{'DD%eq':>6}")
    for sm in SLIP_MULT:
        cents = base_cents * sm
        print(f"\n  -- slippage {sm:.1f}x --")
        print("  " + head)
        print("  " + "-" * len(head))
        for cap in CAPS:
            for sname, (legs_by, pyramid) in stacks.items():
                by = {}
                events_by_day = {}
                for t in tight:
                    legs = legs_by.get(id(t))
                    if not legs:
                        continue
                    d = _tday(t)
                    pnl = 0.0
                    for (ei, xi, en, ex, rps, pyr_i) in legs:
                        target = RISK * days_mult.get(d, 1.0)
                        sh = min(math.floor(target / rps), math.floor(cap / en))
                        if sh <= 0:
                            continue
                        pnl += (ex - en) * sh - 2.0 * cents * sh
                        day, _ = starts[id(t)]
                        ev = events_by_day.setdefault(d, [])
                        ev.append((ei, en * sh, rps * sh))
                        ev.append((xi + 1, -en * sh, -rps * sh))
                        if pyramid and pyr_i is not None:
                            sh2 = sh // 2
                            if sh2 > 0:
                                en2 = en + 2.0 * rps
                                pnl += (ex - en2) * sh2 - 2.0 * cents * sh2
                                ev.append((pyr_i, en2 * sh2, 0.0))
                                ev.append((xi + 1, -en2 * sh2, 0.0))
                    by[d] = by.get(d, 0.0) + pnl
                s = pd.Series({d: by.get(d, 0.0) for d in sorted(days)})
                f = perf(s)
                # exposure walk: minute-resolution concurrent notional/risk
                peak_not, peak_risk, day_peaks = 0.0, 0.0, []
                for d, ev in events_by_day.items():
                    ev.sort(key=lambda x: x[0])
                    cur_n = cur_r = pk = 0.0
                    for _, dn, dr in ev:
                        cur_n += dn
                        cur_r += dr
                        pk = max(pk, cur_n)
                        peak_risk = max(peak_risk, cur_r)
                    day_peaks.append(pk)
                    peak_not = max(peak_not, pk)
                p95 = (statistics.quantiles(day_peaks, n=20)[18]
                       if len(day_peaks) >= 20 else peak_not)
                req_eq = max(peak_not / MARGIN, PDT_FLOOR)
                ddpct = 100.0 * abs(f["maxdd"]) / req_eq
                print(f"  {cap/1000:>5.0f}k {sname:<13}{f['pnl']/MONTHS[w]:>+8,.0f}"
                      f"{f['sharpe']:>7.2f}{f['maxdd']:>8,.0f}{peak_not:>9,.0f}"
                      f"{p95:>9,.0f}{peak_risk:>8,.0f}{req_eq:>9,.0f}{ddpct:>5.1f}%")


def main():
    for w in WINDOWS:
        run_window(w)
    print("\nRecommendation rule (pre-registered): smallest reqEq with 730d >= $1,000/mo")
    print("at 1.0x AND >= $600/mo at 1.5x AND maxDD <= 3% of reqEq AND 180d positive")
    print("at both slips. Slippage grows with size in reality -> ramp gradually,")
    print("measure real fills at each step (the known #1 scaling risk).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
