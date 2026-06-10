"""Time-stop on stagnant trades: scratch what hasn't moved, keep the trailing exit.

Practitioner basis (overnight research 2026-06-10): momentum winners tend to work
soon after entry ("if it doesn't go, it won't go"). Our trailing exit already
crushes losers to ~-1R and lets winners run, but a trade that chops sideways for
hours ties up a slot and usually resolves to a late stop-out or a ~0R EOD close.
A TIME-STOP scratches it early: if, N minutes after entry, the high-water mark
has not reached entry + thr*R, exit at market (bar close in sim).

Tests on the SHIPPED config (tight-OR<=0.5%, trail-1R, trend filter, vol-dial,
$50/$10k, cents-slippage median 0.042R), both windows + OOS halves, 1.0x/1.5x slip:

  trail (base)      : shipped trailing exit, no time-stop
  ts45m/+0.25R      : scratch if HWM < entry+0.25R after 45 min
  ts60m/+0.25R
  ts60m/+0.5R
  ts90m/+0.5R
  ts120m/+0.5R

PRE-REGISTERED GATE: a time-stop arm is a ship candidate only if, in BOTH windows
at 1.0x slip, Sharpe AND PnL >= base with maxDD <= base, h2 not worse than base,
AND the ordering holds at 1.5x slip (scratch exits are market orders and pay the
spread more often, so cost-stress matters most here).

Run (re-simulates exits over the big univ caches, several minutes):
    .venv/Scripts/python.exe backtest/compare_timestop.py
"""
from __future__ import annotations

import math
import statistics
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.compare_selection import _tday  # noqa: E402
from backtest.compare_norefill_trend import trend_eligibility, apply_filter  # noqa: E402
from backtest.compare_exits import load, bucket, EOD, RISK  # noqa: E402
from backtest.compare_volpause import prior_vol_flags, perf  # noqa: E402
from backtest.compare_or_range_realcost import or_pct, risk_ps  # noqa: E402

WINDOWS = [730, 180]
OR_THR = 0.5
TARGET_MEDIAN_R = 0.042
NOTIONAL_CAP = 10_000.0
SLIP_MULT = [1.0, 1.5]
TRAIL_R = 1.0
NS_PER_MIN = 60_000_000_000

ARMS = {
    "trail (base)": None,
    "ts45m/+0.25R": (45, 0.25),
    "ts60m/+0.25R": (60, 0.25),
    "ts60m/+0.5R": (60, 0.50),
    "ts90m/+0.5R": (90, 0.50),
    "ts120m/+0.5R": (120, 0.50),
}


def sim_trail_timestop(day, start, entry, init_stop, eod_ns, ts):
    """Trailing-1R exit with optional time-stop ts=(minutes, thr_R). Mirrors
    compare_exits.sim_long_exit (stop-first ties, HWM through prior bar).

    UNIT GOTCHA: the cached DatetimeIndex is microsecond-resolution (pandas 2.x),
    so asi8 is NOT nanoseconds. The deadline is therefore computed positionally
    via idx.searchsorted(entry_ts + Timedelta), which is unit-safe."""
    ns, hi, lo, cl, idx = day["ns"], day["hi"], day["lo"], day["cl"], day["idx"]
    n = len(ns)
    risk = entry - init_stop
    if risk <= 0:
        return None
    deadline_i = (int(idx.searchsorted(idx[start] + pd.Timedelta(minutes=ts[0]),
                                       side="left")) if ts else n + 1)
    arm_level = entry + ts[1] * risk if ts else None
    hwm, stop = entry, init_stop
    for i in range(start, n):
        if ns[i] >= eod_ns:
            return idx[i], cl[i], (cl[i] - entry) / risk
        s2 = hwm - TRAIL_R * risk
        if s2 > stop:
            stop = s2
        if lo[i] <= stop:
            return idx[i], stop, (stop - entry) / risk
        if hi[i] > hwm:
            hwm = hi[i]
        if ts and i >= deadline_i and hwm < arm_level:
            return idx[i], cl[i], (cl[i] - entry) / risk    # scratch stagnant trade
    return idx[n - 1], cl[n - 1], (cl[n - 1] - entry) / risk


def reexit_ts(trades, buckets, eod_ns_by_date, ts):
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
        res = sim_trail_timestop(day, start, t.entry_price, t.stop_price,
                                 eod_ns_by_date[_tday(t)], ts)
        if res is None:
            continue
        ex_ts, ex_px, pr = res
        out.append(replace(t, exit_time=ex_ts, exit_price=float(ex_px),
                           pnl_r=pr, pnl_dollars=pr * RISK))
    return out


def cap_shares(t, days_mult):
    rps = risk_ps(t)
    target = RISK * days_mult.get(_tday(t), 1.0)
    return min(math.floor(target / rps), math.floor(NOTIONAL_CAP / t.entry_price))


def dollar_series(taken, days, days_mult, cents):
    by = {}
    for t in taken:
        sh = cap_shares(t, days_mult)
        if sh <= 0:
            continue
        pnl = (t.exit_price - t.entry_price) * sh - 2.0 * cents * sh
        by[_tday(t)] = by.get(_tday(t), 0.0) + pnl
    return pd.Series({d: by.get(d, 0.0) for d in sorted(days)})


HEAD = (f"{'arm':<16}{'trades':>7}{'scr%':>6}{'PnL$':>10}{'Sharpe':>8}{'maxDD$':>9}"
        f"   {'h1 PnL':>9}{'h2 PnL':>9}")


def run_window(w):
    all_bars, days, present, trades, closes = load(w)
    elig = trend_eligibility(closes, present, days)
    buckets = bucket(all_bars, present)
    tz = all_bars.index.get_level_values(1).tz
    eod_ns = {d: pd.Timestamp.combine(d, EOD).tz_localize(tz).value for d in days}
    mid = sorted(days)[len(days) // 2]
    prior = prior_vol_flags(closes, days)
    days_mult = {d: (0.5 if prior[d] else 1.0) for d in days}

    print(f"\n=== {w}d: {len(days)} sessions, OOS split {mid} ===")
    results = {}
    for name, ts in ARMS.items():
        sim = [t for t in apply_filter(reexit_ts(trades, buckets, eod_ns, ts), elig)
               if t.side == "long"]
        results[name] = [t for t in sim if or_pct(t) <= OR_THR]
    base = results["trail (base)"]
    base_cents = TARGET_MEDIAN_R * statistics.median(risk_ps(t) for t in base) / 2.0
    base_exits = {(t.symbol, _tday(t)): t.exit_time for t in base}

    for sm in SLIP_MULT:
        cents = base_cents * sm
        print(f"\n  -- slippage {sm:.1f}x --")
        print("  " + HEAD)
        print("  " + "-" * len(HEAD))
        for name, taken in results.items():
            # scratched = exits earlier than the base trail exit for same trade
            scr = sum(1 for t in taken
                      if t.exit_time < base_exits.get((t.symbol, _tday(t)), t.exit_time))
            s = dollar_series(taken, days, days_mult, cents)
            f = perf(s)
            h1 = perf(s[s.index < mid])
            h2 = perf(s[s.index >= mid])
            print(f"  {name:<16}{len(taken):>7}{100*scr/max(len(taken),1):>5.0f}%"
                  f"{f['pnl']:>+10,.0f}{f['sharpe']:>8.2f}{f['maxdd']:>9,.0f}   "
                  f"{h1['pnl']:>+9,.0f}{h2['pnl']:>+9,.0f}")


def main():
    for w in WINDOWS:
        run_window(w)
    print("\nPre-registered gate: Sharpe AND PnL >= base, maxDD <= base, h2 not worse,")
    print("in BOTH windows at 1.0x; ordering must hold at 1.5x. Otherwise reject.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
