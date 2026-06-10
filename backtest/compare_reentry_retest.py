"""Retest entry + same-direction re-entry after stop-out — the practitioner round.

Forum/practitioner sweep (overnight research 2026-06-10, round 2): the single
most-recommended ORB 'upgrade' across trading sites/forums is the RETEST entry —
after a confirmed breakout, don't chase: rest a limit at the broken OR-high and
enter on the pullback ("the highest-probability ORB setup"). The related lore is
the same-direction RE-ENTRY: a stop-out, then price reclaims the OR-high =
'shakeout before the real move' — re-enter once. Our max_flips work only ever
tested OPPOSITE-direction flips; same-direction re-entry is genuinely untested.

Priors stated up front: the retest entry optimizes WIN RATE but risks missing the
runaway breakouts that never pull back — and those trend-day tails are what the
tight-OR + trail-1R edge lives on (same reason touch-entry and marketable-limit
were rejected/deferred). The re-entry arm ADDS right-tail recovery (payoff-shaped,
like everything that ever worked here) at the price of extra round-trips on losers.

Arms (all on tight-OR<=0.5%, trail-1R, trend filter, vol-dial, $50/$10k, cents
slippage, both windows, OOS halves, 1.0x/1.5x slip):
  base            : shipped — enter next-bar open after the confirm close
  retest_ORhigh   : limit at OR-high after the confirm close; enter only if
                    touched before the 11:30 ET cutoff; skip the trade otherwise
  reenter_1x      : base entry; if stopped out and a LATER bar again closes
                    above OR-high before 11:30 ET, re-enter once at next-bar
                    open (same OR-low stop, trail-1R). Re-entered trades pay
                    DOUBLE round-trip costs.

PRE-REGISTERED GATES:
  retest_ORhigh: candidate only if Sharpe AND PnL >= base in BOTH windows at
    BOTH slips (an entry-price change must not surrender the tail).
  reenter_1x: candidate only if PnL >= base AND Sharpe >= base - 0.05 AND
    maxDD <= 1.15x base in BOTH windows at BOTH slips (it adds exposure, so it
    must pay for its extra costs without degrading risk).

Run:
    .venv/Scripts/python.exe backtest/compare_reentry_retest.py
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
from backtest.compare_exits import load, reexit, bucket as bucket_hlc, POLICIES, EOD  # noqa: E402
from backtest.compare_volpause import prior_vol_flags, perf, RISK  # noqa: E402
from backtest.compare_or_range_realcost import or_pct, risk_ps  # noqa: E402

WINDOWS = [730, 180]
OR_THR = 0.5
TARGET_MEDIAN_R = 0.042
NOTIONAL_CAP = 10_000.0
SLIP_MULT = [1.0, 1.5]
TRAIL_R = 1.0
CUTOFF = time(11, 30)
RTH_S, RTH_E = time(9, 30), time(16, 0)


def bucket_full(all_bars, present):
    """Like compare_exits.bucket but with opens + a positional 11:30 cutoff."""
    out = {}
    for sym in present:
        sb = all_bars.xs(sym, level=0)
        t = sb.index.time
        sb = sb[(t >= RTH_S) & (t < RTH_E)]
        d = {}
        for day, g in sb.groupby(sb.index.date):
            idx = g.index
            cut_ts = pd.Timestamp.combine(pd.Timestamp(day), CUTOFF).tz_localize(idx.tz)
            d[day] = {"idx": idx,
                      "op": g["open"].to_numpy(float),
                      "hi": g["high"].to_numpy(float),
                      "lo": g["low"].to_numpy(float),
                      "cl": g["close"].to_numpy(float),
                      "cut_i": int(idx.searchsorted(cut_ts, side="right"))}
        out[sym] = d
    return out


def trail_sim(day, start, entry, init_stop):
    """Trail-1R from bar `start`; returns (exit_i, exit_px, pnl_r). Mirrors
    compare_exits.sim_long_exit (stop-first, HWM through prior bar, exit at
    last bar close — the harness's effective EOD)."""
    hi, lo, cl = day["hi"], day["lo"], day["cl"]
    n = len(cl)
    risk = entry - init_stop
    if risk <= 0:
        return None
    hwm, stop = entry, init_stop
    for i in range(start, n):
        s2 = hwm - TRAIL_R * risk
        if s2 > stop:
            stop = s2
        if lo[i] <= stop:
            return i, stop, (stop - entry) / risk
        if hi[i] > hwm:
            hwm = hi[i]
    return n - 1, cl[n - 1], (cl[n - 1] - entry) / risk


def sim_retest(t, day, start):
    """Rest a limit at OR-high from the base entry bar; fill if touched before
    the 11:30 cutoff (fill at min(open, OR-high) of the touching bar)."""
    lo, op = day["lo"], day["op"]
    end = min(day["cut_i"], len(lo))
    for i in range(start, end):
        if lo[i] <= t.or_high:
            fill = min(op[i], t.or_high)
            res = trail_sim(day, i, fill, t.or_low)
            if res is None:
                return None
            _, ex_px, _ = res
            return fill, ex_px, 1          # 1 round trip
    return None                            # never pulled back -> no trade


def sim_reenter(t, day, start):
    """Base entry; on a stop-out, re-enter once if a later close > OR-high
    before the cutoff. Returns (sum$ per share at 1x, n_round_trips, entry)."""
    cl = day["cl"]
    res = trail_sim(day, start, t.entry_price, t.stop_price)
    if res is None:
        return None
    exit_i, ex_px, pnl_r = res
    legs = [(t.entry_price, ex_px)]
    stopped = ex_px <= t.stop_price + 1e-9 and exit_i < len(cl) - 1
    if stopped:
        end = min(day["cut_i"], len(cl))
        for j in range(exit_i + 1, end):
            if cl[j] > t.or_high:
                if j + 1 < len(cl):
                    entry2 = day["op"][j + 1]
                    if entry2 > t.or_low:  # sanity: positive risk
                        res2 = trail_sim(day, j + 1, entry2, t.or_low)
                        if res2 is not None:
                            legs.append((entry2, res2[1]))
                break
    return legs


def cap_shares(entry, rps, days_mult_d):
    target = RISK * days_mult_d
    return min(math.floor(target / rps), math.floor(NOTIONAL_CAP / entry))


HEAD = (f"{'arm':<16}{'trades':>7}{'fill%':>7}{'PnL$':>10}{'Sharpe':>8}{'maxDD$':>9}"
        f"   {'h1 PnL':>9}{'h2 PnL':>9}")


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

    # per-trade leg lists: [(entry, exit), ...] -> $ via cap-aware shares on leg-1 entry
    arms = {"base": {}, "retest_ORhigh": {}, "reenter_1x": {}}
    fills = {"base": 0, "retest_ORhigh": 0, "reenter_1x": 0}
    for t in tight:
        day = b_full.get(t.symbol, {}).get(_tday(t))
        if day is None:
            continue
        start = int(day["idx"].searchsorted(t.entry_time, side="left"))
        if start >= len(day["cl"]):
            continue
        arms["base"][id(t)] = [(t.entry_price, t.exit_price)]
        fills["base"] += 1
        r = sim_retest(t, day, start)
        if r is not None:
            arms["retest_ORhigh"][id(t)] = [(r[0], r[1])]
            fills["retest_ORhigh"] += 1
        legs = sim_reenter(t, day, start)
        if legs is not None:
            arms["reenter_1x"][id(t)] = legs
            fills["reenter_1x"] += 1

    print(f"\n=== {w}d: {len(days)} sessions, OOS split {mid}  "
          f"({len(tight)} tight-OR trades) ===")
    for sm in SLIP_MULT:
        cents = base_cents * sm
        print(f"\n  -- slippage {sm:.1f}x --")
        print("  " + HEAD)
        print("  " + "-" * len(HEAD))
        for name, legs_by_trade in arms.items():
            by = {}
            n_legs = 0
            for t in tight:
                legs = legs_by_trade.get(id(t))
                if not legs:
                    continue
                rps = t.entry_price - t.stop_price   # sizing risk = original signal
                sh = cap_shares(legs[0][0], rps, days_mult[_tday(t)])
                if sh <= 0:
                    continue
                pnl = sum((ex - en) * sh - 2.0 * cents * sh for en, ex in legs)
                n_legs += len(legs)
                by[_tday(t)] = by.get(_tday(t), 0.0) + pnl
            s = pd.Series({d: by.get(d, 0.0) for d in sorted(days)})
            f = perf(s)
            h1 = perf(s[s.index < mid])
            h2 = perf(s[s.index >= mid])
            fl = 100 * fills[name] / max(len(tight), 1)
            print(f"  {name:<16}{n_legs:>7}{fl:>6.0f}%{f['pnl']:>+10,.0f}"
                  f"{f['sharpe']:>8.2f}{f['maxdd']:>9,.0f}   "
                  f"{h1['pnl']:>+9,.0f}{h2['pnl']:>+9,.0f}")


def main():
    for w in WINDOWS:
        run_window(w)
    print("\nGates: retest needs Sharpe AND PnL >= base (both windows, both slips).")
    print("reenter needs PnL >= base, Sharpe >= base-0.05, maxDD <= 1.15x base.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
