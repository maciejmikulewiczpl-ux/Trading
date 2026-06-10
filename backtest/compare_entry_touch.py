"""Entry mechanics: does a TOUCH entry (resting stop at OR-high) beat CLOSE-CONFIRM?

Fable's review (2026-06-09) flagged entries as "the biggest untouched lever": the
live runner waits for a 1-min bar to CLOSE above OR-high, then market-buys on the
next bar's open. Measured incremental entry slippage ~0.02R, but the bigger hidden
cost is CONTINUATION — the breakout has already run from the signal-bar close to the
next-bar open (~0.06R) before you're in. On tight-OR trades (tiny R) that tax is large.

A resting STOP-LIMIT at OR-high enters the MOMENT price trades through the level —
earlier, at a better price (OR-high vs the higher next-bar open), and with TIGHTER
risk/share (entry-OR_low = the OR range itself), which on the trailing exit means a
winner reaches a bigger R-multiple. The COST: you take every FALSE breakout that
ticks OR-high and reverses (close-confirm never enters those). Net effect is empirical.

This re-detects breakouts from the cached minute bars under BOTH rules with identical
machinery, then runs the SAME trail-1R exit + trend filter + tight-OR<=0.5% cut +
cap-16 portfolio + cents-based slippage (calibrated so the median trade pays 0.042R).
The close-confirm arm should reproduce the known tight-OR trailing result (~+$6.7k /
Sharpe 1.80 @ 730d) as a sanity anchor.

CONSERVATIVE to the touch arm: a touch fill that GAPS the open above OR-high fills at
the open (worse), and its exit scan starts ON the signal bar (so a same-minute reversal
to OR-low books a full -1R). If touch wins even handicapped like this, it's real.

Cheap-ish: cached minute bars (same pkls as compare_or_range_capaware.py).
Run:
    .venv/Scripts/python.exe backtest/compare_entry_touch.py
"""
from __future__ import annotations

import math
import statistics
import sys
from dataclasses import dataclass
from datetime import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.universe_portfolio import portfolio  # noqa: E402
from backtest.compare_selection import _tday  # noqa: E402
from backtest.compare_norefill_trend import trend_eligibility, apply_filter  # noqa: E402
from backtest.compare_exits import load, sim_long_exit  # noqa: E402
from backtest.compare_volpause import prior_vol_flags, perf, RISK, CAP  # noqa: E402

import pandas as pd  # noqa: E402

WINDOWS = [730, 180]
OR_START, OR_END = time(9, 30), time(9, 45)
ENTRY_CUTOFF = time(11, 30)
EOD = time(15, 55)
RTH_E = time(16, 0)
TRAIL = dict(target_R=None, trail_R=1.0, partial=False)
OR_THR = 0.5                 # tight-OR cut (% of entry price)
TARGET_MEDIAN_R = 0.042
NOTIONAL_CAP = 10_000.0
SLIP_MULT = [1.0, 1.5]


@dataclass(frozen=True)
class T:
    symbol: str
    date: object
    side: str
    or_high: float
    or_low: float
    entry_time: object
    entry_price: float
    stop_price: float
    exit_time: object
    exit_price: float
    pnl_r: float


def or_pct(t):
    return (t.or_high - t.or_low) / t.entry_price * 100 if t.entry_price else 0.0


def risk_ps(t):
    return max(t.entry_price - t.stop_price, 1e-6)


def build_days(all_bars, present):
    """{symbol: {date: dict}} with OR levels + post-OR (>=9:45) numpy arrays incl open."""
    out = {}
    for sym in present:
        sb = all_bars.xs(sym, level=0)
        tt = sb.index.time
        sb = sb[(tt >= OR_START) & (tt < RTH_E)]
        d = {}
        for day, g in sb.groupby(sb.index.date):
            gt = g.index.time
            orb = g[(gt >= OR_START) & (gt < OR_END)]
            post = g[gt >= OR_END]
            if orb.empty or post.empty:
                continue
            pt = post.index.time
            n_entry = int((pt <= ENTRY_CUTOFF).sum())   # leading bars within the entry cutoff
            d[day] = {
                "or_high": float(orb["high"].max()),
                "or_low": float(orb["low"].min()),
                "n_entry": n_entry,
                "op": post["open"].to_numpy(float),
                "hi": post["high"].to_numpy(float),
                "lo": post["low"].to_numpy(float),
                "cl": post["close"].to_numpy(float),
                "ns": post.index.asi8,
                "idx": post.index,
            }
        out[sym] = d
    return out


def make_trade(sym, day, dd, entry_i, entry_px, eod_ns):
    """Run trail-1R exit from position entry_i; return a T or None."""
    res = sim_long_exit(dd, entry_i, entry_px, dd["or_low"], eod_ns, TRAIL)
    if res is None:
        return None
    ex_ts, ex_px, pr = res
    return T(sym, day, "long", dd["or_high"], dd["or_low"],
            dd["idx"][entry_i], float(entry_px), dd["or_low"],
            ex_ts, float(ex_px), float(pr))


def detect(days_by_sym, eod_ns_by_date):
    """Return (close_confirm_trades, touch_trades) over all symbol-days."""
    cc, to = [], []
    for sym, dmap in days_by_sym.items():
        for day, dd in dmap.items():
            eod_ns = eod_ns_by_date.get(day)
            if eod_ns is None:
                continue
            oh = dd["or_high"]
            ne, n = dd["n_entry"], len(dd["ns"])
            cl, hi, op = dd["cl"], dd["hi"], dd["op"]

            # close-confirm: first bar (within cutoff) that CLOSES above OR-high;
            # enter at the NEXT bar's open; exits scan from that next bar.
            for i in range(ne):
                if cl[i] > oh:
                    if i + 1 < n:
                        t = make_trade(sym, day, dd, i + 1, op[i + 1], eod_ns)
                        if t:
                            cc.append(t)
                    break

            # touch: first bar (within cutoff) whose HIGH reaches OR-high; fill at
            # max(OR-high, that bar's open) [gap fills worse]; exits scan from THIS bar.
            for i in range(ne):
                if hi[i] >= oh:
                    entry_px = max(oh, op[i])
                    t = make_trade(sym, day, dd, i, entry_px, eod_ns)
                    if t:
                        to.append(t)
                    break
    return cc, to


def dollar_series(taken, days, days_mult, cents):
    by, tot = {}, 0.0
    for t in taken:
        rps = risk_ps(t)
        target = RISK * days_mult.get(_tday(t), 1.0)
        shares = min(math.floor(target / rps), math.floor(NOTIONAL_CAP / t.entry_price))
        if shares <= 0:
            continue
        pnl = (t.exit_price - t.entry_price) * shares - 2.0 * cents * shares
        by[_tday(t)] = by.get(_tday(t), 0.0) + pnl
        tot += pnl
    s = pd.Series({d: by.get(d, 0.0) for d in sorted(days)})
    return s, (tot / len(taken) if taken else 0.0)


HEAD = (f"{'arm':<26}{'trades':>7}{'win%':>6}{'PnL$':>10}{'Sharpe':>8}{'maxDD$':>9}"
        f"   {'h1 PnL':>9}{'h2 PnL':>9}{'avg$/tr':>8}")


def run_window(w):
    all_bars, days, present, _trades, closes = load(w)
    elig = trend_eligibility(closes, present, days)
    tz = all_bars.index.get_level_values(1).tz
    eod_ns = {d: pd.Timestamp.combine(d, EOD).tz_localize(tz).value for d in days}
    mid = sorted(days)[len(days) // 2]
    prior = prior_vol_flags(closes, days)
    days_mult = {d: (0.5 if prior[d] else 1.0) for d in days}

    dbs = build_days(all_bars, present)
    cc, to = detect(dbs, eod_ns)
    # trend filter, then tight-OR cut, both arms identical
    cc = [t for t in apply_filter(cc, elig) if or_pct(t) <= OR_THR]
    to = [t for t in apply_filter(to, elig) if or_pct(t) <= OR_THR]
    base_cents = TARGET_MEDIAN_R * statistics.median(
        risk_ps(t) for t in (cc + to)) / 2.0

    print(f"\n{'='*80}\n=== {w}d | tight-OR<={OR_THR}% | close-confirm {len(cc)} vs touch "
          f"{len(to)} trades | OOS {mid} ===")
    for sm in SLIP_MULT:
        cents = base_cents * sm
        print(f"\n  --- slip {sm:.1f}x (${cents:.3f}/share) ---")
        print("  " + HEAD); print("  " + "-" * len(HEAD))
        for label, arm in (("close-confirm (live)", cc), ("touch (stop @ OR-high)", to)):
            taken = portfolio(arm, CAP)
            s, avgtr = dollar_series(taken, days, days_mult, cents)
            f = perf(s)
            h1 = s[[d for d in s.index if d < mid]].sum()
            h2 = s[[d for d in s.index if d >= mid]].sum()
            win = 100 * sum(1 for t in taken if t.exit_price > t.entry_price) / len(taken) if taken else 0
            print(f"  {label:<26}{len(taken):>7}{win:>5.0f}%{f['pnl']:>+10,.0f}{f['sharpe']:>8.2f}"
                  f"{f['maxdd']:>9,.0f}   {h1:>+9,.0f}{h2:>+9,.0f}{avgtr:>+8.1f}")


def main():
    for w in WINDOWS:
        run_window(w)
    print("\nReads: touch wins ONLY if it lifts Sharpe AND PnL net of its extra false-breakout")
    print("losers, in BOTH windows + the recent OOS half. The close-confirm arm should land")
    print("near the known tight-OR trailing result (~+$6.7k / 1.80 @ 730d) — if not, the")
    print("re-detection drifted from the cached set and the delta is what to trust, not the level.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
