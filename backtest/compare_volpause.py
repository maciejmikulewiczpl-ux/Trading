"""Pause / lower-risk on risky days: PRIOR-day vol (reactive) vs SAME-day signal
(proactive), as full-pause vs half-risk. On the live trailing config, net of cost.

The shipped vol filter is REACTIVE — it reads yesterday's 20d realized vol, so it
sits out the turbulent AFTERMATH but can't see the surprise day. But ORB doesn't
enter until 09:45 (after the opening range), so by entry time we already have
today's gap + first 15 min — a PROACTIVE same-day gate. June 5 (a -2.48% day) was
already -0.77% by 09:45 with a 1.6x-wide opening range, so the morning warned.

This tests, net of the measured ~0.042R round-trip slippage, on the trailing-exit
trades (cap 16, $50), both windows + OOS:
  normal            : no filter (baseline)
  prior_vol pause   : pause days where SPY 20d vol > its 126d median (yesterday)
  prior_vol half    : same, but half risk instead of full pause
  sameday pause     : pause days where SPY is down > 0.5% by 09:45 ET (today)
  sameday half      : same, but half risk
Also reports how many days each flags and the avg $/day ON flagged days (does the
gate actually catch the BAD days?). 'Pause' zeroes a day; 'half' scales it 0.5x.

Run (re-simulates trailing exits, ~couple min):
    .venv/Scripts/python.exe backtest/compare_volpause.py
"""
from __future__ import annotations

import sys
from datetime import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.universe_portfolio import portfolio  # noqa: E402
from backtest.compare_selection import _tday  # noqa: E402
from backtest.compare_norefill_trend import trend_eligibility, apply_filter  # noqa: E402
from backtest.compare_exits import load, bucket, reexit, POLICIES, RISK, EOD  # noqa: E402

WINDOWS = [730, 180]
CAP = 16
COST = 0.042            # measured round-trip slippage (R)
SAMEDAY_MM = -0.005     # flag the day if SPY is down >0.5% by 09:45 ET
OR_START, OR_END, RTH_E = time(9, 30), time(9, 45), time(16, 0)


def prior_vol_flags(closes, days) -> dict:
    spy = closes["SPY"].dropna().sort_index()
    vol = spy.pct_change().rolling(20).std()
    med = vol.rolling(126, min_periods=40).median()
    f = (vol > med).shift(1)
    return {d: (bool(f.get(pd.Timestamp(d))) if pd.notna(f.get(pd.Timestamp(d))) else False) for d in days}


def sameday_flags(all_bars, days) -> dict:
    spy = all_bars.xs("SPY", level=0)
    t = spy.index.time
    rth = spy[(t >= OR_START) & (t < RTH_E)]
    g = rth.groupby(rth.index.date)
    daily = pd.DataFrame({"close": g["close"].last()}).sort_index()
    after = rth[rth.index.time >= OR_END]
    daily["px945"] = after.groupby(after.index.date)["open"].first()
    mm = daily["px945"] / daily["close"].shift(1) - 1.0      # SPY move, prior close -> 09:45
    out = {}
    for d in days:
        v = mm.get(d)
        out[d] = bool(pd.notna(v) and v < SAMEDAY_MM)
    return out


def series(taken, days, mult: dict) -> pd.Series:
    by = {}
    for tr in taken:
        by[_tday(tr)] = by.get(_tday(tr), 0.0) + (tr.pnl_r - COST)
    idx = sorted(days)
    s = pd.Series({d: by.get(d, 0.0) for d in idx})
    m = pd.Series({d: mult.get(d, 1.0) for d in idx})
    return s * RISK * m


def perf(s: pd.Series) -> dict:
    eq = s.cumsum()
    dd = (eq - eq.cummax()).min() if len(eq) else 0.0
    mu, sd = s.mean(), s.std()
    return {"pnl": s.sum(), "sharpe": (mu / sd * (252 ** 0.5)) if sd and sd > 0 else float("nan"), "maxdd": dd}


def three(taken, days, mid, mult):
    d1 = [d for d in days if d < mid]
    d2 = [d for d in days if d >= mid]
    return perf(series(taken, days, mult)), perf(series(taken, d1, mult)), perf(series(taken, d2, mult))


HEAD = f"{'config':<20}{'PnL$':>10}{'Sharpe':>8}{'maxDD$':>10}   {'h1 Sh':>6}{'h2 Sh':>6}{'h2 PnL':>9}"


def prow(label, f, h1, h2):
    print(f"{label:<20}{f['pnl']:>+10,.0f}{f['sharpe']:>8.2f}{f['maxdd']:>10,.0f}   "
          f"{h1['sharpe']:>6.2f}{h2['sharpe']:>6.2f}{h2['pnl']:>+9,.0f}")


def run_window(w):
    all_bars, days, present, trades, closes = load(w)
    mid = sorted(days)[len(days) // 2]
    elig = trend_eligibility(closes, present, days)
    buckets = bucket(all_bars, present)
    tz = all_bars.index.get_level_values(1).tz
    eod_ns = {d: pd.Timestamp.combine(d, EOD).tz_localize(tz).value for d in days}
    trail = apply_filter(reexit(trades, buckets, POLICIES["trail_1R"], eod_ns), elig)
    taken = portfolio(trail, CAP)

    prior = prior_vol_flags(closes, days)
    same = sameday_flags(all_bars, days)
    one = {d: 1.0 for d in days}
    p_pause = {d: (0.0 if prior[d] else 1.0) for d in days}
    p_half = {d: (0.5 if prior[d] else 1.0) for d in days}
    s_pause = {d: (0.0 if same[d] else 1.0) for d in days}
    s_half = {d: (0.5 if same[d] else 1.0) for d in days}

    # how good is each gate at catching bad days? avg $/day on flagged vs not (at full size)
    full = series(taken, days, one)
    def avg_on(flag, val):
        sel = [full[d] for d in sorted(days) if flag[d] == val]
        return (sum(sel) / len(sel)) if sel else 0.0
    print(f"\n=== {w}d: {len(days)} sessions, OOS split {mid}  (trailing, cap {CAP}, ${RISK:.0f}, net of {COST}R) ===")
    print(f"  PRIOR-vol gate: flags {sum(prior.values())} days | avg $/day flagged {avg_on(prior,True):+.0f} vs calm {avg_on(prior,False):+.0f}")
    print(f"  SAME-day gate : flags {sum(same.values())} days | avg $/day flagged {avg_on(same,True):+.0f} vs calm {avg_on(same,False):+.0f}")
    print(HEAD); print("-" * len(HEAD))
    prow("normal (no filter)", *three(taken, days, mid, one))
    prow("prior_vol pause", *three(taken, days, mid, p_pause))
    prow("prior_vol half", *three(taken, days, mid, p_half))
    prow("sameday pause", *three(taken, days, mid, s_pause))
    prow("sameday half", *three(taken, days, mid, s_half))


def main():
    for w in WINDOWS:
        run_window(w)
    print("\nReads: a gate helps if it RAISES Sharpe and CUTS drawdown vs 'normal' while")
    print("keeping PnL up. 'flagged avg $/day' should be clearly NEGATIVE (it's pausing the")
    print("bad days). Same-day flagging negative + better Sharpe = the proactive idea works.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
