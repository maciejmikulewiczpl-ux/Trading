"""Can a FASTER vol measure re-engage sooner without losing the protection?

The shipped dial uses 20-day realized vol, which is mechanically slow to reset —
one big move keeps it elevated for ~20 trading days even if the market calms. This
tests faster resets (10d, 5d, EWMA) against the 20d, on the bull windows AND the
2022 bear, measuring BOTH:
  - protection: Sharpe / drawdown (half-risk dial, net of costs, trailing config)
  - dormancy  : days flagged + avg consecutive-flagged run length (how long it stays
                dialed down after triggering)

The win we're hunting: a measure that keeps ~the same Sharpe/drawdown but with FEWER
flagged days / shorter runs (less time stuck at reduced risk). The risk: faster
resets re-engage INTO the choppy aftermath (vol clusters) and Sharpe/drawdown get
WORSE. If so, the slow 20d window is right and there's no free lunch.

Run (re-sims trailing on each cached set, a few min):
    .venv/Scripts/python.exe backtest/compare_voldial_speed.py
"""
from __future__ import annotations

import sys
from datetime import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from strategies.orb import Params  # noqa: E402
from backtest.run_orb import run_backtest, STARTING_EQUITY  # noqa: E402
from backtest.universe_portfolio import portfolio  # noqa: E402
from backtest.compare_norefill_trend import trend_eligibility, apply_filter  # noqa: E402
from backtest.compare_exits import load, bucket, reexit, POLICIES, EOD  # noqa: E402
from backtest.compare_volpause import series, perf, RISK, CAP  # noqa: E402

# (label, kind, param)
VARIANTS = [
    ("vol20 (shipped)", "window", 20),
    ("vol10", "window", 10),
    ("vol5", "window", 5),
    ("EWMA .94", "ewma", 0.94),
    ("EWMA .90", "ewma", 0.90),
]


def vol_flags(closes, days, kind, param) -> dict:
    spy = closes["SPY"].dropna().sort_index()
    ret = spy.pct_change()
    if kind == "window":
        vol = ret.rolling(param).std()
    else:  # EWMA variance, lambda=param
        vol = ret.pow(2).ewm(alpha=1 - param).mean().pow(0.5)
    med = vol.rolling(126, min_periods=40).median()
    f = (vol > med).shift(1)
    return {d: (bool(f.get(pd.Timestamp(d))) if pd.notna(f.get(pd.Timestamp(d))) else False) for d in days}


def avg_run(flag, days):
    runs, cur = [], 0
    for d in sorted(days):
        if flag[d]:
            cur += 1
        elif cur:
            runs.append(cur); cur = 0
    if cur:
        runs.append(cur)
    return (sum(runs) / len(runs)) if runs else 0.0


def get_set(name):
    if name == "2022":
        from backtest.compare_volpause_bear import load_2022
        all_bars, days, closes = load_2022()
        present = sorted(all_bars.index.get_level_values(0).unique())
        p = Params(or_minutes=15, target_r=2.0, risk_per_trade=100.0, max_position_pct=0.25,
                   max_position_dollars=10_000.0, no_entry_after_time=time(11, 30))
        trades, _ = run_backtest(all_bars, days, present, p, STARTING_EQUITY)
        return all_bars, days, present, trades, closes
    return load(int(name))


def run_set(name):
    all_bars, days, present, trades, closes = get_set(name)
    elig = trend_eligibility(closes, present, days)
    buckets = bucket(all_bars, present)
    tz = all_bars.index.get_level_values(1).tz
    eod_ns = {d: pd.Timestamp.combine(d, EOD).tz_localize(tz).value for d in days}
    taken = portfolio(apply_filter(reexit(trades, buckets, POLICIES["trail_1R"], eod_ns), elig), CAP)

    base = perf(series(taken, days, {d: 1.0 for d in days}))
    print(f"\n=== {name}: {len(days)} sessions  (half-risk dial, trailing, net of cost) ===")
    print(f"  normal (no dial): PnL {base['pnl']:+,.0f}  Sharpe {base['sharpe']:.2f}  maxDD {base['maxdd']:+,.0f}")
    print(f"  {'measure':<18}{'flagged':>8}{'avg run':>8}{'PnL$':>10}{'Sharpe':>8}{'maxDD$':>10}")
    print("  " + "-" * 62)
    for label, kind, param in VARIANTS:
        fl = vol_flags(closes, days, kind, param)
        mult = {d: (0.5 if fl[d] else 1.0) for d in days}
        s = perf(series(taken, days, mult))
        print(f"  {label:<18}{sum(fl.values()):>8}{avg_run(fl, days):>8.1f}"
              f"{s['pnl']:>+10,.0f}{s['sharpe']:>8.2f}{s['maxdd']:>10,.0f}")


def main():
    for name in ("730", "180", "2022"):
        run_set(name)
    print("\nReads: want HIGH Sharpe + LOW drawdown (protection) with FEWER flagged days /")
    print("SHORTER avg run (less dormancy). If faster measures (vol5/EWMA) hold Sharpe+DD")
    print("with shorter runs -> re-engage sooner safely. If Sharpe/DD worsen -> 20d is right.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
