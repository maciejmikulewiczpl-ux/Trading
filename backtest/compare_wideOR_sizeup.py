"""Does 'wider-OR with higher risk' work? (empirical answer: no — you lever a loser.)

Splits the trailing-exit trades into TIGHT (OR<=0.5% of price) vs WIDE (OR>0.8%) and
re-prices each at risk multipliers 1x/2x/4x (real $10k notional cap + cents slippage,
vol-dial, cap 16). The point:
  - Leverage SCALES whatever edge exists; it doesn't create one.
  - TIGHT has a positive edge -> sizing up adds profit (but the $10k cap binds, so
    risk_per_trade barely moves it — you'd raise the CAP, per tightOR_finding).
  - WIDE is net-NEGATIVE -> sizing up just multiplies the LOSS and the drawdown, at
    the SAME (bad) Sharpe (Sharpe is risk-invariant). There is no risk level where a
    losing bucket becomes profitable.

Run:
    .venv/Scripts/python.exe backtest/compare_wideOR_sizeup.py
"""
from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.universe_portfolio import portfolio  # noqa: E402
from backtest.compare_selection import _tday  # noqa: E402
from backtest.compare_norefill_trend import trend_eligibility, apply_filter  # noqa: E402
from backtest.compare_exits import load, bucket, reexit, POLICIES, EOD  # noqa: E402
from backtest.compare_volpause import prior_vol_flags, perf, RISK, CAP  # noqa: E402
from backtest.compare_or_range_realcost import or_pct, risk_ps  # noqa: E402

import pandas as pd  # noqa: E402

WINDOWS = [730, 180]
TARGET_MEDIAN_R = 0.042
NOTIONAL_CAP = 10_000.0
RISK_MULTS = [1.0, 2.0, 4.0]


def dser(taken, days, voldial, cents, risk_mult):
    by = {}
    for t in taken:
        rps = risk_ps(t)
        target = RISK * risk_mult * voldial.get(_tday(t), 1.0)
        shares = min(math.floor(target / rps), math.floor(NOTIONAL_CAP / t.entry_price))
        if shares <= 0:
            continue
        by[_tday(t)] = by.get(_tday(t), 0.0) + (t.pnl_r * shares * rps - 2.0 * cents * shares)
    return pd.Series({d: by.get(d, 0.0) for d in sorted(days)})


def run_window(w):
    all_bars, days, present, trades, closes = load(w)
    elig = trend_eligibility(closes, present, days)
    buckets = bucket(all_bars, present)
    tz = all_bars.index.get_level_values(1).tz
    eod_ns = {d: pd.Timestamp.combine(d, EOD).tz_localize(tz).value for d in days}
    trail = [t for t in apply_filter(reexit(trades, buckets, POLICIES["trail_1R"], eod_ns), elig)
             if t.side == "long"]
    prior = prior_vol_flags(closes, days)
    half = {d: (0.5 if prior[d] else 1.0) for d in days}
    cents = TARGET_MEDIAN_R * statistics.median(risk_ps(t) for t in trail) / 2.0

    tight = [t for t in trail if or_pct(t) <= 0.5]
    wide = [t for t in trail if or_pct(t) > 0.8]

    def avgR(ts):
        rs = [t.pnl_r for t in ts]
        return sum(rs) / len(rs) if rs else 0.0

    print(f"\n========== {w}d  (trailing, real cap-aware $, cents={cents:.3f}/sh) ==========")
    print(f"  TIGHT (OR<=0.5%): {len(tight)} trades, gross avgR {avgR(tight):+.3f}")
    print(f"  WIDE  (OR>0.8%) : {len(wide)} trades, gross avgR {avgR(wide):+.3f}")
    print(f"\n  {'bucket @ risk':<18}{'trades':>7}{'PnL$':>10}{'Sharpe':>8}{'maxDD$':>10}")
    print("  " + "-" * 53)
    for label, ts in [("TIGHT", tight), ("WIDE", wide)]:
        taken = portfolio(ts, CAP)
        for rm in RISK_MULTS:
            s = dser(taken, days, half, cents, rm)
            f = perf(s)
            print(f"  {label+' @ '+str(int(rm))+'x':<18}{len(taken):>7}{f['pnl']:>+10,.0f}{f['sharpe']:>8.2f}{f['maxdd']:>10,.0f}")


def main():
    for w in WINDOWS:
        run_window(w)
    print("\nReads: within a bucket, Sharpe is ~flat across risk multipliers (risk-invariant) —")
    print("leverage only scales $. TIGHT's cap-bound rows barely grow (raise the CAP, not risk).")
    print("WIDE goes MORE NEGATIVE and its drawdown balloons as you size up: levering a loser.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
