"""The true-dollar figure for the tight-OR / trailing win, with the $10k notional cap.

R-space said OR<=0.5% on trailing makes +$17k/730d. But tight OR = tiny risk/share =
LARGE share counts, so the $10k per-position notional cap BINDS hard on exactly these
trades — you can't put the full $50 risk on them, so realized DOLLARS land below the
R-space projection (the Sharpe/edge should hold; the dollar magnitude shrinks). This
re-prices every trailing trade with the runner's real sizing:
    shares = min(floor(risk*mult / risk_per_share), floor($10k / entry))
    net_$  = (exit-entry)*shares  -  2*cents*shares      (cents = calibrated slippage)
mult = vol-dial (0.5 on high-vol days). Real dollars, cap applied, both windows + OOS.

Run (loads minute bars + re-sims trailing exits):
    .venv/Scripts/python.exe backtest/compare_or_range_capaware.py
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
THRESHOLDS = [None, 0.5, 0.4]
TARGET_MEDIAN_R = 0.042
NOTIONAL_CAP = 10_000.0
SLIP_MULT = [1.0, 1.5]

HEAD = f"{'config':<22}{'trades':>7}{'PnL$':>10}{'Sharpe':>8}{'maxDD$':>10}   {'h1 PnL':>9}{'h2 PnL':>9}{'avg$/tr':>8}"


def dollar_series(taken, days, mult, cents):
    """Real $/day: cap-aware shares, trailing exit, cents-based round-trip cost."""
    by = {}
    nsh = 0
    tot = 0.0
    for t in taken:
        rps = risk_ps(t)
        target = RISK * mult.get(_tday(t), 1.0)
        shares = min(math.floor(target / rps), math.floor(NOTIONAL_CAP / t.entry_price))
        if shares <= 0:
            continue
        pnl = (t.exit_price - t.entry_price) * shares - 2.0 * cents * shares
        by[_tday(t)] = by.get(_tday(t), 0.0) + pnl
        nsh += shares
        tot += pnl
    s = pd.Series({d: by.get(d, 0.0) for d in sorted(days)})
    return s, (tot / len(taken) if taken else 0.0)


def main():
    for w in WINDOWS:
        all_bars, days, present, trades, closes = load(w)
        elig = trend_eligibility(closes, present, days)
        buckets = bucket(all_bars, present)
        tz = all_bars.index.get_level_values(1).tz
        eod_ns = {d: pd.Timestamp.combine(d, EOD).tz_localize(tz).value for d in days}
        trail = [t for t in apply_filter(reexit(trades, buckets, POLICIES["trail_1R"], eod_ns), elig)
                 if t.side == "long"]
        mid = sorted(days)[len(days) // 2]
        prior = prior_vol_flags(closes, days)
        mult = {d: (0.5 if prior[d] else 1.0) for d in days}
        base_cents = TARGET_MEDIAN_R * statistics.median(risk_ps(t) for t in trail) / 2.0

        print(f"\n=== {w}d TRAILING, REAL $ (${RISK:.0f} risk, $10k cap), OOS {mid} ===")
        for sm in SLIP_MULT:
            cents = base_cents * sm
            print(f"\n  slip {sm:.1f}x (${cents:.3f}/share)")
            print("  " + HEAD); print("  " + "-" * len(HEAD))
            for thr in THRESHOLDS:
                kept = trail if thr is None else [t for t in trail if or_pct(t) <= thr]
                taken = portfolio(kept, CAP)
                s, avgtr = dollar_series(taken, days, mult, cents)
                d1 = [d for d in days if d < mid]; d2 = [d for d in days if d >= mid]
                f = perf(s)
                h1 = s[[d for d in s.index if d < mid]].sum()
                h2 = s[[d for d in s.index if d >= mid]].sum()
                label = "no filter (baseline)" if thr is None else f"max OR <= {thr:.1f}%"
                print(f"  {label:<22}{len(taken):>7}{f['pnl']:>+10,.0f}{f['sharpe']:>8.2f}"
                      f"{f['maxdd']:>10,.0f}   {h1:>+9,.0f}{h2:>+9,.0f}{avgtr:>+8.1f}")
    print("\nReads: these are the REAL dollars the runner would make (cap applied). PnL$ will be")
    print("below the R-space figure — that's expected. What must HOLD: tight-OR Sharpe clearly")
    print("beats baseline and maxDD is far smaller, in both windows. avg$/tr shows cap erosion.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
