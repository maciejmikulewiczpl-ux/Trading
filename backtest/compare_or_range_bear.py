"""Does the tight-OR / trailing win survive the 2022 BEAR? (robustness check before ship)

The bull-window battery (compare_or_range_robust.py) validated OR<=0.5% on the trailing
exit: smooth gradient, survives 2x slippage, both OOS halves. But ORB is net-NEGATIVE in
a sustained bear (see regime_vol_filter_finding). The honest question here is NOT "is it
profitable in 2022" (the vol-dial/pause hibernates the bot in a bear anyway) but: does the
tight-OR filter make the bear BETTER or WORSE than no filter? If tight-OR cuts the bear
loss too, it's a free robustness win; if it amplifies it, that's a caveat to flag.

Trailing exit, vol-dial half-risk, cap 16, $50, cents-based cost (calibrated + 2x),
OOS split inside 2022.

Run (uses cached 2022 minute bars; re-sims trailing exits):
    .venv/Scripts/python.exe backtest/compare_or_range_bear.py
"""
from __future__ import annotations

import statistics
import sys
from datetime import time as dtime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from strategies.orb import Params  # noqa: E402
from backtest.run_orb import run_backtest, STARTING_EQUITY  # noqa: E402
from backtest.universe_portfolio import portfolio  # noqa: E402
from backtest.compare_norefill_trend import trend_eligibility, apply_filter  # noqa: E402
from backtest.compare_exits import bucket, reexit, POLICIES, EOD  # noqa: E402
from backtest.compare_volpause import prior_vol_flags, RISK, CAP  # noqa: E402
from backtest.compare_volpause_bear import load_2022  # noqa: E402
from backtest.compare_or_range_realcost import or_pct, risk_ps, three_rc  # noqa: E402

import pandas as pd  # noqa: E402

THRESHOLDS = [None, 0.5, 0.4]
TARGET_MEDIAN_R = 0.042
SLIP_MULT = [1.0, 2.0]

HEAD = f"{'config':<22}{'trades':>7}{'PnL$':>10}{'Sharpe':>8}{'maxDD$':>10}   {'h1 Sh':>6}{'h2 Sh':>6}{'h2 PnL':>9}"


def prow(label, n, f, h1, h2):
    print(f"  {label:<22}{n:>7}{f['pnl']:>+10,.0f}{f['sharpe']:>8.2f}{f['maxdd']:>10,.0f}   "
          f"{h1['sharpe']:>6.2f}{h2['sharpe']:>6.2f}{h2['pnl']:>+9,.0f}")


def main():
    all_bars, days, closes = load_2022()
    present = sorted(all_bars.index.get_level_values(0).unique())
    mid = sorted(days)[len(days) // 2]
    params = Params(or_minutes=15, target_r=2.0, risk_per_trade=100.0, max_position_pct=0.25,
                    max_position_dollars=10_000.0, no_entry_after_time=dtime(11, 30))
    trades, _ = run_backtest(all_bars, days, present, params, STARTING_EQUITY)
    elig = trend_eligibility(closes, present, days)
    buckets = bucket(all_bars, present)
    tz = all_bars.index.get_level_values(1).tz
    eod_ns = {d: pd.Timestamp.combine(d, EOD).tz_localize(tz).value for d in days}
    trail = [t for t in apply_filter(reexit(trades, buckets, POLICIES["trail_1R"], eod_ns), elig)
             if t.side == "long"]
    prior = prior_vol_flags(closes, days)
    mult = {d: (0.5 if prior[d] else 1.0) for d in days}
    base_cents = TARGET_MEDIAN_R * statistics.median(risk_ps(t) for t in trail) / 2.0

    spy = closes["SPY"].dropna()
    yr = spy[(spy.index >= pd.Timestamp(2022, 1, 1)) & (spy.index <= pd.Timestamp(2022, 12, 31))]
    print(f"\n=== 2022 BEAR TRAILING: {len(days)} sessions, OOS {mid} | "
          f"SPY {(yr.iloc[-1]/yr.iloc[0]-1)*100:+.1f}% | vol-dial flags {sum(prior.values())} days ===")
    for sm in SLIP_MULT:
        cents = base_cents * sm
        print(f"\n  slip {sm:.1f}x (${cents:.3f}/share)")
        print("  " + HEAD); print("  " + "-" * len(HEAD))
        for thr in THRESHOLDS:
            kept = trail if thr is None else [t for t in trail if or_pct(t) <= thr]
            taken = portfolio(kept, CAP)
            label = "no filter (baseline)" if thr is None else f"max OR <= {thr:.1f}%"
            prow(label, len(taken), *three_rc(taken, days, mid, mult, cents))
    print("\nReads: tight-OR is a free robustness win if it cuts the bear loss (or holds it flat)")
    print("vs baseline. If OR<=0.5% loses MORE than baseline in 2022, flag it. Note the vol-dial")
    print("already hibernates the bot most of a bear, so absolute bear PnL matters less than the sign.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
