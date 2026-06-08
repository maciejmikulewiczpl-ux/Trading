"""Does the tight-OR edge extend PAST the 11:30 entry cutoff? (the user's hunch)

compare_tightOR_revisit showed the edge is NOT front-loaded — later entries are as good
as early ones, monotonically, in both windows. The live cache stops at 11:30 (the live
no_entry_after_time), so we can't see later entries without a re-backtest. This re-runs
the backtest with entries allowed until 15:00, then asks: are tight-OR breakouts that
fire 11:30-13:00 and 13:00-15:00 profitable on their own? If yes, extending the cutoff
adds free volume at the same tiny drawdown — scaling WITHOUT more buying power.

Honest cost: cents calibrated on ALL trailing trades (conservative basis, matches
compare_or_range_capaware), real $10k cap, vol-dial half, both windows + OOS.

Run (RE-RUNS the backtest with a later cutoff, ~10-15 min; then re-sims trailing):
    .venv/Scripts/python.exe backtest/compare_tightOR_latecutoff.py
"""
from __future__ import annotations

import math
import statistics
import sys
from datetime import time as dtime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from strategies.orb import Params  # noqa: E402
from backtest.run_orb import run_backtest, STARTING_EQUITY  # noqa: E402
from backtest.universe_portfolio import portfolio  # noqa: E402
from backtest.compare_selection import _tday  # noqa: E402
from backtest.compare_norefill_trend import trend_eligibility, apply_filter  # noqa: E402
from backtest.compare_exits import load, bucket, reexit, POLICIES, EOD  # noqa: E402
from backtest.compare_volpause import prior_vol_flags, perf, RISK, CAP  # noqa: E402
from backtest.compare_or_range_realcost import or_pct, risk_ps  # noqa: E402
from backtest.compare_or_range_capaware import dollar_series  # noqa: E402

import pandas as pd  # noqa: E402

WINDOWS = [730, 180]
TIGHT = 0.5
LATE_CUTOFF = dtime(15, 0)      # re-backtest allows entries until 15:00
TARGET_MEDIAN_R = 0.042


def et_min(t):
    et = t.entry_time.tz_convert("America/New_York") if t.entry_time.tzinfo else t.entry_time
    return et.hour * 60 + et.minute


def tight(trades):
    return [t for t in trades if or_pct(t) <= TIGHT]


HEAD = f"{'variant':<24}{'trades':>7}{'PnL$':>9}{'Sharpe':>8}{'maxDD$':>9}   {'h1$':>8}{'h2$':>8}{'avg$/tr':>8}"


def row(label, taken, days, mid, mult, cents):
    s, avgtr = dollar_series(taken, days, mult, cents)
    h1 = s[[d for d in s.index if d < mid]].sum()
    h2 = s[[d for d in s.index if d >= mid]].sum()
    f = perf(s)
    print(f"  {label:<24}{len(taken):>7}{f['pnl']:>+9,.0f}{f['sharpe']:>8.2f}{f['maxdd']:>9,.0f}"
          f"   {h1:>+8,.0f}{h2:>+8,.0f}{avgtr:>+8.1f}")


def run_window(w):
    all_bars, days, present, _cached, closes = load(w)
    elig = trend_eligibility(closes, present, days)
    buckets = bucket(all_bars, present)
    tz = all_bars.index.get_level_values(1).tz
    eod_ns = {d: pd.Timestamp.combine(d, EOD).tz_localize(tz).value for d in days}
    mid = sorted(days)[len(days) // 2]
    prior = prior_vol_flags(closes, days)
    half = {d: (0.5 if prior[d] else 1.0) for d in days}

    # RE-BACKTEST with the later cutoff to get post-11:30 entries
    p = Params(or_minutes=15, target_r=2.0, risk_per_trade=100.0, max_position_pct=0.25,
               max_position_dollars=10_000.0, no_entry_after_time=LATE_CUTOFF)
    raw, _ = run_backtest(all_bars, days, present, p, STARTING_EQUITY)
    trail = apply_filter([t for t in reexit(raw, buckets, POLICIES["trail_1R"], eod_ns)
                          if t.side == "long"], elig)
    base = tight(trail)
    cents = TARGET_MEDIAN_R * statistics.median(risk_ps(t) for t in trail) / 2.0   # ALL-trades basis

    print(f"\n========== {w}d  (tight-OR<= {TIGHT}%, entries to {LATE_CUTOFF.strftime('%H:%M')}, "
          f"real cap-aware $, OOS {mid}) ==========")
    print(HEAD); print("  " + "-" * (len(HEAD) - 2))
    print("  -- cumulative cutoff (does extending past 11:30 help?) --")
    for cut, lab in [(690, "<=11:30 (current)"), (750, "<=12:30"), (780, "<=13:00"), (900, "<=15:00 (all)")]:
        row(lab, portfolio([t for t in base if et_min(t) <= cut], CAP), days, mid, half, cents)
    print("  -- marginal LATE-only buckets (are the new trades good on their own?) --")
    for lo, hi, lab in [(690, 750, "11:30-12:30"), (750, 840, "12:30-14:00"), (840, 900, "14:00-15:00")]:
        sub = [t for t in base if lo < et_min(t) <= hi]
        row(lab, portfolio(sub, CAP), days, mid, half, cents)


def main():
    for w in WINDOWS:
        run_window(w)
    print("\nReads: extending the cutoff WINS if the cumulative <=13:00/<=15:00 rows beat <=11:30 on")
    print("Sharpe AND PnL in BOTH windows, and the marginal late-only buckets are positive on their")
    print("own. If late entries dilute Sharpe or run negative, keep 11:30. Late trades = free volume.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
