"""Re-examine every OLD assumption through the NEW tight-OR lens (real cap-aware $).

Every parameter we ever tuned (trend filter, vol-dial, concurrency cap, entry cutoff)
was tuned on the FULL breakout population. The tight-OR filter changes the population,
so those choices may no longer be optimal — an idea that was net-negative across all
trades can be net-positive on the tight-OR subset. This re-tests each, holding the
tight-OR (<=0.5%) + trailing exit fixed, in honest cap-aware dollars, both windows + OOS.

Tested here (all cheap re-slices of one trailing run):
  A. TREND FILTER on vs off   - does tight-OR already select quality (filter redundant)?
  B. VOL-DIAL half/pause/off  - does tight-OR survive vol days on its own now?
  C. CONCURRENCY cap 8..inf   - does cap-16 even bind at ~5 tight trades/day?
  D. ENTRY CUTOFF <=10:30..   - is the edge front-loaded or even across the morning?
  E. DAY-OF-WEEK              - re-check the Thu-weak signal on the tight subset.
(The 11:30 cutoff can only be TIGHTENED here - the cache has no post-11:30 entries;
 loosening it needs a re-backtest, run separately.)

Run (loads minute bars + re-sims trailing exits once per window):
    .venv/Scripts/python.exe backtest/compare_tightOR_revisit.py
"""
from __future__ import annotations

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
from backtest.compare_or_range_capaware import dollar_series  # noqa: E402

import pandas as pd  # noqa: E402

WINDOWS = [730, 180]
TIGHT = 0.5
TARGET_MEDIAN_R = 0.042


def et_min(t):
    et = t.entry_time.tz_convert("America/New_York") if t.entry_time.tzinfo else t.entry_time
    return et.hour * 60 + et.minute, et.strftime("%a")


def tight(trades):
    return [t for t in trades if or_pct(t) <= TIGHT]


HEAD = f"{'variant':<26}{'trades':>7}{'PnL$':>9}{'Sharpe':>8}{'maxDD$':>9}   {'h1$':>8}{'h2$':>8}"


def row(label, taken, days, mid, mult, cents):
    s, _ = dollar_series(taken, days, mult, cents)
    h1 = s[[d for d in s.index if d < mid]].sum()
    h2 = s[[d for d in s.index if d >= mid]].sum()
    f = perf(s)
    print(f"  {label:<26}{len(taken):>7}{f['pnl']:>+9,.0f}{f['sharpe']:>8.2f}{f['maxdd']:>9,.0f}   {h1:>+8,.0f}{h2:>+8,.0f}")


def run_window(w):
    all_bars, days, present, trades, closes = load(w)
    elig = trend_eligibility(closes, present, days)
    buckets = bucket(all_bars, present)
    tz = all_bars.index.get_level_values(1).tz
    eod_ns = {d: pd.Timestamp.combine(d, EOD).tz_localize(tz).value for d in days}
    trail_all = [t for t in reexit(trades, buckets, POLICIES["trail_1R"], eod_ns) if t.side == "long"]
    trail_trend = apply_filter(trail_all, elig)
    mid = sorted(days)[len(days) // 2]
    prior = prior_vol_flags(closes, days)
    half = {d: (0.5 if prior[d] else 1.0) for d in days}
    pause = {d: (0.0 if prior[d] else 1.0) for d in days}
    off = {d: 1.0 for d in days}
    cents = TARGET_MEDIAN_R * statistics.median(risk_ps(t) for t in tight(trail_trend)) / 2.0

    base = tight(trail_trend)   # the shipped-candidate population
    print(f"\n========== {w}d  (tight-OR<= {TIGHT}%, trailing, real cap-aware $, OOS {mid}) ==========")
    print(HEAD); print("  " + "-" * (len(HEAD) - 2))
    print("  -- BASELINE --")
    row("trend+vol-half+cap16", portfolio(base, CAP), days, mid, half, cents)

    print("  -- A. trend filter --")
    row("trend OFF", portfolio(tight(trail_all), CAP), days, mid, half, cents)
    row("trend ON (baseline)", portfolio(base, CAP), days, mid, half, cents)

    print("  -- B. vol-dial --")
    row("vol OFF (full risk)", portfolio(base, CAP), days, mid, off, cents)
    row("vol half (baseline)", portfolio(base, CAP), days, mid, half, cents)
    row("vol full-pause", portfolio(base, CAP), days, mid, pause, cents)

    print("  -- C. concurrency cap --")
    for c in [8, 12, 16, 24, 9999]:
        row(f"cap {c}", portfolio(base, c), days, mid, half, cents)

    print("  -- D. entry cutoff (tighten only) --")
    for cut, lab in [(630, "<=10:30"), (645, "<=10:45"), (660, "<=11:00"), (690, "<=11:30 (all)")]:
        sub = [t for t in base if et_min(t)[0] <= cut]
        row(lab, portfolio(sub, CAP), days, mid, half, cents)

    print("  -- E. day-of-week (baseline, taken) --")
    taken = portfolio(base, CAP)
    by = {}
    for t in taken:
        _, dow = et_min(t)
        by.setdefault(dow, []).append(t)
    for d in ["Mon", "Tue", "Wed", "Thu", "Fri"]:
        sub = by.get(d, [])
        s, _ = dollar_series(sub, days, half, cents)
        print(f"     {d}  n={len(sub):>4}  PnL ${s.sum():>+7,.0f}")


def main():
    for w in WINDOWS:
        run_window(w)
    print("\nReads: a variant 'wins' only if it beats baseline on Sharpe+maxDD in BOTH windows.")
    print("Trend-OFF adding trades w/o hurting Sharpe = filter redundant (free volume to scale).")
    print("vol-OFF not worse = tight-OR self-protects. Cap with rising trades = it binds (room).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
