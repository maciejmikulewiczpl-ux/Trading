"""montecarlo_orb.py -- resampling risk analysis of the SHIPPED ORB config (ChatGPT review #3).

The point estimates (Sharpe, PnL, one maxDD) from a single historical path hide the RANGE of
outcomes the same edge can produce. This bootstraps the shipped-config daily P&L to answer the
questions that actually gate real money:
  - How bad can the drawdown get (95th-percentile worst), not just the one we happened to see?
  - What's the distribution of annualized return?
  - Longest losing streak to expect?
  - Prob-of-ruin: chance of an X% account drawdown at a given account size.

Method: DAY-level block bootstrap. We resample whole trading DAYS with replacement (preserving
same-day cross-position correlation -- the honest unit, since the cap-16 book takes several
correlated momentum trades per day) and lay them in random order to build many equity paths.
i.i.d. TRADE resampling would understate drawdown by breaking that intraday clustering.

Config = exactly what the live runner does: trend filter + tight-OR <=0.5% + cap-16 portfolio +
trailing exit (re-simulated from minute bars) + CAP-AWARE real-dollar sizing (shares limited by
BOTH $50 risk-target AND the $10k/position notional cap) + calibrated cents cost (1x) + vol-dial
half-risk. This is the compare_or_range_capaware.py lens -- the true-dollar P&L, NOT the pessimistic
realcost slippage-stress lens (which over-charges tight-OR trades by capping cost at 0.40R).

Run (loads minute bars + re-sims trailing exits; slower):
    .venv/Scripts/python.exe backtest/montecarlo_orb.py
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.compare_exits import load, bucket, reexit, POLICIES, EOD  # noqa: E402
from backtest.compare_norefill_trend import trend_eligibility, apply_filter  # noqa: E402
from backtest.compare_or_range_capaware import dollar_series, TARGET_MEDIAN_R  # noqa: E402
from backtest.compare_or_range_realcost import or_pct, risk_ps  # noqa: E402
from backtest.compare_volpause import prior_vol_flags, RISK, CAP  # noqa: E402
from backtest.universe_portfolio import portfolio  # noqa: E402

OR_THRESH = 0.5          # the shipped tight-OR cut
N_BOOT = 20000
START_CAPITAL = 25_000.0  # nominal real-money account for %-DD / CAGR / ruin framing (adjustable)
RUIN_LEVELS = [0.05, 0.10, 0.15, 0.20]  # account-drawdown thresholds for prob-of-ruin


def shipped_daily_pnl(w: int):
    """Cap-aware real-$ daily P&L series for the live config (the compare_or_range_capaware lens)."""
    all_bars, days, present, trades, closes = load(w)
    elig = trend_eligibility(closes, present, days)
    buckets = bucket(all_bars, present)
    tz = all_bars.index.get_level_values(1).tz
    eod_ns = {d: pd.Timestamp.combine(d, EOD).tz_localize(tz).value for d in days}
    trail = [t for t in apply_filter(reexit(trades, buckets, POLICIES["trail_1R"], eod_ns), elig)
             if t.side == "long"]
    prior = prior_vol_flags(closes, days)
    mult = {d: (0.5 if prior[d] else 1.0) for d in days}
    base_cents = TARGET_MEDIAN_R * statistics.median(risk_ps(t) for t in trail) / 2.0
    kept = [t for t in trail if or_pct(t) <= OR_THRESH]
    taken = portfolio(kept, CAP)
    daily, _avgtr = dollar_series(taken, days, mult, base_cents)   # pd.Series $, indexed by day
    return daily, len(taken), len(days)


def path_stats(daily_vals: np.ndarray):
    """maxDD ($, positive number), final PnL ($), longest losing streak (days) for one path."""
    equity = np.cumsum(daily_vals)
    running_max = np.maximum.accumulate(equity)
    dd = running_max - equity
    max_dd = float(dd.max()) if len(dd) else 0.0
    final = float(equity[-1]) if len(equity) else 0.0
    # longest run of negative days
    streak = best = 0
    for v in daily_vals:
        if v < 0:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return max_dd, final, best


def pct(a, q):
    return float(np.percentile(a, q))


def run_window(w: int):
    daily, n_taken, n_days = shipped_daily_pnl(w)
    vals = daily.to_numpy(dtype=float)
    years = w / 365.25

    # observed (single historical path)
    obs_dd, obs_final, obs_streak = path_stats(vals)

    rng = np.random.default_rng(20260630)
    n = len(vals)
    finals = np.empty(N_BOOT)
    dds = np.empty(N_BOOT)
    streaks = np.empty(N_BOOT)
    for i in range(N_BOOT):
        sample = vals[rng.integers(0, n, size=n)]   # resample days w/ replacement, random order
        d, f, s = path_stats(sample)
        dds[i] = d
        finals[i] = f
        streaks[i] = s

    print(f"\n=== {w}d window: {n_days} sessions, {n_taken} trades, ~{years:.1f}y | "
          f"start ${START_CAPITAL:,.0f}, RISK ${RISK:.0f}/trade, cap {CAP} ===")
    print(f"  OBSERVED path:   PnL ${obs_final:+,.0f}  maxDD ${obs_dd:,.0f} "
          f"({obs_dd/START_CAPITAL*100:.1f}%)  longest losing streak {obs_streak}d")
    print(f"  ANNUALIZED return on ${START_CAPITAL:,.0f}: "
          f"median {pct(finals,50)/years/START_CAPITAL*100:+.1f}%/yr  "
          f"[5th {pct(finals,5)/years/START_CAPITAL*100:+.1f}%, "
          f"95th {pct(finals,95)/years/START_CAPITAL*100:+.1f}%]")
    print(f"  TOTAL PnL:       median ${pct(finals,50):+,.0f}  "
          f"[5th ${pct(finals,5):+,.0f}, 95th ${pct(finals,95):+,.0f}]")
    print(f"  MAX DRAWDOWN:    median ${pct(dds,50):,.0f} ({pct(dds,50)/START_CAPITAL*100:.1f}%)  "
          f"| 95th-worst ${pct(dds,95):,.0f} ({pct(dds,95)/START_CAPITAL*100:.1f}%)  "
          f"| 99th ${pct(dds,99):,.0f} ({pct(dds,99)/START_CAPITAL*100:.1f}%)")
    print(f"  LOSING STREAK:   median {pct(streaks,50):.0f}d  | 95th {pct(streaks,95):.0f}d  "
          f"| worst {int(streaks.max())}d")
    print(f"  PROB-OF-RUIN (chance maxDD exceeds X% of ${START_CAPITAL:,.0f}):")
    for lv in RUIN_LEVELS:
        p = float((dds > lv * START_CAPITAL).mean())
        print(f"      >{lv*100:>4.0f}% drawdown: {p*100:5.1f}%")
    # prob of a losing 2yr overall
    print(f"  P(window net LOSS): {float((finals < 0).mean())*100:.1f}%")


def main():
    print("Monte Carlo (day-level block bootstrap) on the SHIPPED ORB config.")
    print(f"{N_BOOT:,} resamples. maxDD/streak use random-order day paths (preserves same-day corr).")
    for w in (730, 180):
        run_window(w)
    print("\nRead: the 95th-worst drawdown + prob-of-ruin size the real-money account (not the one")
    print("historical path). If 95th-worst DD is tolerable at START_CAPITAL, the sizing is safe; if")
    print("not, cut RISK or raise capital. Streak stat = the psychological drawdown to expect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
