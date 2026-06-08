"""Dig #2: does EXPANDING the universe add profitable tight-OR volume (net of worse fills)?

The cap doesn't bind and the cutoff can't extend, so more tight-OR trades can only come
from more NAMES. This concatenates the ~120 tier-2 names (fetch_universe_expanded.py) onto
the live ~100, re-backtests tight-OR (<=0.5%, trailing), and charges the NEW names HIGHER
slippage (they're less liquid) via a tiered cents model. The honest question: do the extra
trades earn their (worse) keep, and does the cap start binding (room to scale)?

Tier-1 (live names): base cents (median trade = 0.042R). Tier-2 (new names): TIER2_MULT x
base. Real $10k cap, vol-half, both windows + OOS. Expansion 'wins' if the full universe
beats the live-only universe on PnL while holding Sharpe, in both windows.

Run (needs EXP caches from fetch_universe_expanded.py; re-backtests ~220 names — heavy):
    .venv/Scripts/python.exe backtest/compare_tightOR_universe.py
"""
from __future__ import annotations

import math
import pickle
import statistics
import sys
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

import pandas as pd  # noqa: E402

WINDOWS = [730, 180]
TIGHT = 0.5
TARGET_MEDIAN_R = 0.042
NOTIONAL_CAP = 10_000.0
TIER2_MULT = 2.0          # new names slip ~2x the liquid tier (conservative)
CAPS = [16, 24, 32]
PARAMS = Params(or_minutes=15, target_r=2.0, risk_per_trade=100.0, max_position_pct=0.25,
                max_position_dollars=10_000.0, no_entry_after_time=__import__("datetime").time(11, 30))


def tight(trades):
    return [t for t in trades if or_pct(t) <= TIGHT]


def dser(taken, days, mult, cents_of):
    by = {}
    for t in taken:
        rps = risk_ps(t)
        shares = min(math.floor(RISK * mult.get(_tday(t), 1.0) / rps), math.floor(NOTIONAL_CAP / t.entry_price))
        if shares <= 0:
            continue
        by[_tday(t)] = by.get(_tday(t), 0.0) + (t.pnl_r * shares * rps - 2.0 * cents_of(t.symbol) * shares)
    return pd.Series({d: by.get(d, 0.0) for d in sorted(days)})


HEAD = f"{'config':<26}{'names':>6}{'trades':>7}{'PnL$':>9}{'Sharpe':>8}{'maxDD$':>9}   {'h1$':>8}{'h2$':>8}"


def run_window(w):
    all_bars, days, present, cached15, closes = load(w)
    tier1 = set(present)
    # bolt on the expansion caches
    exp_min = pickle.load(open(ROOT / "backtest" / f".bars_cache_univ_EXP_{w}d.pkl", "rb"))["bars"]
    exp_day = pickle.load(open(ROOT / "backtest" / f".bars_cache_daily_EXP_{w}d.pkl", "rb"))
    full_bars = pd.concat([all_bars, exp_min]).sort_index()
    full_present = sorted(full_bars.index.get_level_values(0).unique())
    full_closes = closes.join(exp_day[[c for c in exp_day.columns if c not in closes.columns]], how="outer")
    tier2 = set(full_present) - tier1

    buckets = bucket(full_bars, full_present)
    tz = full_bars.index.get_level_values(1).tz
    eod_ns = {d: pd.Timestamp.combine(d, EOD).tz_localize(tz).value for d in days}
    mid = sorted(days)[len(days) // 2]
    prior = prior_vol_flags(full_closes, days)
    half = {d: (0.5 if prior[d] else 1.0) for d in days}

    def trail_for(bars, pres, cl):
        raw, _ = run_backtest(bars, days, pres, PARAMS, STARTING_EQUITY)
        elig = trend_eligibility(cl, pres, days)
        return apply_filter([t for t in reexit(raw, buckets, POLICIES["trail_1R"], eod_ns)
                             if t.side == "long"], elig)

    live_trail = trail_for(all_bars, present, closes)        # live universe only
    full_trail = trail_for(full_bars, full_present, full_closes)
    base_cents = TARGET_MEDIAN_R * statistics.median(risk_ps(t) for t in tight(live_trail)) / 2.0
    cents_of = lambda sym: base_cents * (TIER2_MULT if sym in tier2 else 1.0)

    n_t2_trades = sum(1 for t in tight(full_trail) if t.symbol in tier2)
    print(f"\n========== {w}d  (tight-OR<= {TIGHT}%, trailing, tiered $ tier2={TIER2_MULT}x, OOS {mid}) ==========")
    print(f"  tier1 {len(tier1)} names, tier2 {len(tier2)} names | tight trades from tier2: {n_t2_trades}")
    print(HEAD); print("  " + "-" * (len(HEAD) - 2))

    def emit(label, trail, nm, cap):
        taken = portfolio(tight(trail), cap)
        s = dser(taken, days, half, cents_of)
        h1 = s[[d for d in s.index if d < mid]].sum(); h2 = s[[d for d in s.index if d >= mid]].sum()
        f = perf(s)
        print(f"  {label:<26}{nm:>6}{len(taken):>7}{f['pnl']:>+9,.0f}{f['sharpe']:>8.2f}{f['maxdd']:>9,.0f}   {h1:>+8,.0f}{h2:>+8,.0f}")

    emit("live univ (cap16)", live_trail, len(tier1), 16)
    for c in CAPS:
        emit(f"expanded (cap{c})", full_trail, len(full_present), c)


def main():
    for w in WINDOWS:
        run_window(w)
    print("\nReads: expansion wins if 'expanded' beats 'live univ' on PnL while holding Sharpe, in")
    print("BOTH windows — and if higher caps keep adding PnL (the cap now binds = room to scale).")
    print("If tier-2's 2x slippage eats the extra trades (Sharpe drops), the liquid-only universe wins.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
