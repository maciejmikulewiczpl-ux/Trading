"""Dig #2: does EXPANDING the universe add profitable tight-OR volume (net of worse fills)?

The cap doesn't bind and the cutoff can't extend, so more tight-OR trades can only come
from more NAMES. Memory-frugal (7.9GB-safe): tier-2 trailing trades are pre-built in
batches by build_tier2_trades.py; here we just MERGE them with tier-1 at the trade level
(no big-bar concat) and price the new names at HIGHER slippage (they're less liquid).

Tier-1 (live names): base cents (median tight trade = 0.042R). Tier-2 (new names):
TIER2_MULT x base. Real $10k cap, vol-half, both windows + OOS. Expansion 'wins' if the
full universe beats live-only on PnL while holding Sharpe in both windows, and higher caps
keep adding PnL (the cap now binds = room to scale).

Run AFTER build_tier2_trades.py:
    .venv/Scripts/python.exe backtest/compare_tightOR_universe.py
"""
from __future__ import annotations

import gc
import math
import pickle
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
TIGHT = 0.5
TARGET_MEDIAN_R = 0.042
NOTIONAL_CAP = 10_000.0
TIER2_MULT = 2.0
CAPS = [16, 24, 32]


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
    # tier-1 trailing trades from the existing cache (loads big bars once, then freed)
    all_bars, days, present, trades, closes = load(w)
    elig = trend_eligibility(closes, present, days)
    buckets = bucket(all_bars, present)
    tz = all_bars.index.get_level_values(1).tz
    eod_ns = {d: pd.Timestamp.combine(d, EOD).tz_localize(tz).value for d in days}
    tier1_trail = apply_filter([t for t in reexit(trades, buckets, POLICIES["trail_1R"], eod_ns)
                                if t.side == "long"], elig)
    mid = sorted(days)[len(days) // 2]
    prior = prior_vol_flags(closes, days)
    half = {d: (0.5 if prior[d] else 1.0) for d in days}
    tier1_names = set(present)
    del all_bars, buckets, trades; gc.collect()

    tier2_trail = pickle.load(open(ROOT / "backtest" / f".tier2_trail_{w}d.pkl", "rb"))
    tier2_names = {t.symbol for t in tier2_trail}
    full_trail = tier1_trail + tier2_trail

    base_cents = TARGET_MEDIAN_R * statistics.median(risk_ps(t) for t in tight(tier1_trail)) / 2.0
    cents_of = lambda sym: base_cents * (TIER2_MULT if sym in tier2_names else 1.0)
    n_t2 = sum(1 for t in tight(full_trail) if t.symbol in tier2_names)

    print(f"\n========== {w}d  (tight-OR<= {TIGHT}%, trailing, tiered $ tier2={TIER2_MULT}x, OOS {mid}) ==========")
    print(f"  tier1 {len(tier1_names)} names, tier2 {len(tier2_names)} names | tight trades from tier2: {n_t2}")
    print(HEAD); print("  " + "-" * (len(HEAD) - 2))

    def emit(label, trail, nm, cap):
        taken = portfolio(tight(trail), cap)
        s = dser(taken, days, half, cents_of)
        h1 = s[[d for d in s.index if d < mid]].sum(); h2 = s[[d for d in s.index if d >= mid]].sum()
        f = perf(s)
        print(f"  {label:<26}{nm:>6}{len(taken):>7}{f['pnl']:>+9,.0f}{f['sharpe']:>8.2f}{f['maxdd']:>9,.0f}   {h1:>+8,.0f}{h2:>+8,.0f}")

    emit("live univ (cap16)", tier1_trail, len(tier1_names), 16)
    for c in CAPS:
        emit(f"expanded (cap{c})", full_trail, len(tier1_names) + len(tier2_names), c)


def main():
    for w in WINDOWS:
        run_window(w)
    print("\nReads: expansion wins if 'expanded' beats 'live univ' on PnL while holding Sharpe in")
    print("BOTH windows, and higher caps keep adding PnL. If tier-2's 2x slippage eats the extra")
    print("trades (Sharpe drops), the liquid-only universe wins. (Note: live 10s poll-cycle limits")
    print("how many names the runner can actually watch — a deployment constraint, flagged separately.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
