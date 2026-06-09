"""Digs #3 + #4: re-tune OR LENGTH and STOP PLACEMENT under the tight-OR lens.

Both were set on the OLD full population. Through the tight-OR (<=0.5%) + trailing lens:
  #3 OR LENGTH (or_minutes): a SHORTER opening range makes smaller ranges -> more trades
     qualify as tight (free volume) but noisier levels; a LONGER one, fewer/cleaner. 5/10/
     15/30 min.
  #4 STOP PLACEMENT (stop_offset_pct): we stop exactly at OR_low. A buffer BEYOND it
     (0.10/0.25/0.50 of the OR range) dodges the liquidity-sweep at the round number every
     retail stop sits on — fewer whipsaw stop-outs, at the cost of bigger risk/share. The
     tight-OR set is UNCHANGED by the stop (or_pct is range/price), so this is a clean test:
     same trades, different stop.

Honest apples-to-apples: slippage cents is a MARKET constant, calibrated ONCE on the
baseline (OR15/offset0, median trade = 0.042R) and applied to every variant — so a wider
stop doesn't get to pretend its bigger R makes slippage vanish. Real $10k cap, vol-half,
both windows + OOS. A variant ships only if it beats baseline on Sharpe AND maxDD in BOTH.

Run (RE-BACKTESTS each variant, ~45-60 min; one process to avoid thrashing):
    .venv/Scripts/python.exe backtest/compare_tightOR_params.py
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

import pandas as pd  # noqa: E402

import os
WINDOWS = [int(x) for x in os.environ.get("ORB_WINDOWS", "730,180").split(",")]
TIGHT = 0.5
TARGET_MEDIAN_R = 0.042
NOTIONAL_CAP = 10_000.0
CUTOFF = dtime(11, 30)
OR_LENGTHS = [int(x) for x in os.environ.get("ORB_ORLEN", "5,10,15,30").split(",")]
STOP_OFFSETS = [float(x) for x in os.environ.get("ORB_OFFSETS", "0.0,0.10,0.25,0.50").split(",")]


def tight(trades):
    return [t for t in trades if or_pct(t) <= TIGHT]


def dser(taken, days, mult, cents):
    by = {}
    for t in taken:
        rps = risk_ps(t)
        shares = min(math.floor(RISK * mult.get(_tday(t), 1.0) / rps), math.floor(NOTIONAL_CAP / t.entry_price))
        if shares <= 0:
            continue
        by[_tday(t)] = by.get(_tday(t), 0.0) + (t.pnl_r * shares * rps - 2.0 * cents * shares)
    return pd.Series({d: by.get(d, 0.0) for d in sorted(days)})


HEAD = f"{'variant':<22}{'trades':>7}{'PnL$':>9}{'Sharpe':>8}{'maxDD$':>9}   {'h1$':>8}{'h2$':>8}{'avg$/tr':>8}"


def run_window(w):
    all_bars, days, present, cached15, closes = load(w)
    elig = trend_eligibility(closes, present, days)
    buckets = bucket(all_bars, present)
    tz = all_bars.index.get_level_values(1).tz
    eod_ns = {d: pd.Timestamp.combine(d, EOD).tz_localize(tz).value for d in days}
    mid = sorted(days)[len(days) // 2]
    prior = prior_vol_flags(closes, days)
    half = {d: (0.5 if prior[d] else 1.0) for d in days}

    def backtest(or_min, offset):
        if or_min == 15 and offset == 0.0:
            raw = cached15
        else:
            p = Params(or_minutes=or_min, target_r=2.0, risk_per_trade=100.0, max_position_pct=0.25,
                       max_position_dollars=10_000.0, no_entry_after_time=CUTOFF, stop_offset_pct=offset)
            raw, _ = run_backtest(all_bars, days, present, p, STARTING_EQUITY)
        return apply_filter([t for t in reexit(raw, buckets, POLICIES["trail_1R"], eod_ns)
                             if t.side == "long"], elig)

    base = tight(backtest(15, 0.0))
    cents = TARGET_MEDIAN_R * statistics.median(risk_ps(t) for t in base) / 2.0   # fixed, baseline-calibrated

    def emit(label, trail):
        taken = portfolio(tight(trail), CAP)
        s = dser(taken, days, half, cents)
        h1 = s[[d for d in s.index if d < mid]].sum(); h2 = s[[d for d in s.index if d >= mid]].sum()
        f = perf(s); avg = s.sum() / len(taken) if taken else 0.0
        star = "  <- live" if label == "OR=15 offset=0.00" else ""
        print(f"  {label:<22}{len(taken):>7}{f['pnl']:>+9,.0f}{f['sharpe']:>8.2f}{f['maxdd']:>9,.0f}"
              f"   {h1:>+8,.0f}{h2:>+8,.0f}{avg:>+8.1f}{star}")

    print(f"\n========== {w}d  (tight-OR<= {TIGHT}%, trailing, real cap-aware $, OOS {mid}) ==========")
    print(HEAD); print("  " + "-" * (len(HEAD) - 2))
    print("  -- #3 OR LENGTH (offset 0) --")
    for orm in OR_LENGTHS:
        emit(f"OR={orm} offset=0.00", backtest(orm, 0.0))
    print("  -- #4 STOP OFFSET (OR=15) --")
    for off in STOP_OFFSETS:
        emit(f"OR=15 offset={off:.2f}", backtest(15, off))


def main():
    for w in WINDOWS:
        run_window(w)
    print("\nReads: a variant ships only if it beats the live (OR=15 offset=0.00) on Sharpe AND")
    print("maxDD in BOTH windows. Shorter OR adding trades w/o hurting Sharpe = free volume.")
    print("A stop buffer helping = tight-OR was dying to liquidity sweeps at the OR low.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
