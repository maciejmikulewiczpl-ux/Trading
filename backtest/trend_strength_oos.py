"""trend_strength_oos.py -- pre-registered OOS test of the TREND-STRENGTH drawdown filter.

The exploratory confluence_tail.py found that dropping the weakest-trend tight-OR trades (by distance
above the 200d SMA) cut drawdown ~57% for ~0 PnL. BUT that used (a) a percentile threshold computed
over the whole sample (lookahead), (b) applied post-cap (partitioning the taken book, not re-running
selection). This is the honest version:
  - ABSOLUTE threshold sma_dist >= X%  (a live gate can't know future percentiles)
  - applied as an ELIGIBILITY GATE before portfolio(cap) -> freed slots get reused (like the bot)
  - threshold CHOSEN ON TRAIN, scored on HELD-OUT TEST (both directions), cap-aware real-$ lens.

The existing trend filter is binary (prior close > SMA200 => sma_dist>0); this gate strengthens it to
sma_dist >= X. Ships only if a train-chosen X beats no-gate on the TEST half's Sharpe AND drawdown.

Run:
    .venv/Scripts/python.exe backtest/trend_strength_oos.py
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
from backtest.compare_norefill_trend import trend_eligibility, apply_filter, SMA_DAYS  # noqa: E402
from backtest.compare_or_range_capaware import dollar_series, TARGET_MEDIAN_R  # noqa: E402
from backtest.compare_or_range_realcost import or_pct, risk_ps  # noqa: E402
from backtest.compare_selection import _tday  # noqa: E402
from backtest.compare_volpause import prior_vol_flags, perf, RISK, CAP  # noqa: E402
from backtest.universe_portfolio import portfolio  # noqa: E402

OR_THRESH = 0.5
GRID = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0]   # absolute sma_dist% thresholds (0 = no gate)


def sma_dist_map(trail, closes):
    """{id(trade): dist-above-200SMA % at prior close} (no lookahead)."""
    sma = closes.rolling(SMA_DAYS).mean()
    out = {}
    for t in trail:
        d = pd.Timestamp(_tday(t))
        prior = closes.index[closes.index < d]
        v = np.nan
        if len(prior):
            last = prior[-1]
            pc = closes[t.symbol].get(last, np.nan)
            s = sma[t.symbol].get(last, np.nan)
            if pc == pc and s == s and s > 0:
                v = (pc - s) / s * 100.0
        out[id(t)] = v
    return out


def build(w):
    all_bars, days, present, trades, closes = load(w)
    elig = trend_eligibility(closes, present, days)
    buckets = bucket(all_bars, present)
    tz = all_bars.index.get_level_values(1).tz
    eod_ns = {d: pd.Timestamp.combine(d, EOD).tz_localize(tz).value for d in days}
    longs = [t for t in apply_filter(reexit(trades, buckets, POLICIES["trail_1R"], eod_ns), elig)
             if t.side == "long"]
    prior = prior_vol_flags(closes, days)
    mult = {d: (0.5 if prior[d] else 1.0) for d in days}
    # calibrate slippage on the FULL longs population (market constant, matches capaware) -- NOT the
    # tight-OR subset (whose smaller risk/share would undercharge cost and inflate PnL).
    cents = TARGET_MEDIAN_R * statistics.median(risk_ps(t) for t in longs) / 2.0
    trail = [t for t in longs if or_pct(t) <= OR_THRESH]
    sd = sma_dist_map(trail, closes)
    return trail, days, mult, cents, sd


def score(trail, days, mult, cents, sd, thr):
    """cap-aware perf over `days` for the gate sma_dist >= thr (nan = fail-open, keep)."""
    kept = [t for t in trail if (sd[id(t)] != sd[id(t)]) or sd[id(t)] >= thr]
    taken = portfolio(kept, CAP)
    taken = [t for t in taken if _tday(t) in set(days)]
    s, _ = dollar_series(taken, days, mult, cents)
    f = perf(s)
    return f, len(taken)


def run_window(w):
    trail, days, mult, cents, sd = build(w)
    mid = sorted(days)[len(days) // 2]
    h1 = [d for d in days if d < mid]
    h2 = [d for d in days if d >= mid]

    print(f"\n=== {w}d, cap-aware real-$, split at {mid} ===")
    # full-window grid (context)
    print("  full-window grid (context, in-sample):")
    print(f"    {'gate sma_dist>=':<16}{'trades':>7}{'PnL$':>9}{'Sharpe':>8}{'maxDD$':>9}")
    for thr in GRID:
        f, n = score(trail, days, mult, cents, sd, thr)
        tag = "  (no gate)" if thr == 0 else ""
        print(f"    >= {thr:>4.0f}%       {n:>7}{f['pnl']:>+9,.0f}{f['sharpe']:>8.2f}{f['maxdd']:>9,.0f}{tag}")

    for train, test, lbl in [(h1, h2, "train h1 -> test h2"), (h2, h1, "train h2 -> test h1")]:
        # choose threshold on TRAIN by Sharpe
        best = max(GRID, key=lambda thr: score(trail, train, mult, cents, sd, thr)[0]["sharpe"])
        f_base_te, n_base = score(trail, test, mult, cents, sd, 0.0)
        f_gate_te, n_gate = score(trail, test, mult, cents, sd, best)
        f_base_tr, _ = score(trail, train, mult, cents, sd, 0.0)
        f_gate_tr, _ = score(trail, train, mult, cents, sd, best)
        print(f"\n  [{lbl}]  train-chosen gate = sma_dist >= {best:.0f}%")
        print(f"    TRAIN: no-gate Sharpe {f_base_tr['sharpe']:+.2f} PnL {f_base_tr['pnl']:+,.0f} "
              f"DD {f_base_tr['maxdd']:,.0f}  ->  gate Sharpe {f_gate_tr['sharpe']:+.2f} "
              f"PnL {f_gate_tr['pnl']:+,.0f} DD {f_gate_tr['maxdd']:,.0f}")
        print(f"    TEST : no-gate Sharpe {f_base_te['sharpe']:+.2f} PnL {f_base_te['pnl']:+,.0f} "
              f"DD {f_base_te['maxdd']:,.0f}  ->  gate Sharpe {f_gate_te['sharpe']:+.2f} "
              f"PnL {f_gate_te['pnl']:+,.0f} DD {f_gate_te['maxdd']:,.0f}")
        dd_cut = f_base_te["maxdd"] - f_gate_te["maxdd"]
        pnl_gv = f_base_te["pnl"] - f_gate_te["pnl"]
        verdict = ("PASS (test Sharpe up AND DD down)" if
                   (f_gate_te["sharpe"] > f_base_te["sharpe"] and f_gate_te["maxdd"] < f_base_te["maxdd"])
                   else "FAIL (no clean OOS win)")
        print(f"    OOS: DD cut ${dd_cut:+,.0f}, PnL give-up ${pnl_gv:+,.0f}  =>  {verdict}")


def main():
    print("Pre-registered OOS test: trend-strength eligibility gate (sma_dist >= X%), cap-aware $.")
    for w in (730, 180):
        run_window(w)
    print("\nRead: SHIP only if the TRAIN-chosen gate beats no-gate on the TEST half's Sharpe AND")
    print("drawdown, in BOTH split directions. If the gate only helps in-sample or one direction,")
    print("the 57%-DD-cut was overfit to where the drawdown happened to land.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
