"""Regime-conditioned shorts: do ORB shorts pay on HIGH-VOL days (where longs bleed)?

Standalone always-on shorts failed OOS (validate_short_oos.py) and are OFF. But
they were never conditioned on volatility — and we now know high-vol days are
exactly where the long book suffers (compare_regime_filter.py). Hypothesis: trade
the SHORT side only on high-vol days. This tests it NET OF COSTS from the start,
because shorts clear a higher bar than longs:

  cost/trade (R) = BASE (measured long entry+exit slippage ~0.042R)
                 + BORROW (short borrow/locate)
                 + HIVOL_EXTRA (fast/gappy fills on high-vol days)

Method: regenerate short-only ORB trades over the universe (cached), tag each day
calm vs high-vol by SPY 20d realized vol vs its trailing-126d median (lookahead-
free), and report short performance gross AND net on all / calm / high-vol days,
both windows + OOS. Longs on the same split shown for contrast (the premise).

A regime-short only earns a place if HIGH-VOL-day shorts are NET-positive (avgR &
Sharpe > 0) in both windows and both OOS halves. Otherwise it's dead.

Run (regenerates short trades on first run, ~minutes; then cached):
    .venv/Scripts/python.exe backtest/compare_regime_shorts.py
"""
from __future__ import annotations

import pickle
import sys
from datetime import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from strategies.orb import Params  # noqa: E402
from backtest.run_orb import STARTING_EQUITY, run_backtest  # noqa: E402
from backtest.universe_portfolio import portfolio  # noqa: E402
from backtest.compare_selection import _tday  # noqa: E402

WINDOWS = [730, 180]
RISK = 50.0
VOL_WIN, VOL_MED_WIN = 20, 126
# Cost model in R-units (baked in):
BASE_COST = 0.042     # measured long-side entry+exit slippage
BORROW = 0.010        # short borrow/locate per round trip (rough)
HIVOL_EXTRA = 0.030   # extra fill slippage on high-vol days


def load_bars(w):
    bars = pickle.load(open(ROOT / "backtest" / f".bars_cache_univ_{w}d.pkl", "rb"))
    closes = pickle.load(open(ROOT / "backtest" / f".bars_cache_daily_{w}d.pkl", "rb"))
    long_trades = pickle.load(open(ROOT / "backtest" / f".bars_cache_trades_{w}d.pkl", "rb"))
    all_bars, days = bars["bars"], bars["days"]
    present = sorted(all_bars.index.get_level_values(0).unique())
    return all_bars, days, present, closes, long_trades


def short_trades(w, all_bars, days, present):
    cache = ROOT / "backtest" / f".bars_cache_shorttrades_{w}d.pkl"
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    p = Params(or_minutes=15, target_r=2.0, risk_per_trade=100.0, max_position_pct=0.25,
               max_position_dollars=10_000.0, no_entry_after_time=time(11, 30),
               enable_long=False, enable_short=True)
    print(f"  regenerating short trades for {w}d (slow, one-time)...", flush=True)
    trades, _ = run_backtest(all_bars, days, present, p, STARTING_EQUITY)
    pickle.dump(trades, open(cache, "wb"))
    return trades


def hivol_days(closes, days) -> dict:
    spy = closes["SPY"].dropna().sort_index()
    vol = spy.pct_change().rolling(VOL_WIN).std()
    med = vol.rolling(VOL_MED_WIN, min_periods=40).median()
    flag = (vol > med).shift(1)        # decide a day from the prior close
    out = {}
    for d in days:
        v = flag.get(pd.Timestamp(d))
        out[d] = (bool(v) if pd.notna(v) else False)   # unknown -> treat as calm
    return out


def stats(trades, days, cost_fn):
    """Net daily-R series; cost_fn(t)->R cost subtracted per trade."""
    by = {}
    for t in trades:
        by[_tday(t)] = by.get(_tday(t), 0.0) + (t.pnl_r - cost_fn(t))
    s = pd.Series(by).reindex(sorted(days), fill_value=0.0) if by else pd.Series(0.0, index=sorted(days))
    dollar = s * RISK
    mu, sd = s.mean(), s.std()
    sharpe = (mu / sd * (252 ** 0.5)) if sd and sd > 0 else float("nan")
    gross = sum(t.pnl_r for t in trades)
    return {"n": len(trades), "avgR_gross": (gross / len(trades) if trades else 0.0),
            "avgR_net": (s.sum() / len(trades) if trades else 0.0),
            "sumR_net": s.sum(), "pnl": dollar.sum(), "sharpe": sharpe}


def row(label, s):
    if not s.get("n"):
        print(f"  {label:<22}{'(none)':>10}"); return
    print(f"  {label:<22}{s['n']:>6}{s['avgR_gross']:>+9.3f}{s['avgR_net']:>+9.3f}"
          f"{s['sumR_net']:>+8.1f}{s['pnl']:>+10,.0f}{s['sharpe']:>8.2f}")


def main():
    short_cost = lambda hv: (lambda t: BASE_COST + BORROW + (HIVOL_EXTRA if hv else 0.0))
    long_cost = lambda hv: (lambda t: BASE_COST + (HIVOL_EXTRA if hv else 0.0))
    verdict = {}
    for w in WINDOWS:
        all_bars, days, present, closes, longs = load_bars(w)
        shorts = short_trades(w, all_bars, days, present)
        hv = hivol_days(closes, days)
        mid = sorted(days)[len(days) // 2]
        nhv = sum(hv.values())
        print(f"\n=== {w}d: {len(days)} sessions ({nhv} high-vol), "
              f"{len(shorts)} short signals, {len(longs)} long ===")
        print(f"  {'subset':<22}{'n':>6}{'avgR_gr':>9}{'avgR_net':>9}{'sumR_n':>8}{'PnL$':>10}{'Sharpe':>8}")
        print("  " + "-" * 71)

        def sub(trades, pred):
            return [t for t in trades if pred(_tday(t))]
        # shorts: cost depends on whether that trade's day is hi-vol
        sh_all = stats(shorts, days, lambda t: BASE_COST + BORROW + (HIVOL_EXTRA if hv[_tday(t)] else 0.0))
        sh_calm = stats(sub(shorts, lambda d: not hv[d]), [d for d in days if not hv[d]],
                        lambda t: BASE_COST + BORROW)
        sh_hv = stats(sub(shorts, lambda d: hv[d]), [d for d in days if hv[d]],
                      lambda t: BASE_COST + BORROW + HIVOL_EXTRA)
        ln_hv = stats(sub(longs, lambda d: hv[d]), [d for d in days if hv[d]],
                      lambda t: BASE_COST + HIVOL_EXTRA)
        ln_calm = stats(sub(longs, lambda d: not hv[d]), [d for d in days if not hv[d]],
                        lambda t: BASE_COST)
        print("  SHORTS:")
        row("all days", sh_all)
        row("calm days", sh_calm)
        row("HIGH-VOL days", sh_hv)
        print("  LONGS (contrast):")
        row("calm days", ln_calm)
        row("HIGH-VOL days", ln_hv)

        # OOS halves on the key cell: shorts on high-vol days
        hv_days = [d for d in days if hv[d]]
        d1 = [d for d in hv_days if d < mid]
        d2 = [d for d in hv_days if d >= mid]
        sc = lambda t: BASE_COST + BORROW + HIVOL_EXTRA
        h1 = stats(sub(shorts, lambda d: hv[d] and d < mid), d1, sc)
        h2 = stats(sub(shorts, lambda d: hv[d] and d >= mid), d2, sc)
        print("  HIGH-VOL shorts OOS:")
        row("  first half", h1)
        row("  second half", h2)
        verdict[w] = (sh_hv, h1, h2)

    print("\n" + "=" * 64)
    print("VERDICT: high-vol-day shorts must be NET-positive (avgR_net & Sharpe > 0)")
    print(f"in full + both OOS halves, both windows. Cost = {BASE_COST}+{BORROW}+{HIVOL_EXTRA}R on hi-vol.")
    print("=" * 64)
    ok = True
    for w in WINDOWS:
        full, h1, h2 = verdict[w]
        segs = []
        for nm, s in (("full", full), ("h1", h1), ("h2", h2)):
            good = s.get("n", 0) > 0 and s["avgR_net"] > 0 and (s["sharpe"] > 0 or s["sharpe"] != s["sharpe"])
            segs.append(f"{nm}:{'ok' if good else 'X'}")
            ok &= good
        print(f"  {w}d  [{'  '.join(segs)}]")
    print("-" * 64)
    print("PURSUE regime-shorts" if ok else "DROP regime-shorts — not net-positive after costs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
