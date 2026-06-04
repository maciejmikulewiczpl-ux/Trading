"""Pre-registered validation of ONE selection config (no grid scanning).

compare_selection.py showed (a) the live cap-8 first-come-with-refill mechanism
is fragile/negative over 2yr, (b) most of the "selection" gain is just trading
fewer names (random captures it), (c) only `or_rvol` beats random robustly in
BOTH windows, and (d) the best grid cell is unstable (overfitting). So instead
of shipping the best square, we PRE-REGISTER a single config from theory and
demand it clear a high bar before it earns a place in the live runner:

  PRE-REGISTERED CONFIG
    universe : top-12 names/day by or_rvol (opening-range relative volume),
               lookahead-free, computed at 09:45 ET. No gap signals.
    fill     : first-come, cap 8, NO slot-refill (a freed slot is not handed to
               a later low-quality breakout — the churn fix).

  PASS BAR (all must hold): PRE-REG beats BOTH the live mechanism AND random-12
  on PnL, in the full window AND both OOS halves, for BOTH the 180d and 730d
  windows. Anything less = do not ship.

Reference rows (not part of the bar): live cap-8-with-refill, and static-100
no-refill, to show how much comes from the churn fix vs the or_rvol filter.

Needs the bar caches from compare_selection.py (run it for 180d and with
SELECT_LOOKBACK_DAYS=730 first). Then:
    .venv/Scripts/python.exe backtest/validate_selection.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path
from time import time as _t

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from strategies.orb import Params  # noqa: E402
from datetime import time  # noqa: E402
from backtest.run_orb import STARTING_EQUITY, run_backtest  # noqa: E402
from backtest.universe_portfolio import perf, portfolio  # noqa: E402
from backtest.compare_selection import (  # noqa: E402
    build_signals, daily_cap, eligible_pool, filter_to_sets, topk_by_column, _tday,
)

WINDOWS = [180, 730]
K = 12               # pre-registered watchlist size
CAP = 8              # live concurrency cap
SEEDS = 25           # random-12 control samples
PARAMS = Params(or_minutes=15, target_r=2.0, risk_per_trade=100.0,
                max_position_pct=0.25, max_position_dollars=10_000.0,
                no_entry_after_time=time(11, 30))


def cache_path(w):
    return ROOT / "backtest" / f".bars_cache_univ_{w}d.pkl"


def perf_on(taken, days):
    """perf over `days`, scoring only the taken trades whose session is in `days`."""
    dset = set(days)
    return perf([t for t in taken if _tday(t) in dset], days)


def three_window_perf(taken, days, mid):
    d1 = [d for d in days if d < mid]
    d2 = [d for d in days if d >= mid]
    return perf_on(taken, days), perf_on(taken, d1), perf_on(taken, d2)


def mean_perf(dicts):
    keys = ("n", "win", "sum_r", "pnl", "ret_pct", "max_dd", "sharpe")
    good = [d for d in dicts if d.get("n", 0)]
    if not good:
        return {"n": 0}
    return {k: float(np.mean([d[k] for d in good])) for k in keys}


HEAD = (f"{'config':<26}{'PnL$':>11}{'Sharpe':>8}   "
        f"{'PnL h1':>9}{'Sh h1':>7}   {'PnL h2':>9}{'Sh h2':>7}")


def prow(label, full, h1, h2):
    def c(s, k, fmt):
        return format(s[k], fmt) if s.get("n", 0) else "—"
    print(f"{label:<26}{c(full,'pnl','>+11,.0f')}{c(full,'sharpe','>8.2f')}   "
          f"{c(h1,'pnl','>+9,.0f')}{c(h1,'sharpe','>7.2f')}   "
          f"{c(h2,'pnl','>+9,.0f')}{c(h2,'sharpe','>7.2f')}")


def run_window(w) -> dict:
    cp = cache_path(w)
    if not cp.exists():
        print(f"  ! no cache {cp.name} — run compare_selection.py for {w}d first.")
        return {}
    with open(cp, "rb") as f:
        d = pickle.load(f)
    all_bars, trading_days = d["bars"], d["days"]
    present = sorted(all_bars.index.get_level_values(0).unique())
    mid = sorted(trading_days)[len(trading_days) // 2]

    all_trades, _ = run_backtest(all_bars, trading_days, present, PARAMS, STARTING_EQUITY)
    sig = build_signals(all_bars, present)
    pool = eligible_pool(sig)

    # --- the three principals + two references ---
    live = three_window_perf(portfolio(all_trades, CAP), trading_days, mid)  # refill

    norefill_all = three_window_perf(
        daily_cap(all_trades, CAP, lambda dd, t: t.entry_time, False), trading_days, mid)

    orvol_sets = topk_by_column(sig, "or_rvol", K)
    prereg = three_window_perf(
        daily_cap(filter_to_sets(all_trades, orvol_sets), CAP, lambda dd, t: t.entry_time, False),
        trading_days, mid)

    # random-12, no-refill, averaged over seeds (full + each half)
    rfull, rh1, rh2 = [], [], []
    for seed in range(SEEDS):
        rng = np.random.default_rng(seed)
        rsets = {}
        for dd, syms in pool.items():
            rsets[dd] = set(syms) if len(syms) <= K else set(rng.choice(syms, K, replace=False))
        taken = daily_cap(filter_to_sets(all_trades, rsets), CAP, lambda dd, t: t.entry_time, False)
        a, b, c = three_window_perf(taken, trading_days, mid)
        rfull.append(a); rh1.append(b); rh2.append(c)
    rand = (mean_perf(rfull), mean_perf(rh1), mean_perf(rh2))

    print(f"\n=== {w}d window: {len(present)} names, {len(trading_days)} sessions, "
          f"OOS split {mid} ===")
    print(HEAD)
    print("-" * len(HEAD))
    prow("LIVE (cap8, refill)", *live)
    prow("static-100, no-refill", *norefill_all)
    prow(f"RANDOM-{K}, no-refill (x{SEEDS})", *rand)
    prow(f"PRE-REG or_rvol-{K}", *prereg)

    return {"live": live, "rand": rand, "prereg": prereg}


def main() -> int:
    t0 = _t()
    results = {}
    for w in WINDOWS:
        results[w] = run_window(w)

    # ---- pass/fail on the pre-registered bar ----
    print("\n" + "=" * 64)
    print("PRE-REGISTERED PASS BAR")
    print("PRE-REG must beat LIVE and RANDOM on PnL in full + both halves, both windows.")
    print("=" * 64)
    all_pass = True
    for w in WINDOWS:
        r = results.get(w)
        if not r:
            print(f"  {w}d: SKIPPED (no cache)"); all_pass = False; continue
        checks = []
        for i, seg in enumerate(("full", "h1", "h2")):
            pr, lv, rn = r["prereg"][i], r["live"][i], r["rand"][i]
            if pr.get("n", 0) == 0:
                checks.append(False); continue
            checks.append(pr["pnl"] > lv.get("pnl", 0) and pr["pnl"] > rn.get("pnl", 0))
        ok = all(checks)
        all_pass &= ok
        seg_str = "  ".join(f"{s}:{'ok' if c else 'X'}" for s, c in zip(("full", "h1", "h2"), checks))
        print(f"  {w}d: {'PASS' if ok else 'FAIL'}   [{seg_str}]")

    print("-" * 64)
    if all_pass:
        print("VERDICT: PASS — the pre-registered config clears the bar. Worth wiring")
        print("         into the live runner (behind a config flag, paper-first).")
    else:
        print("VERDICT: FAIL — does not robustly beat live AND random everywhere.")
        print("         Do NOT ship the watchlist change. (The churn fix alone may")
        print("         still be worth it — compare the no-refill reference row.)")
    print(f"\n({_t() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
