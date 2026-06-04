"""Can a MARKET-regime gate (not equity-curve streaks) tell us when to pause ORB?

Streak/equity-curve timing was a dead end (daily P&L autocorrelation ~0). This
tests the more promising cousin: is there a *market state* — measured from SPY,
lookahead-free — in which ORB systematically does better or worse, so we can
stand down on the bad days?

Gates (each computed from SPY daily closes STRICTLY BEFORE the session):
  vol_high : 20d realized vol > its trailing-126d median  (breakouts like vol)
  trend_up : SPY close > 200d SMA                          (risk-on tape)
  combo    : vol_high AND trend_up
  (+ the inverses, as a sanity check — if a gate helps, its inverse should hurt)

Method: take the live-like trade stream (refill + the live 200d trend filter),
then PAUSE every trade on unfavorable-regime days and score the result over the
full window + both OOS halves. The decisive control is RANDOM-PAUSE: skip the
same NUMBER of days at random (avg over seeds). A gate only earns its keep if it
beats random-pause on PnL AND Sharpe — i.e. it knows *which* days to skip, not
just that trading fewer days changes the numbers.

Needs the caches from compare_selection.py / compare_norefill_trend.py (minute,
trades, daily). Run:
    .venv/Scripts/python.exe backtest/compare_regime_filter.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.universe_portfolio import perf, portfolio  # noqa: E402
from backtest.compare_selection import _tday  # noqa: E402
from backtest.compare_norefill_trend import trend_eligibility, apply_filter  # noqa: E402

WINDOWS = [730, 180]
CAP = 8
SEEDS = 25
VOL_WIN = 20          # realized-vol lookback (days)
VOL_MED_WIN = 126     # trailing window for the vol threshold (~6 months)
SMA_WIN = 200


def load(w):
    bars = pickle.load(open(ROOT / "backtest" / f".bars_cache_univ_{w}d.pkl", "rb"))
    trades = pickle.load(open(ROOT / "backtest" / f".bars_cache_trades_{w}d.pkl", "rb"))
    closes = pickle.load(open(ROOT / "backtest" / f".bars_cache_daily_{w}d.pkl", "rb"))
    all_bars, days = bars["bars"], bars["days"]
    present = sorted(all_bars.index.get_level_values(0).unique())
    return all_bars, days, present, trades, closes


def regime_flags(closes: pd.DataFrame, trading_days):
    """Per session date: favorable (=trade) booleans, using only prior closes.

    The signal from the first pass: ORB does BETTER in calm tape, so the gates
    here PAUSE on high-vol days (favorable = low realized vol), tested at a few
    thresholds for robustness, plus an uptrend gate and the combination. NaN
    (insufficient history) -> True (fail-open: trade).
    """
    spy = closes["SPY"].dropna().sort_index()
    ret = spy.pct_change()
    vol = ret.rolling(VOL_WIN).std()
    sma = spy.rolling(SMA_WIN, min_periods=100).mean()
    f = pd.DataFrame(index=spy.index)
    for q in (0.4, 0.5, 0.6):
        thr = vol.rolling(VOL_MED_WIN, min_periods=40).quantile(q)
        f[f"vol<q{int(q * 100)}"] = vol < thr          # trade only the calmer days
    f["trend_up"] = spy > sma
    f["vol<q50 & up"] = f["vol<q50"] & f["trend_up"]
    cols = list(f.columns)
    f = f.shift(1)  # decide today from yesterday's close -> lookahead-free
    out = {}
    for d in trading_days:
        row = f.loc[pd.Timestamp(d)] if pd.Timestamp(d) in f.index else None
        out[d] = {c: (True if row is None or pd.isna(row[c]) else bool(row[c])) for c in cols}
    return out, cols


def three(taken, days, mid):
    dset1 = [d for d in days if d < mid]
    dset2 = [d for d in days if d >= mid]
    sset = set(days)
    f = perf([t for t in taken if _tday(t) in sset], days)
    h1 = perf([t for t in taken if _tday(t) in set(dset1)], dset1)
    h2 = perf([t for t in taken if _tday(t) in set(dset2)], dset2)
    return f, h1, h2


def gate_perf(filtered, all_days, paused: set, mid):
    keep = set(all_days) - paused
    return three(portfolio([t for t in filtered if _tday(t) in keep], CAP), all_days, mid)


def mean_perf(ds):
    keys = ("n", "win", "pnl", "ret_pct", "max_dd", "sharpe")
    good = [d for d in ds if d.get("n", 0)]
    if not good:
        return {"n": 0}
    return {k: float(np.mean([d[k] for d in good])) for k in keys}


def random_pause(filtered, all_days, n_pause, mid):
    fs, h1s, h2s = [], [], []
    for s in range(SEEDS):
        rng = np.random.default_rng(s)
        idx = rng.choice(len(all_days), size=min(n_pause, len(all_days)), replace=False)
        paused = {all_days[i] for i in idx}
        a, b, c = gate_perf(filtered, all_days, paused, mid)
        fs.append(a); h1s.append(b); h2s.append(c)
    return mean_perf(fs), mean_perf(h1s), mean_perf(h2s)


HEAD = (f"{'config':<22}{'days':>5}{'PnL$':>10}{'Sharpe':>8}{'maxDD$':>10}   "
        f"{'h1 PnL':>8}{'h1 Sh':>7}   {'h2 PnL':>8}{'h2 Sh':>7}")


def prow(label, days_traded, full, h1, h2):
    def c(s, k, fmt):
        return format(s[k], fmt) if s.get("n", 0) else "—"
    print(f"{label:<22}{days_traded:>5}{c(full,'pnl','>+10,.0f')}{c(full,'sharpe','>8.2f')}"
          f"{c(full,'max_dd','>10,.0f')}   {c(h1,'pnl','>+8,.0f')}{c(h1,'sharpe','>7.2f')}   "
          f"{c(h2,'pnl','>+8,.0f')}{c(h2,'sharpe','>7.2f')}")


def run_window(w) -> dict:
    all_bars, days, present, trades, closes = load(w)
    mid = sorted(days)[len(days) // 2]
    elig = trend_eligibility(closes, present, days)
    filtered = apply_filter(trades, elig)
    flags, gates = regime_flags(closes, days)

    base = three(portfolio(filtered, CAP), days, mid)
    print(f"\n=== {w}d: {len(days)} sessions, OOS split {mid} ===")
    print(HEAD)
    print("-" * len(HEAD))
    prow("always-on (LIVE)", len(days), *base)

    res = {}
    for g in gates:
        fav = {d for d in days if flags[d][g]}
        paused = set(days) - fav
        res[g] = (gate_perf(filtered, days, paused, mid), len(fav), len(paused))
        prow(f"  pause-hi-vol: {g}", len(fav), *res[g][0])

    # random-pause control matched to the best gate's paused-day count (by full Sharpe)
    best = max(gates, key=lambda g: res[g][0][0].get("sharpe", -9) if res[g][0][0].get("n", 0) else -9)
    npause = res[best][2]
    rnd = random_pause(filtered, days, npause, mid)
    prow(f"RANDOM-pause {npause}d", len(days) - npause, *rnd)
    return {"base": base, "best": best, "best_perf": res[best][0], "rnd": rnd}


def main() -> int:
    out = {w: run_window(w) for w in WINDOWS}
    print("\n" + "=" * 64)
    print("VERDICT: a gate must beat always-on (Sharpe) AND random-pause (PnL+Sharpe),")
    print("both windows, both OOS halves.")
    print("=" * 64)
    for w in WINDOWS:
        r = out[w]
        bf, base, rnd = r["best_perf"], r["base"], r["rnd"]
        checks = []
        for i in range(3):
            checks.append(bf[i].get("n", 0) > 0
                          and bf[i]["sharpe"] >= base[i].get("sharpe", -9)
                          and bf[i]["pnl"] > rnd[i].get("pnl", 0)
                          and bf[i]["sharpe"] > rnd[i].get("sharpe", -9))
        seg = "  ".join(f"{s}:{'ok' if c else 'X'}" for s, c in zip(("full", "h1", "h2"), checks))
        print(f"  {w}d best gate '{r['best']}': {'PASS' if all(checks) else 'FAIL'}   [{seg}]")
    print("-" * 64)
    print("If FAIL: the regime gate doesn't reliably pick the bad days; pausing on it")
    print("just trades fewer days. Keep the risk-based daily_loss_cap; don't add a gate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
