"""lottery_ignition.py -- can a price/volume IGNITION score predict next-day
cross-sectional top-decile gainers? A 10-year classification-lift study on the
existing .swing_daily_cache.pkl (NO fetching).

This is Track A of the lottery experiment (see experiments/lottery/README.md). It
tests, falsifiably, whether the mechanical "ignition" composite -- green-day streak +
volume acceleration + 52-week-high proximity + prior-day-winner momentum -- carries
information about WHICH names print the biggest moves the NEXT day.

LIQUID-UNIVERSE BOUND (important caveat): the cache is the ~liquid swing universe
(top-dollar-volume large/mid caps). The real lottery winners (sub-$2 squeezes, fresh
micro-cap meme spikes) are NOT in here. So this study can only show whether ignition
predicts the biggest gainers AMONG liquid names. A positive result here is a lower
bound on the live small/micro-cap edge; a null result does NOT kill the live signal,
because the live universe is exactly the part this cache can't see. The forward test
(experiments/lottery/) is the real arbiter; this is the cheap pre-flight.

CORRECTNESS (audited):
  - All ignition features for day T are computed STRICTLY from data <= T's close.
    vol_accel uses volume through T; streak uses closes through T; 52w-high proximity
    uses the trailing 252-session high through T. No same-day-of-T+1 leakage.
  - label = next_ret = close(T) -> close(T+1) return.
  - winner = within-day cross-sectional TOP DECILE of next_ret (so the base rate is
    ~10% by construction; lift is what matters).
  - The score for day T predicts T+1's winner flag.

Run (NO network):
    .venv/Scripts/python.exe backtest/lottery_ignition.py
    .venv/Scripts/python.exe backtest/lottery_ignition.py --perms 1000 --boot 1000
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "backtest" / ".swing_daily_cache.pkl"

TOP_DECILE = 0.90          # winner = next_ret in the top 10% that day
MIN_NAMES_PER_DAY = 30     # need a real cross-section to rank within a day
MIN_PRICE = 3.0            # ignore sub-$3 noise (and the cache has few anyway)
STREAK_LB = 5              # green-streak look-back cap
VOL_FAST, VOL_SLOW = 5, 20 # volume accel = mean(5d)/mean(20d)
HIGH_LB = 252              # 52-week high window


def load_cache() -> dict:
    with open(CACHE, "rb") as f:
        return pickle.load(f)


def build_panel(data: dict) -> pd.DataFrame:
    """Stack every symbol into one long DataFrame with lookahead-free ignition
    features for day T and the label next_ret (close T -> close T+1).

    Columns: date, sym, close, next_ret, ig_streak, ig_volaccel, ig_high_prox,
             ig_prevwin, ignition (0..4 integer count), score (continuous blend).
    """
    frames = []
    for sym, df in data["symbols"].items():
        if df is None or len(df) < HIGH_LB + 5:
            continue
        d = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        d = d.sort_index()
        close = d["Close"]
        vol = d["Volume"]

        # --- label: next-day return (close T -> close T+1). shift(-1) is the ONLY
        #     forward-looking column; it is the thing we predict, never a feature. ---
        next_ret = close.shift(-1) / close - 1.0

        # --- feature 1: green-day streak through T (consecutive up-closes, capped) ---
        up = (close > close.shift(1)).astype(int)
        # running streak length ending at T
        streak = up.copy()
        run = np.zeros(len(up), dtype=float)
        c = 0
        upv = up.values
        for i in range(len(upv)):
            c = c + 1 if upv[i] == 1 else 0
            run[i] = c
        streak = pd.Series(run, index=close.index).clip(upper=STREAK_LB)

        # --- feature 2: volume acceleration = mean vol last 5d / mean vol last 20d.
        #     Both windows END at T (inclusive) -> uses only data <= T. ---
        v_fast = vol.rolling(VOL_FAST, min_periods=VOL_FAST).mean()
        v_slow = vol.rolling(VOL_SLOW, min_periods=VOL_SLOW).mean()
        volaccel = v_fast / v_slow.replace(0, np.nan)

        # --- feature 3: 52w-high proximity = close(T) / trailing-252 high(incl T).
        #     ~1.0 means at/near the highs. ---
        high252 = close.rolling(HIGH_LB, min_periods=HIGH_LB).max()
        high_prox = close / high252.replace(0, np.nan)

        # --- feature 4: prior-day winner momentum = was T itself an up day of >=2%? ---
        ret_today = close / close.shift(1) - 1.0
        prevwin = (ret_today >= 0.02).astype(float)

        f = pd.DataFrame({
            "sym": sym,
            "close": close,
            "next_ret": next_ret,
            "ig_streak": streak,
            "ig_volaccel": volaccel,
            "ig_high_prox": high_prox,
            "ig_prevwin": prevwin,
        }, index=close.index)
        f["date"] = f.index
        frames.append(f)

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.dropna(subset=["next_ret", "ig_volaccel", "ig_high_prox"])
    panel = panel[panel["close"] >= MIN_PRICE]

    # ignition = integer count of binary triggers (the spec's "ignition>=3" composite):
    #   streak >= 3 green days, volaccel >= 1.5, high_prox >= 0.95, prevwin == 1
    panel["t_streak"] = (panel["ig_streak"] >= 3).astype(int)
    panel["t_volaccel"] = (panel["ig_volaccel"] >= 1.5).astype(int)
    panel["t_highprox"] = (panel["ig_high_prox"] >= 0.95).astype(int)
    panel["t_prevwin"] = (panel["ig_prevwin"] >= 1).astype(int)
    panel["ignition"] = (panel["t_streak"] + panel["t_volaccel"]
                         + panel["t_highprox"] + panel["t_prevwin"])

    # continuous score for percentile-bucket lift: blend of the four z-ish pieces
    # (computed per-day below). Here just keep the raw components.
    return panel


def label_winners(panel: pd.DataFrame) -> pd.DataFrame:
    """Within each date, flag the cross-sectional top-decile of next_ret as winner=1.
    Days with too few names are dropped (can't rank a thin cross-section)."""
    out = []
    for dt, g in panel.groupby("date"):
        if len(g) < MIN_NAMES_PER_DAY:
            continue
        thr = g["next_ret"].quantile(TOP_DECILE)
        g = g.copy()
        g["winner"] = (g["next_ret"] >= thr).astype(int)
        # per-day continuous score = mean of within-day percentile ranks of the 4 feats
        for col, dst in [("ig_streak", "p_streak"), ("ig_volaccel", "p_volaccel"),
                         ("ig_high_prox", "p_highprox"), ("ig_prevwin", "p_prevwin")]:
            g[dst] = g[col].rank(pct=True)
        g["score"] = g[["p_streak", "p_volaccel", "p_highprox", "p_prevwin"]].mean(axis=1)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def lift_table(panel: pd.DataFrame) -> tuple:
    base = panel["winner"].mean()
    # bucket by score quintile
    panel = panel.copy()
    panel["score_q"] = pd.qcut(panel["score"], 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    buckets = []
    for q, g in panel.groupby("score_q", observed=True):
        rate = g["winner"].mean()
        buckets.append((int(q), len(g), rate, rate / base if base else float("nan")))
    # ignition >= 3 bucket
    hi = panel[panel["ignition"] >= 3]
    ig_rate = hi["winner"].mean() if len(hi) else float("nan")
    ig_lift = (ig_rate / base) if base else float("nan")
    return base, buckets, (len(hi), ig_rate, ig_lift)


def per_year(panel: pd.DataFrame) -> list:
    panel = panel.copy()
    panel["year"] = pd.to_datetime(panel["date"]).dt.year
    rows = []
    for yr, g in panel.groupby("year"):
        base = g["winner"].mean()
        hi = g[g["ignition"] >= 3]
        if len(hi) == 0 or base == 0:
            rows.append((int(yr), len(g), base, float("nan"), float("nan")))
            continue
        rate = hi["winner"].mean()
        rows.append((int(yr), len(g), base, rate, rate / base))
    return rows


def permutation_p(panel: pd.DataFrame, n_perms: int, rng: np.random.Generator) -> float:
    """Shuffle the winner labels WITHIN each day, recompute the ignition>=3 hit COUNT,
    and ask how often the shuffled count >= observed. Within-day shuffle preserves each
    day's base rate and cross-sectional structure -- tests only the score->winner
    association, not the (trivial) 10% marginal.

    Implemented as a per-day HYPERGEOMETRIC draw, which is mathematically identical to a
    within-day label shuffle (the number of ig>=3 rows that land on winners, given each
    day's #winners and #ig>=3 rows) but vastly faster than O(perms x rows) shuffling.
    The observed statistic is the TOTAL count of (ig>=3 AND winner) across all days."""
    hi = (panel["ignition"] >= 3).values
    win = panel["winner"].values == 1
    if hi.sum() == 0:
        return float("nan")
    obs_count = int((hi & win).sum())

    # per-day (N total rows, K winners, n ig>=3 rows) for the hypergeometric
    days = panel.groupby("date")
    Ns, Ks, ns = [], [], []
    for _, g in days:
        Ns.append(len(g))
        Ks.append(int((g["winner"] == 1).sum()))
        ns.append(int((g["ignition"] >= 3).sum()))
    Ns = np.array(Ns); Ks = np.array(Ks); ns = np.array(ns)

    ge = 0
    for _ in range(n_perms):
        # draw, per day, #ig>=3-rows that are winners ~ Hypergeometric(K, N-K, n)
        drawn = rng.hypergeometric(Ks, Ns - Ks, ns).sum()
        if drawn >= obs_count:
            ge += 1
    return (ge + 1) / (n_perms + 1)


def bootstrap_ci(panel: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> tuple:
    """Resample DAYS with replacement; recompute ignition>=3 lift each time -> 95% CI.
    Resampling days (not rows) respects the within-day dependence. Vectorized: precompute
    each day's (wins_hi, n_hi, wins_all, n_all) once, then index with a bootstrap matrix."""
    hi_mask = (panel["ignition"] >= 3)
    win = (panel["winner"] == 1)
    g = panel.groupby("date")
    wins_hi = g.apply(lambda d: int(((d["ignition"] >= 3) & (d["winner"] == 1)).sum())).values.astype(float)
    n_hi = g.apply(lambda d: int((d["ignition"] >= 3).sum())).values.astype(float)
    wins_all = g["winner"].sum().values.astype(float)
    n_all = g.size().values.astype(float)
    nd = len(n_all)
    if nd == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, nd, size=(n_boot, nd))
    sum_wh = wins_hi[idx].sum(axis=1)
    sum_nh = n_hi[idx].sum(axis=1)
    sum_wa = wins_all[idx].sum(axis=1)
    sum_na = n_all[idx].sum(axis=1)
    good = (sum_nh > 0) & (sum_wa > 0)
    rate_hi = sum_wh[good] / sum_nh[good]
    rate_all = sum_wa[good] / sum_na[good]
    lifts = rate_hi / rate_all
    if lifts.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(lifts, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260612)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print("=== lottery_ignition: next-day top-decile prediction on liquid cache ===")
    print("CAVEAT: liquid swing universe only -- a LOWER BOUND on the live micro-cap edge.")
    data = load_cache()
    print(f"loaded cache: {len(data['symbols'])} symbols, run_date {data.get('run_date')}")

    panel = build_panel(data)
    panel = label_winners(panel)
    n_days = panel["date"].nunique()
    print(f"panel: {len(panel):,} (symbol,day) rows over {n_days:,} trading days, "
          f">= {MIN_NAMES_PER_DAY} names/day, price >= ${MIN_PRICE:.0f}")

    base, buckets, (n_hi, ig_rate, ig_lift) = lift_table(panel)
    print(f"\nBASE RATE (winner = top-decile next_ret): {base*100:.2f}%  "
          f"(by construction ~10%)")
    print("\nPer-score-quintile lift (continuous combined ignition percentile score):")
    print("  quintile     n      win-rate   lift")
    for q, n, rate, lift in buckets:
        print(f"  Q{q}      {n:>8,}   {rate*100:6.2f}%   {lift:4.2f}x")

    print(f"\nIGNITION >= 3 triggers (streak3 + volaccel1.5 + highprox.95 + prevwin2%):")
    print(f"  n={n_hi:,}   win-rate {ig_rate*100:.2f}%   LIFT {ig_lift:.2f}x")

    print("\nYear-by-year stability (ignition>=3 lift):")
    print("  year      n     base     ig>=3 rate   lift")
    for yr, n, b, rate, lift in per_year(panel):
        rate_s = f"{rate*100:6.2f}%" if rate == rate else "   n/a"
        lift_s = f"{lift:4.2f}x" if lift == lift else "  n/a"
        print(f"  {yr}   {n:>7,}   {b*100:5.2f}%   {rate_s}     {lift_s}")

    print(f"\nPermutation test ({args.perms} within-day label shuffles)...")
    p = permutation_p(panel, args.perms, rng)
    print(f"  p(shuffled ig>=3 hit-rate >= observed) = {p:.4f}  "
          f"({'SIGNIFICANT' if p < 0.05 else 'not significant'} at 0.05)")

    print(f"\nBootstrap 95% CI on ig>=3 lift ({args.boot} day-resamples)...")
    lo, hi = bootstrap_ci(panel, args.boot, rng)
    print(f"  lift 95% CI = [{lo:.2f}x, {hi:.2f}x]")

    print("\nVERDICT GUIDE: ignition 'works' on liquid names iff lift materially > 1.0,")
    print("p < 0.05, CI lower bound > 1.0, and the per-year lifts are mostly > 1.0.")
    print("Remember this is the LIQUID lower bound; the live micro-cap forward test decides.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
