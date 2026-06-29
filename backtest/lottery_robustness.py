"""lottery_robustness.py -- does the ignition edge survive (1) different MARKET REGIMES
(high-vol / down markets, not just the calm bull we've live-traded) and (2) TRANSACTION
COSTS? Historical, on the same liquid cache as lottery_ignition.py. Feeds the real-money
decision (we've only traded ~2 calm weeks live).

(1) REGIME: build a market proxy from the cross-sectional mean daily return, classify each
    day by volatility (20d realized vol vs its median) and trend (index vs its 100d MA),
    then recompute the ignition>=3 top-decile lift WITHIN each regime. The edge is durable
    only if it holds in high-vol AND down regimes, not just calm/bull.

(2) COST: the bot harvests via a trailing stop (asymmetric: ride winners, cut losers), so
    a naive "avg return - cost" understates it. We faithfully simulate the trailing-stop
    harvest on the daily bars (buy at the T+1 open on ignition>=3 names, 10% trail + T+3
    time-stop), then net out round-trip cost at 0/25/50/100 bps and see where it dies.

LIQUID LOWER BOUND (same caveat as lottery_ignition): large/mid caps, not the live
micro-cap universe. Directional; the live forward test decides.

Run (NO network):  .venv/Scripts/python.exe backtest/lottery_robustness.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtest.lottery_ignition import (load_cache, build_panel, label_winners,  # noqa: E402
                                        permutation_p)


def _lift(sub: pd.DataFrame) -> tuple:
    base = sub["winner"].mean()
    hi = sub[sub["ignition"] >= 3]
    if len(hi) == 0 or base == 0:
        return len(hi), float("nan"), base
    return len(hi), hi["winner"].mean() / base, base


def regime_section(panel: pd.DataFrame, perms: int, rng) -> None:
    # market proxy: cross-sectional mean of next_ret per day -> index, vol, trend
    mkt = panel.groupby("date")["next_ret"].mean().sort_index()
    idx = (1 + mkt).cumprod()
    vol20 = mkt.rolling(20).std()
    ma100 = idx.rolling(100).mean()
    reg = pd.DataFrame({"hi_vol": (vol20 > vol20.median()), "bull": (idx > ma100)})
    reg = reg.dropna()
    panel = panel.merge(reg, left_on="date", right_index=True, how="inner")

    print("=== (1) REGIME conditioning of the ignition>=3 lift ===")
    print("market proxy = cross-sectional mean daily return; vol=20d std vs median; "
          "trend=index vs 100d MA\n")
    print(f"  {'regime':24}{'days':>7}{'ig>=3 n':>10}{'lift':>8}{'perm p':>9}")
    cells = [("ALL", panel),
             ("low-vol (calm)", panel[~panel["hi_vol"]]),
             ("HIGH-vol", panel[panel["hi_vol"]]),
             ("bull (up trend)", panel[panel["bull"]]),
             ("BEAR (down trend)", panel[~panel["bull"]]),
             ("calm+bull", panel[(~panel["hi_vol"]) & panel["bull"]]),
             ("HIGH-vol+BEAR", panel[panel["hi_vol"] & (~panel["bull"])])]
    for name, sub in cells:
        if len(sub) == 0:
            print(f"  {name:24}{'--':>7}"); continue
        n_hi, lift, _b = _lift(sub)
        p = permutation_p(sub, perms, rng) if n_hi else float("nan")
        d = sub["date"].nunique()
        ls = f"{lift:.2f}x" if lift == lift else "n/a"
        ps = f"{p:.4f}" if p == p else "n/a"
        print(f"  {name:24}{d:>7,}{n_hi:>10,}{ls:>8}{ps:>9}")
    print("  READ: durable iff lift stays >1 (ideally significant) in HIGH-vol AND BEAR, "
          "not just calm/bull.\n")


def harvest_section(data: dict, panel: pd.DataFrame, trail: float, max_days: int) -> None:
    bars = {s: df.sort_index() for s, df in data["symbols"].items() if df is not None}
    grs = []
    ev = panel[panel["ignition"] >= 3][["sym", "date"]].itertuples(index=False)
    for sym, T in ev:
        df = bars.get(sym)
        if df is None:
            continue
        try:
            pos = df.index.get_loc(T)
        except KeyError:
            continue
        fut = df.iloc[pos + 1: pos + 2 + max_days]          # T+1 .. T+1+max_days
        if len(fut) < 2:
            continue
        o = fut["Open"].values; h = fut["High"].values
        lo = fut["Low"].values; cl = fut["Close"].values
        entry = float(o[0])
        if entry <= 0:
            continue
        hw, exit_px = entry, float(cl[-1])
        for i in range(len(fut)):
            stop = hw * (1 - trail / 100)
            if i > 0 and lo[i] <= stop:
                exit_px = stop; break
            hw = max(hw, h[i])
            if i >= max_days:
                exit_px = float(cl[i]); break
        grs.append(exit_px / entry - 1.0)
    g = np.array(grs)
    if g.size == 0:
        print("=== (2) COST sensitivity: no harvestable events ==="); return
    print(f"=== (2) COST sensitivity of the trailing-stop harvest "
          f"({trail:.0f}% trail, T+{max_days}) ===")
    print(f"buy ignition>=3 at the T+1 open, trailing-stop harvest; n={g.size:,} trades\n")
    print(f"  gross avg/trade {g.mean()*100:+.3f}%  win {np.mean(g > 0)*100:.1f}%  "
          f"median {np.median(g)*100:+.3f}%")
    print(f"  {'round-trip cost':>18}{'net avg/trade':>16}{'still +?':>10}")
    for bps in (0, 25, 50, 100):
        net = g - bps / 10000.0          # round-trip cost in return terms
        m = net.mean() * 100
        print(f"  {str(bps)+' bps':>18}{m:>+15.3f}%{('yes' if m > 0 else 'NO'):>10}")
    print("  READ: the trailing harvest is the bot's real engine (tail capture). Find the "
          "cost level where net avg/trade crosses 0 — that's the slippage budget.\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--trail", type=float, default=10.0)
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260629)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print("=== lottery_robustness: regime + cost survival of the ignition edge (liquid) ===")
    print("CAVEAT: liquid large/mid-cap lower bound; live micro-cap forward test decides.\n")
    data = load_cache()
    panel = label_winners(build_panel(data))
    print(f"panel: {len(panel):,} rows over {panel['date'].nunique():,} days\n")

    regime_section(panel, args.perms, rng)
    harvest_section(data, panel, args.trail, args.days)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
