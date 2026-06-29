"""lottery_confluence.py -- does combining MECHANICAL signals beat one signal? And is a
premarket gap a "gap-and-go" or a "gap-fade"? A historical cross-sectional study on the
same liquid daily cache as lottery_ignition.py (NO fetching), so it has real statistical
power (years), unlike the ~2-week live sample.

The "confluence" question dissolved on 31 live trades; the MECHANICAL part of it
(momentum x gap) IS testable on history. The SOCIAL signals (WSB/StockTwits/Trends) are
NOT here -- no point-in-time history -- so they stay with the live forward test.

FRAME (mirrors the bot + lookahead-free):
  - board day T: ignition (0..4) computed STRICTLY from data <= close(T) -- the morning
    signal, exactly as lottery_ignition.py.
  - trade day T+1: you observe the GAP at the open (open(T+1)/close(T)-1) BEFORE entering,
    then capture the INTRADAY return open(T+1)->close(T+1) (~ the bot's 9:45->close W1).
  - So selection features = ignition(T) [known at T close] + gap(T+1) [known at T+1 open,
    before entry]; the label = intraday open->close [after entry]. No leakage.
  - winner = within-(trade-day) cross-sectional TOP DECILE of the intraday return.

LIQUID LOWER BOUND (same caveat as lottery_ignition): the cache is large/mid caps, not the
live micro-cap lottery universe. A positive result is a lower bound; a null doesn't kill
the live signal. The forward test decides.

Run (NO network):
    .venv/Scripts/python.exe backtest/lottery_confluence.py
    .venv/Scripts/python.exe backtest/lottery_confluence.py --perms 2000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtest.lottery_ignition import (load_cache, TOP_DECILE, MIN_NAMES_PER_DAY,  # noqa: E402
                                        MIN_PRICE, STREAK_LB, VOL_FAST, VOL_SLOW, HIGH_LB)


def build_panel(data: dict) -> pd.DataFrame:
    """ignition(T) board signal + gap(T+1) + intraday-label(T+1), all lookahead-free."""
    frames = []
    for sym, df in data["symbols"].items():
        if df is None or len(df) < HIGH_LB + 5:
            continue
        d = df[["Open", "High", "Low", "Close", "Volume"]].sort_index()
        op, close, vol = d["Open"], d["Close"], d["Volume"]

        # --- ignition components through close(T) (identical spec to lottery_ignition) ---
        up = (close > close.shift(1)).astype(int).values
        run = np.zeros(len(up), dtype=float)
        c = 0
        for i in range(len(up)):
            c = c + 1 if up[i] == 1 else 0
            run[i] = c
        streak = pd.Series(run, index=close.index).clip(upper=STREAK_LB)
        volaccel = vol.rolling(VOL_FAST, min_periods=VOL_FAST).mean() / \
            vol.rolling(VOL_SLOW, min_periods=VOL_SLOW).mean().replace(0, np.nan)
        high_prox = close / close.rolling(HIGH_LB, min_periods=HIGH_LB).max().replace(0, np.nan)
        prevwin = ((close / close.shift(1) - 1.0) >= 0.02).astype(float)
        ignition = ((streak >= 3).astype(int) + (volaccel >= 1.5).astype(int)
                    + (high_prox >= 0.95).astype(int) + (prevwin >= 1).astype(int))

        # --- trade day T+1: gap (open vs prior close) is the pre-entry feature; the label
        #     is the intraday open->close move that an entry near the open would capture ---
        gap = op.shift(-1) / close - 1.0
        intraday = close.shift(-1) / op.shift(-1) - 1.0

        f = pd.DataFrame({"sym": sym, "close": close, "ignition": ignition,
                          "gap": gap, "label": intraday}, index=close.index)
        f["date"] = f.index
        frames.append(f)

    panel = pd.concat(frames, ignore_index=True).dropna(subset=["label", "gap", "ignition"])
    return panel[panel["close"] >= MIN_PRICE]


def label_winners(panel: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _dt, g in panel.groupby("date"):
        if len(g) < MIN_NAMES_PER_DAY:
            continue
        g = g.copy()
        g["winner"] = (g["label"] >= g["label"].quantile(TOP_DECILE)).astype(int)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def cell(panel, mask, base) -> tuple:
    sub = panel[mask]
    if len(sub) == 0:
        return 0, float("nan"), float("nan"), float("nan")
    rate = sub["winner"].mean()
    return len(sub), rate, (rate / base if base else float("nan")), sub["label"].mean() * 100


def permutation_p(panel, mask, n_perms, rng) -> float:
    """Within-day label shuffle (hypergeometric) for an arbitrary selection mask."""
    win = (panel["winner"] == 1).values
    sel = mask.values
    if sel.sum() == 0:
        return float("nan")
    obs = int((sel & win).sum())
    Ns, Ks, ns = [], [], []
    for _, g in panel.groupby("date"):
        Ns.append(len(g)); Ks.append(int((g["winner"] == 1).sum()))
        ns.append(int(mask.loc[g.index].sum()))
    Ns, Ks, ns = np.array(Ns), np.array(Ks), np.array(ns)
    ge = sum(1 for _ in range(n_perms)
             if rng.hypergeometric(Ks, Ns - Ks, ns).sum() >= obs)
    return (ge + 1) / (n_perms + 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260629)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print("=== lottery_confluence: does momentum x gap beat momentum alone? (liquid cache) ===")
    print("CAVEAT: liquid large/mid-cap universe -- a LOWER BOUND on the live micro-cap edge.\n")
    panel = label_winners(build_panel(load_cache()))
    base = panel["winner"].mean()
    nd = panel["date"].nunique()
    print(f"panel: {len(panel):,} (name,trade-day) rows over {nd:,} days | base winner rate "
          f"{base*100:.2f}% (~top decile)\n")

    def row(name, mask):
        n, r, lift, avg = cell(panel, mask, base)
        print(f"  {name:30} n={n:>7,}  win {r*100:5.2f}%  lift {lift:4.2f}x  avg-intraday {avg:+.2f}%")
        return lift

    print("SINGLE signals:")
    ig = panel["ignition"] >= 3
    row("ignition>=3", ig)
    row("gap > 0 (any gap up)", panel["gap"] > 0)
    row("gap >= +2%", panel["gap"] >= 0.02)
    row("gap >= +5%", panel["gap"] >= 0.05)

    print("\nGAP DIRECTION (gap-and-go vs gap-fade — read avg-intraday):")
    row("gap <= -2% (gap down)", panel["gap"] <= -0.02)
    row("gap -2%..+2% (flat)", (panel["gap"] > -0.02) & (panel["gap"] < 0.02))
    row("gap +2%..+5%", (panel["gap"] >= 0.02) & (panel["gap"] < 0.05))
    row("gap >= +5%", panel["gap"] >= 0.05)

    print("\nCONFLUENCE (does the gap ADD to ignition?):")
    ig_lift = row("ignition>=3 (alone)", ig)
    up_lift = row("ignition>=3 AND gap>0", ig & (panel["gap"] > 0))
    dn_lift = row("ignition>=3 AND gap<=0", ig & (panel["gap"] <= 0))
    row("ignition>=3 AND gap>=+2%", ig & (panel["gap"] >= 0.02))

    print(f"\nPermutation test ({args.perms} within-day shuffles):")
    for name, mask in [("ignition>=3", ig),
                       ("ignition>=3 AND gap>0", ig & (panel["gap"] > 0)),
                       ("gap>=+5%", panel["gap"] >= 0.05)]:
        p = permutation_p(panel, mask, args.perms, rng)
        print(f"  {name:28} p = {p:.4f}  ({'SIGNIFICANT' if p < 0.05 else 'ns'})")

    print("\nREAD: confluence helps iff (ignition AND gap>0) lift is materially ABOVE "
          "ignition-alone. Gap-and-go iff high-gap avg-intraday is POSITIVE (fade if negative).")
    print("Liquid lower bound — the social signals + micro-caps are decided by the live test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
