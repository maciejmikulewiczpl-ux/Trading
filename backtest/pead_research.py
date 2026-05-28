"""Deeper analysis of the PEAD event table (reads backtest/pead_events.csv).

run_pead.py found that surprise magnitude is noise but the earnings GAP is the
one near-significant, monotonic factor (big gap-ups drift further up). This
OOS-validates the gap signal — the actual gate — plus a few combinations,
without re-fetching anything.

  .venv-openbb\\Scripts\\python.exe backtest\\pead_research.py
(any python with pandas/numpy works — no network, no yfinance needed.)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CSV = Path(__file__).with_name("pead_events.csv")
HOLDS = [10, 21]


def stat(s):
    s = s.dropna()
    if len(s) == 0:
        return (0, np.nan, np.nan, np.nan)
    mean = s.mean()
    se = s.std() / np.sqrt(len(s)) if len(s) > 1 else np.nan
    t = mean / se if se and se > 0 else np.nan
    return (len(s), mean, t, (s > 0).mean() * 100)


def line(label, s):
    n, m, t, w = stat(s)
    print(f"    {label:<22}{n:>5}{m:>+11.3f}{t:>+7.2f}{w:>6.1f}%")


def main():
    if not CSV.exists():
        print(f"Missing {CSV}; run run_pead.py first.", file=sys.stderr)
        return 1
    df = pd.read_csv(CSV, parse_dates=["edate", "entry_day"])
    beats = df[df["surprise_pct"] > 0].copy().sort_values("edate")
    mid = beats["edate"].iloc[len(beats) // 2]
    first, second = beats[beats["edate"] < mid], beats[beats["edate"] >= mid]
    print(f"Beats: {len(beats)}  | split at {mid.date()}  "
          f"(first {len(first)}, second {len(second)})")

    # ---- GAP factor: OOS top vs bottom tercile, both holds ----
    for hold in HOLDS:
        col = f"rel_{hold}"
        print(f"\n{'='*60}\nGAP factor, hold={hold}d — OOS top vs bottom tercile\n{'='*60}")
        print(f"    {'group':<22}{'n':>5}{'mean_rel%':>11}{'t':>7}{'win%':>7}")
        for name, part in [("FULL", beats), ("first half", first), ("second half", second)]:
            p = part.dropna(subset=["gap_pct", col]).copy()
            if len(p) < 30:
                print(f"  {name}: too few"); continue
            p["b"] = pd.qcut(p["gap_pct"], 3, labels=["low", "mid", "high"])
            print(f"  [{name}]")
            line("gap LOW (small/neg)", p[p["b"] == "low"][col])
            line("gap HIGH (big gap-up)", p[p["b"] == "high"][col])
            # long-high / short-low spread (the tradeable signal)
            spread = p[p["b"] == "high"][col].mean() - p[p["b"] == "low"][col].mean()
            print(f"    -> high-minus-low spread: {spread:+.3f}%")

    # ---- Combo: big gap-up AND down-regime (both looked good) ----
    print(f"\n{'='*60}\nCOMBO: gap-up tercile x regime, hold=10d\n{'='*60}")
    col = "rel_10"
    p = beats.dropna(subset=["gap_pct", col, "regime_up"]).copy()
    p["b"] = pd.qcut(p["gap_pct"], 3, labels=["low", "mid", "high"])
    print(f"    {'group':<22}{'n':>5}{'mean_rel%':>11}{'t':>7}{'win%':>7}")
    line("gap-HIGH, regime up", p[(p["b"] == "high") & (p["regime_up"] == True)][col])
    line("gap-HIGH, regime down", p[(p["b"] == "high") & (p["regime_up"] == False)][col])

    print("\nVERDICT NOTES:")
    print("- Gap signal is real only if BOTH halves show a positive high-minus-low")
    print("  spread with sane n. If it flips sign across halves -> noise, don't ship.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
