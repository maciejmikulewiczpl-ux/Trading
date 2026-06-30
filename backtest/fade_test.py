"""fade_test.py -- the screenshot runs FADE (VWAP-rejection) and CONTINUATION (momentum)
strategies side by side, both at 70%+ win rates. Those are contradictory edges; on history
we can ask which (if either) actually holds CROSS-SECTIONALLY on the liquid cache.

Two falsifiable reads (lookahead-free, daily bars):
  - EXTENSION: rank names by how stretched they are above their 20d MA at close(T); does the
    most-extended decile UNDER-perform next day (fade/mean-reversion edge -> shorts) or
    OUT-perform (momentum/continuation)?
  - TODAY'S MOVE: rank by today's return; do today's big gainers reverse or continue next day?

CAVEAT: the screenshot's edge (if real) is INTRADAY microstructure (1-min VWAP/order-flow)
that daily bars cannot see. This only tests the cross-sectional momentum-vs-reversion claim,
not the scalp. Liquid universe (same cache + lower-bound caveat as lottery_ignition).

Run:  .venv/Scripts/python.exe backtest/fade_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtest.lottery_ignition import load_cache, MIN_NAMES_PER_DAY, MIN_PRICE  # noqa: E402


def build(data: dict) -> pd.DataFrame:
    frames = []
    for sym, df in data["symbols"].items():
        if df is None or len(df) < 40:
            continue
        d = df.sort_index()
        close = d["Close"]
        ma20 = close.rolling(20, min_periods=20).mean()
        f = pd.DataFrame({
            "sym": sym, "close": close,
            "ext": close / ma20 - 1.0,                       # stretch above 20d MA (at T)
            "ret0": close / close.shift(1) - 1.0,            # today's move (at T)
            "next_ret": close.shift(-1) / close - 1.0,       # label: T -> T+1 (the ONLY shift(-1))
        }, index=close.index)
        f["date"] = f.index
        frames.append(f)
    p = pd.concat(frames, ignore_index=True).dropna(subset=["ext", "ret0", "next_ret"])
    return p[p["close"] >= MIN_PRICE]


def decile_table(panel: pd.DataFrame, by: str, label: str) -> None:
    rows = []
    for _dt, g in panel.groupby("date"):
        if len(g) < MIN_NAMES_PER_DAY:
            continue
        g = g.copy()
        g["dec"] = pd.qcut(g[by].rank(method="first"), 10, labels=False, duplicates="drop")
        rows.append(g)
    p = pd.concat(rows, ignore_index=True)
    print(f"\n=== next-day return by {label} decile ===")
    print(f"  {'decile':>14}{'n':>9}{'avg next_ret':>15}{'win%':>8}")
    grp = p.groupby("dec")
    for dec, gg in grp:
        nm = ("D1 (lowest)" if dec == 0 else "D10 (highest)" if dec == 9 else f"D{dec+1}")
        print(f"  {nm:>14}{len(gg):>9,}{gg['next_ret'].mean()*100:>+14.3f}%"
              f"{(gg['next_ret'] > 0).mean()*100:>7.0f}%")
    top = grp.get_group(9)["next_ret"].mean() * 100
    bot = grp.get_group(0)["next_ret"].mean() * 100
    allm = p["next_ret"].mean() * 100
    print(f"  spread D10-D1 = {top - bot:+.3f}%/day  (all-names avg {allm:+.3f}%)")
    if by == "ext":
        verb = "CONTINUATION (extended keeps rising)" if top > allm else "FADE (extended reverts)"
    else:
        verb = "CONTINUATION (winners keep running)" if top > allm else "FADE (winners revert)"
    print(f"  -> most-{label} names: {verb}")


def main() -> int:
    print("=== fade_test: does the liquid cross-section FADE or CONTINUE? (liquid lower bound) ===")
    print("Screenshot ran fade AND continuation both at 70%+ WR -- both can't be a real edge.\n")
    panel = build(load_cache())
    print(f"panel: {len(panel):,} rows over {panel['date'].nunique():,} days")
    decile_table(panel, "ext", "extension-above-20dMA")
    decile_table(panel, "ret0", "today's-move")
    print("\nREAD: a real FADE-short edge needs the most-extended/biggest-gainer decile to "
          "UNDER-perform meaningfully next day. If they continue (or it's a wash), the "
          "screenshot's 70%+ fade win rate isn't a cross-sectional reality. Daily proxy only "
          "-- intraday VWAP/order-flow microstructure is invisible here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
