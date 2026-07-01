"""hype_gap_distribution.py -- quantify the OVERNIGHT GAP tail for the Hype book (reviews' #3 tail).

The Hype bot holds ~T+3 with a trailing stop. A trailing stop protects INTRADAY, but an overnight
gap-DOWN fills at the next open -- BELOW the stop -- so realized loss can exceed the intended stop.
The reviews (all 3) flagged this as the Hype bot's real tail risk. This measures it: pull daily
OHLC for the names the bot actually traded (top-3 combined_score) and characterize the overnight
gap distribution = open_t / close_{t-1} - 1, vs SPY as a liquid benchmark.

Then translate to an expected extra loss: for a position stopped by an overnight gap, the slippage
beyond the stop ~ the gap-down magnitude. We report the gap-down tail (p1/p5, freq of <-5/-10/-20%,
worst) and a crude expected-overnight-gap-loss = P(gap<0) * E[gap | gap<0], per name and pooled.

Run (needs yfinance):
    .venv-openbb/Scripts/python.exe backtest/hype_gap_distribution.py
"""
from __future__ import annotations

import json
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
PICKS = ROOT / "experiments" / "lottery" / "picks"


def traded_symbols():
    syms = set()
    for f in sorted(glob.glob(str(PICKS / "*.json"))):
        d = json.load(open(f))
        ranked = sorted([p for p in d["picks"] if p.get("combined_score") is not None],
                        key=lambda x: -x["combined_score"])
        for p in ranked[:3]:
            syms.add(p["symbol"])
    return sorted(syms)


def gaps_for(df: pd.DataFrame) -> np.ndarray:
    """Overnight gaps = open_t / close_{t-1} - 1, as a fraction."""
    if df is None or df.empty or "Open" not in df or "Close" not in df:
        return np.array([])
    o = df["Open"].to_numpy(dtype=float)
    c = df["Close"].to_numpy(dtype=float)
    prev_c = c[:-1]
    op = o[1:]
    m = (prev_c > 0) & np.isfinite(prev_c) & np.isfinite(op)
    return op[m] / prev_c[m] - 1.0


def tail_line(name, g, n_names=None):
    if len(g) == 0:
        print(f"  {name:<26} no data")
        return
    down = g[g < 0]
    exp_gap_loss = (len(down) / len(g)) * (down.mean() if len(down) else 0.0)
    extra = f"  [{n_names} names]" if n_names else ""
    print(f"  {name:<26}{len(g):>7}{np.median(np.abs(g))*100:>8.2f}{np.percentile(g,5)*100:>+8.1f}"
          f"{np.percentile(g,1)*100:>+8.1f}{g.min()*100:>+8.1f}"
          f"{100*(g<=-0.05).mean():>7.1f}{100*(g<=-0.10).mean():>7.1f}{100*(g<=-0.20).mean():>7.1f}"
          f"{exp_gap_loss*100:>+9.3f}{extra}")


def main():
    syms = traded_symbols()
    print(f"=== overnight-gap tail: {len(syms)} Hype-traded names + SPY, ~2y daily ===")
    data = yf.download(syms + ["SPY"], period="2y", interval="1d", group_by="ticker",
                       auto_adjust=False, progress=False, threads=True)

    hdr = (f"  {'name':<26}{'nights':>7}{'med|g|%':>8}{'p5%':>8}{'p1%':>8}{'worst%':>8}"
           f"{'<-5%':>7}{'<-10%':>7}{'<-20%':>7}{'E[gaploss]%':>10}")
    print("\n" + hdr); print("  " + "-" * (len(hdr) - 2))

    pooled = []
    spy_g = np.array([])
    per_name = []
    for s in syms + ["SPY"]:
        try:
            df = data[s] if isinstance(data.columns, pd.MultiIndex) else data
        except Exception:
            df = None
        g = gaps_for(df)
        if s == "SPY":
            spy_g = g
        else:
            pooled.append(g)
            if len(g):
                per_name.append((s, g))

    pooled_arr = np.concatenate(pooled) if pooled else np.array([])
    tail_line("HYPE POOLED", pooled_arr, n_names=len(per_name))
    tail_line("SPY (benchmark)", spy_g)
    print()
    # worst individual names by 1st-percentile gap-down
    worst = sorted(per_name, key=lambda x: np.percentile(x[1], 1))[:8]
    print("  worst 8 names by p1 gap-down:")
    for s, g in worst:
        tail_line(s, g)

    if len(pooled_arr) and len(spy_g):
        print(f"\nRead: HYPE median |gap| {np.median(np.abs(pooled_arr))*100:.2f}% vs SPY "
              f"{np.median(np.abs(spy_g))*100:.2f}% "
              f"(~{np.median(np.abs(pooled_arr))/max(np.median(np.abs(spy_g)),1e-9):.0f}x). "
              f"HYPE p1 gap-down {np.percentile(pooled_arr,1)*100:.1f}% vs SPY "
              f"{np.percentile(spy_g,1)*100:.1f}%.")
        print("A trailing stop does NOT cap this: an overnight gap-down fills at the open, below the")
        print("stop. E[gaploss] is the avg per-night drag from gaps; multiply by ~2-3 overnight holds")
        print("per Hype trade + position size for the expected gap bleed. This is why Hype needs")
        print("SMALL per-name notional + the liquidity/spread guard, not a tighter stop.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
