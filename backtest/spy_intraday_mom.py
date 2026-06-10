"""SPY last-half-hour intraday momentum — candidate DECORRELATED 2nd engine.

Literature basis (overnight research 2026-06-10): Gao, Han, Li & Zhou,
"Market Intraday Momentum" (J. Financial Economics 2018; SSRN 2440866): on SPY
1993-2013, the FIRST half-hour return (measured from the prior close, so it
includes the overnight gap) significantly predicts the LAST half-hour return
(R^2 ~1.6%, rising to ~2.6% when combined with the 12th half-hour, 15:00-15:30).
Stronger on volatile/high-volume days. Mechanism: late-informed trading +
infrequent rebalancing near the close.

Why it matters for us: a one-trade-per-day, index-only, 25-minute-hold strategy
that fires at 15:30 ET is structurally DECORRELATED from morning ORB longs (the
meanrev 2nd-engine attempt failed on entry quality; this one has a JFE-published
entry). It would reuse the existing runner's plumbing (market orders, EOD flat).

Test, on cached SPY/QQQ minute bars (730d + 180d):
  fh        : long/short last half-hour in sign(first half-hour ret incl. gap)
  r12       : sign of the 15:00->15:30 return
  fh+r12    : trade only when both agree (the paper's strongest combo)
Entry 15:30 bar open proxy (15:30 bar close), exit 15:55 (our EOD-flat time, not
16:00 — conservative, gives up the final 5 min). $10k notional per trade, 2 bps
round-trip cost (tight index spreads + market orders).

PRE-REGISTERED BAR (exploratory — this is a NEW engine, not a tweak): report as
a candidate only if, in BOTH windows, an arm has positive net PnL, Sharpe >= 1.0,
AND |corr(daily $, tight-OR ORB daily $)| <= 0.30 (the decorrelation is the whole
point). Shipping would still need its own runner work + paper forward-test.

Run AFTER compare_sip_gates.py (reads its .daily_base_tightOR_{w}d.pkl for the
decorrelation check; skips that check if missing):
    .venv/Scripts/python.exe backtest/spy_intraday_mom.py
"""
from __future__ import annotations

import pickle
import sys
from datetime import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.compare_volpause import perf  # noqa: E402

WINDOWS = [730, 180]
SYMBOLS = ["SPY", "QQQ"]
NOTIONAL = 10_000.0
COST_BPS = 2.0          # round-trip, market orders on penny-spread index ETFs
T_FH_END = time(10, 0)
T_12_S, T_12_E = time(15, 0), time(15, 30)
T_EXIT = time(15, 55)


def session_marks(sym_bars):
    """Per day: prior close, px@10:00, px@15:00, px@15:30, px@15:55."""
    t = sym_bars.index.time
    rth = sym_bars[(t >= time(9, 30)) & (t < time(16, 0))]
    g = rth.groupby(rth.index.date)

    def px_at(tt):
        # last bar close strictly before tt = price at tt (1-min bars)
        sel = rth[rth.index.time < tt]
        return sel.groupby(sel.index.date)["close"].last()

    out = pd.DataFrame({
        "close": g["close"].last(),
        "p10": px_at(T_FH_END),
        "p15": px_at(T_12_S),
        "p1530": px_at(T_12_E),
        "p1555": px_at(T_EXIT),
    }).sort_index()
    out["prior_close"] = out["close"].shift(1)
    return out.dropna()


def run_symbol(sym, all_bars, w):
    m = session_marks(all_bars.xs(sym, level=0))
    r_fh = m["p10"] / m["prior_close"] - 1.0          # incl. overnight gap (paper spec)
    r_12 = m["p1530"] / m["p15"] - 1.0
    r_last = m["p1555"] / m["p1530"] - 1.0            # what we'd capture

    cost = COST_BPS / 1e4
    arms = {
        "fh": r_fh.apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0),
        "r12": r_12.apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0),
        "fh+r12": ((r_fh > 0) & (r_12 > 0)).astype(int) - ((r_fh < 0) & (r_12 < 0)).astype(int),
    }
    mid = m.index[len(m) // 2]
    print(f"\n  {sym} ({len(m)} sessions)  "
          f"{'arm':<8}{'n':>5}{'win%':>6}{'avg bps':>9}{'PnL$@10k':>10}{'Sharpe':>8}"
          f"{'maxDD$':>8}  {'h2 PnL':>8}")
    out = {}
    for name, sig in arms.items():
        ret = sig * r_last - sig.abs() * cost
        dollars = ret * NOTIONAL
        traded = sig != 0
        f = perf(dollars)
        h2 = perf(dollars[dollars.index >= mid])
        n = int(traded.sum())
        win = 100 * float((ret[traded] > 0).mean()) if n else 0.0
        avg = 1e4 * float(ret[traded].mean()) if n else 0.0
        print(f"  {'':9}{name:<8}{n:>5}{win:>5.0f}%{avg:>+9.2f}{f['pnl']:>+10,.0f}"
              f"{f['sharpe']:>8.2f}{f['maxdd']:>8,.0f}  {h2['pnl']:>+8,.0f}")
        out[name] = dollars
    return out


def run_window(w):
    bars = pickle.load(open(ROOT / "backtest" / f".bars_cache_univ_{w}d.pkl", "rb"))
    all_bars = bars["bars"]
    print(f"\n=== {w}d ===")
    series = {}
    for sym in SYMBOLS:
        series[sym] = run_symbol(sym, all_bars, w)
    # decorrelation vs the shipped ORB (daily $ from compare_sip_gates base arm)
    p = ROOT / "backtest" / f".daily_base_tightOR_{w}d.pkl"
    if p.exists():
        orb = pickle.load(open(p, "rb"))
        orb.index = pd.to_datetime(orb.index)
        print(f"\n  corr(daily $, tight-OR ORB daily $), {w}d:")
        for sym in SYMBOLS:
            for name, d in series[sym].items():
                dd = d.copy()
                dd.index = pd.to_datetime(dd.index)
                j = pd.concat([dd, orb], axis=1, join="inner")
                c = j.corr().iloc[0, 1] if len(j) > 20 else float("nan")
                print(f"    {sym} {name:<8}: {c:+.2f}  ({len(j)} common days)")
    else:
        print("  (no ORB daily series cached — run compare_sip_gates.py first)")


def main():
    for w in WINDOWS:
        run_window(w)
    print("\nPre-registered bar: net PnL > 0, Sharpe >= 1.0, |corr with ORB| <= 0.30,")
    print("in BOTH windows -> candidate 2nd engine (own runner + paper test required).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
