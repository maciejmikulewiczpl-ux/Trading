"""confluence_tail.py -- re-examine confluence through the TAIL, not the mean (DeepSeek review #9).

Confluence (require a 2nd signal to co-fire) was rejected 3x on MEAN R -- it never lifted average
return. But the reviewer's nuance: a 2nd signal might still cut the LEFT TAIL (the deep-loss
outliers / drawdown clusters) even if it doesn't move the mean. Tight-OR trailing losses are mostly
capped near -1R, but the cached tight-OR book has a real left tail (p1 ~ -2R, worst -41R gap-through)
and cluster drawdowns -- exactly what a tail filter would target.

Method: take the SHIPPED tight-OR taken book, compute 3 candidate 2nd-signals per trade from daily
closes (no lookahead -- uses only closes strictly before the session):
  - gap%      : overnight gap into the breakout = (OR_mid - prior_close)/prior_close
  - mom20     : symbol 20d return minus SPY 20d return at prior close (relative momentum)
  - sma_dist% : (prior_close - 200d SMA)/SMA (trend strength)
Split the taken trades at the signal's median. For each side report the TAIL: mean R, p5 R,
%<=-2R (deep losers), and the R-equity max drawdown. Then the decision metric: does dropping the
UNFAVORABLE side reduce the FULL book's drawdown (and at what cost to PnL/Sharpe)?

Ships only if a filter meaningfully cuts drawdown for little PnL give-up. Cached trades; fast.

Run:
    .venv/Scripts/python.exe backtest/confluence_tail.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.compare_or_range_realcost import load_cached, or_pct  # noqa: E402
from backtest.compare_selection import _tday  # noqa: E402
from backtest.compare_norefill_trend import SMA_DAYS, RET_DAYS  # noqa: E402
from backtest.compare_volpause import CAP  # noqa: E402
from backtest.universe_portfolio import portfolio  # noqa: E402

OR_THRESH = 0.5


def signals(taken, closes):
    """Per-trade dict of 2nd-signals using only closes strictly before the session (no lookahead)."""
    sma = closes.rolling(SMA_DAYS).mean()
    ret = closes.pct_change(RET_DAYS)
    out = []
    for t in taken:
        d = pd.Timestamp(_tday(t))
        prior = closes.index[closes.index < d]
        rec = {"t": t, "gap": np.nan, "mom20": np.nan, "sma_dist": np.nan}
        if len(prior):
            last = prior[-1]
            pc = closes[t.symbol].get(last, np.nan)
            if pc and pc == pc and pc > 0:
                or_mid = (t.or_high + t.or_low) / 2.0
                rec["gap"] = (or_mid - pc) / pc * 100.0
                s = sma[t.symbol].get(last, np.nan)
                if s == s and s > 0:
                    rec["sma_dist"] = (pc - s) / s * 100.0
                r = ret[t.symbol].get(last, np.nan)
                spy = ret["SPY"].get(last, np.nan) if "SPY" in ret.columns else np.nan
                if r == r and spy == spy:
                    rec["mom20"] = (r - spy) * 100.0
        out.append(rec)
    return out


def maxdd_R(trs):
    by = {}
    for t in trs:
        by[_tday(t)] = by.get(_tday(t), 0.0) + t.pnl_r
    if not by:
        return 0.0
    eq = np.cumsum([by[d] for d in sorted(by)])
    return float((np.maximum.accumulate(eq) - eq).max())


def tail_row(label, trs):
    r = np.array([t.pnl_r for t in trs]) if trs else np.array([0.0])
    return (f"  {label:<22}{len(trs):>6}{r.mean():>+8.3f}{np.percentile(r,5):>+8.2f}"
            f"{100*(r<=-2).mean():>7.0f}%{maxdd_R(trs):>9.1f}{r.sum():>+9.1f}")


def main():
    trades, closes, days = load_cached(730)
    kept = [t for t in trades if or_pct(t) <= OR_THRESH]
    taken = portfolio(kept, CAP)
    sig = signals(taken, closes)
    full_dd = maxdd_R(taken)
    full_pnl = sum(t.pnl_r for t in taken)
    print(f"=== confluence via the TAIL: {len(taken)} tight-OR taken trades (730d) ===")
    print(f"FULL book: PnL {full_pnl:+.1f}R  maxDD {full_dd:.1f}R  mean {full_pnl/len(taken):+.3f}R\n")
    print(f"  {'subgroup':<22}{'n':>6}{'meanR':>8}{'p5R':>8}{'<=-2R':>8}{'maxDD_R':>9}{'sumR':>9}")
    print("  " + "-" * 70)

    for key, name in [("gap", "overnight gap%"), ("mom20", "rel mom20 vs SPY"), ("sma_dist", "dist above 200SMA%")]:
        vals = np.array([s[key] for s in sig], dtype=float)
        ok = ~np.isnan(vals)
        med = np.nanmedian(vals)
        hi = [s["t"] for s in sig if s[key] == s[key] and s[key] >= med]   # favorable = higher
        lo = [s["t"] for s in sig if s[key] == s[key] and s[key] < med]
        print(f"\n  [{name}]  median={med:+.2f}  (coverage {ok.sum()}/{len(sig)})")
        print(tail_row(f"HIGH (>= med)", hi))
        print(tail_row(f"LOW  (< med)", lo))
        # decision: keep only HIGH side -> what happens to full-book DD & PnL?
        keep = hi
        kdd, kpnl = maxdd_R(keep), sum(t.pnl_r for t in keep)
        print(f"    -> drop LOW side: book PnL {full_pnl:+.1f}R -> {kpnl:+.1f}R "
              f"({kpnl-full_pnl:+.1f}R), maxDD {full_dd:.1f}R -> {kdd:.1f}R ({kdd-full_dd:+.1f}R)")

    print("\nRead: a filter is worth it only if dropping the unfavorable side CUTS maxDD materially")
    print("for a small PnL give-up. If PnL falls ~proportionally to DD (just fewer trades), it's not")
    print("a tail filter -- it's just de-risking by trading less, which position-sizing does better.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
