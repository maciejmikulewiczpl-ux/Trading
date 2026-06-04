"""Risk-budget diversification: same total risk, more positions — does Sharpe rise?

The 8-slot cap harvests only a sliver of the strategy's gross edge. The fix isn't
selection (that failed) — it's spreading the SAME total daily risk across MORE
simultaneous breakouts. This holds the risk budget constant at the current
8 x $100 = $800/day and re-splits it: 12 x $67, 16 x $50, 20 x $40, ...

Everything is in R-space (each trade's capital-agnostic R-multiple), then scaled
by risk-per-trade = BUDGET / N. Because Sharpe is scale-invariant, the Sharpe of
a config depends only on WHICH trades it takes (the cap), not the per-trade $ —
so a rising Sharpe as N grows is the diversification benefit, clean. PnL and
drawdown are reported in dollars at the constant $800 budget.

Win condition: as N rises at constant budget, Sharpe goes UP and max drawdown
($) goes DOWN while PnL holds — i.e. same return, less risk => you can then size
the whole budget up toward a profit target at the SAME drawdown as today.
(If n barely grows with N, the cap rarely binds after the trend filter and there
is little to diversify — the table will show that immediately.)

Applied on top of the live 200d trend filter. Needs the caches from
compare_selection.py / compare_norefill_trend.py. Run:
    .venv/Scripts/python.exe backtest/compare_diversification.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.universe_portfolio import portfolio  # noqa: E402
from backtest.compare_selection import _tday  # noqa: E402
from backtest.compare_norefill_trend import trend_eligibility, apply_filter  # noqa: E402

WINDOWS = [730, 180]
BUDGET = 800.0                       # current 8 x $100/day total risk
CAPS = [8, 12, 16, 20, 25, 40]


def load(w):
    bars = pickle.load(open(ROOT / "backtest" / f".bars_cache_univ_{w}d.pkl", "rb"))
    trades = pickle.load(open(ROOT / "backtest" / f".bars_cache_trades_{w}d.pkl", "rb"))
    closes = pickle.load(open(ROOT / "backtest" / f".bars_cache_daily_{w}d.pkl", "rb"))
    all_bars, days = bars["bars"], bars["days"]
    present = sorted(all_bars.index.get_level_values(0).unique())
    return days, present, trades, closes


def stats(taken, days, rpt):
    """Daily R series (scaled to $ by rpt = risk/trade), over all sessions."""
    by = {}
    for t in taken:
        by[_tday(t)] = by.get(_tday(t), 0.0) + t.pnl_r
    s = pd.Series(by).reindex(sorted(days), fill_value=0.0) if by else pd.Series(0.0, index=sorted(days))
    dollar = s * rpt
    eq = dollar.cumsum()
    dd = (eq - eq.cummax()).min() if len(eq) else 0.0
    mu, sd = s.mean(), s.std()
    sharpe = (mu / sd * (252 ** 0.5)) if sd and sd > 0 else float("nan")
    active = sum(1 for v in by.values() if v != 0)
    return {"n": len(taken), "sumR": s.sum(), "pnl": dollar.sum(), "maxdd": dd,
            "sharpe": sharpe, "tpd": (len(taken) / max(1, active))}


def three(taken, days, mid, rpt):
    d1 = [d for d in days if d < mid]
    d2 = [d for d in days if d >= mid]
    return (stats(taken, days, rpt),
            stats([t for t in taken if _tday(t) < mid], d1, rpt),
            stats([t for t in taken if _tday(t) >= mid], d2, rpt))


HEAD = (f"{'config':<18}{'trades':>7}{'sumR':>8}{'PnL$':>10}{'Sharpe':>8}{'maxDD$':>10}"
        f"{'t/day':>6}   {'h1 Sh':>6}{'h2 Sh':>6}{'h2 PnL$':>9}")


def prow(label, full, h1, h2):
    def c(s, k, fmt):
        return format(s[k], fmt) if s.get("n", 0) else "—"
    print(f"{label:<18}{full['n']:>7}{c(full,'sumR','>+8.1f')}{c(full,'pnl','>+10,.0f')}"
          f"{c(full,'sharpe','>8.2f')}{c(full,'maxdd','>10,.0f')}{c(full,'tpd','>6.1f')}   "
          f"{c(h1,'sharpe','>6.2f')}{c(h2,'sharpe','>6.2f')}{c(h2,'pnl','>+9,.0f')}")


def run_window(w):
    days, present, trades, closes = load(w)
    mid = sorted(days)[len(days) // 2]
    filtered = apply_filter(trades, trend_eligibility(closes, present, days))
    print(f"\n=== {w}d: {len(present)} names, {len(days)} sessions, OOS split {mid} ===")
    print(f"(constant total risk budget ${BUDGET:,.0f}/day; risk/trade = budget / N)")
    print(HEAD)
    print("-" * len(HEAD))
    base = None
    for n in CAPS:
        rpt = BUDGET / n
        taken = portfolio(filtered, n)
        f, h1, h2 = three(taken, days, mid, rpt)
        label = f"{n} x ${rpt:,.0f}" + ("  (LIVE)" if n == 8 else "")
        prow(label, f, h1, h2)
        if n == 8:
            base = f
    return base


def main():
    for w in WINDOWS:
        run_window(w)
    print("\nReads: compare each row to '8 x $100 (LIVE)'. If Sharpe rises and maxDD$")
    print("shrinks as N grows at the SAME budget, diversification works — then the whole")
    print("$800 can be scaled up for more profit at today's drawdown. If 'trades' barely")
    print("grows with N, the cap rarely binds after the filter and there's little to gain.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
