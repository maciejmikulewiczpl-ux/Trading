"""Cap-aware confirmation of the risk-budget diversification result.

compare_diversification.py worked in idealized R-space (pnl_r x risk, ignoring the
$10k per-position notional cap). This re-prices every trade with the REAL sizing
the runner would use — shares = min(floor(risk/risk_per_share), floor($10k/entry)),
trade dropped if that's 0 — so the dollar PnL / Sharpe / drawdown are what you'd
actually get. Same total risk budget ($800/day) split across more positions, on
top of the live trend filter, both windows + OOS halves.

Re-sizing the cached $100-risk trades is exact: the breakout set is sizing-
independent (a trade only ever drops when shares hit 0), so the $50/$40 trade set
is the $100 set re-priced, minus the few names whose risk-per-share exceeds the
smaller budget. No backtest re-run needed.

Run (uses caches from compare_selection.py / compare_norefill_trend.py):
    .venv/Scripts/python.exe backtest/compare_capaware.py
"""
from __future__ import annotations

import math
import pickle
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.universe_portfolio import perf, portfolio  # noqa: E402
from backtest.compare_selection import _tday  # noqa: E402
from backtest.compare_norefill_trend import trend_eligibility, apply_filter  # noqa: E402

WINDOWS = [730, 180]
BUDGET = 800.0
NOTIONAL_CAP = 10_000.0
# (concurrent positions, risk per trade) — all ~$800/day total risk
CONFIGS = [(8, 100), (12, 67), (16, 50), (20, 40), (25, 32)]


def load(w):
    bars = pickle.load(open(ROOT / "backtest" / f".bars_cache_univ_{w}d.pkl", "rb"))
    trades = pickle.load(open(ROOT / "backtest" / f".bars_cache_trades_{w}d.pkl", "rb"))
    closes = pickle.load(open(ROOT / "backtest" / f".bars_cache_daily_{w}d.pkl", "rb"))
    all_bars, days = bars["bars"], bars["days"]
    present = sorted(all_bars.index.get_level_values(0).unique())
    return days, present, trades, closes


def resize(trades, risk_per_trade):
    """Re-price each trade at `risk_per_trade` with the real $10k notional cap."""
    out = []
    for t in trades:
        rps = abs(t.entry_price - t.stop_price)
        if rps <= 0:
            continue
        shares = min(math.floor(risk_per_trade / rps), math.floor(NOTIONAL_CAP / t.entry_price))
        if shares <= 0:
            continue                      # can't size it at this risk -> not taken
        pnl = ((t.exit_price - t.entry_price) if t.side == "long"
               else (t.entry_price - t.exit_price)) * shares
        out.append(replace(t, shares=shares, pnl_dollars=pnl, risk_dollars=shares * rps))
    return out


def three(taken, days, mid):
    d1 = [d for d in days if d < mid]
    d2 = [d for d in days if d >= mid]
    s = set(days)
    return (perf([t for t in taken if _tday(t) in s], days),
            perf([t for t in taken if _tday(t) < mid], d1),
            perf([t for t in taken if _tday(t) >= mid], d2))


HEAD = (f"{'config':<16}{'trades':>7}{'PnL$':>10}{'Sharpe':>8}{'maxDD$':>10}{'win%':>7}   "
        f"{'h1 PnL':>8}{'h1 Sh':>6}   {'h2 PnL':>8}{'h2 Sh':>6}")


def prow(label, f, h1, h2):
    def c(s, k, fmt):
        return format(s[k], fmt) if s.get("n", 0) else "—"
    print(f"{label:<16}{f.get('n',0):>7}{c(f,'pnl','>+10,.0f')}{c(f,'sharpe','>8.2f')}"
          f"{c(f,'max_dd','>10,.0f')}{c(f,'win','>6.1f')}%   "
          f"{c(h1,'pnl','>+8,.0f')}{c(h1,'sharpe','>6.2f')}   {c(h2,'pnl','>+8,.0f')}{c(h2,'sharpe','>6.2f')}")


def run_window(w):
    days, present, trades, closes = load(w)
    mid = sorted(days)[len(days) // 2]
    elig = trend_eligibility(closes, present, days)
    print(f"\n=== {w}d: {len(present)} names, {len(days)} sessions, OOS split {mid} "
          f"(REAL sizing, $10k notional cap) ===")
    print(HEAD)
    print("-" * len(HEAD))
    for n, rpt in CONFIGS:
        filt = apply_filter(resize(trades, rpt), elig)
        taken = portfolio(filt, n)
        f, h1, h2 = three(taken, days, mid)
        print_lbl = f"{n} x ${rpt}" + ("  (LIVE)" if n == 8 else "")
        prow(print_lbl, f, h1, h2)


def main():
    for w in WINDOWS:
        run_window(w)
    print("\nThese are REAL dollars (the $10k cap applied). Confirm vs idealized run:")
    print("does 16 x $50 still beat 8 x $100 on Sharpe + drawdown at the same ~$800 budget?")
    return 0


if __name__ == "__main__":
    sys.exit(main())
