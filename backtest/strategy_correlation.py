"""strategy_correlation.py — are the three bots actually independent, or one momentum bet
in three costumes? (ChatGPT review's #1 concern, and the gate to any portfolio work.)

Pulls each bot's account daily-return series (Alpaca portfolio history), aligns on common
dates, and reports pairwise correlation + same-day-LOSS overlap (do they bleed together?).

CAVEAT: News/Hype inception ~2026-06-15, so the common window is only ~2 weeks (~10 days) —
LOW statistical power; read this as directional + a process to re-run as data grows. The
STRUCTURAL prior is strong regardless: all three are long-momentum breakouts, so positive
correlation (and joint drawdowns in a momentum-off regime) is expected.

Run:  .venv/Scripts/python.exe backtest/strategy_correlation.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BOTS = [("ORB", ".env"), ("News-Edge", ".env.news"), ("Hype", ".env.lottery")]


def _keys(envfile: str):
    f = ROOT / envfile
    k = s = None
    if f.exists():
        for line in f.read_text().splitlines():
            line = line.strip()
            if line.startswith("ALPACA_API_KEY"):
                k = line.split("=", 1)[1].strip().strip('"').strip("'")
            elif line.startswith("ALPACA_SECRET_KEY"):
                s = line.split("=", 1)[1].strip().strip('"').strip("'")
    return k, s


def daily_returns(envfile: str) -> pd.Series:
    """Daily account return series from Alpaca portfolio history (profit_loss / prior equity)."""
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetPortfolioHistoryRequest
    k, s = _keys(envfile)
    if not k:
        return pd.Series(dtype=float)
    tc = TradingClient(k, s, paper=True)
    ph = tc.get_portfolio_history(GetPortfolioHistoryRequest(period="1M", timeframe="1D"))
    import datetime as dt
    rows = {}
    eq = ph.equity or []
    pl = ph.profit_loss or []
    ts = ph.timestamp or []
    for i in range(len(ts)):
        d = dt.datetime.fromtimestamp(ts[i], dt.timezone.utc).date()
        prior_eq = (eq[i] - pl[i]) if (i < len(eq) and i < len(pl) and eq[i] is not None) else None
        if prior_eq and prior_eq > 0 and i < len(pl) and pl[i] is not None:
            rows[d] = pl[i] / prior_eq
    return pd.Series(rows).sort_index()


def main() -> int:
    print("=== inter-strategy correlation: are the 3 bots independent? ===")
    series = {}
    for name, env in BOTS:
        try:
            r = daily_returns(env)
            series[name] = r
            nz = (r != 0).sum()
            print(f"  {name:10} ({env}): {len(r)} days, {nz} non-flat")
        except Exception as e:
            print(f"  {name:10} ({env}): ERR {str(e)[:60]}")
    df = pd.DataFrame(series).dropna(how="all")
    # keep days where at least 2 bots have a non-flat return (a real co-trading day)
    active = df[(df != 0).sum(axis=1) >= 2]
    print(f"\ncommon active days (>=2 bots trading): {len(active)}")
    if len(active) < 4:
        print("  too few common active days for correlation — re-run as the live sample grows.")
        return 0
    print("\nPairwise daily-return correlation:")
    corr = active.corr()
    print(corr.round(2).to_string())
    print("\nSame-day LOSS overlap (both bots red on the same day, of days both traded):")
    names = list(active.columns)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = active[names[i]], active[names[j]]
            both = active[(a != 0) & (b != 0)]
            if len(both):
                both_red = ((both[names[i]] < 0) & (both[names[j]] < 0)).mean()
                print(f"  {names[i]} & {names[j]}: {both_red*100:.0f}% of {len(both)} shared days both down")
    print("\nREAD: high positive correlation + high joint-loss overlap = NOT 3 independent bots,")
    print("but 3 expressions of ONE momentum factor -> the '3-bot' diversification is largely")
    print("illusory, and a momentum-off regime hits all three at once. That reframes position")
    print("sizing + makes PORTFOLIO allocation (not a 4th filter) the priority. Tiny sample —")
    print("directional; re-run as the live history grows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
