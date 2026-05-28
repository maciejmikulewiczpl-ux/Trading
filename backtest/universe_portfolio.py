"""Realistic capital-constrained portfolio sim over the broad universe.

universe_scan.py showed a robust per-signal edge across 55 names, but its
+147 R / +$13k is UNCAPPED (assumes every simultaneous breakout is taken,
needing far more than $100k of buying power). This answers the real question:
on $100k, with a cap on concurrent open positions, what does ORB-on-breadth
actually return, and at what drawdown / Sharpe?

Method: take the broad-universe trades, sort by entry time, and greedily fill
a portfolio that allows at most N concurrent open positions (a breakout is
skipped if all slots are full at its entry). $100 risk / $10k cap per trade,
as live. Reports realized PnL, return%, max DD, and an annualized Sharpe from
the daily PnL series, for several caps — vs the uncapped ceiling and the
current 5-name concentration.

Run (data fetch ~1-2 min):
    .venv/Scripts/python.exe backtest/universe_portfolio.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from strategies.orb import Params, Trade  # noqa: E402
from backtest.run_orb import (  # noqa: E402
    LOOKBACK_DAYS, STARTING_EQUITY, get_trading_days, load_env, run_backtest, to_et,
)
from backtest.universe_scan import UNIVERSE, fetch_chunked  # noqa: E402

ET = ZoneInfo("America/New_York")

CAPS = [5, 10, 15, 20, None]  # None = uncapped ceiling


def portfolio(trades: list[Trade], cap):
    """Greedy fill: take trades in entry-time order, max `cap` concurrent.
    Returns the list of taken trades."""
    taken, open_exits = [], []
    for t in sorted(trades, key=lambda x: x.entry_time):
        open_exits = [x for x in open_exits if x > t.entry_time]
        if cap is None or len(open_exits) < cap:
            taken.append(t)
            open_exits.append(t.exit_time)
    return taken


def perf(trades: list[Trade], trading_days) -> dict:
    if not trades:
        return {"n": 0}
    df = pd.DataFrame([{"exit_date": t.exit_time.date(),
                        "pnl": t.pnl_dollars, "r": t.pnl_r} for t in trades])
    # Daily PnL series across all sessions (0 on days with no closed trades).
    daily = df.groupby("exit_date")["pnl"].sum()
    idx = pd.Index(sorted(d for d in trading_days))
    daily = daily.reindex(idx, fill_value=0.0)
    eq = STARTING_EQUITY + daily.cumsum()
    dd = (eq - eq.cummax()).min()
    mu, sd = daily.mean(), daily.std()
    sharpe = (mu / sd * (252 ** 0.5)) if sd > 0 else float("nan")
    return {
        "n": len(df),
        "win": (df["r"] > 0).mean() * 100,
        "sum_r": df["r"].sum(),
        "pnl": df["pnl"].sum(),
        "ret_pct": df["pnl"].sum() / STARTING_EQUITY * 100,
        "max_dd": dd,
        "sharpe": sharpe,
    }


def main() -> int:
    load_env()
    key, sec = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    if not key or not sec:
        print("ERROR: API keys missing", file=sys.stderr)
        return 1
    dc = StockHistoricalDataClient(key, sec)
    tc = TradingClient(key, sec, paper=True)

    end = datetime.now(tz=ET)
    start = end - timedelta(days=LOOKBACK_DAYS)
    trading_days = get_trading_days(tc, start, end)
    print(f"Universe: {len(UNIVERSE)} names, {len(trading_days)} sessions. Fetching...")
    raw = fetch_chunked(dc, UNIVERSE, start, end)
    all_bars = to_et(raw)
    present = sorted(all_bars.index.get_level_values(0).unique())

    params = Params(or_minutes=15, target_r=2.0, risk_per_trade=100.0,
                    max_position_pct=0.25, max_position_dollars=10_000.0,
                    no_entry_after_time=time(11, 30))
    all_trades, _ = run_backtest(all_bars, trading_days, present, params, STARTING_EQUITY)
    print(f"Total broad-universe signals: {len(all_trades)}\n")

    # Reference: current live 5-name concentration.
    base5 = {"SPY", "QQQ", "AAPL", "NVDA", "TSLA"}
    p5 = perf([t for t in all_trades if t.symbol in base5], trading_days)

    print(f"{'config':<22}{'n':>5}{'win%':>7}{'sumR':>8}{'PnL$':>11}"
          f"{'ret%':>8}{'maxDD$':>11}{'Sharpe':>8}")
    print("-" * 80)
    print(f"{'current 5-name':<22}{p5['n']:>5}{p5['win']:>6.1f}%{p5['sum_r']:>+8.1f}"
          f"{p5['pnl']:>+11,.0f}{p5['ret_pct']:>+7.2f}%{p5['max_dd']:>+11,.0f}"
          f"{p5['sharpe']:>8.2f}")
    print("-" * 80)
    for cap in CAPS:
        taken = portfolio(all_trades, cap)
        s = perf(taken, trading_days)
        label = f"broad, cap={cap}" if cap is not None else "broad, UNCAPPED"
        print(f"{label:<22}{s['n']:>5}{s['win']:>6.1f}%{s['sum_r']:>+8.1f}"
              f"{s['pnl']:>+11,.0f}{s['ret_pct']:>+7.2f}%{s['max_dd']:>+11,.0f}"
              f"{s['sharpe']:>8.2f}")

    print("\nNotes:")
    print("- $100 risk / $10k cap per trade (live params). cap=N -> <=N concurrent.")
    print("- $100k cash -> ~10 concurrent at $10k each; Reg-T margin -> ~20.")
    print("- Sharpe is annualized from daily PnL (252). Window is ~6 months so treat")
    print("  it as indicative, not precise. PnL/ret are over the full window, not annual.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
