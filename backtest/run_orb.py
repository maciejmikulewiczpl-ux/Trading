"""Backtest ORB on the watchlist over the last 6 months of 1-min IEX bars."""
from __future__ import annotations

import os
import sys
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetCalendarRequest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from strategies.orb import Params, Trade, simulate_session  # noqa: E402

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

WATCHLIST = ["SPY", "QQQ", "AAPL", "NVDA", "TSLA"]
LOOKBACK_DAYS = 180
STARTING_EQUITY = 100_000.0
OUTPUT_CSV = Path(__file__).parent / "orb_trades.csv"


def load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def get_trading_days(trading_client: TradingClient, start: datetime, end: datetime):
    cal = trading_client.get_calendar(GetCalendarRequest(start=start.date(), end=end.date()))
    return [c.date for c in cal]


def fetch_bars(client: StockHistoricalDataClient, symbols, start, end) -> pd.DataFrame:
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Minute,
        start=start.astimezone(UTC),
        end=end.astimezone(UTC),
        feed=DataFeed.IEX,
    )
    return client.get_stock_bars(req).df


def to_et(bars: pd.DataFrame) -> pd.DataFrame:
    sym = bars.index.get_level_values(0)
    ts = bars.index.get_level_values(1).tz_convert(ET)
    bars = bars.copy()
    bars.index = pd.MultiIndex.from_arrays([sym, ts], names=["symbol", "timestamp"])
    return bars


def run_backtest(
    all_bars: pd.DataFrame,
    trading_days,
    watchlist,
    params: Params,
    starting_equity: float,
) -> tuple[list[Trade], float]:
    """Run the ORB backtest in-memory; returns (trades, final_equity)."""
    trades: list[Trade] = []
    running_equity = starting_equity
    symbols_in_data = set(all_bars.index.get_level_values(0).unique())

    for session_date in trading_days:
        day_equity = running_equity
        day_trades: list[Trade] = []
        for symbol in watchlist:
            if symbol not in symbols_in_data:
                continue
            sym_bars = all_bars.xs(symbol, level=0)
            day_bars = sym_bars[sym_bars.index.date == session_date]
            if day_bars.empty:
                continue
            t = day_bars.index.time
            day_bars = day_bars[(t >= time(9, 30)) & (t < time(16, 0))]
            if day_bars.empty:
                continue
            for trade in simulate_session(day_bars, symbol, day_equity, params):
                trades.append(trade)
                day_trades.append(trade)
        for tr in day_trades:
            running_equity += tr.pnl_dollars
    return trades, running_equity


def load_all_bars(verbose: bool = True, lookback_days: int | None = None):
    """Fetch trading-day calendar and 1-min IEX bars for the watchlist. Returns (bars, days).

    lookback_days overrides the module default (LOOKBACK_DAYS) — pass a large
    value for multi-year backtests. IEX minute history on the basic tier starts
    ~2021-01; older windows return empty.
    """
    load_env()
    api_key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret:
        raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY not set in .env")

    data_client = StockHistoricalDataClient(api_key, secret)
    trading_client = TradingClient(api_key, secret, paper=True)

    days_back = LOOKBACK_DAYS if lookback_days is None else lookback_days
    end = datetime.now(tz=ET)
    start = end - timedelta(days=days_back)
    if verbose:
        print(f"Window   : {start.date()} -> {end.date()}  ({days_back} calendar days)")

    trading_days = get_trading_days(trading_client, start, end)
    if verbose:
        print(f"Sessions : {len(trading_days)} trading days")
        print("Pulling 1-min IEX bars...")
    raw_bars = fetch_bars(data_client, WATCHLIST, start, end)
    if raw_bars.empty:
        raise RuntimeError("No bars returned. Check API key / feed permissions.")
    all_bars = to_et(raw_bars)
    if verbose:
        print(f"Bars     : {len(all_bars):,} rows across {len(all_bars.index.get_level_values(0).unique())} symbols")
    return all_bars, trading_days


def main() -> int:
    print(f"Universe : {WATCHLIST}")
    try:
        all_bars, trading_days = load_all_bars()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    params = Params()
    trades, running_equity = run_backtest(all_bars, trading_days, WATCHLIST, params, STARTING_EQUITY)

    if not trades:
        print("No trades generated.")
        return 0

    df = pd.DataFrame([{
        "symbol": t.symbol,
        "date": t.date.date(),
        "side": t.side,
        "or_high": round(t.or_high, 2),
        "or_low": round(t.or_low, 2),
        "entry_time": t.entry_time.tz_convert(ET).strftime("%Y-%m-%d %H:%M"),
        "entry_price": round(t.entry_price, 2),
        "stop_price": round(t.stop_price, 2),
        "target_price": round(t.target_price, 2),
        "shares": t.shares,
        "risk_dollars": round(t.risk_dollars, 2),
        "exit_time": t.exit_time.tz_convert(ET).strftime("%Y-%m-%d %H:%M"),
        "exit_price": round(t.exit_price, 2),
        "exit_reason": t.exit_reason,
        "pnl_dollars": round(t.pnl_dollars, 2),
        "pnl_r": round(t.pnl_r, 2),
    } for t in trades])
    df.to_csv(OUTPUT_CSV, index=False)

    print()
    print(f"=== ORB backtest: {len(trades)} trades ===")
    print()
    print(f"{'symbol':<7} {'n':>4} {'win%':>6} {'avg_R':>7} {'pnl $':>12}")
    for sym, grp in df.groupby("symbol"):
        n = len(grp)
        wins = (grp["pnl_r"] > 0).sum()
        avg_r = grp["pnl_r"].mean()
        tot = grp["pnl_dollars"].sum()
        print(f"{sym:<7} {n:>4} {wins/n*100:>5.1f}% {avg_r:>+7.2f} {tot:>+12,.2f}")

    # Per-side breakdown (only interesting once shorts are enabled).
    if df["side"].nunique() > 1:
        print()
        print(f"{'side':<7} {'n':>4} {'win%':>6} {'avg_R':>7} {'pnl $':>12}")
        for side, grp in df.groupby("side"):
            n = len(grp)
            wins = (grp["pnl_r"] > 0).sum()
            avg_r = grp["pnl_r"].mean()
            tot = grp["pnl_dollars"].sum()
            print(f"{side:<7} {n:>4} {wins/n*100:>5.1f}% {avg_r:>+7.2f} {tot:>+12,.2f}")

    n = len(df)
    wins = (df["pnl_r"] > 0).sum()
    print()
    print("Aggregate:")
    print(f"  Trades        : {n}")
    print(f"  Win rate      : {wins/n*100:.1f}%")
    print(f"  Avg R         : {df['pnl_r'].mean():+.2f}")
    print(f"  Total PnL     : ${df['pnl_dollars'].sum():+,.2f}")
    print(f"  Final equity  : ${running_equity:,.2f}  (start ${STARTING_EQUITY:,.0f})")

    df_sorted = df.sort_values("exit_time")
    eq_curve = STARTING_EQUITY + df_sorted["pnl_dollars"].cumsum()
    running_max = eq_curve.cummax()
    dd = (eq_curve - running_max).min()
    print(f"  Max drawdown  : ${dd:+,.2f}")

    print()
    print(f"Trades CSV    : {OUTPUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
