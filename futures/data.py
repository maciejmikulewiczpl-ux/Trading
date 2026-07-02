"""MES (Micro E-mini S&P 500) data + contract constants for the futures bot.

Phase-1 (backtest gate) uses the continuous E-mini S&P front-month proxy `ES=F` from yfinance for
DAILY history (long, free, reliable). MES tracks the same index at 1/10 the multiplier, so the PRICE
SERIES is identical -- only the $/point differs. Intraday (minute) history is NOT available from
yfinance beyond ~60 days; the intraday loader is a stub until we wire IBKR historical or Databento
(see plan). Keep every strategy's cost model in real MES dollars via the constants below.

    .venv-openbb/Scripts/python.exe futures/data.py        # smoke-test the loader
"""
from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd

ET = ZoneInfo("America/New_York")

# --- MES contract constants (Micro E-mini S&P 500) ---
POINT_VALUE = 5.0        # $ per 1.00 index point (ES = $50; MES = $5)
TICK = 0.25              # minimum price increment, index points
TICK_VALUE = TICK * POINT_VALUE   # $1.25 per tick
# Round-turn friction assumption (conservative for a very liquid micro):
COMMISSION_RT = 1.50     # $ commission+fees round-turn (IBKR MES ~ $0.60/side)
SLIP_TICKS_RT = 2        # ticks of slippage round-turn (~1 tick each side)
COST_RT_USD = COMMISSION_RT + SLIP_TICKS_RT * TICK_VALUE   # ~$4.00 round-turn per contract


def load_mes_daily(start: str = "2005-01-01", symbol: str = "ES=F") -> pd.DataFrame:
    """Daily OHLCV for the E-mini S&P proxy (price == MES price). Columns: open/high/low/close/volume,
    tz-naive DatetimeIndex, ascending. Raises on empty."""
    import yfinance as yf
    raw = yf.download(symbol, start=start, auto_adjust=True, progress=False)
    if raw is None or len(raw) == 0:
        raise RuntimeError(f"no data returned for {symbol}")
    # yfinance returns a MultiIndex (field, ticker) even for one symbol -> flatten to fields.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.sort_index().dropna(subset=["open", "high", "low", "close"])


def load_mes_intraday(interval: str = "5m", period: str = "60d", symbol: str = "ES=F") -> pd.DataFrame:
    """Intraday OHLCV for the E-mini S&P proxy, tz-aware in America/New_York (ascending).

    SHORT-WINDOW ONLY: yfinance caps intraday history at ~60 days for sub-hourly bars, so this returns
    ~50 trading days -- enough to SMOKE-TEST the ORB logic and get a preliminary read, NOT enough for a
    verdict. Deep intraday history (years) comes from IBKR via futures/broker_ibkr.py once the account
    is live; that path caches to futures/data/ and this loader can be pointed at the cache."""
    import yfinance as yf
    raw = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
    if raw is None or len(raw) == 0:
        raise RuntimeError(f"no intraday data for {symbol} ({interval}/{period})")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].copy()
    idx = pd.to_datetime(df.index)
    df.index = idx.tz_convert(ET) if idx.tz is not None else idx.tz_localize("UTC").tz_convert(ET)
    return df.sort_index().dropna(subset=["open", "high", "low", "close"])


def load_mes_intraday_cache(path: str | None = None) -> pd.DataFrame:
    """Load deep intraday history from a local parquet cache written by broker_ibkr.py (Phase 2).
    Returns the same schema/tz as load_mes_intraday. Raises if the cache does not exist yet."""
    from pathlib import Path
    p = Path(path) if path else Path(__file__).resolve().parent / "data" / "mes_intraday.parquet"
    if not p.exists():
        raise FileNotFoundError(f"no intraday cache at {p} -- fetch it with broker_ibkr.py once IBKR is live")
    df = pd.read_parquet(p)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(ET)
    return df.sort_index()


def _smoke() -> int:
    df = load_mes_daily()
    print(f"MES(ES=F) daily: {len(df)} rows  {df.index.min().date()} -> {df.index.max().date()}")
    print(f"last close {df['close'].iloc[-1]:.2f}  -> MES notional ~${df['close'].iloc[-1]*POINT_VALUE:,.0f}")
    print(f"round-turn friction assumption: ${COST_RT_USD:.2f}/contract "
          f"(${COMMISSION_RT:.2f} comm + {SLIP_TICKS_RT} ticks slip)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_smoke())
