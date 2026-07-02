"""IBKR adapter for the MES futures bot (data + orders) via `ib_async` (the maintained ib_insync fork).

Two jobs:
  1) DATA (needed now): pull deep intraday MES history and cache it to futures/data/mes_intraday.parquet
     so futures/backtest_orb.py can run on years, not the ~50 free yfinance days.
  2) ORDERS (Phase 3+): market entry + a native IBKR trailing stop, position/flatten helpers.

Requires IB Gateway (or TWS) running and logged into the PAPER account, with API enabled
(Settings > API > Enable ActiveX and Socket Clients). Default paper socket port = 4002 (Gateway) /
7497 (TWS). No API keys -- IBKR auth is the Gateway login. Install: `pip install ib_async`.

    # once Gateway is up on the paper account:
    .venv-openbb/Scripts/python.exe futures/broker_ibkr.py fetch    # cache deep intraday history
    .venv-openbb/Scripts/python.exe futures/broker_ibkr.py ping     # connection smoke test
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

CACHE = Path(__file__).resolve().parent / "data" / "mes_intraday.parquet"
# Paper defaults: IB Gateway=4002, TWS=7497 (live would be 4001/7496). clientId is arbitrary but unique.
# Overridable via .env.futures (IB_HOST/IB_PORT/IB_CLIENT_ID) loaded by run_mes_bot.py.
HOST = os.environ.get("IB_HOST", "127.0.0.1")
PORT = int(os.environ.get("IB_PORT", "4002"))
CLIENT_ID = int(os.environ.get("IB_CLIENT_ID", "17"))


def mes_contract(ib):
    """Front-month MES continuous future, qualified against IBKR's contract DB.
    ContFuture stitches the front month for history; for live orders resolve the dated Future."""
    from ib_async import ContFuture
    c = ContFuture("MES", "CME", currency="USD")
    ib.qualifyContracts(c)
    return c


def connect(host: str = HOST, port: int = PORT, client_id: int = CLIENT_ID):
    """Return a connected IB handle (raises if Gateway/TWS isn't up + API-enabled)."""
    from ib_async import IB
    ib = IB()
    ib.connect(host, port, clientId=client_id, timeout=15)
    return ib


def fetch_intraday(ib, duration: str = "2 Y", bar_size: str = "5 mins",
                   rth: bool = False) -> pd.DataFrame:
    """Historical MES bars -> OHLCV DataFrame, tz America/New_York. duration e.g. '2 Y'/'6 M';
    bar_size e.g. '5 mins'/'1 min'. IBKR paces long requests; may need a CME market-data subscription
    for full depth (flagged to the user). rth=False keeps the overnight session too."""
    c = mes_contract(ib)
    bars = ib.reqHistoricalData(c, endDateTime="", durationStr=duration, barSizeSetting=bar_size,
                                whatToShow="TRADES", useRTH=rth, formatDate=1)
    if not bars:
        raise RuntimeError("no bars returned (check market-data permissions / Gateway)")
    from ib_async import util
    df = util.df(bars).rename(columns={"date": "dt"})
    df = df.set_index(pd.DatetimeIndex(pd.to_datetime(df["dt"])))
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("America/New_York")
    return df[["open", "high", "low", "close", "volume"]].sort_index()


def cache_deep_history(duration: str = "2 Y", bar_size: str = "5 mins") -> int:
    """Fetch deep intraday history and write the parquet cache backtest_orb reads via
    data.load_mes_intraday_cache(). Prints the coverage it landed."""
    ib = connect()
    try:
        df = fetch_intraday(ib, duration=duration, bar_size=bar_size)
    finally:
        ib.disconnect()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE)
    print(f"cached {len(df)} bars ({bar_size}) {df.index.min()} -> {df.index.max()} to {CACHE}")
    return 0


# --- ORDER side (Phase 3+; used by run_mes_bot.py) ---
def market_with_trailing(ib, action: str, qty: int, trail_points: float):
    """Submit a market entry + an attached native IBKR trailing stop (trailStopPrice by points).
    action = 'BUY'/'SELL'. Returns (parent_trade, stop_trade). UNTESTED until Gateway is live."""
    from ib_async import MarketOrder, Order
    c = mes_contract(ib)
    parent = MarketOrder(action, qty, transmit=False)
    parent_trade = ib.placeOrder(c, parent)
    stop_action = "SELL" if action == "BUY" else "BUY"
    trail = Order(orderType="TRAIL", action=stop_action, totalQuantity=qty,
                  auxPrice=trail_points, parentId=parent.orderId, transmit=True)
    stop_trade = ib.placeOrder(c, trail)
    return parent_trade, stop_trade


def position(ib) -> int:
    """Signed MES position (contracts). 0 if flat."""
    for p in ib.positions():
        if getattr(p.contract, "symbol", None) == "MES":
            return int(p.position)
    return 0


def flatten(ib):
    """Market-close any open MES position (kill switch)."""
    from ib_async import MarketOrder
    pos = position(ib)
    if pos == 0:
        return None
    c = mes_contract(ib)
    return ib.placeOrder(c, MarketOrder("SELL" if pos > 0 else "BUY", abs(pos)))


def main(argv) -> int:
    cmd = argv[1] if len(argv) > 1 else "ping"
    if cmd == "fetch":
        dur = argv[2] if len(argv) > 2 else "2 Y"
        return cache_deep_history(duration=dur)
    if cmd == "ping":
        ib = connect()
        try:
            c = mes_contract(ib)
            print(f"connected. MES resolved: {c.localSymbol or c.symbol} on {c.exchange}; "
                  f"position={position(ib)}")
        finally:
            ib.disconnect()
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
