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
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ET = ZoneInfo("America/New_York")

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


def fetch_intraday(ib, duration: str = "30 D", bar_size: str = "5 mins",
                   rth: bool = False, end="") -> pd.DataFrame | None:
    """One historical request -> OHLCV DataFrame (tz America/New_York), or None if empty. IBKR caps
    intraday requests (~1 month for 5-min bars) so keep `duration` small and walk `end` backward for
    depth (see cache_deep_history). `end` = "" (now) or a tz-aware datetime. rth=False keeps overnight."""
    c = mes_contract(ib)
    bars = ib.reqHistoricalData(c, endDateTime=end, durationStr=duration, barSizeSetting=bar_size,
                                whatToShow="TRADES", useRTH=rth, formatDate=1)
    if not bars:
        return None
    from ib_async import util
    df = util.df(bars).rename(columns={"date": "dt"})
    df = df.set_index(pd.DatetimeIndex(pd.to_datetime(df["dt"])))
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("America/New_York")
    return df[["open", "high", "low", "close", "volume"]].sort_index()


def cache_deep_history(months: int = 24, bar_size: str = "5 mins", chunk: str = "30 D",
                       pace_s: float = 11.0) -> int:
    """Deep intraday history via CHUNKED walk-back: request `chunk` at a time, stepping endDateTime
    to the earliest bar seen, ~`months` chunks, respecting IBKR pacing. Stitches + dedupes and writes
    the parquet cache that data.load_mes_intraday_cache() / backtest_orb read."""
    import time as _t
    ib = connect()
    frames: list[pd.DataFrame] = []
    end = ""
    try:
        for i in range(months):
            df = fetch_intraday(ib, duration=chunk, bar_size=bar_size, end=end)
            if df is None or df.empty:
                print(f"  chunk {i+1}: no more data -- stopping."); break
            frames.append(df)
            earliest = df.index.min()
            print(f"  chunk {i+1}: {len(df)} bars back to {earliest}")
            end = earliest.tz_convert("UTC").to_pydatetime()   # ib_async formats tz-aware datetimes
            if i < months - 1:
                _t.sleep(pace_s)
    finally:
        ib.disconnect()
    if not frames:
        raise RuntimeError("no bars returned at all (check Gateway / market-data permissions)")
    full = pd.concat(frames)
    full = full[~full.index.duplicated(keep="last")].sort_index()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(CACHE)
    print(f"cached {len(full)} bars ({bar_size}) {full.index.min()} -> {full.index.max()} to {CACHE}")
    return 0


def _third_friday(year: int, month: int) -> date:
    """MES quarterly expiry = 3rd Friday of the contract month."""
    d = date(year, month, 1)
    first_fri = 1 + (4 - d.weekday()) % 7
    return date(year, month, first_fri + 14)


def _quarterly_months(n: int, today: date | None = None) -> list[str]:
    """The last n MES quarterly contract months (YYYYMM, newest first). MES expires H/M/U/Z."""
    today = today or datetime.now(ET).date()
    qm = ((today.month - 1) // 3 + 1) * 3          # next quarter month in {3,6,9,12}
    cy, cm, out = (today.year + (1 if qm > 12 else 0)), (qm if qm <= 12 else 12), []
    for _ in range(n):
        out.append(f"{cy}{cm:02d}")
        cm -= 3
        if cm < 1:
            cm += 12
            cy -= 1
    return out


def cache_deep_history_dated(quarters: int = 8, bar_size: str = "30 mins",
                             chunk: str = "3 M", subchunks: int = 1, pace_s: float = 11.0) -> int:
    """Deep intraday history by STITCHING dated quarterly contracts (ContFuture blocks endDateTime).
    Each contract contributes its front-month period; newest wins on overlap. No roll back-adjustment
    (candidates are intraday-only, so each day is self-contained). `subchunks` walks endDateTime back
    WITHIN a contract (needed for fine bars: 5-min caps at ~1 month/request, so use chunk='1 M',
    subchunks=3). Writes the parquet cache."""
    import time as _t
    from ib_async import Future, util
    ib = connect()
    frames: list[pd.DataFrame] = []
    earliest = None
    now = datetime.now(ET)

    def _proc(bars) -> pd.DataFrame:
        df = util.df(bars).rename(columns={"date": "dt"})
        df = df.set_index(pd.DatetimeIndex(pd.to_datetime(df["dt"])))
        df.index = (df.index.tz_localize("UTC") if df.index.tz is None else df.index).tz_convert(ET)
        return df[["open", "high", "low", "close", "volume"]].sort_index()

    try:
        for i, ym in enumerate(_quarterly_months(quarters)):
            c = Future(symbol="MES", lastTradeDateOrContractMonth=ym, exchange="CME",
                       currency="USD", includeExpired=True)
            if not ib.qualifyContracts(c):
                print(f"  {ym}: could not qualify -- skip."); continue
            exp = datetime.combine(_third_friday(int(ym[:4]), int(ym[4:6])),
                                   datetime.min.time()).replace(hour=16, tzinfo=ET)
            c_end = min(exp, now)
            got = 0
            for _j in range(subchunks):
                bars = ib.reqHistoricalData(c, endDateTime=c_end, durationStr=chunk,
                                            barSizeSetting=bar_size, whatToShow="TRADES",
                                            useRTH=False, formatDate=1, timeout=90)
                if not bars:
                    break
                raw = _proc(bars)
                raw_min = raw.index.min()
                keep = raw[raw.index < earliest] if earliest is not None else raw
                if not keep.empty:
                    frames.append(keep)
                    earliest = keep.index.min()
                    got += len(keep)
                c_end = raw_min.tz_convert("UTC").to_pydatetime()   # step back within the contract
                _t.sleep(pace_s)
            if got:
                print(f"  {c.localSymbol}: +{got} bars, earliest now {earliest.date()}")
    finally:
        ib.disconnect()
    if not frames:
        raise RuntimeError("no bars from any dated contract")
    full = pd.concat(frames)
    full = full[~full.index.duplicated(keep="first")].sort_index()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(CACHE)
    print(f"cached {len(full)} bars ({bar_size}) {full.index.min().date()} -> "
          f"{full.index.max().date()} to {CACHE}")
    return 0


# --- ORDER side (used by run_mes_bot.py) ---
def _front_month(today: date | None = None, roll_buffer: int = 8) -> str:
    """Nearest quarterly month (YYYYMM) whose 3rd-Fri expiry is > today+roll_buffer days (rolls the
    front ~a week before expiry, matching MES liquidity)."""
    today = today or datetime.now(ET).date()
    y, m = today.year, today.month
    for _ in range(12):
        if m in (3, 6, 9, 12) and (_third_friday(y, m) - today).days > roll_buffer:
            return f"{y}{m:02d}"
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return f"{y}{m:02d}"


def mes_front_contract(ib):
    """Current front-month DATED MES future -- for ORDERS (you cannot trade a ContFuture)."""
    from ib_async import Future
    c = Future(symbol="MES", lastTradeDateOrContractMonth=_front_month(), exchange="CME", currency="USD")
    ib.qualifyContracts(c)
    return c


def order_market(ib, action: str, qty: int):
    """Market order on the front MES contract. action='BUY'/'SELL'. Returns the Trade. Fields set
    explicitly to avoid TWS order-preset rejection (err 10349): DAY tif, allow extended hours, and
    clear the legacy eTradeOnly/firmQuoteOnly flags that some TWS builds reject."""
    from ib_async import MarketOrder
    o = MarketOrder(action, qty)
    o.tif = "DAY"
    o.outsideRth = True
    o.eTradeOnly = False
    o.firmQuoteOnly = False
    return ib.placeOrder(mes_front_contract(ib), o)


def position(ib) -> int:
    """Signed MES position (contracts). 0 if flat."""
    for p in ib.positions():
        if getattr(p.contract, "symbol", None) == "MES":
            return int(p.position)
    return 0


def account_snapshot(ib) -> dict:
    """Paper-account P&L snapshot for the dashboard (NetLiquidation / Unrealized / Realized). Best-effort."""
    out: dict = {}
    try:
        for r in ib.accountSummary():
            if r.tag in ("NetLiquidation", "UnrealizedPnL", "RealizedPnL", "AvailableFunds"):
                out[r.tag] = float(r.value)
    except Exception:
        pass
    return out


def reconcile_to(ib, target: int) -> str | None:
    """Place a market order to move the current MES position to `target` (-1/0/+1...). Returns a
    description of the order sent, or None if already there. This is how the momentum bot enters,
    flips, and exits -- idempotent, so a missed pass self-corrects on the next one."""
    cur = position(ib)
    delta = target - cur
    if delta == 0:
        return None
    order_market(ib, "BUY" if delta > 0 else "SELL", abs(delta))
    return f"{'BUY' if delta > 0 else 'SELL'} {abs(delta)} (pos {cur} -> {target})"


def flatten(ib):
    """Market-close any open MES position (kill switch / EOD)."""
    return reconcile_to(ib, 0)


def main(argv) -> int:
    cmd = argv[1] if len(argv) > 1 else "ping"
    if cmd == "fetch":
        months = int(argv[2]) if len(argv) > 2 else 24
        return cache_deep_history(months=months)
    if cmd == "deep":   # dated-contract stitched deep history (recommended)
        quarters = int(argv[2]) if len(argv) > 2 else 8
        bar = argv[3] if len(argv) > 3 else "30 mins"
        chunk = argv[4] if len(argv) > 4 else "3 M"
        subchunks = int(argv[5]) if len(argv) > 5 else 1
        return cache_deep_history_dated(quarters=quarters, bar_size=bar, chunk=chunk, subchunks=subchunks)
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
