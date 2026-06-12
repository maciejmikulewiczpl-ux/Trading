"""Probe 2: do MONTHLY SPY puts have daily bars at 30-60 DTE?

Picks ~5% OTM puts on 3rd-Friday expiries across the window and counts daily
bars in the 60 days before expiry. If liquid monthlies are markable most days
from 45 DTE in, the credit-spread backtest is feasible on Alpaca bars alone.

Run:
    .venv/Scripts/python.exe backtest/probe_options_monthly.py
"""
from __future__ import annotations

import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.run_orb import load_env  # noqa: E402

from alpaca.trading.client import TradingClient  # noqa: E402
from alpaca.trading.requests import GetOptionContractsRequest  # noqa: E402
from alpaca.trading.enums import AssetStatus, ContractType  # noqa: E402
from alpaca.data.historical.option import OptionHistoricalDataClient  # noqa: E402
from alpaca.data.historical.stock import StockHistoricalDataClient  # noqa: E402
from alpaca.data.requests import OptionBarsRequest, StockBarsRequest  # noqa: E402
from alpaca.data.timeframe import TimeFrame  # noqa: E402

# 3rd Fridays (or Thursday when Good Friday), Feb 2024 .. May 2026 sample
MONTHLIES = [date(2024, 3, 15), date(2024, 8, 16), date(2024, 12, 20),
             date(2025, 4, 17), date(2025, 8, 15), date(2026, 1, 16),
             date(2026, 5, 15)]


def main() -> None:
    load_env()
    key, sec = os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"]
    tc = TradingClient(key, sec, paper=True)
    oc = OptionHistoricalDataClient(key, sec)
    sc = StockHistoricalDataClient(key, sec)

    spy = sc.get_stock_bars(StockBarsRequest(
        symbol_or_symbols="SPY", timeframe=TimeFrame.Day,
        start=datetime(2024, 1, 1))).df.reset_index().set_index("timestamp")

    print(f"{'expiry':<12}{'strike':>8}{'bars<=60DTE':>11}{'first bar(DTE)':>15}"
          f"{'bars 45-21DTE':>14}{'medvol':>8}")
    for exp in MONTHLIES:
        # SPY close ~60 days before expiry -> pick 5% OTM strike
        ref_day = exp - timedelta(days=60)
        px = float(spy.loc[spy.index.date <= ref_day, "close"].iloc[-1])
        target_strike = round(px * 0.95)
        req = GetOptionContractsRequest(
            underlying_symbols=["SPY"], status=AssetStatus.INACTIVE,
            type=ContractType.PUT,
            expiration_date_gte=exp, expiration_date_lte=exp, limit=1000,
        )
        cs = tc.get_option_contracts(req).option_contracts or []
        if not cs:
            print(f"{exp!s:<12}  no contracts")
            continue
        c = min(cs, key=lambda c: abs(float(c.strike_price) - target_strike))
        bars = oc.get_option_bars(OptionBarsRequest(
            symbol_or_symbols=c.symbol, timeframe=TimeFrame.Day,
            start=datetime.combine(exp - timedelta(days=60), datetime.min.time()),
        )).df
        if bars.empty:
            print(f"{exp!s:<12}{c.strike_price:>8}  NO BARS")
            continue
        ts = bars.index.get_level_values("timestamp")
        dte_first = (exp - ts.min().date()).days
        n_4521 = sum(21 <= (exp - t.date()).days <= 45 for t in ts)
        print(f"{exp!s:<12}{float(c.strike_price):>8.0f}{len(bars):>11}{dte_first:>15}"
              f"{n_4521:>14}{bars['volume'].median():>8.0f}")


if __name__ == "__main__":
    main()
