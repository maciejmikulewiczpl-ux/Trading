"""Feasibility probe: how good is Alpaca's options data for backtesting?

Questions (answered empirically, printed as a report):
  1. CONTRACTS — can we enumerate EXPIRED SPY contracts (needed to reconstruct
     historical chains)?  How far back do expirations go?
  2. BARS — how far back do daily option bars go?  (Alpaca docs say Feb 2024.)
  3. QUALITY — for a typical ~30-45 DTE OTM SPY put, are daily bars gap-free
     enough to mark a position every day?  What did bid/ask look like (latest
     quote, as a spread-cost sanity check)?

Run:
    .venv/Scripts/python.exe backtest/probe_options_data.py
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.run_orb import load_env  # noqa: E402

import os  # noqa: E402

from alpaca.trading.client import TradingClient  # noqa: E402
from alpaca.trading.requests import GetOptionContractsRequest  # noqa: E402
from alpaca.trading.enums import AssetStatus, ContractType  # noqa: E402
from alpaca.data.historical.option import OptionHistoricalDataClient  # noqa: E402
from alpaca.data.requests import OptionBarsRequest, OptionLatestQuoteRequest  # noqa: E402
from alpaca.data.timeframe import TimeFrame  # noqa: E402


def main() -> None:
    load_env()
    key, sec = os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"]
    tc = TradingClient(key, sec, paper=True)
    dc = OptionHistoricalDataClient(key, sec)

    # --- 1. expired-contract enumeration, walking back year by year ---------
    print("=" * 70)
    print("1. EXPIRED SPY PUT CONTRACTS (chain reconstruction)")
    for exp_start in [date(2024, 2, 1), date(2024, 6, 1), date(2025, 4, 1),
                      date(2025, 6, 1), date(2026, 1, 1)]:
        req = GetOptionContractsRequest(
            underlying_symbols=["SPY"],
            status=AssetStatus.INACTIVE,
            type=ContractType.PUT,
            expiration_date_gte=exp_start,
            expiration_date_lte=exp_start + timedelta(days=14),
            limit=500,
        )
        try:
            res = tc.get_option_contracts(req)
            n = len(res.option_contracts or [])
            sample = (res.option_contracts or [None])[0]
            samp = f"e.g. {sample.symbol} strike={sample.strike_price}" if sample else ""
            print(f"  expiries {exp_start} +14d: {n:>4} expired puts  {samp}")
        except Exception as e:  # noqa: BLE001
            print(f"  expiries {exp_start} +14d: ERROR {e}")

    # --- 2. earliest daily bars --------------------------------------------
    print("=" * 70)
    print("2. DAILY-BAR HISTORY DEPTH (per contract)")
    # pick one real expired contract per era and ask for all its bars
    probes = []
    for exp in [date(2024, 3, 28), date(2024, 9, 20), date(2025, 4, 17),
                date(2025, 9, 19), date(2026, 3, 20)]:
        req = GetOptionContractsRequest(
            underlying_symbols=["SPY"], status=AssetStatus.INACTIVE,
            type=ContractType.PUT,
            expiration_date_gte=exp - timedelta(days=3),
            expiration_date_lte=exp + timedelta(days=3),
            limit=500,
        )
        cs = tc.get_option_contracts(req).option_contracts or []
        if not cs:
            print(f"  expiry ~{exp}: no contracts returned")
            continue
        # mid-of-the-pack strike ~ ATM-ish
        cs = sorted(cs, key=lambda c: float(c.strike_price))
        probes.append(cs[len(cs) // 2].symbol)
    for sym in probes:
        req = OptionBarsRequest(symbol_or_symbols=sym, timeframe=TimeFrame.Day,
                                start=datetime(2023, 1, 1))
        try:
            bars = dc.get_option_bars(req).df
            if bars.empty:
                print(f"  {sym:<24} NO BARS")
            else:
                idx = bars.index.get_level_values("timestamp")
                print(f"  {sym:<24} {len(bars):>3} bars  {idx.min().date()} -> {idx.max().date()}")
        except Exception as e:  # noqa: BLE001
            print(f"  {sym:<24} ERROR {e}")

    # --- 3. bar continuity for a typical short-put candidate ----------------
    print("=" * 70)
    print("3. BAR CONTINUITY — ~5% OTM SPY put, Apr-2025 expiry (vol-spike era)")
    req = GetOptionContractsRequest(
        underlying_symbols=["SPY"], status=AssetStatus.INACTIVE,
        type=ContractType.PUT,
        expiration_date_gte=date(2025, 4, 14), expiration_date_lte=date(2025, 4, 21),
        limit=500,
    )
    cs = tc.get_option_contracts(req).option_contracts or []
    cs = sorted(cs, key=lambda c: float(c.strike_price))
    if cs:
        # SPY was ~560-570 in early Mar 2025; 5% OTM put ~ strike 535
        target = min(cs, key=lambda c: abs(float(c.strike_price) - 535))
        req = OptionBarsRequest(symbol_or_symbols=target.symbol, timeframe=TimeFrame.Day,
                                start=datetime(2025, 2, 1))
        bars = dc.get_option_bars(req).df
        idx = bars.index.get_level_values("timestamp")
        ndays = (idx.max() - idx.min()).days
        print(f"  {target.symbol}  strike {target.strike_price}")
        print(f"  {len(bars)} daily bars over {ndays} calendar days "
              f"({idx.min().date()} -> {idx.max().date()})")
        print(f"  volume: median {bars['volume'].median():.0f}, min {bars['volume'].min():.0f}")
        print(bars.tail(8)[["open", "high", "low", "close", "volume"]].to_string())
    else:
        print("  no contracts found for Apr 2025!")

    # --- 4. current spread sanity (live quote on an active OTM put) ---------
    print("=" * 70)
    print("4. SPREAD CHECK — live quote, active ~30-45 DTE OTM SPY put")
    req = GetOptionContractsRequest(
        underlying_symbols=["SPY"], status=AssetStatus.ACTIVE, type=ContractType.PUT,
        expiration_date_gte=date.today() + timedelta(days=30),
        expiration_date_lte=date.today() + timedelta(days=45),
        limit=500,
    )
    cs = tc.get_option_contracts(req).option_contracts or []
    if cs:
        strikes = sorted({float(c.strike_price) for c in cs})
        print(f"  {len(cs)} active puts, strikes {strikes[0]:.0f}..{strikes[-1]:.0f}, "
              f"{len(strikes)} distinct strikes")
        # grab a handful around 4-6% OTM (rough — just for spread sizing)
        spy_px = float(os.environ.get("PROBE_SPY_PX", "0")) or None
        sample = [c for c in cs if 0.90 <= float(c.strike_price) / strikes[-1] <= 0.95][:4]
        for c in sample:
            try:
                q = dc.get_option_latest_quote(
                    OptionLatestQuoteRequest(symbol_or_symbols=c.symbol))[c.symbol]
                mid = (q.bid_price + q.ask_price) / 2
                spr = q.ask_price - q.bid_price
                pct = 100 * spr / mid if mid else float("nan")
                print(f"  {c.symbol:<24} bid {q.bid_price:>6.2f} ask {q.ask_price:>6.2f} "
                      f"spread {spr:.2f} ({pct:.1f}% of mid)")
            except Exception as e:  # noqa: BLE001
                print(f"  {c.symbol:<24} quote ERROR {e}")
    print("=" * 70)


if __name__ == "__main__":
    main()
