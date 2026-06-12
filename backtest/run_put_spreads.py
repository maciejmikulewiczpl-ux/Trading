"""SPY put-credit-spread feasibility backtest (options engine, door #1).

PRE-REGISTERED SPEC (written before running — do not tune after the fact):
  Underlying SPY, MONTHLY expiries only, ONE position at a time.
  Entry: first day with no open position where a monthly expiry sits 28-52 DTE
         (nearest such expiry). Short strike = closest to 30-delta (delta from
         Black-Scholes, IV implied from the option's own close). Long leg $10
         lower. Fill at bar closes.
  Manage daily on closes:
    PROFIT  spread value <= 50% of credit          -> buy back
    TIME    DTE <= 21                              -> buy back
    STOP    spread value >= 3x credit (2x-credit loss) -> buy back
  Costs: $0.05/leg/side slippage ($20 per spread round trip per contract)
         + $2.60 fees per spread round trip. Conservative for SPY.
  Window: 2024-02-01 (start of Alpaca options data) -> today.

GATES (all must hold to call the avenue "viable"):
  G1 data    >=95% of position-days markable from bars (carry-forward <=1 day)
  G2 income  net PnL > 0 after costs over the full window
  G3 tail    no blow-through: worst realized loss <= 4x its trade's credit
             (i.e. the daily-close stop rule actually contains gap risk)
  Sanity (not a gate): annualized return on $1k collateral in the 5-9% band
  the PUT index would suggest; far outside either way = look for bugs.

Run:
    .venv/Scripts/python.exe backtest/run_put_spreads.py          # uses cache
    .venv/Scripts/python.exe backtest/run_put_spreads.py --fetch  # refetch
"""
from __future__ import annotations

import math
import os
import pickle
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.run_orb import load_env  # noqa: E402

CACHE = ROOT / "backtest" / ".put_spread_chain_cache.pkl"
START = date(2024, 2, 1)
RISK_FREE = 0.045
DIV_YIELD = 0.012
WIDTH = 10.0
SLIP_RT = 0.20          # $/share spread round trip (4 legs x $0.05)
FEES_RT = 2.60          # $ per spread round trip
PROFIT_TAKE = 0.50
STOP_MULT = 3.0         # exit when value >= 3x credit
DTE_EXIT = 21
DTE_MIN, DTE_MAX = 28, 52
TARGET_DELTA = -0.30


# ---------------------------------------------------------------- BS helpers
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_put(spot: float, strike: float, t: float, vol: float) -> float:
    if t <= 0 or vol <= 0:
        return max(strike - spot, 0.0)
    d1 = (math.log(spot / strike) + (RISK_FREE - DIV_YIELD + 0.5 * vol * vol) * t) / (vol * math.sqrt(t))
    d2 = d1 - vol * math.sqrt(t)
    return strike * math.exp(-RISK_FREE * t) * _norm_cdf(-d2) - spot * math.exp(-DIV_YIELD * t) * _norm_cdf(-d1)


def put_delta(spot: float, strike: float, t: float, vol: float) -> float:
    if t <= 0 or vol <= 0:
        return -1.0 if strike > spot else 0.0
    d1 = (math.log(spot / strike) + (RISK_FREE - DIV_YIELD + 0.5 * vol * vol) * t) / (vol * math.sqrt(t))
    return -math.exp(-DIV_YIELD * t) * _norm_cdf(-d1)


def implied_vol(price: float, spot: float, strike: float, t: float) -> float | None:
    lo, hi = 0.01, 3.0
    if not (bs_put(spot, strike, t, lo) <= price <= bs_put(spot, strike, t, hi)):
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        if bs_put(spot, strike, t, mid) < price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ---------------------------------------------------------------- data fetch
def third_friday(y: int, m: int) -> date:
    d = date(y, m, 15)
    while d.weekday() != 4:
        d += timedelta(days=1)
    return d


def fetch_chains() -> dict:
    """Per monthly expiry: {expiry, strikes->{date->close}} + SPY daily closes."""
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetOptionContractsRequest
    from alpaca.trading.enums import AssetStatus, ContractType
    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import OptionBarsRequest, StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    key, sec = os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"]
    tc = TradingClient(key, sec, paper=True)
    oc = OptionHistoricalDataClient(key, sec)
    sc = StockHistoricalDataClient(key, sec)

    spy = sc.get_stock_bars(StockBarsRequest(
        symbol_or_symbols="SPY", timeframe=TimeFrame.Day,
        start=datetime(2023, 12, 1))).df.reset_index()
    spy["d"] = spy["timestamp"].dt.date
    spy_close = dict(zip(spy["d"], spy["close"]))

    today = date.today()
    expiries = []
    y, m = START.year, START.month
    while (y, m) <= (today.year, today.month):
        tf = third_friday(y, m)
        if tf > START:
            expiries.append(tf)
        m += 1
        if m == 13:
            y, m = y + 1, 1
    chains = {}
    for exp in expiries:
        ref = exp - timedelta(days=55)
        ref_px = next((spy_close[ref - timedelta(days=k)]
                       for k in range(7) if ref - timedelta(days=k) in spy_close), None)
        if ref_px is None:
            continue
        lo_k, hi_k = ref_px * 0.82, ref_px * 1.02
        status = AssetStatus.INACTIVE if exp <= today else AssetStatus.ACTIVE
        symbols, actual_exp, token = [], None, None
        while True:
            req = GetOptionContractsRequest(
                underlying_symbols=["SPY"], status=status, type=ContractType.PUT,
                expiration_date_gte=exp - timedelta(days=1),
                expiration_date_lte=exp, limit=1000, page_token=token)
            res = tc.get_option_contracts(req)
            for c in res.option_contracts or []:
                if lo_k <= float(c.strike_price) <= hi_k:
                    symbols.append((c.symbol, float(c.strike_price), c.expiration_date))
            token = res.next_page_token
            if not token:
                break
        if not symbols:
            print(f"  {exp}: no contracts in strike range — skipped")
            continue
        # monthly = latest expiry present (Thu if Fri is a holiday)
        actual_exp = max(s[2] for s in symbols)
        symbols = [s for s in symbols if s[2] == actual_exp]
        sym_list = [s[0] for s in symbols]
        strikes = {s[0]: s[1] for s in symbols}
        closes: dict[float, dict[date, float]] = {}
        for i in range(0, len(sym_list), 100):
            chunk = sym_list[i:i + 100]
            bars = oc.get_option_bars(OptionBarsRequest(
                symbol_or_symbols=chunk, timeframe=TimeFrame.Day,
                start=datetime.combine(actual_exp - timedelta(days=60), datetime.min.time()),
            )).df
            if bars.empty:
                continue
            for (sym, ts), row in bars.iterrows():
                closes.setdefault(strikes[sym], {})[ts.date()] = float(row["close"])
        chains[actual_exp] = closes
        print(f"  {actual_exp}: {len(closes)} strikes with bars")
    return {"chains": chains, "spy": spy_close}


# ---------------------------------------------------------------- simulation
@dataclass
class Trade:
    entry: date
    expiry: date
    short_k: float
    long_k: float
    credit: float
    exit: date | None = None
    exit_value: float | None = None
    reason: str = ""
    marks: list = field(default_factory=list)

    @property
    def pnl(self) -> float:  # per contract, $
        return (self.credit - self.exit_value) * 100 - SLIP_RT * 100 - FEES_RT


def leg_close(chain: dict, strike: float, d: date, lookback: int = 1) -> float | None:
    """Close for strike on day d, carrying forward at most `lookback` days."""
    for k in range(lookback + 1):
        v = chain.get(strike, {}).get(d - timedelta(days=k))
        if v is not None:
            return v
    return None


def simulate(data: dict) -> tuple[list[Trade], dict]:
    chains: dict[date, dict] = data["chains"]
    spy: dict[date, float] = data["spy"]
    days = sorted(d for d in spy if d >= START)
    expiries = sorted(chains)

    trades: list[Trade] = []
    open_t: Trade | None = None
    stale_marks = total_marks = 0

    for d in days:
        if open_t is not None:
            ch = chains[open_t.expiry]
            sv = leg_close(ch, open_t.short_k, d, 0)
            lv = leg_close(ch, open_t.long_k, d, 0)
            total_marks += 1
            if sv is None or lv is None:
                stale_marks += 1
                sv = leg_close(ch, open_t.short_k, d, 1)
                lv = leg_close(ch, open_t.long_k, d, 1)
                if sv is None or lv is None:
                    continue
            value = sv - lv
            open_t.marks.append((d, value))
            dte = (open_t.expiry - d).days
            reason = None
            if value <= PROFIT_TAKE * open_t.credit:
                reason = "profit"
            elif value >= STOP_MULT * open_t.credit:
                reason = "stop"
            elif dte <= DTE_EXIT:
                reason = "time"
            elif d >= open_t.expiry:
                reason = "expiry"
                value = max(open_t.short_k - spy[d], 0) - max(open_t.long_k - spy[d], 0)
            if reason:
                open_t.exit, open_t.exit_value, open_t.reason = d, value, reason
                open_t = None
                continue

        if open_t is None and d in spy:
            spot = spy[d]
            cand = [e for e in expiries if DTE_MIN <= (e - d).days <= DTE_MAX]
            if not cand:
                continue
            exp = cand[0]
            ch = chains[exp]
            t_yrs = (exp - d).days / 365.0
            best, best_gap = None, 1e9
            for k in sorted(ch):
                if k >= spot or (k - WIDTH) not in ch:
                    continue
                px = ch[k].get(d)
                if px is None or px < 0.10:
                    continue
                iv = implied_vol(px, spot, k, t_yrs)
                if iv is None:
                    continue
                delta = put_delta(spot, k, t_yrs, iv)
                if abs(delta - TARGET_DELTA) < best_gap:
                    best, best_gap = k, abs(delta - TARGET_DELTA)
            if best is None:
                continue
            sv, lv = ch[best].get(d), ch[best - WIDTH].get(d)
            if sv is None or lv is None or sv - lv <= 0.15:
                continue
            open_t = Trade(entry=d, expiry=exp, short_k=best,
                           long_k=best - WIDTH, credit=sv - lv)
            trades.append(open_t)

    if open_t is not None and open_t.exit is None:
        trades.remove(open_t)  # still open — drop from stats
    stats = {"stale_pct": 100 * stale_marks / max(total_marks, 1)}
    return trades, stats


def report(trades: list[Trade], stats: dict, data: dict) -> None:
    print(f"\n{'entry':<12}{'exit':<12}{'expiry':<12}{'K':>5}{'credit':>8}"
          f"{'exitval':>8}{'PnL$':>8}  reason")
    for t in trades:
        print(f"{t.entry!s:<12}{t.exit!s:<12}{t.expiry!s:<12}{t.short_k:>5.0f}"
              f"{t.credit:>8.2f}{t.exit_value:>8.2f}{t.pnl:>+8.0f}  {t.reason}")

    pnls = [t.pnl for t in trades]
    cum = pd.Series(pnls).cumsum()
    maxdd = (cum - cum.cummax()).min()
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    worst = min(trades, key=lambda t: t.pnl)
    days_span = (trades[-1].exit - trades[0].entry).days
    ann_ret = sum(pnls) / 1000.0 / (days_span / 365.0) * 100  # on $1k collateral

    spy = data["spy"]
    d0, d1 = trades[0].entry, trades[-1].exit
    spy_ret = (spy[max(d for d in spy if d <= d1)] / spy[min(d for d in spy if d >= d0)] - 1) * 100

    print("\n" + "=" * 64)
    print(f"trades {len(trades)}  |  wins {len(wins)}  losses {len(losses)}  "
          f"win rate {100 * len(wins) / len(trades):.0f}%")
    print(f"net PnL ${sum(pnls):+,.0f}/contract  |  avg win ${pd.Series(wins).mean():.0f}  "
          f"avg loss ${pd.Series(losses).mean():.0f}")
    print(f"max cum drawdown ${maxdd:,.0f}  |  worst trade ${worst.pnl:+,.0f} "
          f"({worst.entry}, {worst.reason}, {worst.pnl / (worst.credit * 100):+.1f}x credit)")
    print(f"annualized return on $1k max-loss collateral: {ann_ret:+.1f}%  "
          f"(SPY buy-hold same window: {spy_ret:+.1f}%)")
    print(f"exit mix: " + ", ".join(f"{r}={sum(1 for t in trades if t.reason == r)}"
                                    for r in ("profit", "time", "stop", "expiry")))
    print("-" * 64)
    g1 = stats["stale_pct"] <= 5.0
    g2 = sum(pnls) > 0
    g3 = all((t.credit - t.exit_value) * 100 >= -4 * t.credit * 100 for t in trades)
    print(f"G1 data   stale marks {stats['stale_pct']:.1f}% (<=5%):        {'PASS' if g1 else 'FAIL'}")
    print(f"G2 income net PnL > 0 after costs:           {'PASS' if g2 else 'FAIL'}")
    print(f"G3 tail   worst loss <= 4x credit:           {'PASS' if g3 else 'FAIL'}")
    print("=" * 64)


def main() -> None:
    load_env()
    if "--fetch" in sys.argv or not CACHE.exists():
        print("fetching chains from Alpaca (cached afterwards)...")
        data = fetch_chains()
        pickle.dump(data, open(CACHE, "wb"))
    else:
        data = pickle.load(open(CACHE, "rb"))
    trades, stats = simulate(data)
    if not trades:
        print("NO TRADES — check data")
        return
    report(trades, stats, data)


if __name__ == "__main__":
    main()
