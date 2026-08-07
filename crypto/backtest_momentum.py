"""Crypto TREND-following sleeve -- validation (mirrors futures/backtest_momentum + diversification).

TWO questions, both required for this to be worth building:
  1. Does long-only time-series MOMENTUM on a crypto basket beat buy-hold RISK-ADJUSTED (Sharpe up,
     drawdown WAY down through the 2022 bear)? Crypto buy-hold is high-return but −80% DD; trend
     should sidestep the bears.
  2. Is the crypto-trend return stream UNCORRELATED with EQUITIES (SPY)? That's the whole point --
     a new uncorrelated sleeve for the diversified momentum portfolio (equity ORB + futures TSMOM +
     crypto). If it just tracks SPY, it adds nothing.

Strategy: for each coin, LONG if trailing-LOOKBACK return > 0 (time-series momentum), else FLAT;
inverse-vol weighted across the coins currently long; daily rebalance; long-only (crypto shorting
is costly). Frictionless daily bars (Alpaca, 2021-) -> directional; a real sleeve nets fees/spread.
Run: .venv/Scripts/python.exe crypto/backtest_momentum.py"""
from __future__ import annotations
import os, sys, datetime as dt, statistics, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtest.run_orb import load_env

PAIRS = ["BTC/USD", "ETH/USD", "SOL/USD", "LTC/USD", "BCH/USD", "AVAX/USD", "LINK/USD", "UNI/USD", "DOGE/USD", "AAVE/USD"]
LOOKBACKS = [30, 60, 90, 120]        # time-series-momentum lookback (days) to sweep
VOL_WIN = 30                          # inverse-vol weighting window


def load():
    load_env()
    from alpaca.data.historical.crypto import CryptoHistoricalDataClient
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    cc = CryptoHistoricalDataClient()
    px = {}
    cdf = cc.get_crypto_bars(CryptoBarsRequest(symbol_or_symbols=PAIRS, timeframe=TimeFrame.Day,
          start=dt.datetime(2021, 1, 1))).df
    for p in PAIRS:
        try:
            px[p] = {d.date(): float(r.close) for d, r in cdf.xs(p, level=0).iterrows()}
        except KeyError:
            pass
    # SPY daily for the equity-correlation test
    sc = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    sdf = sc.get_stock_bars(StockBarsRequest(symbol_or_symbols="SPY", timeframe=TimeFrame.Day,
          start=dt.datetime(2021, 1, 1), feed=DataFeed.IEX)).df
    spy = {d.tz_convert("UTC").date(): float(r.close) for d, r in sdf.xs("SPY", level=0).iterrows()}
    return px, spy


def series(px):
    dates = sorted(set().union(*[set(px[p]) for p in px]))
    return dates


def run(px, lookback):
    coins = list(px)
    dates = series(px)
    port = []          # (date, portfolio daily return)
    for i in range(max(lookback, VOL_WIN) + 1, len(dates)):
        d, dprev = dates[i], dates[i - 1]
        legs = []
        for c in coins:
            s = px[c]
            if d not in s or dprev not in s:
                continue
            dref = dates[i - lookback]
            if dref not in s:
                continue
            mom = s[dprev] / s[dref] - 1.0            # trailing return known at yesterday's close (no lookahead)
            if mom <= 0:
                continue
            # inverse-vol weight from the trailing VOL_WIN daily returns (through yesterday)
            rets = []
            for j in range(i - VOL_WIN, i - 1):
                a, b = dates[j], dates[j + 1]
                if a in s and b in s and s[a]:
                    rets.append(s[b] / s[a] - 1.0)
            vol = statistics.pstdev(rets) if len(rets) > 3 else None
            if not vol or vol <= 0:
                continue
            todret = s[d] / s[dprev] - 1.0
            legs.append((1.0 / vol, todret))
        if legs:
            wsum = sum(w for w, _ in legs)
            port.append((d, sum(w * r for w, r in legs) / wsum))
        else:
            port.append((d, 0.0))
    return port


def stats(rets, ann=365):
    if not rets:
        return None
    mean, sd = statistics.mean(rets), statistics.pstdev(rets)
    sharpe = (mean / sd * math.sqrt(ann)) if sd else 0
    # equity curve + maxDD
    eq = 1.0; peak = 1.0; mdd = 0.0
    for r in rets:
        eq *= (1 + r); peak = max(peak, eq); mdd = min(mdd, eq / peak - 1)
    cagr = eq ** (ann / len(rets)) - 1
    return dict(n=len(rets), cagr=cagr, sharpe=sharpe, mdd=mdd, total=eq - 1)


def buyhold(px, coins):
    dates = series(px)
    rets = []
    for i in range(1, len(dates)):
        d, dp = dates[i], dates[i - 1]
        legs = [px[c][d] / px[c][dp] - 1 for c in coins if d in px[c] and dp in px[c] and px[c][dp]]
        if legs:
            rets.append(sum(legs) / len(legs))
    return rets


def main():
    px, spy = load()
    print(f"loaded {len(px)} coins, {len(series(px))} days ({series(px)[0]}..{series(px)[-1]})")
    print("\n=== CRYPTO TREND (long-only TSMOM, inverse-vol) vs BUY-HOLD ===")
    print(f"  {'strategy':<22}{'CAGR':>8}{'Sharpe':>8}{'maxDD':>8}{'total':>9}")
    bh = stats(buyhold(px, list(px)))
    print(f"  {'buy-hold basket':<22}{bh['cagr']*100:>7.0f}%{bh['sharpe']:>8.2f}{bh['mdd']*100:>7.0f}%{bh['total']*100:>8.0f}%")
    bhbtc = stats([px['BTC/USD'][d]/px['BTC/USD'][dp]-1 for dp,d in zip(series({'BTC/USD':px['BTC/USD']}), series({'BTC/USD':px['BTC/USD']})[1:]) if dp in px['BTC/USD'] and d in px['BTC/USD']])
    print(f"  {'buy-hold BTC':<22}{bhbtc['cagr']*100:>7.0f}%{bhbtc['sharpe']:>8.2f}{bhbtc['mdd']*100:>7.0f}%{bhbtc['total']*100:>8.0f}%")
    best = None
    for lb in LOOKBACKS:
        port = run(px, lb)
        st = stats([r for _, r in port])
        if best is None or st['sharpe'] > best[1]['sharpe']:
            best = (lb, st, port)
        print(f"  {'trend '+str(lb)+'d':<22}{st['cagr']*100:>7.0f}%{st['sharpe']:>8.2f}{st['mdd']*100:>7.0f}%{st['total']*100:>8.0f}%")

    # THE diversification test: correlation of the best crypto-trend vs SPY
    lb, st, port = best
    pmap = dict(port)
    pairs = []
    sdates = sorted(spy)
    for i in range(1, len(sdates)):
        d, dp = sdates[i], sdates[i - 1]
        if d in pmap and spy.get(dp):
            pairs.append((pmap[d], spy[d] / spy[dp] - 1))
    if len(pairs) > 30:
        cx = [a for a, _ in pairs]; cy = [b for _, b in pairs]
        mx, my = statistics.mean(cx), statistics.mean(cy)
        cov = sum((a-mx)*(b-my) for a, b in pairs) / len(pairs)
        corr = cov / (statistics.pstdev(cx) * statistics.pstdev(cy)) if statistics.pstdev(cx) and statistics.pstdev(cy) else 0
        print(f"\n=== DIVERSIFICATION: best trend ({lb}d) vs SPY ===")
        print(f"  correlation with SPY daily returns: {corr:+.2f}  (n={len(pairs)} overlapping days)")
        print(f"  -> {'UNCORRELATED = real diversifier' if abs(corr)<0.3 else 'CORRELATED = adds little'}")
    print("\n[caveats] frictionless daily bars; long-only; 2021- (one full cycle); crypto fees/spread ~"
          "10-30bps/trade not modeled. Relative read. A real sleeve = weekly rebalance + cost model.")


if __name__ == "__main__":
    main()
