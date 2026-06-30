"""trend_following.py -- a cross-asset TIME-SERIES MOMENTUM (trend-following) backtest on a
diversified basket of liquid ETFs that proxy the major futures markets. The legit cousin of
the hype bot: same DNA (rank/ride momentum, risk-weight, cut losers) but on ASSET CLASSES
over a slow (monthly) horizon -- the documented managed-futures / TSMOM strategy class.

Method (canonical, lookahead-free, monthly rebalance):
  - signal_i = sign(12-month return) at month-end t  -> long (+1) / short (-1) each market
  - inverse-volatility weighting (trailing 12mo) so no single market dominates the risk
  - portfolio return(t+1) = sum_i position_i(t) * return_i(t+1)   [positions lag returns]
  - also a LONG-ONLY variant (skip shorts) for comparison
Benchmarks: SPY buy-and-hold, and a 60/40 (60% SPY / 40% TLT).

HONEST CAVEATS: ETF proxies (UNLEVERED -- real futures would scale these up AND down via
margin); frictionless (trend-following is low-turnover so costs are small, but not zero);
these ETFs all survived (mild survivorship); single lookback (12mo). Trend-following is a
REAL but MODEST + CYCLICAL edge (great 2008/2022, dull in chop) -- not a +600% machine.

Run (yfinance):  .venv-openbb/Scripts/python.exe backtest/trend_following.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# diversified basket across equity / bond / commodity / FX (liquid, long history)
TICKERS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "SLV", "USO", "DBC", "UUP"]
START = "2007-01-01"
MOM_LB = 12          # months of trailing return for the trend signal
VOL_LB = 12          # months for inverse-vol weighting


def _stats(monthly_ret: pd.Series, spy: pd.Series) -> dict:
    r = monthly_ret.dropna()
    if len(r) < 12:
        return {}
    eq = (1 + r).cumprod()
    yrs = len(r) / 12.0
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(12)
    sharpe = (r.mean() * 12) / vol if vol else float("nan")
    dd = (eq / eq.cummax() - 1).min()
    corr = r.corr(spy.reindex(r.index)) if spy is not None else float("nan")
    return {"CAGR": cagr, "vol": vol, "sharpe": sharpe, "maxDD": dd, "corr_SPY": corr,
            "eq": eq}


def main() -> int:
    import yfinance as yf
    print("=== cross-asset trend-following (TSMOM) on liquid ETF proxies ===")
    print(f"basket: {', '.join(TICKERS)}\n")
    raw = yf.download(TICKERS, start=START, auto_adjust=True, progress=False)
    px = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    px = px[[t for t in TICKERS if t in px.columns]].dropna(how="all")
    m = px.resample("ME").last()
    rets = m.pct_change()
    mom = m / m.shift(MOM_LB) - 1.0
    vol = rets.rolling(VOL_LB).std()
    invvol = 1.0 / vol.replace(0, np.nan)

    def run(signal: pd.DataFrame, name: str):
        w = invvol.where(signal.notna())
        w = w.div(w.sum(axis=1), axis=0)            # normalize over markets with a signal
        pos = signal * w
        port = (pos.shift(1) * rets).sum(axis=1)
        port[pos.shift(1).abs().sum(axis=1) == 0] = np.nan   # no positions yet -> NaN
        return name, port

    spy_m = rets["SPY"]
    runs = [
        run(np.sign(mom), "TSMOM long/short"),
        run((mom > 0).astype(float).where(mom.notna()), "TSMOM long-only"),
    ]
    bench = [("SPY buy&hold", spy_m),
             ("60/40 SPY/TLT", 0.6 * rets["SPY"] + 0.4 * rets["TLT"])]

    print(f"  {'strategy':>22}{'CAGR':>8}{'vol':>8}{'Sharpe':>8}{'maxDD':>9}{'corrSPY':>9}")
    eqs = {}
    for name, series in runs + bench:
        st = _stats(series, spy_m)
        if not st:
            print(f"  {name:>22}  (insufficient)"); continue
        eqs[name] = st["eq"]
        print(f"  {name:>22}{st['CAGR']*100:>7.1f}%{st['vol']*100:>7.1f}%{st['sharpe']:>8.2f}"
              f"{st['maxDD']*100:>8.1f}%{st['corr_SPY']:>9.2f}")

    # per-year returns: trend long/short vs SPY
    ls = dict(runs)["TSMOM long/short"]
    print("\nPer-year return — TSMOM long/short vs SPY (the diversification test):")
    print(f"  {'year':>6}{'TSMOM':>10}{'SPY':>10}")
    yr_ls = (1 + ls).groupby(ls.index.year).prod() - 1
    yr_spy = (1 + spy_m).groupby(spy_m.index.year).prod() - 1
    for y in sorted(set(yr_ls.index) | set(yr_spy.index)):
        a = yr_ls.get(y); b = yr_spy.get(y)
        print(f"  {y:>6}{(f'{a*100:+.1f}%' if a==a else '   -'):>10}{(f'{b*100:+.1f}%' if b==b else '   -'):>10}")

    print("\nREAD: trend-following EARNS ITS KEEP via Sharpe + LOW correlation to SPY + smaller")
    print("drawdowns + winning in SPY's bad years (2008/2022), NOT via beating SPY's CAGR in a")
    print("bull market. Unlevered ETF proxy; futures would scale it. Frictionless (low turnover).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
