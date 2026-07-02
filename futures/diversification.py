"""Multi-asset DIVERSIFICATION study (the legitimate path to more risk-adjusted profit).

Two US equity indices are 0.78 correlated -> no diversification. Real diversification needs UNCORRELATED
sleeves. This runs TIME-SERIES MOMENTUM (managed-futures TSMOM) on a cross-asset ETF basket (equities +
bonds + metals + commodities + dollar), then asks: does combining the sleeves beat any single one on a
RISK-ADJUSTED basis, and by how much? Daily data via yfinance (no TWS needed).

Method (lookahead-free, monthly rebalance): signal_i = sign(MOM_LB-month return); inverse-vol weight;
portfolio_ret(t+1) = sum_i pos_i(t) * ret_i(t+1). Long/short and long-only variants. Reports per-sleeve
Sharpe, the correlation matrix of the SLEEVE returns (not raw assets), the combined portfolio vs SPY
buy-hold, crisis-year returns, and a leverage-to-SPY-return comparison (the "more profit, less DD" case).

    .venv-openbb/Scripts/python.exe futures/diversification.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BASKET = {"SPY": "US large", "QQQ": "US tech", "IWM": "US small", "EFA": "Dev intl", "EEM": "EM",
          "TLT": "Long bond", "IEF": "7-10y bond", "GLD": "Gold", "SLV": "Silver",
          "DBC": "Commodities", "USO": "Oil", "UUP": "US dollar"}
START = "2007-01-01"
MOM_LB, VOL_LB = 12, 12


def _stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 24:
        return {}
    eq = (1 + r).cumprod()
    yrs = len(r) / 12.0
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(12)
    return {"CAGR": cagr, "vol": vol, "sharpe": (r.mean() * 12) / vol if vol else float("nan"),
            "maxDD": (eq / eq.cummax() - 1).min(), "r": r, "eq": eq}


def main() -> int:
    import yfinance as yf
    print(f"=== Cross-asset TSMOM diversification | {len(BASKET)} sleeves | {START}-> ===")
    raw = yf.download(list(BASKET), start=START, auto_adjust=True, progress=False)
    px = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    px = px[[t for t in BASKET if t in px.columns]].dropna(how="all")
    m = px.resample("ME").last()
    rets = m.pct_change()
    mom = m / m.shift(MOM_LB) - 1.0
    vol = rets.rolling(VOL_LB).std()
    invvol = 1.0 / vol.replace(0, np.nan)
    sig = np.sign(mom)                       # long/short trend signal

    # per-sleeve trend return (single-asset long/short, lagged -> no lookahead)
    sleeve = (sig.shift(1) * rets).dropna(how="all")
    print("\n  per-sleeve TSMOM (long/short):")
    print(f"    {'sleeve':<12}{'CAGR':>8}{'vol':>7}{'Sharpe':>8}{'maxDD':>8}")
    sl_sharpe = {}
    for t in sleeve.columns:
        s = _stats(sleeve[t])
        if s:
            sl_sharpe[t] = s["sharpe"]
            print(f"    {t+' '+BASKET[t]:<12}{s['CAGR']*100:>+7.1f}%{s['vol']*100:>6.1f}%"
                  f"{s['sharpe']:>+8.2f}{s['maxDD']*100:>+7.0f}%")

    # correlation of the SLEEVE returns (this is what diversification actually depends on)
    corr = sleeve.corr()
    print(f"\n  sleeve-return correlation: mean pairwise = {corr.values[np.triu_indices(len(corr),1)].mean():.2f}")
    print("    least-correlated pairs (the real diversifiers):")
    pairs = [(corr.index[i], corr.columns[j], corr.iloc[i, j])
             for i in range(len(corr)) for j in range(i + 1, len(corr))]
    for a, b, c in sorted(pairs, key=lambda x: x[2])[:6]:
        print(f"      {a}-{b}: {c:+.2f}")

    # combined portfolios (inverse-vol / equal-risk), positions lag returns
    w = invvol.div(invvol.sum(axis=1), axis=0)
    port_ls = (sig.shift(1) * w.shift(1) * rets).sum(axis=1)          # long/short, risk-weighted
    port_lo = ((sig.shift(1).clip(lower=0)) * w.shift(1) * rets).sum(axis=1)   # long-only variant
    spy_bh = rets["SPY"] if "SPY" in rets else None
    bench = {"TSMOM L/S (all sleeves)": _stats(port_ls), "TSMOM long-only": _stats(port_lo),
             "SPY buy&hold": _stats(spy_bh) if spy_bh is not None else {}}
    print("\n  PORTFOLIO vs benchmark:")
    print(f"    {'strategy':<26}{'CAGR':>8}{'vol':>7}{'Sharpe':>8}{'maxDD':>8}")
    for name, s in bench.items():
        if s:
            print(f"    {name:<26}{s['CAGR']*100:>+7.1f}%{s['vol']*100:>6.1f}%{s['sharpe']:>+8.2f}"
                  f"{s['maxDD']*100:>+7.0f}%")

    # crisis-year returns (does the diversified trend hedge equity crashes?)
    print("\n  crisis-year total return (portfolio L/S vs SPY):")
    for yr in (2008, 2020, 2022):
        p = port_ls[port_ls.index.year == yr]
        sp = spy_bh[spy_bh.index.year == yr] if spy_bh is not None else pd.Series(dtype=float)
        if len(p):
            pr = (1 + p).prod() - 1
            spr = (1 + sp).prod() - 1 if len(sp) else float("nan")
            print(f"    {yr}: TSMOM {pr*100:+.0f}%   SPY {spr*100:+.0f}%")

    # leverage-to-target: lever the BEST-Sharpe diversified book to SPY's CAGR; compare drawdown.
    sp = _stats(spy_bh)
    best_name, best = max((("TSMOM L/S", _stats(port_ls)), ("TSMOM long-only", _stats(port_lo))),
                          key=lambda x: (x[1] or {"sharpe": -9})["sharpe"])
    if best and sp and best["CAGR"] > 0:
        lev = sp["CAGR"] / best["CAGR"]
        lev_lo = _stats((port_lo if "long-only" in best_name else port_ls) * lev)
        print(f"\n  LEVERAGE-TO-TARGET (best book = {best_name}, Sharpe {best['sharpe']:.2f}):")
        print(f"    to match SPY's {sp['CAGR']*100:.1f}%/yr, lever it {lev:.1f}x -> maxDD "
              f"{lev_lo['maxDD']*100:+.0f}%  vs  SPY buy&hold maxDD {sp['maxDD']*100:+.0f}%")
        print(f"    => SAME return as SPY with {'LESS' if lev_lo['maxDD']>sp['maxDD'] else 'MORE'} "
              f"drawdown ({abs(lev_lo['maxDD']/sp['maxDD']):.0%} of SPY's DD). THAT is the diversification win.")

    print("\nRead: diversification helps iff the COMBINED Sharpe > the best single sleeve AND sleeve corr")
    print("is low. The real 'more profit' argument = a high-Sharpe diversified book can be LEVERED to a")
    print("target return with SMALLER drawdown than the single asset. ETF proxies (unlevered), frictionless,")
    print("survivorship-mild; RELATIVE ranking is the signal. Directional.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
