"""The real portfolio: does the diversified DAILY-trend book (Sharpe ~0.94) combine with our MES
INTRADAY momentum (Sharpe ~1.1) to beat either alone? Two edges on totally different horizons should
be ~uncorrelated -> combining two uncorrelated ~1.0-Sharpe books lifts the aggregate Sharpe.

Builds monthly return series for both over their overlap, measures correlation + the equal-risk combo
Sharpe, and reports the theoretical combo using the more-robust full-period Sharpes. yfinance + the MES
5-min cache (no TWS). SMALL overlap (~22 months) -> directional.

    .venv-openbb/Scripts/python.exe futures/portfolio_combo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASKET = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "SLV", "DBC", "USO", "UUP"]
MOM_LB, VOL_LB = 12, 12
FULL_SHARPE = {"MES_intraday_momentum": 1.10, "diversified_TSMOM": 0.94}   # from prior full-period runs


def _mes_monthly() -> pd.Series:
    import futures.backtest_momentum as bm
    from futures.data import load_mes_intraday_cache
    bm.POINT_VALUE, bm.COST_SIDE_USD = 5.0, 2.0
    df = load_mes_intraday_cache()
    notional = df["close"].iloc[-1] * 5.0                       # ~ MES $ notional
    tr = bm.backtest(df, trail_k=999, gap_adj=True)             # validated config
    tr["date"] = pd.to_datetime(tr["date"])
    daily = tr.set_index("date")["ret_usd"] / notional         # daily return on notional
    return daily.resample("ME").sum()                           # monthly (returns are tiny -> sum ~ compound)


def _tsmom_monthly() -> pd.Series:
    import yfinance as yf
    raw = yf.download(BASKET, start="2007-01-01", auto_adjust=True, progress=False)
    px = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    px = px[[t for t in BASKET if t in px.columns]].dropna(how="all")
    m = px.resample("ME").last()
    rets = m.pct_change()
    sig = np.sign(m / m.shift(MOM_LB) - 1.0).clip(lower=0)      # long-only trend (the better book)
    invvol = 1.0 / rets.rolling(VOL_LB).std().replace(0, np.nan)
    w = invvol.div(invvol.sum(axis=1), axis=0)
    return (sig.shift(1) * w.shift(1) * rets).sum(axis=1)


def _sh(x):
    x = x.dropna()
    return x.mean() / x.std() * np.sqrt(12) if x.std() else float("nan")


def main() -> int:
    mes = _mes_monthly().rename("MES_intraday_momentum")
    tsm = _tsmom_monthly().rename("diversified_TSMOM")
    both = pd.concat([mes, tsm], axis=1).dropna()
    print(f"=== PORTFOLIO COMBO: MES intraday momentum + diversified TSMOM ===")
    print(f"overlap: {both.index.min().date()} -> {both.index.max().date()} ({len(both)} months)\n")
    if len(both) < 6:
        print("insufficient overlap."); return 0
    rho = both.iloc[:, 0].corr(both.iloc[:, 1])
    print(f"  correlation of the two books (monthly): {rho:+.2f}")
    print(f"  {'book':<26}{'Sharpe(overlap)':>16}{'Sharpe(full-period)':>21}")
    for c in both.columns:
        print(f"  {c:<26}{_sh(both[c]):>+16.2f}{FULL_SHARPE.get(c, float('nan')):>+21.2f}")

    # equal-RISK (inverse-vol) combo over the overlap
    w1, w2 = 1 / both.iloc[:, 0].std(), 1 / both.iloc[:, 1].std()
    combo = (w1 * both.iloc[:, 0] + w2 * both.iloc[:, 1]) / (w1 + w2)
    print(f"\n  equal-risk COMBO Sharpe (overlap): {_sh(combo):+.2f}  "
          f"vs best single {max(_sh(both.iloc[:,0]), _sh(both.iloc[:,1])):+.2f}")

    # theoretical combo using the robust full-period Sharpes + empirical corr (equal-risk):
    s1, s2 = FULL_SHARPE["MES_intraday_momentum"], FULL_SHARPE["diversified_TSMOM"]
    theo = (s1 + s2) / np.sqrt(2 + 2 * rho) if rho > -1 else float("nan")
    print(f"  theoretical COMBO Sharpe (full-period S={s1}/{s2}, corr {rho:+.2f}): {theo:+.2f}")
    print(f"    -> {'BEATS' if theo > max(s1, s2) else 'does NOT beat'} the best single book "
          f"({max(s1, s2):.2f}); uplift {(theo/max(s1,s2)-1)*100:+.0f}%")

    print("\nRead: if the two books are ~uncorrelated (|corr| low), the combo Sharpe exceeds either alone")
    print("= genuine diversification (deploy more capital at the same risk / hit a target with less DD).")
    print("Overlap is short (~22mo) so the OVERLAP Sharpes are noisy; the full-period + corr theory is the")
    print("more robust read. Two different-horizon edges (intraday vs monthly) = the real portfolio.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
