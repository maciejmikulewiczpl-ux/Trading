"""TSMOM sleeve SIGNAL: current long-only cross-asset trend target weights.

The diversifier half of the portfolio (futures/diversification.py, [[futures_diversification_wip]]):
long-only 12-month time-series momentum over a cross-asset ETF basket, inverse-vol weighted. This is
the slow, monthly sleeve that runs alongside the fast MES intraday-momentum bot -- the two are
NEGATIVELY correlated (-0.36) so combining lifts the aggregate Sharpe ~1.1->~1.8.

target_weights() -> {ETF: weight} for THIS month (weights sum to <=1; cash = 1 - sum when few uptrends).
Uses yfinance daily (monthly-resampled) for the signal -- robust, no market-data subscription needed.

    .venv-openbb/Scripts/python.exe tsmom/weights.py     # print this month's target
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BASKET = {"SPY": "US large", "QQQ": "US tech", "IWM": "US small", "EFA": "Dev intl", "EEM": "EM",
          "TLT": "Long bond", "IEF": "7-10y bond", "GLD": "Gold", "SLV": "Silver",
          "DBC": "Commodities", "USO": "Oil", "UUP": "US dollar"}
MOM_LB, VOL_LB = 12, 12   # months


def weights_from_daily(px: pd.DataFrame, gross: float = 1.0) -> dict[str, float]:
    """Pure computation: given a DAILY adjusted-close DataFrame (cols=ETFs), return this month's
    long-only TSMOM target weights (inverse-vol among up-trending names; downtrending names -> cash).
    Data-source-agnostic so the bot (Alpaca) and research (yfinance) share the exact same logic."""
    px = px[[t for t in BASKET if t in px.columns]].dropna(how="all")
    m = px.resample("ME").last()
    rets = m.pct_change()
    mom = m / m.shift(MOM_LB) - 1.0
    invvol = 1.0 / rets.rolling(VOL_LB).std().replace(0, np.nan)
    iv_all = invvol.iloc[-1]
    total_iv = iv_all.sum()
    iv_up = iv_all[mom.iloc[-1] > 0].dropna()     # long-only: only names in an uptrend
    if iv_up.empty or total_iv <= 0:
        return {}                                 # everything downtrending -> all cash
    w = (iv_up / total_iv) * gross                # share of whole-basket inv-vol -> downtrends = cash
    return {t: round(float(v), 4) for t, v in w.items() if v > 0}


def target_weights(gross: float = 1.0) -> dict[str, float]:
    """Standalone (yfinance) wrapper for research/testing on .venv-openbb. The bot uses Alpaca data +
    weights_from_daily() directly (see tsmom/rebalance.py) so it needs no yfinance on the VM."""
    import yfinance as yf
    raw = yf.download(list(BASKET), start="2018-01-01", auto_adjust=True, progress=False)
    px = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    return weights_from_daily(px, gross)


def main() -> int:
    w = target_weights()
    inv = sum(w.values())
    print(f"=== TSMOM target this month | invested {inv*100:.0f}% / cash {100-inv*100:.0f}% ===")
    for t, wt in sorted(w.items(), key=lambda x: -x[1]):
        print(f"  LONG {t+' '+BASKET.get(t,''):<16}{wt*100:>5.1f}%")
    if not w:
        print("  (all sleeves in downtrend -> hold cash)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
