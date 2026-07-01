"""factor_exposure.py -- is the ORB edge genuine ALPHA or just factor beta? (reviews' factor ask)

Regress the shipped-config daily backtest return on standard equity factors. If the return is fully
explained by loadings on market/momentum/size/vol (high R^2, insignificant intercept), the "edge" is
just factor exposure you could buy with an ETF. If there's a significant positive INTERCEPT (alpha)
after controlling for factors, the edge is genuine selection/timing.

Factors (daily), built as SPREADS vs the market to cut collinearity:
  MKT   = SPY return                 (market beta)
  MOM   = MTUM - SPY                 (momentum tilt -- our strat is a long-momentum breakout)
  SIZE  = IWM - SPY                  (small minus big)
  LOWVOL= USMV - SPY                 (defensive/low-vol tilt)
  dVIX  = ^VIX daily % change        (vol / crash exposure)

Dependent var = daily strat return = PnL / START_CAPITAL. OLS with hand-rolled t-stats (no statsmodels
dependency). Reads: alpha t>2 and positive => real edge beyond factors.

Run (needs the CSV from montecarlo_orb.shipped_daily_pnl, generated in .venv first; yfinance here):
    .venv-openbb/Scripts/python.exe backtest/factor_exposure.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
PNL_CSV = ROOT / "backtest" / ".orb_daily_pnl_730.csv"
START_CAPITAL = 25_000.0


def main():
    pnl = pd.read_csv(PNL_CSV, index_col=0)["pnl"]
    pnl.index = pd.to_datetime(pnl.index)
    y_full = (pnl / START_CAPITAL)   # daily strat return

    tickers = ["SPY", "MTUM", "IWM", "USMV", "^VIX"]
    px = yf.download(tickers, period="3y", interval="1d", auto_adjust=True, progress=False)["Close"]
    rets = px.pct_change()
    f = pd.DataFrame({
        "MKT": rets["SPY"],
        "MOM": rets["MTUM"] - rets["SPY"],
        "SIZE": rets["IWM"] - rets["SPY"],
        "LOWVOL": rets["USMV"] - rets["SPY"],
        "dVIX": rets["^VIX"],
    })

    df = pd.concat([y_full.rename("y"), f], axis=1).dropna()
    y = df["y"].to_numpy()
    names = ["alpha", "MKT", "MOM", "SIZE", "LOWVOL", "dVIX"]
    X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy() for c in names[1:]])
    n, k = X.shape

    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    sigma2 = (resid @ resid) / (n - k)
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    tstat = beta / se
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid @ resid) / ss_tot

    print(f"=== ORB factor-exposure regression: {n} days, R^2 = {r2:.3f} ===")
    print(f"(daily strat return = PnL/${START_CAPITAL:,.0f}; factors are SPY-relative spreads)\n")
    print(f"  {'factor':<10}{'beta':>12}{'t-stat':>9}{'sig':>5}")
    print("  " + "-" * 36)
    for nm, b, t in zip(names, beta, tstat):
        sig = "***" if abs(t) > 2.58 else ("**" if abs(t) > 1.96 else ("*" if abs(t) > 1.64 else ""))
        if nm == "alpha":
            print(f"  {nm:<10}{b*252*100:>+11.1f}%{t:>9.2f}{sig:>5}   (annualized)")
        else:
            print(f"  {nm:<10}{b:>+12.3f}{t:>9.2f}{sig:>5}")

    ann_alpha = beta[0] * 252 * 100
    print(f"\nRead: R^2={r2:.3f} => factors explain {r2*100:.0f}% of daily variance. "
          f"Annualized alpha {ann_alpha:+.1f}% (t={tstat[0]:.2f}).")
    if tstat[0] > 1.96:
        print("Alpha is significant AND positive => the edge is NOT just factor beta; genuine selection/timing.")
    elif tstat[0] > 1.64:
        print("Alpha marginally significant => mostly genuine, watch power.")
    else:
        print("Alpha not significant => the return may be explained by factor loadings; investigate.")
    print("Significant factor betas show WHICH exposures the strat carries (a long-momentum breakout")
    print("should load + on MKT/MOM). Low R^2 is EXPECTED for an intraday strat (daily factors miss")
    print("intraday timing) and itself argues the edge is not a static factor tilt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
