"""Trend-following / dual-momentum backtest on a diversified ETF set.

The Part-B survey named this the robust "bedrock" sleeve: modest Sharpe but
decades-tested, uncorrelated to the intraday ORB, and strong drawdown control
(crisis alpha). This tests the canonical variants on ~2007->now MONTHLY data
and compares them to buy & hold SPY and a static 60/40.

Self-contained: yfinance + pandas + numpy only (runs in .venv-openbb). yfinance
gives ~20y of ETF history (vs Alpaca IEX's short window) — essential for trend
following, whose whole value is surviving 2008 / 2020 / 2022.

  .venv-openbb\\Scripts\\python.exe backtest\\run_trend.py

No-lookahead: a month-end signal sets weights w[t]; the realized return is the
NEXT month's return, i.e. strat_ret = (w.shift(1) * monthly_ret).sum(axis=1).

Variants:
  - SPY buy & hold                 (benchmark)
  - 60/40 SPY/AGG (monthly reb)    (benchmark)
  - SPY 10-month SMA timing        (simplest absolute-momentum trend)
  - GEM dual momentum              (SPY vs EFA, 12m; winner if > cash, else AGG)
  - Faber GTAA-5                   (SPY/EFA/AGG/VNQ/DBC, each in if > 10m SMA else cash)
  - Diversified 12m momentum top-3 (rank risk assets, hold top-3 that beat cash)
"""
from __future__ import annotations

import io
import sys

import numpy as np
import pandas as pd
import yfinance as yf

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

START = "2005-01-01"          # download start (warmup for 12m mom / 10m SMA)
EVAL_START = "2007-07-01"     # metrics measured from here (all ETFs live + warmup done)
CASH = "SHY"                  # 1-3y treasuries = cash/risk-free proxy
RISK = ["SPY", "QQQ", "EFA", "EEM", "VNQ", "GLD", "DBC", "TLT", "AGG"]
ALL = sorted(set(RISK + [CASH, "EFA", "AGG"]))


def load_monthly():
    raw = yf.download(ALL, start=START, auto_adjust=True, progress=False, group_by="ticker")
    closes = {}
    for s in ALL:
        try:
            c = raw[s]["Close"].dropna()
        except Exception:
            continue
        if not c.empty:
            closes[s] = c
    px = pd.DataFrame(closes)
    px.index = pd.to_datetime(px.index)
    monthly = px.resample("ME").last()
    return monthly


def evaluate(weights: pd.DataFrame, monthly_ret: pd.DataFrame, rf: pd.Series):
    # No lookahead: weights set at month-end t earn month t+1's return.
    aligned = weights.shift(1).reindex(monthly_ret.index).fillna(0.0)
    strat = (aligned * monthly_ret.reindex(columns=aligned.columns).fillna(0.0)).sum(axis=1)
    strat = strat.loc[EVAL_START:].dropna()
    if strat.empty:
        return None
    eq = (1 + strat).cumprod()
    n = len(strat)
    cagr = eq.iloc[-1] ** (12 / n) - 1
    vol = strat.std() * np.sqrt(12)
    rf_m = rf.reindex(strat.index).fillna(0.0)
    sharpe = (strat - rf_m).mean() / strat.std() * np.sqrt(12) if strat.std() > 0 else np.nan
    dd = (eq / eq.cummax() - 1).min()
    pos = (strat > 0).mean() * 100
    by_year = (1 + strat).groupby(strat.index.year).prod() - 1
    return {
        "cagr": cagr * 100, "vol": vol * 100, "sharpe": sharpe,
        "maxdd": dd * 100, "pos_mo": pos,
        "worst_yr": by_year.min() * 100, "best_yr": by_year.max() * 100,
        "y2008": by_year.get(2008, np.nan) * 100,
        "y2022": by_year.get(2022, np.nan) * 100,
        "by_year": by_year,
    }


def w_buyhold(monthly, asset):
    w = pd.DataFrame(0.0, index=monthly.index, columns=monthly.columns)
    w[asset] = 1.0
    return w


def w_6040(monthly):
    w = pd.DataFrame(0.0, index=monthly.index, columns=monthly.columns)
    w["SPY"], w["AGG"] = 0.6, 0.4
    return w


def w_sma_timing(monthly, asset, months=10):
    sma = monthly[asset].rolling(months).mean()
    in_mkt = monthly[asset] > sma
    w = pd.DataFrame(0.0, index=monthly.index, columns=monthly.columns)
    w[asset] = in_mkt.astype(float)
    w[CASH] = (~in_mkt).astype(float)
    return w


def w_gem(monthly, lb=12):
    mom = monthly / monthly.shift(lb) - 1
    w = pd.DataFrame(0.0, index=monthly.index, columns=monthly.columns)
    for t in monthly.index:
        m_spy, m_efa, m_cash = mom.loc[t, "SPY"], mom.loc[t, "EFA"], mom.loc[t, CASH]
        if np.isnan(m_spy) or np.isnan(m_efa) or np.isnan(m_cash):
            continue
        winner = "SPY" if m_spy >= m_efa else "EFA"
        if mom.loc[t, winner] > m_cash:
            w.loc[t, winner] = 1.0
        else:
            w.loc[t, "AGG"] = 1.0
    return w


def w_faber(monthly, assets=("SPY", "EFA", "AGG", "VNQ", "DBC"), months=10):
    sma = monthly.rolling(months).mean()
    w = pd.DataFrame(0.0, index=monthly.index, columns=monthly.columns)
    each = 1.0 / len(assets)
    for a in assets:
        on = monthly[a] > sma[a]
        w[a] += on.astype(float) * each
        w[CASH] += (~on).astype(float) * each
    return w


def w_div_momentum(monthly, lb=12, topk=3):
    mom = monthly / monthly.shift(lb) - 1
    w = pd.DataFrame(0.0, index=monthly.index, columns=monthly.columns)
    for t in monthly.index:
        mc = mom.loc[t, CASH]
        if np.isnan(mc):
            continue
        scores = {a: mom.loc[t, a] for a in RISK if not np.isnan(mom.loc[t, a])}
        eligible = {a: s for a, s in scores.items() if s > mc}  # absolute filter
        top = sorted(eligible, key=eligible.get, reverse=True)[:topk]
        if top:
            for a in top:
                w.loc[t, a] = 1.0 / topk
            w.loc[t, CASH] = 1.0 - len(top) / topk  # unfilled slots -> cash
        else:
            w.loc[t, CASH] = 1.0
    return w


def main():
    print(f"Loading monthly ETF data {START}->now (yfinance)...")
    monthly = load_monthly()
    monthly_ret = monthly.pct_change()
    rf = monthly_ret[CASH]
    print(f"Assets: {list(monthly.columns)}")
    print(f"Months: {len(monthly)}  | metrics from {EVAL_START}\n")

    variants = {
        "SPY buy&hold": w_buyhold(monthly, "SPY"),
        "60/40 SPY/AGG": w_6040(monthly),
        "SPY 10mo-SMA timing": w_sma_timing(monthly, "SPY"),
        "GEM dual-momentum": w_gem(monthly),
        "Faber GTAA-5": w_faber(monthly),
        "Diversified mom top-3": w_div_momentum(monthly),
    }

    rows = {name: evaluate(w, monthly_ret, rf) for name, w in variants.items()}

    hdr = (f"{'strategy':<24}{'CAGR%':>7}{'vol%':>7}{'Sharpe':>8}{'maxDD%':>8}"
           f"{'pos-mo':>8}{'worst-yr':>9}{'2008':>8}{'2022':>8}")
    print(hdr)
    print("-" * len(hdr))
    for name, m in rows.items():
        if m is None:
            print(f"{name:<24}  (no data)"); continue
        print(f"{name:<24}{m['cagr']:>7.2f}{m['vol']:>7.2f}{m['sharpe']:>8.2f}"
              f"{m['maxdd']:>8.1f}{m['pos_mo']:>7.1f}%{m['worst_yr']:>+9.1f}"
              f"{m['y2008']:>+8.1f}{m['y2022']:>+8.1f}")

    print("\nReadout: trend/dual-momentum should show LOWER maxDD and far better")
    print("2008/2022 than SPY buy&hold, at a similar-or-better Sharpe — that")
    print("drawdown control (not raw CAGR) is the edge and the diversification value.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
