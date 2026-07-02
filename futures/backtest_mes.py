"""Phase-1 GATE: does any simple MES strategy show a real edge on DAILY bars, net of costs, OOS?

Tests the daily-testable candidates from the plan on the ES=F/MES daily series:
  buy_hold    : benchmark (long always) -- also a sanity check the framework matches reality.
  trend_ma    : long when close > 200d SMA, else flat        (trend, long-only)
  trend_ma_ls : +1 above 200d SMA / -1 below                 (trend, long/short)
  tsmom_12m   : sign(12-month return), long/short            (canonical TSMOM, our prior)
  mr_1d       : long the day after a down close, else flat    (S&P short-term mean-reversion)
  donchian    : long a 20d-high breakout, exit on 10d low     (breakout, long-only)
  NEG_random  : seeded random long/short -- must be ~0 net of costs (false-positive canary)

The intraday time-series ORB candidate is BLOCKED on an intraday data source (see futures/data.py) and
is NOT tested here. All returns are per-NOTIONAL (1 contract vs full notional = unlevered) and net of
the round-turn friction in data.py. Lookahead-free: signal decided at close_t is held over day t+1.

    .venv-openbb/Scripts/python.exe futures/backtest_mes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from futures.data import COST_RT_USD, POINT_VALUE, load_mes_daily  # noqa: E402

COST_SIDE_USD = COST_RT_USD / 2.0   # per unit |Δposition| (one side of a round-turn)
MONTHLY_TARGET = 0.03               # the user's "3%/month" target -> annualized for the reality lens
ANN_TARGET = (1 + MONTHLY_TARGET) ** 12 - 1


def net_daily(df: pd.DataFrame, signal: pd.Series) -> pd.Series:
    """Net per-notional daily return of a {-1,0,1} signal. signal_t (at close_t) is HELD over day t+1;
    costs hit when the held position changes, priced at the prior close. Lookahead-free."""
    ret = df["close"].pct_change()
    held = signal.shift(1).fillna(0.0)                      # position active during day t
    gross = held * ret
    trades = held.diff().abs().fillna(held.abs())           # contracts traded at start of day t
    cost = trades * COST_SIDE_USD / (df["close"].shift(1) * POINT_VALUE)
    return (gross - cost).fillna(0.0)


def stats(daily: pd.Series) -> dict:
    r = daily.dropna()
    if len(r) < 60:
        return {}
    eq = (1 + r).cumprod()
    yrs = len(r) / 252.0
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(252)
    sharpe = (r.mean() * 252) / vol if vol else float("nan")
    dd = (eq / eq.cummax() - 1).min()
    mar = cagr / abs(dd) if dd < 0 else float("nan")
    return {"CAGR": cagr, "vol": vol, "sharpe": sharpe, "maxDD": dd, "MAR": mar,
            "totret": eq.iloc[-1] - 1, "n": len(r)}


def _trades_count(signal: pd.Series) -> int:
    held = signal.shift(1).fillna(0.0)
    return int(held.diff().abs().fillna(held.abs()).sum() / 2)   # round-turns


def signals(df: pd.DataFrame) -> dict[str, pd.Series]:
    c = df["close"]
    ret = c.pct_change()
    sma200 = c.rolling(200).mean()
    mom12 = c / c.shift(252) - 1.0
    # Donchian 20-high entry / 10-low exit, stateful long-only
    hi20 = c.rolling(20).max()
    lo10 = c.rolling(10).min()
    don = pd.Series(0.0, index=c.index)
    pos = 0.0
    for i in range(len(c)):
        if pos == 0.0 and c.iloc[i] >= hi20.iloc[i] and not np.isnan(hi20.iloc[i]):
            pos = 1.0
        elif pos == 1.0 and c.iloc[i] <= lo10.iloc[i]:
            pos = 0.0
        don.iloc[i] = pos
    rng = np.random.default_rng(20260702)
    neg = pd.Series(rng.choice([-1.0, 0.0, 1.0], size=len(c)), index=c.index)
    return {
        "buy_hold": pd.Series(1.0, index=c.index),
        "trend_ma": (c > sma200).astype(float),
        "trend_ma_ls": np.sign(c - sma200).replace(0, np.nan).ffill().fillna(0.0),
        "tsmom_12m": np.sign(mom12).fillna(0.0),
        "mr_1d": (ret < 0).astype(float),
        "donchian": don,
        "NEG_random": neg,
    }


def _fmt(name: str, s: dict) -> str:
    if not s:
        return f"  {name:<12} (insufficient data)"
    return (f"  {name:<12} CAGR {s['CAGR']*100:+6.1f}%  Sharpe {s['sharpe']:+5.2f}  "
            f"maxDD {s['maxDD']*100:6.1f}%  MAR {s['MAR']:5.2f}  totret {s['totret']*100:+7.0f}%")


def main() -> int:
    df = load_mes_daily()
    sigs = signals(df)
    print(f"=== MES daily backtest gate: {len(df)} days  {df.index.min().date()} -> "
          f"{df.index.max().date()}  (per-notional, net of ${COST_RT_USD:.0f}/RT) ===\n")

    # framework sanity check: buy_hold must ~match actual ES=F total return
    bh_actual = df["close"].iloc[-1] / df["close"].iloc[0] - 1
    bh_model = stats(net_daily(df, sigs["buy_hold"]))["totret"]
    print(f"SANITY buy_hold: model totret {bh_model*100:+.0f}% vs actual price {bh_actual*100:+.0f}% "
          f"(diff = costs/compounding; should be close)\n")

    print(f"{'FULL SAMPLE':<12}")
    results = {}
    for name, sig in sigs.items():
        d = net_daily(df, sig)
        s = stats(d)
        results[name] = (d, s)
        print(_fmt(name, s) + f"  trades {_trades_count(sig)}")

    # OOS: split in half by date
    mid = df.index[len(df) // 2]
    print(f"\nOOS split at {mid.date()} (H1 = in-sample-ish / H2 = out-of-sample):")
    print(f"  {'strategy':<12}{'H1 Sharpe':>11}{'H2 Sharpe':>11}{'H1 CAGR':>10}{'H2 CAGR':>10}"
          f"{'both>0?':>9}")
    for name in ["trend_ma", "trend_ma_ls", "tsmom_12m", "mr_1d", "donchian", "NEG_random"]:
        d = results[name][0]
        h1, h2 = stats(d[d.index < mid]), stats(d[d.index >= mid])
        if h1 and h2:
            both = "YES" if (h1["sharpe"] > 0 and h2["sharpe"] > 0) else "no"
            print(f"  {name:<12}{h1['sharpe']:>+11.2f}{h2['sharpe']:>+11.2f}"
                  f"{h1['CAGR']*100:>+9.1f}%{h2['CAGR']*100:>+9.1f}%{both:>9}")

    # reality lens: leverage needed to hit ~3%/month, and the drawdown it implies
    print(f"\nREALITY LENS (target {MONTHLY_TARGET*100:.0f}%/mo = {ANN_TARGET*100:.0f}%/yr):")
    print(f"  {'strategy':<12}{'unlev CAGR':>11}{'lev x needed':>14}{'=> acct maxDD':>15}")
    for name in ["buy_hold", "trend_ma", "trend_ma_ls", "tsmom_12m", "mr_1d", "donchian"]:
        s = results[name][1]
        if not s or s["CAGR"] <= 0:
            print(f"  {name:<12}{(s['CAGR']*100 if s else float('nan')):>+10.1f}%   "
                  f"target unreachable (non-positive edge)")
            continue
        lev = ANN_TARGET / s["CAGR"]
        acct_dd = lev * s["maxDD"]
        print(f"  {name:<12}{s['CAGR']*100:>+10.1f}%{lev:>13.1f}x{acct_dd*100:>+14.0f}%")

    print("\nRead: a candidate is interesting only if BOTH OOS halves have Sharpe>0 AND it beats")
    print("buy_hold risk-adjusted (higher Sharpe/MAR). NEG_random must be ~flat/negative. The reality")
    print("lens shows the leverage 3%/mo demands and the account drawdown that leverage creates.")
    print("NOTE: the intraday time-series ORB candidate is NOT here -- it needs an intraday feed (data.py).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
