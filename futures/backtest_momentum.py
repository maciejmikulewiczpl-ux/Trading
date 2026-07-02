"""Intraday MOMENTUM candidate for MES -- our top research lead (Zarattini/Concretum "Beat the Market",
SSRN 4824172; SPY 2007-2024: +19.6%/yr, Sharpe 1.33 net of costs). Adapted to MES.

Rule (v1): each day, open = 09:30 ET RTH open. For each decision time tau (every :00/:30), a dynamic
"noise band" = open x (1 +/- sigma_tau), where sigma_tau = the 14-day trailing average of the typical
absolute move-from-open BY THAT TIME OF DAY (|close_tau/open - 1|). If price breaks ABOVE the upper
band -> go long; BELOW the lower band -> go short; ride until an opposite breakout or the close (a
noise-band re-entry acts as the stop). Flat at 15:55. The band WIDENS intraday (sigma grows), so late
breakouts need a bigger move -- the paper's key feature.

Data-agnostic: filters any bar size to the :00/:30 decision grid; uses the IBKR deep cache if present
(futures/data.load_mes_intraday_cache), else the free ~50d yfinance window (smoke test). Net of costs.

    .venv-openbb/Scripts/python.exe futures/backtest_momentum.py
"""
from __future__ import annotations

import sys
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from futures.data import COST_RT_USD, POINT_VALUE  # noqa: E402

RTH_OPEN, EOD = time(9, 30), time(15, 55)
SIGMA_LB = 14        # trailing days for the time-of-day move estimate
COST_SIDE_USD = COST_RT_USD / 2.0


def _load() -> tuple[pd.DataFrame, str]:
    from futures.data import load_mes_intraday, load_mes_intraday_cache
    try:
        return load_mes_intraday_cache(), "IBKR cache"
    except FileNotFoundError:
        return load_mes_intraday("5m", "60d"), "yfinance ~50d (SMOKE)"


def _decision_bars(rth: pd.DataFrame) -> pd.DataFrame:
    """RTH bars at :00 or :30 (the decision grid), excluding the 09:30 open bar itself."""
    t = rth.index
    mask = ((t.minute == 0) | (t.minute == 30)) & (rth.index.time > RTH_OPEN)
    return rth[mask]


def build_sigma(days: dict[object, pd.DataFrame]) -> dict:
    """{(date, hh_mm): sigma} = 14-day trailing avg of |close_tau/open - 1| at each time-of-day."""
    per_tod: dict[str, list] = {}
    hist: dict[str, list] = {}
    sigma = {}
    for d in sorted(days):
        rth = days[d]
        if rth.empty:
            continue
        op = rth["open"].iloc[0]
        for ts, row in _decision_bars(rth).iterrows():
            key = f"{ts.hour:02d}:{ts.minute:02d}"
            past = hist.get(key, [])
            if len(past) >= SIGMA_LB:
                sigma[(d, key)] = float(np.mean(past[-SIGMA_LB:]))
            hist.setdefault(key, []).append(abs(row["close"] / op - 1.0) if op else np.nan)
    return sigma


def backtest(df: pd.DataFrame) -> pd.DataFrame:
    rth_all = df[(df.index.time >= RTH_OPEN) & (df.index.time <= EOD)]
    days = {d: rth_all[rth_all.index.date == d] for d in sorted({t.date() for t in rth_all.index})}
    sigma = build_sigma(days)
    rows = []
    for d in sorted(days):
        rth = days[d]
        if len(rth) < 3:
            continue
        op = rth["open"].iloc[0]
        pos, entry, ret_usd, extreme = 0, None, 0.0, None
        for ts, row in _decision_bars(rth).iterrows():
            key = f"{ts.hour:02d}:{ts.minute:02d}"
            s = sigma.get((d, key))
            if s is None:
                continue
            up, lo = op * (1 + s), op * (1 - s)
            px = row["close"]
            new = pos
            if px > up:
                new = 1
            elif px < lo:
                new = -1
            if new != pos:
                if pos != 0 and entry is not None:      # close the old leg
                    ret_usd += (px - entry) * pos * POINT_VALUE - COST_SIDE_USD * 2
                pos, entry, extreme = new, px, px
        # mark to close
        if pos != 0 and entry is not None:
            close = rth["close"].iloc[-1]
            ret_usd += (close - entry) * pos * POINT_VALUE - COST_SIDE_USD * 2
            rows.append({"date": d, "ret_usd": ret_usd})
        elif ret_usd != 0.0:
            rows.append({"date": d, "ret_usd": ret_usd})
    return pd.DataFrame(rows)


def _report(trades: pd.DataFrame, n_days: int, label: str) -> None:
    if trades.empty:
        print(f"  [{label}] no trades (need >{SIGMA_LB} days of history)."); return
    r = trades["ret_usd"]
    eq = r.cumsum()
    dd = (eq - eq.cummax()).min()
    sharpe = (r.mean() / r.std() * np.sqrt(252)) if r.std() else float("nan")
    pf = r[r > 0].sum() / -r[r < 0].sum() if (r < 0).any() else float("inf")
    print(f"  [{label}] trade-days {len(trades)}/{n_days} | net ${r.sum():+.0f} "
          f"(per day ${r.mean():+.1f}) | win {100*(r>0).mean():.0f}% | PF {pf:.2f} "
          f"| Sharpe~{sharpe:+.2f} | maxDD ${dd:+.0f} | best ${r.max():+.0f} worst ${r.min():+.0f}")


def main() -> int:
    df, src = _load()
    n_days = len({t.date() for t in df.index})
    print(f"=== MES intraday MOMENTUM (Zarattini-style): {len(df)} bars, {n_days} days "
          f"[{src}] {df.index.min().date()} -> {df.index.max().date()} ===")
    print(f"setup: noise band = open x (1 +/- 14d-avg move-by-time); :00/:30 breakout; ride to "
          f"opposite band / close; net ${COST_RT_USD:.0f}/RT\n")
    _report(backtest(df), n_days, "intraday-momentum")
    print("\nRead: PRELIMINARY on short data unless the IBKR cache is deep. Target = the paper's "
          "Sharpe ~1.33. Judge net$, PF, Sharpe, and the tail (best). Deep history makes it decisive.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
