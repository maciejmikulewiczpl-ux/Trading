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
    """All RTH bars after the 09:30 open -> sigma is computed at EVERY 5-min time-of-day, so any
    decision grid (30/15/5-min) can look it up. The strategy's actual grid is chosen in backtest()."""
    return rth[rth.index.time > RTH_OPEN]


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


def backtest(df: pd.DataFrame, trail_k: float = 1.5, gap_adj: bool = True,
             take_profit: float | None = None, decide_minutes: tuple = (0, 30),
             exit_inside: bool = False) -> pd.DataFrame:
    """Bar-by-bar. Decisions at :00/:30 vs the (optionally gap-adjusted) noise band; between decisions
    a dynamic trailing stop = trail_k * sigma_entry * open guards the position. trail_k huge = 'ride to
    opposite band/close' (the v1 behaviour). gap_adj: raise upper by an overnight gap-down / lower the
    lower by a gap-up (the paper's asymmetry, damping counter-gap entries). take_profit (e.g. 0.005 =
    +0.5%): lock the win when price reaches entry*(1+/-tp) -- the 'focus on short runs' idea."""
    rth_all = df[(df.index.time >= RTH_OPEN) & (df.index.time <= EOD)]
    days_l = sorted({t.date() for t in rth_all.index})
    day_df = {d: rth_all[rth_all.index.date == d] for d in days_l}
    sigma = build_sigma(day_df)
    closes = {d: day_df[d]["close"].iloc[-1] for d in days_l if len(day_df[d])}
    rows, prev_d = [], None
    for d in days_l:
        rth = day_df[d]
        if len(rth) < 3:
            prev_d = d
            continue
        op = rth["open"].iloc[0]
        pc = closes.get(prev_d)
        gap = (op / pc - 1.0) if (pc and gap_adj) else 0.0
        pos, entry, extreme, s_ent, ret = 0, None, None, None, 0.0
        for ts, row in rth.iterrows():
            # 0) take-profit check intra-bar (the 'lock a fixed % win / short runs' idea)
            if pos != 0 and take_profit is not None:
                if pos == 1 and row["high"] >= entry * (1 + take_profit):
                    ret += (entry * (1 + take_profit) - entry) * POINT_VALUE - COST_SIDE_USD * 2
                    pos, entry, extreme = 0, None, None
                elif pos == -1 and row["low"] <= entry * (1 - take_profit):
                    ret += (entry - entry * (1 - take_profit)) * POINT_VALUE - COST_SIDE_USD * 2
                    pos, entry, extreme = 0, None, None
            # 1) trailing-stop check intra-bar (uses this bar's high/low)
            if pos != 0:
                trail = trail_k * s_ent * op
                if pos == 1:
                    extreme = max(extreme, row["high"])
                    if row["low"] <= extreme - trail:
                        ret += (extreme - trail - entry) * POINT_VALUE - COST_SIDE_USD * 2
                        pos, entry, extreme = 0, None, None
                else:
                    extreme = min(extreme, row["low"])
                    if row["high"] >= extreme + trail:
                        ret += (entry - (extreme + trail)) * POINT_VALUE - COST_SIDE_USD * 2
                        pos, entry, extreme = 0, None, None
            # 2) decision at each grid mark (band breakout -> set/flip target)
            if ts.minute in decide_minutes and ts.time() > RTH_OPEN:
                s = sigma.get((d, f"{ts.hour:02d}:{ts.minute:02d}"))
                if s is not None:
                    up, lo = op * (1 + s), op * (1 - s)
                    if gap < 0:
                        up += -gap * op        # overnight gap-down -> harder to go long
                    elif gap > 0:
                        lo -= gap * op         # overnight gap-up -> harder to go short
                    px = row["close"]
                    # inside the band: hold (default) or EXIT-TO-FLAT + re-enter on a fresh breakout
                    # (the "exit when momentum fades, re-enter on new momentum" idea)
                    inside = 0 if exit_inside else pos
                    tgt = 1 if px > up else (-1 if px < lo else inside)
                    if tgt != pos:
                        if pos != 0:
                            ret += (px - entry) * pos * POINT_VALUE - COST_SIDE_USD * 2
                        pos, entry, extreme, s_ent = tgt, (px if tgt else None), (px if tgt else None), s
        if pos != 0:
            close = rth["close"].iloc[-1]
            ret += (close - entry) * pos * POINT_VALUE - COST_SIDE_USD * 2
        if ret != 0.0:
            rows.append({"date": d, "ret_usd": ret})
        prev_d = d
    return pd.DataFrame(rows)


def _stats(trades: pd.DataFrame) -> dict | None:
    if trades.empty or len(trades) < 5:
        return None
    r = trades["ret_usd"]
    eq = r.cumsum()
    return {"n": len(r), "net": r.sum(), "perday": r.mean(), "win": 100 * (r > 0).mean(),
            "pf": (r[r > 0].sum() / -r[r < 0].sum()) if (r < 0).any() else float("inf"),
            "sharpe": (r.mean() / r.std() * np.sqrt(252)) if r.std() else float("nan"),
            "dd": (eq - eq.cummax()).min(), "best": r.max(), "worst": r.min()}


def _line(name: str, s: dict | None) -> str:
    if not s:
        return f"  {name:<16} n<5"
    return (f"  {name:<16} n={s['n']:3d} net${s['net']:+6.0f} perday${s['perday']:+5.1f} "
            f"win{s['win']:3.0f}% PF{s['pf']:.2f} Sharpe~{s['sharpe']:+.2f} "
            f"maxDD${s['dd']:+6.0f} best${s['best']:+5.0f}")


def main() -> int:
    df, src = _load()
    n_days = len({t.date() for t in df.index})
    print(f"=== MES intraday MOMENTUM (Zarattini-style): {len(df)} bars, {n_days} days "
          f"[{src}] {df.index.min().date()} -> {df.index.max().date()} ===")
    print(f"setup: noise band = open x (1 +/- 14d avg move-by-time); :00/:30 breakout; net "
          f"${COST_RT_USD:.0f}/RT. Comparing mechanics (trail_k huge = v1 ride-only):\n")
    variants = [("ride-only (v1)", dict(trail_k=999, gap_adj=False)),
                ("+gap-adjust", dict(trail_k=999, gap_adj=True)),
                ("+stop k=2.5", dict(trail_k=2.5, gap_adj=True)),
                ("+stop k=1.5", dict(trail_k=1.5, gap_adj=True)),
                ("+stop k=1.0", dict(trail_k=1.0, gap_adj=True))]
    results = {}
    for name, kw in variants:
        tr = backtest(df, **kw)
        results[name] = tr
        print(_line(name, _stats(tr)))
    # TAKE-PROFIT test (the 'focus on short runs' idea): same strategy (gap-adjust, no trailing stop),
    # only the exit differs -- ride to close vs lock a fixed % win. Isolates the cap-the-winner effect.
    print("\nTake-profit test (gap-adjust, no stop; only the exit differs):")
    print(_line("ride to close", _stats(backtest(df, trail_k=999, gap_adj=True, take_profit=None))))
    for tp in (0.003, 0.005, 0.010):
        print(_line(f"take-profit +{tp*100:.1f}%",
                    _stats(backtest(df, trail_k=999, gap_adj=True, take_profit=tp))))

    # OOS split on the best variant by Sharpe
    best = max(results, key=lambda k: (_stats(results[k]) or {"sharpe": -9})["sharpe"])
    tr = results[best].copy()
    tr["date"] = pd.to_datetime(tr["date"])
    mid = tr["date"].quantile(0.5)
    print(f"\nOOS split of best [{best}]:")
    print(_line("H1", _stats(tr[tr["date"] < mid])))
    print(_line("H2", _stats(tr[tr["date"] >= mid])))
    print("\nRead: does a trailing stop / gap-adjust beat v1 ride-only on Sharpe + maxDD without "
          "killing the tail (best)? Target ~1.33. Both OOS halves +ve = robust. Deep 5-min = fair test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
