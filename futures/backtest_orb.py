"""Intraday time-series ORB candidate for MES (the ORB-heritage strategy from the plan).

Our stock ORB edge is CROSS-SECTIONAL (tight-OR picks the best breakout across 122 names). One
instrument has no cross-section, so we convert the tight-OR filter to a TIME-SERIES gate: trade only
DAYS whose opening range is small vs recent daily ATR. Applies our shipped lessons -- trailing-stop
exit (let winners run) and the vol-regime idea (native since MES=S&P).

Setup (RTH, ET):
  - Opening range = 09:30-09:45 ET (first OR_MIN minutes). OR_high/OR_low from those bars.
  - tight-OR-day filter: only trade if OR_range/price <= the OR_TIGHT_PCTL percentile of recent days
    (the time-series analogue of tight-OR).
  - Entry: first RTH bar after the OR that closes beyond OR_high (long) / OR_low (short).
  - Exit: trailing stop = TRAIL_R x OR_range from the running extreme, else flat at 15:55 ET (same-day;
    intraday-only avoids overnight gap risk and matches our short data window).
  - One trade/day/side; risk unit R = OR_range (points). Records return in MES $ and in R.

HONEST CAVEAT: runs on yfinance ~50-trading-day intraday (SMOKE TEST for the logic + a preliminary
read, NOT a verdict). Deep history (years) via broker_ibkr.py will make this decisive. Costs applied
per round-turn from data.py. Lookahead-free: entry uses a bar's close, fills at next bar's open.

    .venv-openbb/Scripts/python.exe futures/backtest_orb.py
"""
from __future__ import annotations

import sys
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from futures.data import COST_RT_USD, POINT_VALUE, TICK_VALUE, load_mes_intraday  # noqa: E402

RTH_OPEN, OR_END, EOD = time(9, 30), time(9, 45), time(15, 55)
OR_TIGHT_PCTL = 0.5      # trade only days with OR range in the tightest half (time-series tight-OR)
TRAIL_R = 1.0            # trailing stop distance = TRAIL_R * OR_range
DIRECTION = "both"       # "long", "short", or "both"


def _rth(day_df: pd.DataFrame) -> pd.DataFrame:
    t = day_df.index.time
    return day_df[(t >= RTH_OPEN) & (t <= EOD)]


def _simulate_day(day_df: pd.DataFrame, allow_long: bool, allow_short: bool):
    """Return (ret_usd, ret_R, side, or_range) for one day, or None if no trade."""
    rth = _rth(day_df)
    if len(rth) < 5:
        return None
    orbars = rth[rth.index.time < OR_END]
    if len(orbars) == 0:
        return None
    or_hi, or_lo = orbars["high"].max(), orbars["low"].min()
    or_range = or_hi - or_lo
    if or_range <= 0:
        return None
    post = rth[rth.index.time >= OR_END]
    if len(post) < 2:
        return None

    # entry: first bar closing beyond the OR; fill at NEXT bar open (lookahead-free)
    side = 0
    entry_px = None
    bars = post.reset_index(drop=True)
    for i in range(len(bars) - 1):
        c = bars.loc[i, "close"]
        if allow_long and c > or_hi:
            side = 1
        elif allow_short and c < or_lo:
            side = -1
        if side != 0:
            entry_px = bars.loc[i + 1, "open"]
            entry_i = i + 1
            break
    if side == 0 or entry_px is None:
        return None

    # trailing stop = TRAIL_R * or_range from the running favorable extreme; else exit at EOD close
    trail = TRAIL_R * or_range
    extreme = entry_px
    exit_px = bars.loc[len(bars) - 1, "close"]   # default EOD
    for j in range(entry_i, len(bars)):
        hi, lo = bars.loc[j, "high"], bars.loc[j, "low"]
        if side == 1:
            extreme = max(extreme, hi)
            if lo <= extreme - trail:
                exit_px = extreme - trail
                break
        else:
            extreme = min(extreme, lo)
            if hi >= extreme + trail:
                exit_px = extreme + trail
                break

    pts = (exit_px - entry_px) * side
    ret_usd = pts * POINT_VALUE - COST_RT_USD
    return ret_usd, pts / or_range, side, or_range


def backtest(df: pd.DataFrame, tight_pctl: float = OR_TIGHT_PCTL) -> pd.DataFrame:
    """Per-day trade table with the tight-OR-day filter applied. df = intraday OHLCV (ET)."""
    days = sorted({ts.date() for ts in df.index})
    # precompute each day's OR range %, to set the tight-OR threshold on a trailing basis
    or_pct = {}
    for d in days:
        dd = _rth(df[df.index.date == d])
        ob = dd[dd.index.time < OR_END]
        if len(ob) and dd["close"].iloc[-1] > 0:
            or_pct[d] = (ob["high"].max() - ob["low"].min()) / dd["close"].iloc[0]
    rows = []
    allow_long, allow_short = DIRECTION in ("long", "both"), DIRECTION in ("short", "both")
    hist = []   # trailing OR% for the percentile gate
    for d in days:
        op = or_pct.get(d)
        tight = None
        if op is not None:
            if len(hist) >= 10:
                thr = np.quantile(hist, tight_pctl)
                tight = bool(op <= thr)   # bool() -> avoid numpy-bool identity gotcha below
            hist.append(op)
        if not tight:                  # None (no history yet) or not-tight -> skip
            continue
        res = _simulate_day(df[df.index.date == d], allow_long, allow_short)
        if res:
            rows.append({"date": d, "ret_usd": res[0], "ret_R": res[1], "side": res[2]})
    return pd.DataFrame(rows)


def _stats(trades: pd.DataFrame, n_days: int) -> None:
    if trades.empty:
        print("  no trades (need >=10 days of history for the tight-OR gate).")
        return
    r = trades["ret_usd"]
    wins = (r > 0).sum()
    gross_w = r[r > 0].sum()
    gross_l = -r[r < 0].sum()
    pf = gross_w / gross_l if gross_l > 0 else float("inf")
    eq = r.cumsum()
    dd = (eq - eq.cummax()).min()
    print(f"  trades {len(trades)} over {n_days} days ({len(trades)/n_days*100:.0f}% of days)")
    print(f"  net $ {r.sum():+.0f}  (per trade {r.mean():+.1f}, in R {trades['ret_R'].mean():+.2f})")
    print(f"  win% {wins/len(trades)*100:.0f}  PF {pf:.2f}  maxDD ${dd:+.0f}  "
          f"best ${r.max():+.0f}  worst ${r.min():+.0f}")
    # tail lens (our house metric): SUM and best matter more than mean for a small book
    print(f"  R: mean {trades['ret_R'].mean():+.2f}  best {trades['ret_R'].max():+.2f}  "
          f"worst {trades['ret_R'].min():+.2f}  (1R = the day's OR range; cost ~{COST_RT_USD/TICK_VALUE:.0f} ticks)")


def main() -> int:
    from futures.data import load_mes_intraday_cache
    try:
        df = load_mes_intraday_cache()          # deep IBKR 5-min cache if present
        src = "IBKR cache"
    except FileNotFoundError:
        df = load_mes_intraday(interval="5m", period="60d")
        src = "yfinance ~50d SMOKE"
    n_days = len({ts.date() for ts in df.index})
    print(f"=== MES intraday ORB [{src}]: {len(df)} 5m bars, {n_days} days "
          f"{df.index.min().date()} -> {df.index.max().date()} ===")
    print(f"setup: OR 09:30-09:45 ET | tight-OR gate <= p{OR_TIGHT_PCTL*100:.0f} | "
          f"trail {TRAIL_R}xOR | EOD 15:55 | dir={DIRECTION} | net of ${COST_RT_USD:.0f}/RT\n")
    print(f"[tight-OR-day filtered <= p{OR_TIGHT_PCTL*100:.0f}]")
    _stats(backtest(df, OR_TIGHT_PCTL), n_days)

    # contrast: no tight-OR filter (trade every day) -- does the gate help, like on stocks?
    print("\n[ALL days, no tight-OR filter -- contrast]")
    _stats(backtest(df, 1.0), n_days)

    print("\nRead: PRELIMINARY only (~50 days). Looking for net$>0, PF>1, and the tight-OR gate beating")
    print("the all-days version (our stock finding). A fat 'best R' with modest mean = the tail we want.")
    print("Deep IBKR history will make this decisive; this just proves the logic and hints at sign.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
