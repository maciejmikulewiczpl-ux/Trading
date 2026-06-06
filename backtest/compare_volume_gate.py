"""A reactive VOLUME-regime gate vs the shipped VOLATILITY dial — redundant or additive?

Gemini's market-structure points are all about VOLUME, but our shipped dial uses
VOLATILITY (realized price swings). Volume and volatility are highly correlated, so
a volume gate may just re-flag the same days. This tests it head-to-head on the
live trailing config, net of costs, both windows + OOS:

  vol gate    : pause/half when SPY 20d realized VOL > its 126d median (shipped)
  volume gate : pause/half when SPY 20d avg VOLUME > its 126d median (new)

Reports, for each gate: how many days it flags, the avg $/day ON flagged days (a
good gate flags the LOW-edge days), and the pause/half configs. Plus the OVERLAP
(days both gates flag) — high overlap = the volume gate is redundant with the dial
we already have. SPY volume is aggregated from the cached IEX minute bars (a stable
fraction of consolidated, so the RELATIVE measure is a fair proxy).

Run (re-simulates trailing, ~couple min; uses existing caches):
    .venv/Scripts/python.exe backtest/compare_volume_gate.py
"""
from __future__ import annotations

import sys
from datetime import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.universe_portfolio import portfolio  # noqa: E402
from backtest.compare_norefill_trend import trend_eligibility, apply_filter  # noqa: E402
from backtest.compare_exits import load, bucket, reexit, POLICIES, EOD  # noqa: E402
from backtest.compare_volpause import (  # noqa: E402
    prior_vol_flags, series, three, HEAD, prow, RISK, CAP,
)

WINDOWS = [730, 180]
VOL_WIN, VOL_MED_WIN = 20, 126


def volume_flags(all_bars, days) -> dict:
    """High-volume regime: SPY 20d avg daily volume > its trailing 126d median,
    decided from the prior close. Volume aggregated from cached SPY minute bars."""
    spy = all_bars.xs("SPY", level=0)
    t = spy.index.time
    rth = spy[(t >= time(9, 30)) & (t < time(16, 0))]
    dv = rth.groupby(rth.index.date)["volume"].sum().sort_index()
    avg = dv.rolling(VOL_WIN).mean()
    med = avg.rolling(VOL_MED_WIN, min_periods=40).median()
    f = (avg > med).shift(1)
    return {d: (bool(f.get(d)) if pd.notna(f.get(d)) else False) for d in days}


def run_window(w):
    all_bars, days, present, trades, closes = load(w)
    mid = sorted(days)[len(days) // 2]
    elig = trend_eligibility(closes, present, days)
    buckets = bucket(all_bars, present)
    tz = all_bars.index.get_level_values(1).tz
    eod_ns = {d: pd.Timestamp.combine(d, EOD).tz_localize(tz).value for d in days}
    taken = portfolio(apply_filter(reexit(trades, buckets, POLICIES["trail_1R"], eod_ns), elig), CAP)

    vol = prior_vol_flags(closes, days)      # VOLATILITY gate (shipped)
    vlm = volume_flags(all_bars, days)       # VOLUME gate (new)
    one = {d: 1.0 for d in days}
    full = series(taken, days, one)
    def avg_on(flag, val):
        sel = [full[d] for d in sorted(days) if flag[d] == val]
        return (sum(sel) / len(sel)) if sel else 0.0
    both = sum(1 for d in days if vol[d] and vlm[d])
    vol_only = sum(1 for d in days if vol[d] and not vlm[d])
    vlm_only = sum(1 for d in days if vlm[d] and not vol[d])

    print(f"\n=== {w}d: {len(days)} sessions, OOS split {mid} (trailing, net of cost) ===")
    print(f"  VOLATILITY gate: flags {sum(vol.values())} | avg $/day flagged {avg_on(vol,True):+.0f} vs calm {avg_on(vol,False):+.0f}")
    print(f"  VOLUME gate    : flags {sum(vlm.values())} | avg $/day flagged {avg_on(vlm,True):+.0f} vs calm {avg_on(vlm,False):+.0f}")
    print(f"  OVERLAP: both {both} | vol-only {vol_only} | volume-only {vlm_only}  "
          f"(high overlap => volume is redundant with the shipped dial)")
    print(HEAD); print("-" * len(HEAD))
    prow("normal", *three(taken, days, mid, one))
    prow("VOL pause", *three(taken, days, mid, {d: (0.0 if vol[d] else 1.0) for d in days}))
    prow("VOL half", *three(taken, days, mid, {d: (0.5 if vol[d] else 1.0) for d in days}))
    prow("VOLUME pause", *three(taken, days, mid, {d: (0.0 if vlm[d] else 1.0) for d in days}))
    prow("VOLUME half", *three(taken, days, mid, {d: (0.5 if vlm[d] else 1.0) for d in days}))


def main():
    for w in WINDOWS:
        run_window(w)
    print("\nReads: if the VOLUME gate flags ~the same days (high overlap) and gives ~the")
    print("same Sharpe/drawdown lift as the VOLATILITY gate, it's redundant — keep the dial")
    print("we shipped. If it flags DIFFERENT low-edge days and adds lift, it's complementary.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
