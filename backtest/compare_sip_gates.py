"""'Stocks in play' gates on the shipped tight-OR config: RVOL / VWAP / bar-volume.

Literature basis (overnight research 2026-06-10): Zarattini, Barbon & Aziz (SSRN
4729284, "A Profitable Day Trading Strategy For The U.S. Equity Market", 7,000
stocks 2016-2023) report that the ENTIRE ORB edge concentrates in names whose
opening-range RELATIVE VOLUME is elevated: RVOL<1.0 -> -0.02R/trade, RVOL>1.0 ->
+0.08R/trade. Practitioner consensus adds VWAP alignment (breakout above session
VWAP = institutional backing) and breakout-bar volume confirmation. None of these
was ever tested as a GATE on our shipped config (or_rvol was only tried as a
RANKING for cap-8 queue ordering, a different mechanism — see
selection_scanner_findings).

This tests, on the SHIPPED config (tight-OR<=0.5%, trail-1R, trend filter,
vol-dial half, $50 risk / $10k cap, cents-slippage with median trade = 0.042R):

  base               : shipped config, no extra gate
  rvol>=1.0/1.5/2.0  : OR-window (09:30-09:45) volume vs its trailing-14-session
                       mean for that symbol (point-in-time, needs >=5 prior days)
  vwap_confirm       : signal-bar close > session VWAP at the signal bar
  barvol>=1.5        : signal-bar volume >= 1.5x mean per-bar OR volume
  rvol1.0+vwap       : both gates

Also prints the Zarattini diagnostic: avg net-$/trade by RVOL bucket, to see if
the monotone RVOL->PnL relation exists at all in our tight-OR universe.

All arms (incl. base) drop the first 15 sessions of each window so every trade
has RVOL history — common eval window, no arm-dependent asymmetry.

PRE-REGISTERED GATE (set before seeing results): an arm is a ship candidate only
if, in BOTH windows and at BOTH slippage levels (1.0x, 1.5x):
  Sharpe >= base + 0.10, maxDD <= base maxDD, PnL >= 0.85 x base PnL,
  h2 (recent OOS half) PnL >= base h2 - 10%.
A filter cuts trades, so it must pay for lost volume with risk-adjusted quality.

Run (loads the big univ caches, several minutes):
    .venv/Scripts/python.exe backtest/compare_sip_gates.py
"""
from __future__ import annotations

import math
import pickle
import statistics
import sys
from datetime import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.compare_selection import _tday  # noqa: E402
from backtest.compare_norefill_trend import trend_eligibility, apply_filter  # noqa: E402
from backtest.compare_exits import load, bucket, reexit, POLICIES, EOD  # noqa: E402
from backtest.compare_volpause import prior_vol_flags, perf, RISK  # noqa: E402
from backtest.compare_or_range_realcost import or_pct, risk_ps  # noqa: E402

WINDOWS = [730, 180]
OR_THR = 0.5
TARGET_MEDIAN_R = 0.042
NOTIONAL_CAP = 10_000.0
SLIP_MULT = [1.0, 1.5]
RVOL_LOOKBACK = 14
RVOL_MIN_HIST = 5
WARMUP_SESSIONS = 15
OR_END = time(9, 45)
RTH_S, RTH_E = time(9, 30), time(16, 0)


def gate_stats(all_bars, present):
    """Per (symbol, date): OR volume, trailing RVOL, and signal-bar lookup arrays.

    Returns {sym: {date: dict(rvol, idx, cl, vol, cumvwap, or_barvol_mean)}}.
    rvol is None when fewer than RVOL_MIN_HIST prior sessions exist.
    """
    out = {}
    for sym in present:
        sb = all_bars.xs(sym, level=0)
        t = sb.index.time
        sb = sb[(t >= RTH_S) & (t < RTH_E)]
        days = {}
        or_vols = []          # (date, or_vol) in session order
        for day, g in sb.groupby(sb.index.date):
            or_g = g[g.index.time < OR_END]
            if or_g.empty:
                continue
            or_vol = float(or_g["volume"].sum())
            vol = g["volume"].to_numpy(float)
            # session VWAP path: cum(bar_vwap*vol)/cum(vol), lookahead-free per bar
            pv = (g["vwap"].to_numpy(float) * vol).cumsum()
            cv = vol.cumsum()
            cumvwap = pv / pd.Series(cv).replace(0.0, float("nan")).to_numpy()
            days[day] = {
                "or_vol": or_vol,
                "or_barvol_mean": or_vol / max(len(or_g), 1),
                "idx": g.index, "cl": g["close"].to_numpy(float),
                "vol": vol, "cumvwap": cumvwap, "rvol": None,
            }
            or_vols.append((day, or_vol))
        # trailing RVOL (strictly prior sessions)
        for i, (day, ov) in enumerate(or_vols):
            hist = [v for _, v in or_vols[max(0, i - RVOL_LOOKBACK):i]]
            if len(hist) >= RVOL_MIN_HIST and sum(hist) > 0:
                days[day]["rvol"] = ov / (sum(hist) / len(hist))
        out[sym] = days
    return out


def signal_features(t, gs):
    """(rvol, vwap_ok, barvol_ratio) for a trade, or None if unavailable."""
    day = gs.get(t.symbol, {}).get(_tday(t))
    if day is None:
        return None
    pos = int(day["idx"].searchsorted(t.entry_time, side="left")) - 1
    if pos < 0 or pos >= len(day["cl"]):
        return None
    cw = day["cumvwap"][pos]
    vwap_ok = bool(pd.notna(cw) and day["cl"][pos] > cw)
    barvol = day["vol"][pos] / day["or_barvol_mean"] if day["or_barvol_mean"] > 0 else 0.0
    return day["rvol"], vwap_ok, barvol


def cap_shares(t, days_mult):
    rps = risk_ps(t)
    target = RISK * days_mult.get(_tday(t), 1.0)
    return min(math.floor(target / rps), math.floor(NOTIONAL_CAP / t.entry_price))


def dollar_series(taken, days, days_mult, cents):
    by = {}
    for t in taken:
        sh = cap_shares(t, days_mult)
        if sh <= 0:
            continue
        pnl = (t.exit_price - t.entry_price) * sh - 2.0 * cents * sh
        by[_tday(t)] = by.get(_tday(t), 0.0) + pnl
    return pd.Series({d: by.get(d, 0.0) for d in sorted(days)})


HEAD = (f"{'arm':<18}{'trades':>7}{'PnL$':>10}{'Sharpe':>8}{'maxDD$':>9}"
        f"   {'h1 PnL':>9}{'h2 PnL':>9}")


def show(label, taken, days, mid, days_mult, cents):
    s = dollar_series(taken, days, days_mult, cents)
    f = perf(s)
    h1 = perf(s[s.index < mid])
    h2 = perf(s[s.index >= mid])
    print(f"{label:<18}{len(taken):>7}{f['pnl']:>+10,.0f}{f['sharpe']:>8.2f}"
          f"{f['maxdd']:>9,.0f}   {h1['pnl']:>+9,.0f}{h2['pnl']:>+9,.0f}")
    return s


def run_window(w):
    all_bars, days, present, trades, closes = load(w)
    elig = trend_eligibility(closes, present, days)
    buckets = bucket(all_bars, present)
    gs = gate_stats(all_bars, present)
    tz = all_bars.index.get_level_values(1).tz
    eod_ns = {d: pd.Timestamp.combine(d, EOD).tz_localize(tz).value for d in days}
    trail = [t for t in apply_filter(reexit(trades, buckets, POLICIES["trail_1R"], eod_ns), elig)
             if t.side == "long"]
    tight = [t for t in trail if or_pct(t) <= OR_THR]

    # common eval window: skip warmup sessions so RVOL history always exists
    days_eval = sorted(days)[WARMUP_SESSIONS:]
    dset = set(days_eval)
    mid = days_eval[len(days_eval) // 2]
    prior = prior_vol_flags(closes, days)
    days_mult = {d: (0.5 if prior[d] else 1.0) for d in days}
    base_cents = TARGET_MEDIAN_R * statistics.median(risk_ps(t) for t in tight) / 2.0

    # attach features; drop trades we can't evaluate from ALL arms
    feats = {}
    pool = []
    for t in tight:
        if _tday(t) not in dset:
            continue
        f = signal_features(t, gs)
        if f is None or f[0] is None:
            continue
        feats[id(t)] = f
        pool.append(t)

    arms = {
        "base": lambda f: True,
        "rvol>=1.0": lambda f: f[0] >= 1.0,
        "rvol>=1.5": lambda f: f[0] >= 1.5,
        "rvol>=2.0": lambda f: f[0] >= 2.0,
        "vwap_confirm": lambda f: f[1],
        "barvol>=1.5": lambda f: f[2] >= 1.5,
        "rvol1.0+vwap": lambda f: f[0] >= 1.0 and f[1],
    }

    print(f"\n=== {w}d: {len(days_eval)} eval sessions (of {len(days)}), OOS split {mid} ===")
    print(f"    pool: {len(pool)} tight-OR trades with RVOL history "
          f"(of {len(tight)} tight)")
    # Zarattini diagnostic: net-$/trade by RVOL bucket (1.0x slip)
    print(f"    RVOL bucket      {'n':>6}{'avg$/tr':>9}{'win%':>7}")
    for lo, hi in [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 1e9)]:
        grp = [t for t in pool if lo <= feats[id(t)][0] < hi]
        if not grp:
            continue
        tot, wins = 0.0, 0
        for t in grp:
            sh = cap_shares(t, days_mult)
            if sh <= 0:
                continue
            p = (t.exit_price - t.entry_price) * sh - 2.0 * base_cents * sh
            tot += p
            wins += p > 0
        lab = f"{lo:g}-{hi:g}" if hi < 1e9 else f"{lo:g}+"
        print(f"    {lab:<16}{len(grp):>6}{tot/len(grp):>+9.1f}{100*wins/len(grp):>6.0f}%")

    base_series = None
    for sm in SLIP_MULT:
        cents = base_cents * sm
        print(f"\n  -- slippage {sm:.1f}x --")
        print("  " + HEAD)
        print("  " + "-" * len(HEAD))
        for name, gate in arms.items():
            taken = [t for t in pool if gate(feats[id(t)])]
            s = show("  " + name, taken, days_eval, mid, days_mult, cents)
            if name == "base" and sm == 1.0:
                base_series = s
    # persist the base daily series for the SPY-momentum decorrelation check
    pickle.dump(base_series, open(ROOT / "backtest" / f".daily_base_tightOR_{w}d.pkl", "wb"))


def main():
    for w in WINDOWS:
        run_window(w)
    print("\nPre-registered gate: Sharpe >= base+0.10 AND maxDD <= base AND PnL >= 0.85x")
    print("base AND h2 >= base h2 - 10%, in BOTH windows at BOTH slips. Otherwise reject.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
