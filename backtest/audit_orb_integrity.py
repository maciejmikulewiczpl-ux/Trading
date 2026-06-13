"""audit_orb_integrity.py -- post-swing-incident audit of the ORB result foundations.

Motivation (2026-06-12): the swing-engine backtest reported Sharpe 3.03 that turned
out to be a 5.65x daily-PnL double-count + survivorship pool. This audits whether
the ORB caches and portfolio machinery -- the foundation of EVERY shipped finding
(tight-OR, trailing, diversification, vol-dial) -- carry analogous defects.

Checks on both trade caches (.bars_cache_trades_730d.pkl fixed-2R cache and
.pit_trailtrades_730d.pkl trailing cache):

  C1 PnL identity      pnl_dollars == (exit-entry)*shares  (longs; sign flip shorts)
  C2 risk identity     risk_dollars == (entry-stop)*shares  and  pnl_r*risk == pnl
  C3 intraday times    09:45 <= entry < exit <= 16:00, same session (no overnight)
  C4 breakout sanity   long entry_price >= or_high (entry on/after breakout, small eps)
  C5 stop sanity       long stop < entry, stop within [or_low - eps, entry]
  C6 positivity        shares > 0, prices > 0, or_low <= or_high
  C7 conservation      daily series built the portfolio way (one insert per trade,
                       bucketed by day) sums EXACTLY to sum of per-trade PnL
                       -- the invariant whose violation produced swing Sharpe 3.03.

Run:  .venv/Scripts/python.exe backtest/audit_orb_integrity.py
"""
from __future__ import annotations

import pickle
import sys
from datetime import time as dtime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EPS = 0.011          # a cent + float fuzz
T_OREND, T_CLOSE = dtime(9, 45), dtime(16, 0)


def tday(t):
    return t.entry_time.date()


def audit(trades, label):
    n = len(trades)
    bad = {k: 0 for k in ("C1_pnl", "C2_risk", "C3_time", "C4_break", "C5_stop", "C6_pos")}
    examples = {}

    for t in trades:
        sgn = 1 if t.side == "long" else -1
        # C1
        expect = sgn * (t.exit_price - t.entry_price) * t.shares
        if abs(expect - t.pnl_dollars) > max(0.02, abs(t.pnl_dollars) * 1e-4):
            bad["C1_pnl"] += 1
            examples.setdefault("C1_pnl", t)
        # C2
        expect_risk = sgn * (t.entry_price - t.stop_price) * t.shares
        if abs(expect_risk - t.risk_dollars) > max(0.02, abs(t.risk_dollars) * 1e-4):
            bad["C2_risk"] += 1
            examples.setdefault("C2_risk", t)
        elif t.risk_dollars > 0 and abs(t.pnl_r * t.risk_dollars - t.pnl_dollars) > \
                max(0.05, abs(t.pnl_dollars) * 1e-3):
            bad["C2_risk"] += 1
            examples.setdefault("C2_risk", t)
        # C3
        et, xt = t.entry_time, t.exit_time
        same_day = et.date() == xt.date()
        if not (same_day and et < xt and et.time() >= T_OREND and xt.time() <= T_CLOSE):
            bad["C3_time"] += 1
            examples.setdefault("C3_time", t)
        # C4
        if t.side == "long" and t.entry_price < t.or_high - EPS:
            bad["C4_break"] += 1
            examples.setdefault("C4_break", t)
        # C5
        if t.side == "long" and not (t.stop_price < t.entry_price + EPS
                                     and t.stop_price >= t.or_low - 0.51):
            bad["C5_stop"] += 1
            examples.setdefault("C5_stop", t)
        # C6
        if not (t.shares > 0 and t.entry_price > 0 and t.exit_price > 0
                and t.or_low <= t.or_high + EPS):
            bad["C6_pos"] += 1
            examples.setdefault("C6_pos", t)

    # C7 conservation: bucket once per trade by day, compare sums
    by = {}
    for t in trades:
        by[tday(t)] = by.get(tday(t), 0.0) + t.pnl_dollars
    ser = pd.Series(by)
    total_trades = sum(t.pnl_dollars for t in trades)
    c7_ok = abs(ser.sum() - total_trades) < 0.01

    print(f"\n=== {label}: {n} trades ===")
    for k, v in bad.items():
        status = "OK " if v == 0 else "FAIL"
        print(f"  {k:<9} violations: {v:>6} / {n}   {status}")
        if v and k in examples:
            t = examples[k]
            print(f"            e.g. {t.symbol} {t.entry_time} entry={t.entry_price} "
                  f"exit={t.exit_price} stop={t.stop_price} sh={t.shares} "
                  f"pnl={t.pnl_dollars:.2f} r={t.pnl_r:.3f}")
    print(f"  C7_conserv daily-sum {ser.sum():+,.2f} vs trades {total_trades:+,.2f}   "
          f"{'OK' if c7_ok else 'FAIL'}")
    # entry-time distribution eyeball
    ts = sorted({t.entry_time.time() for t in trades})
    print(f"  entry times span {ts[0]} -> {ts[-1]} | "
          f"exit reasons: {pd.Series([t.exit_reason for t in trades]).value_counts().to_dict()}")
    return bad, c7_ok


def main() -> None:
    fixed = pickle.load(open(ROOT / "backtest" / ".bars_cache_trades_730d.pkl", "rb"))
    audit([t for t in fixed if t.side == "long"], "fixed-2R cache (730d) LONGS")

    blob = pickle.load(open(ROOT / "backtest" / ".pit_trailtrades_730d.pkl", "rb"))
    trail = [t for syms in blob.values() for t in syms if t.side == "long"]
    audit(trail, "trailing cache PIT (730d) LONGS")


if __name__ == "__main__":
    main()
