"""Dissect the ORB trades we already have — where is the edge, where does it leak?

Instead of guessing new signals, slice the existing trades by entry characteristics
and find sub-populations that are systematically strong or (net) negative. A
net-negative slice = something to cut; a much-stronger slice = something to lean on.
Cheap (cached fixed-2R trades + trend filter; no re-backtest). Net of ~0.042R cost.

Dimensions: entry time-of-day, breakout strength (how far past OR the fill landed),
OR-range size (% of price), day of week. avgR is the capital-agnostic edge metric.

Run:
    .venv/Scripts/python.exe backtest/trade_anatomy.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.compare_norefill_trend import trend_eligibility, apply_filter  # noqa: E402
from backtest.compare_selection import _tday  # noqa: E402

COST = 0.042
W = "730"


def main():
    trades = pickle.load(open(ROOT / "backtest" / f".bars_cache_trades_{W}d.pkl", "rb"))
    closes = pickle.load(open(ROOT / "backtest" / f".bars_cache_daily_{W}d.pkl", "rb"))
    present = sorted({t.symbol for t in trades})
    days = sorted({_tday(t) for t in trades})
    trades = apply_filter(trades, trend_eligibility(closes, present, days))
    trades = [t for t in trades if t.side == "long"]

    rows = []
    for t in trades:
        et = t.entry_time.tz_convert("America/New_York") if t.entry_time.tzinfo else t.entry_time
        orr = t.or_high - t.or_low
        rows.append({
            "net_r": t.pnl_r - COST,
            "win": 1 if t.pnl_r > 0 else 0,
            "entry_min": et.hour * 60 + et.minute,
            "hhmm": et.strftime("%H:%M"),
            "breakout_str": (t.entry_price - t.or_high) / orr if orr > 0 else 0.0,  # OR-range units past OR_high
            "or_pct": orr / t.entry_price * 100 if t.entry_price else 0.0,           # OR range as % of price
            "dow": et.strftime("%a"),
        })
    df = pd.DataFrame(rows)
    print(f"=== {W}d: {len(df)} long trades (trend-filtered), net of {COST}R cost ===")
    print(f"OVERALL: avgR {df.net_r.mean():+.3f}  win {df.win.mean()*100:.1f}%  sumR {df.net_r.sum():+.0f}\n")

    def show(name, col, bins, labels):
        df["_b"] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
        g = df.groupby("_b", observed=True)
        print(f"-- by {name} --")
        print(f"   {'bucket':<14}{'n':>6}{'win%':>7}{'avgR':>8}{'sumR':>8}")
        for b, sub in g:
            print(f"   {str(b):<14}{len(sub):>6}{sub.win.mean()*100:>6.1f}%{sub.net_r.mean():>+8.3f}{sub.net_r.sum():>+8.0f}")
        print()

    show("entry time (ET)", "entry_min",
         [585, 600, 615, 630, 660, 690],
         ["9:45-10:00", "10:00-10:15", "10:15-10:30", "10:30-11:00", "11:00-11:30"])
    show("breakout strength (OR units past OR_high)", "breakout_str",
         [-1, 0.05, 0.15, 0.30, 0.60, 100], ["~0 (at OR)", ".05-.15", ".15-.30", ".30-.60", ">.60 (chase)"])
    show("OR range (% of price)", "or_pct",
         [0, 0.3, 0.5, 0.8, 1.2, 100], ["<0.3% tight", "0.3-0.5%", "0.5-0.8%", "0.8-1.2%", ">1.2% wide"])
    print("-- by day of week --")
    g = df.groupby("dow", observed=True)
    for d in ["Mon", "Tue", "Wed", "Thu", "Fri"]:
        if d in g.groups:
            sub = g.get_group(d)
            print(f"   {d}  n={len(sub):>4}  win {sub.win.mean()*100:.1f}%  avgR {sub.net_r.mean():+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
