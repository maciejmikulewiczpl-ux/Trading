"""Hype bot HOLD-HORIZON + ENTRY-TIMING curves (review #3 + SECOND_WAVE_SPEC Tier 1).

Answers two questions the reviews (all 3) + our own data raised:
  1. OPTIMAL HOLD: entering at 09:45, what's the expectancy at each exit horizon
     (10:00 ... close, then +1d/+2d/+3d/+5d/+10d)? -> validates the same-day-exit change +
     shows where the edge peaks / decays.
  2. ENTRY TIMING: is 09:45 too early? Entering at 10:30 / 12:00 / 14:00 and holding to the
     SAME-day close -- does a later entry on the same names beat 09:45? (DeepSeek: "09:45
     arrives after the party.")

MEASUREMENT ONLY -- reads the immutable picks files, pulls fresh IEX bars, computes in memory,
prints the curves. Does NOT mutate picks files or touch the live bot. Reports the bot's traded
set (top-3 by combined_score) and all scored picks.

Run:
    .venv/Scripts/python.exe experiments/lottery/horizon_curve.py
"""
from __future__ import annotations

import json
import statistics as st
import sys
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from backtest.run_orb import load_env  # noqa: E402

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
PICKS_DIR = Path(__file__).resolve().parent / "picks"

ENTRY_945 = time(9, 45)
# hold-horizon exits (entering 09:45): intraday times, then multi-day handled separately
INTRADAY_EXITS = [time(10, 0), time(10, 30), time(11, 0), time(12, 0),
                  time(13, 0), time(14, 0), time(15, 0), time(15, 55)]
DAILY_HORIZONS = [1, 2, 3, 5, 10]           # trading-day closes after the entry day
ENTRY_TIMES = [time(9, 45), time(10, 30), time(12, 0), time(14, 0)]  # entry-timing curve


def _client():
    import os
    from alpaca.data.historical import StockHistoricalDataClient
    load_env()
    return StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])


def _price_at_or_after(sb_et, t, field="open"):
    """First bar at/after time t (same day). None if none."""
    m = sb_et[[ts.time() >= t for ts in sb_et.index]]
    return float(m[field].iloc[0]) if len(m) else None


def _price_at_or_before(sb_et, t, field="close"):
    """Last bar at/before time t. None if none."""
    m = sb_et[[ts.time() <= t for ts in sb_et.index]]
    return float(m[field].iloc[-1]) if len(m) else None


def _agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return len(vals), st.mean(vals), st.median(vals), 100 * sum(1 for v in vals if v > 0) / len(vals)


def collect():
    """Per pick: {rank, hold[label]->ret%, entry[label]->ret%}. Fresh IEX bars per day."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    dc = _client()
    rows = []
    files = sorted(PICKS_DIR.glob("*.json"))
    for f in files:
        rec = json.load(open(f))
        picks = [p for p in rec["picks"] if p.get("combined_score") is not None]
        if not picks:
            continue
        picks.sort(key=lambda x: -x["combined_score"])
        rank = {p["symbol"]: i + 1 for i, p in enumerate(picks)}
        syms = [p["symbol"] for p in picks]
        d = datetime.fromisoformat(rec["date"]).date()
        mstart = datetime.combine(d, time(9, 30), ET); mend = datetime.combine(d, time(16, 0), ET)
        dstart = datetime.combine(d, time(0, 0), ET); dend = datetime.combine(d + timedelta(days=18), time(23, 59), ET)
        try:
            mdf = dc.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=syms, timeframe=TimeFrame.Minute,
                start=mstart.astimezone(UTC), end=mend.astimezone(UTC), feed=DataFeed.IEX)).df
            ddf = dc.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=syms, timeframe=TimeFrame.Day,
                start=dstart.astimezone(UTC), end=dend.astimezone(UTC), feed=DataFeed.IEX)).df
        except Exception as e:
            print(f"  {rec['date']}: fetch failed ({str(e)[:50]}) -- skip day")
            continue
        for sym in syms:
            row = {"date": rec["date"], "sym": sym, "rank": rank[sym], "hold": {}, "entry": {}}
            try:
                sb = mdf.xs(sym, level=0)
                sb = sb.set_index(sb.index.tz_convert(ET))
            except Exception:
                sb = None
            entry945 = _price_at_or_after(sb, ENTRY_945) if sb is not None else None
            # hold-horizon (enter 09:45 -> intraday exits)
            if sb is not None and entry945:
                for t in INTRADAY_EXITS:
                    px = _price_at_or_before(sb, t)
                    row["hold"][t.strftime("%H:%M")] = (px / entry945 - 1) * 100 if px else None
            # entry-timing (enter at T -> same-day close)
            close_px = _price_at_or_before(sb, time(15, 55)) if sb is not None else None
            if sb is not None and close_px:
                for t in ENTRY_TIMES:
                    ep = _price_at_or_after(sb, t)
                    row["entry"][t.strftime("%H:%M")] = (close_px / ep - 1) * 100 if ep else None
            # multi-day (enter 09:45 -> close[d+N]) from daily bars
            if entry945:
                try:
                    db = ddf.xs(sym, level=0).sort_index()
                    dts = [ts.tz_convert(ET).date() for ts in db.index]
                    if d in dts:
                        i = dts.index(d)
                        for n in DAILY_HORIZONS:
                            if i + n < len(db):
                                cN = float(db["close"].iloc[i + n])
                                row["hold"][f"+{n}d"] = (cN / entry945 - 1) * 100
                except Exception:
                    pass
            rows.append(row)
    return rows


def curve(rows, keys, field, subset):
    print(f"\n  {'horizon':<10}{'n':>5}{'mean%':>9}{'median%':>10}{'win%':>7}")
    for k in keys:
        vals = [r[field].get(k) for r in rows if subset(r) and k in r[field]]
        a = _agg(vals)
        if a:
            print(f"  {k:<10}{a[0]:>5}{a[1]:>+9.2f}{a[2]:>+10.2f}{a[3]:>7.0f}")


def main():
    rows = collect()
    if not rows:
        print("no rows collected."); return 0
    top3 = lambda r: r["rank"] <= 3
    allp = lambda r: True
    hold_keys = [t.strftime("%H:%M") for t in INTRADAY_EXITS] + [f"+{n}d" for n in DAILY_HORIZONS]
    entry_keys = [t.strftime("%H:%M") for t in ENTRY_TIMES]

    print("=" * 64)
    print("HOLD-HORIZON curve (enter 09:45, exit at ...) -- BOT top-3")
    curve(rows, hold_keys, "hold", top3)
    print("\nHOLD-HORIZON curve -- ALL scored picks")
    curve(rows, hold_keys, "hold", allp)

    print("\n" + "=" * 64)
    print("ENTRY-TIMING curve (enter at ..., exit same-day close) -- BOT top-3")
    curve(rows, entry_keys, "entry", top3)
    print("\nENTRY-TIMING curve -- ALL scored picks")
    curve(rows, entry_keys, "entry", allp)

    print("\nRead: HOLD curve peak = optimal exit (validates/tunes the same-day close).")
    print("ENTRY curve: if a later entry beats 09:45 to close, 09:45 is too early.")
    print("Directional until n_days>=30; small n per cell. Buy-and-hold (no trailing stop).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
