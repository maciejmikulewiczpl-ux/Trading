"""Entry-timing check (Gemini review kernel #2): is the 09:45 entry "buying the retail exhaustion
top"? Gemini claims 09:30-09:45 is a retail markup we buy into, then the stock mean-reverts.

For the bot's TRADED basket (top-3 by logged combined_score each day), fetch IEX minute bars and
compute, per name:
  drift_open_945 = 09:30 open -> 09:45 open   (the pre-entry markup Gemini says we chase)
  ret_945_close  = 09:45 open -> 15:55 close  (current entry; already logged, recomputed here)
  ret_open_close = 09:30 open -> 15:55 close  (what a MOO/open entry would have captured)

Exhaustion thesis predicts: drift_open_945 strongly POSITIVE and ret_945_close NEGATIVE (we buy the
top and it fades). PROFIT test: if ret_open_close > ret_945_close, entering at the open earns more
(we're leaking the drift by waiting). MEASUREMENT ONLY; does not touch the live bot.

Run:
    .venv/Scripts/python.exe experiments/lottery/entry_timing.py
"""
from __future__ import annotations

import json
import statistics as st
import sys
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from backtest.run_orb import load_env  # noqa: E402

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
PICKS_DIR = Path(__file__).resolve().parent / "picks"
OPEN_T, ENTRY_T, EXIT_T = time(9, 30), time(9, 45), time(15, 55)


def _client():
    import os
    from alpaca.data.historical import StockHistoricalDataClient
    load_env()
    return StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])


def _agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return (len(vals), st.mean(vals), st.median(vals),
            100 * sum(1 for v in vals if v > 0) / len(vals), sum(vals))


def main() -> int:
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    dc = _client()
    rows = []  # (date, sym, drift, r945, ropen)
    for f in sorted(PICKS_DIR.glob("*.json")):
        rec = json.load(open(f))
        scored = [p for p in rec["picks"] if p.get("combined_score") is not None]
        top3 = sorted(scored, key=lambda x: -x["combined_score"])[:3]
        syms = [p["symbol"] for p in top3]
        if not syms:
            continue
        d = datetime.fromisoformat(rec["date"]).date()
        start = datetime.combine(d, OPEN_T, ET)
        end = datetime.combine(d, time(16, 0), ET)
        try:
            req = StockBarsRequest(symbol_or_symbols=syms, timeframe=TimeFrame.Minute,
                                   start=start.astimezone(UTC), end=end.astimezone(UTC),
                                   feed=DataFeed.IEX)
            df = dc.get_stock_bars(req).df
        except Exception as e:
            print(f"{rec['date']}: fetch failed ({str(e)[:50]})")
            continue
        for sym in syms:
            try:
                sb = df.xs(sym, level=0)
                t = sb.index.tz_convert(ET).time
                o930 = sb[t >= OPEN_T]["open"].iloc[0]
                o945 = sb[t >= ENTRY_T]["open"].iloc[0]
                close = sb[t <= EXIT_T]["close"].iloc[-1]
                drift = (o945 / o930 - 1.0) * 100
                r945 = (close / o945 - 1.0) * 100
                ropen = (close / o930 - 1.0) * 100
                rows.append((rec["date"], sym, drift, r945, ropen))
            except Exception:
                continue

    if not rows:
        print("no bars."); return 0
    drift = _agg([r[2] for r in rows])
    r945 = _agg([r[3] for r in rows])
    ropen = _agg([r[4] for r in rows])
    n_days = len({r[0] for r in rows})
    print(f"=== ENTRY-TIMING: {len(rows)} traded names over {n_days} days (top-3 by combined_score) ===\n")
    print(f"{'leg':<28}{'n':>4}{'mean%':>8}{'med%':>8}{'pos%':>7}{'SUM%':>9}")
    print(f"{'open(9:30)->9:45 drift':<28}{drift[0]:>4}{drift[1]:>+8.2f}{drift[2]:>+8.2f}{drift[3]:>7.0f}{drift[4]:>+9.1f}")
    print(f"{'9:45->close (CURRENT entry)':<28}{r945[0]:>4}{r945[1]:>+8.2f}{r945[2]:>+8.2f}{r945[3]:>7.0f}{r945[4]:>+9.1f}")
    print(f"{'open->close (MOO entry)':<28}{ropen[0]:>4}{ropen[1]:>+8.2f}{ropen[2]:>+8.2f}{ropen[3]:>7.0f}{ropen[4]:>+9.1f}")

    # exhaustion split: do HIGH pre-entry-drift names fade harder after 09:45?
    med_drift = st.median([r[2] for r in rows])
    hi = [r[3] for r in rows if r[2] >= med_drift]
    lo = [r[3] for r in rows if r[2] < med_drift]
    ah, al = _agg(hi), _agg(lo)
    print("\n  Exhaustion split (09:45->close return, by pre-entry drift):")
    if ah:
        print(f"    high open->9:45 drift  n={ah[0]:>3} 9:45->close mean={ah[1]:+.2f}% SUM={ah[4]:+.1f}%")
    if al:
        print(f"    low  open->9:45 drift  n={al[0]:>3} 9:45->close mean={al[1]:+.2f}% SUM={al[4]:+.1f}%")

    d945, dopen = r945[1], ropen[1]
    print("\nRead: exhaustion thesis needs drift>0 AND 9:45->close<0 (we buy the top, it fades). If")
    print("open->close (MOO) SUM <= 9:45->close SUM, entering earlier does NOT earn more -> reject #2.")
    print(f"Verdict: MOO {'BEATS' if dopen > d945 else 'does NOT beat'} 09:45 on mean "
          f"({dopen:+.2f}% vs {d945:+.2f}%). Directional, small n; IEX feed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
