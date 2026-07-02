"""Overnight vs intraday decomposition (Fable PnL #1 diagnostic). Every reviewer wants a same-day
exit to kill the -0.64%/night gap bleed. The counter-argument is that the +1/+2 day TAIL is what pays,
and it may live in the overnight gaps. This settles it: for the traded top-3 names, decompose the
multi-day hold return (close_T -> close_T+3, the ret_3d window) into its OVERNIGHT legs
(close_{d-1} -> open_d) and INTRADAY legs (open_d -> close_d), in log space so they sum exactly.

  If OVERNIGHT is a net drag AND intraday carries the mean+tail -> a same-day-style exit is defensible.
  If OVERNIGHT carries the right tail (moonshots gap up overnight) -> same-day exit AMPUTATES the tail,
  confirming the earlier revert. MEASUREMENT ONLY.

Run:
    .venv/Scripts/python.exe experiments/lottery/overnight_decomp.py
"""
from __future__ import annotations

import json
import math
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
HOLD = 3   # trading days (T -> T+3, matches ret_3d)


def _client():
    import os
    from alpaca.data.historical import StockHistoricalDataClient
    load_env()
    return StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])


def _agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return (len(vals), st.mean(vals), st.median(vals), sum(vals), max(vals), min(vals))


def main() -> int:
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    dc = _client()
    rows = []   # (sym, date, overnight_pct, intraday_pct, total_pct)
    for f in sorted(PICKS_DIR.glob("*.json")):
        rec = json.load(open(f))
        scored = [p for p in rec["picks"] if p.get("combined_score") is not None]
        top3 = sorted(scored, key=lambda x: -x["combined_score"])[:3]
        syms = [p["symbol"] for p in top3]
        if not syms:
            continue
        d = datetime.fromisoformat(rec["date"]).date()
        start = datetime.combine(d - timedelta(days=2), time(0, 0), ET)
        end = datetime.combine(d + timedelta(days=10), time(23, 59), ET)
        try:
            req = StockBarsRequest(symbol_or_symbols=syms, timeframe=TimeFrame.Day,
                                   start=start.astimezone(UTC), end=end.astimezone(UTC),
                                   feed=DataFeed.IEX)
            df = dc.get_stock_bars(req).df
        except Exception as e:
            print(f"{rec['date']}: fetch failed ({str(e)[:50]})")
            continue
        for sym in syms:
            try:
                sb = df.xs(sym, level=0).sort_index()
                dates = [ts.tz_convert(ET).date() for ts in sb.index]
                if d not in dates:
                    continue
                i = dates.index(d)
                if i + HOLD >= len(sb):
                    continue   # not enough forward days yet
                on = intr = 0.0
                for k in range(1, HOLD + 1):
                    c_prev = float(sb["close"].iloc[i + k - 1])
                    o_cur = float(sb["open"].iloc[i + k])
                    c_cur = float(sb["close"].iloc[i + k])
                    if c_prev <= 0 or o_cur <= 0:
                        raise ValueError
                    on += math.log(o_cur / c_prev)       # overnight leg
                    intr += math.log(c_cur / o_cur)      # intraday leg
                rows.append((sym, rec["date"], (math.exp(on) - 1) * 100,
                             (math.exp(intr) - 1) * 100, (math.exp(on + intr) - 1) * 100))
            except Exception:
                continue

    if not rows:
        print("no decomposable holds yet."); return 0
    on = _agg([r[2] for r in rows])
    intr = _agg([r[3] for r in rows])
    tot = _agg([r[4] for r in rows])
    print(f"=== OVERNIGHT vs INTRADAY decomposition: {len(rows)} traded holds (close_T->close_T+{HOLD}) ===\n")
    print(f"{'leg':<12}{'n':>4}{'mean%':>8}{'med%':>8}{'SUM%':>9}{'best%':>8}{'worst%':>8}")
    print(f"{'overnight':<12}{on[0]:>4}{on[1]:>+8.2f}{on[2]:>+8.2f}{on[3]:>+9.1f}{on[4]:>+8.1f}{on[5]:>+8.1f}")
    print(f"{'intraday':<12}{intr[0]:>4}{intr[1]:>+8.2f}{intr[2]:>+8.2f}{intr[3]:>+9.1f}{intr[4]:>+8.1f}{intr[5]:>+8.1f}")
    print(f"{'TOTAL':<12}{tot[0]:>4}{tot[1]:>+8.2f}{tot[2]:>+8.2f}{tot[3]:>+9.1f}{tot[4]:>+8.1f}{tot[5]:>+8.1f}")

    # Where does the right tail live? Look at the top-5 total-return holds and split their legs.
    top = sorted(rows, key=lambda r: -r[4])[:5]
    print("\n  Top-5 holds by total return — which leg drove them:")
    for sym, dt, o, ix, t in top:
        drv = "overnight" if o > ix else "intraday"
        print(f"    {sym:<6} {dt}  total {t:+.1f}%  =  overnight {o:+.1f}%  +  intraday {ix:+.1f}%   [{drv}-driven]")
    print("\nRead: if the overnight SUM is negative AND the best/tail holds are intraday-driven, a")
    print("same-day exit is defensible. If the tail holds are overnight-driven, same-day exit amputates")
    print("the payoff (confirms the revert). Small n; IEX daily bars.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
