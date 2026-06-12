"""Lottery outcomes scorer. Mirrors experiments/news_edge/newsedge.py cmd_outcomes
(NEVER modifies it) for the lottery picks schema.

score_day(date):   fills ret_945_close (09:45 ET open -> 15:55 ET close, IEX minute bars)
                   for every picked name. Thin/illiquid names -> None (tolerated; the
                   1d/3d daily-bar horizons are the robust fallback).
backfill(days_back): on later runs, fills ret_1d (close T -> close T+1) and ret_3d
                   (close T -> close T+3) from DAILY bars for prior picks files.
                   Idempotent -- only fills nulls.

1d/3d are the HEADLINE metrics for thin names (W2/W3 winners).

Run:
    .venv/Scripts/python.exe experiments/lottery/outcomes.py score 2026-06-12
    .venv/Scripts/python.exe experiments/lottery/outcomes.py backfill 7
"""
from __future__ import annotations

import json
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
ENTRY_T, EXIT_T = time(9, 45), time(15, 55)


def _client():
    import os
    from alpaca.data.historical import StockHistoricalDataClient
    load_env()
    return StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])


def score_day(date_str: str) -> int:
    """Fill ret_945_close for date_str's picks (09:45 open -> 15:55 close, IEX minutes)."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    f = PICKS_DIR / f"{date_str}.json"
    if not f.exists():
        print(f"no picks file for {date_str}")
        return 1
    rec = json.load(open(f))
    syms = [p["symbol"] for p in rec["picks"]]
    if not syms:
        print(f"{date_str}: no picks")
        return 0
    d = datetime.fromisoformat(date_str).date()
    start = datetime.combine(d, time(9, 30), ET)
    end = datetime.combine(d, time(16, 0), ET)
    dc = _client()
    req = StockBarsRequest(symbol_or_symbols=syms, timeframe=TimeFrame.Minute,
                           start=start.astimezone(UTC), end=end.astimezone(UTC),
                           feed=DataFeed.IEX)
    df = dc.get_stock_bars(req).df
    for p in rec["picks"]:
        try:
            sb = df.xs(p["symbol"], level=0)
            t = sb.index.tz_convert(ET).time
            entry = sb[t >= ENTRY_T]["open"].iloc[0]
            exit_ = sb[t <= EXIT_T]["close"].iloc[-1]
            p["ret_945_close"] = round((exit_ / entry - 1.0) * 100, 3)
        except Exception:
            p["ret_945_close"] = None
    rec["scored_at"] = datetime.now(ET).isoformat(timespec="seconds")
    json.dump(rec, open(f, "w"), indent=2)
    got = [p for p in rec["picks"] if p.get("ret_945_close") is not None]
    print(f"{date_str}: ret_945_close for {len(got)}/{len(rec['picks'])} names"
          + (f" (avg {sum(p['ret_945_close'] for p in got)/len(got):+.2f}%)" if got else ""))
    return 0


def backfill(days_back: int = 7) -> int:
    """Fill ret_1d (close T -> T+1) and ret_3d (close T -> T+3) from DAILY bars for the
    last `days_back` picks files. Idempotent -- only touches nulls. Needs the entry day's
    close + the +1/+3 trading-day closes to exist."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    files = sorted(PICKS_DIR.glob("*.json"))[-days_back:]
    if not files:
        print("no picks files to backfill")
        return 0
    dc = _client()
    for f in files:
        rec = json.load(open(f))
        d = datetime.fromisoformat(rec["date"]).date()
        # need closes up to ~T+6 calendar days to cover 3 trading days + weekends
        if (datetime.now(ET).date() - d).days < 1:
            continue   # too soon; T+1 close not in yet
        syms = [p["symbol"] for p in rec["picks"]]
        if not syms:
            continue
        start = datetime.combine(d - timedelta(days=2), time(0, 0), ET)
        end = datetime.combine(d + timedelta(days=10), time(23, 59), ET)
        try:
            req = StockBarsRequest(symbol_or_symbols=syms, timeframe=TimeFrame.Day,
                                   start=start.astimezone(UTC), end=end.astimezone(UTC),
                                   feed=DataFeed.IEX)
            df = dc.get_stock_bars(req).df
        except Exception as e:
            print(f"{rec['date']}: daily fetch failed ({str(e)[:50]})")
            continue
        filled1 = filled3 = 0
        for p in rec["picks"]:
            try:
                sb = df.xs(p["symbol"], level=0).sort_index()
                dates = [ts.tz_convert(ET).date() for ts in sb.index]
                if d not in dates:
                    continue
                i = dates.index(d)
                c0 = float(sb["close"].iloc[i])
                if p.get("ret_1d") is None and i + 1 < len(sb):
                    c1 = float(sb["close"].iloc[i + 1])
                    p["ret_1d"] = round((c1 / c0 - 1.0) * 100, 3) if c0 else None
                    if p["ret_1d"] is not None:
                        filled1 += 1
                if p.get("ret_3d") is None and i + 3 < len(sb):
                    c3 = float(sb["close"].iloc[i + 3])
                    p["ret_3d"] = round((c3 / c0 - 1.0) * 100, 3) if c0 else None
                    if p["ret_3d"] is not None:
                        filled3 += 1
            except Exception:
                continue
        rec["backfilled_at"] = datetime.now(ET).isoformat(timespec="seconds")
        json.dump(rec, open(f, "w"), indent=2)
        print(f"{rec['date']}: filled ret_1d x{filled1}, ret_3d x{filled3}")
    return 0


def main(argv) -> int:
    if len(argv) >= 3 and argv[1] == "score":
        return score_day(argv[2])
    if argv[1:2] == ["backfill"]:
        return backfill(int(argv[2]) if len(argv) > 2 else 7)
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
