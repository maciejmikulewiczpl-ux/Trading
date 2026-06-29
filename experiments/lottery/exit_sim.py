"""Exit-rule simulator for the Hype bot — replays the bot's ACTUAL traded names under
different trailing-stop widths and time-stop horizons, to answer "would T+4 / a wider
trail have done better?" Path-dependent, so it reconstructs the daily price path from
Alpaca (entry dates come from the trade ledger); whatif.py can't do this (it only has the
logged EOD/1d/3d point returns).

For each traded name it walks daily bars from entry and applies a trailing stop + a
time-stop:
  - high-water ratchets up with each day's high; stop = high_water * (1 - trail%)
  - if a day's LOW pierces the stop -> exit at the stop (trailing-stop hit)
  - else at max_days -> exit at that day's close (time-stop)
Then sweeps trail% x max_days and reports avg realized %, win rate, total.

HONEST LIMITS: DAILY resolution (intraday trigger is approximated by the day's low vs the
stop — fine for comparing horizons with a consistent convention, not for cent-precise
fills); IEX bars are thin on micro-caps; SMALL SAMPLE. Compares are apples-to-apples (same
names, same convention) so the RELATIVE ranking is the signal, not the absolute level.

Run:  .venv/Scripts/python.exe experiments/lottery/exit_sim.py
"""
from __future__ import annotations

import csv
import os
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
LEDGER = ROOT / "logs" / "lottery_trade_ledger.csv"
CUR_TRAIL, CUR_DAYS = 10.0, 3        # the bot's current config (reference)


def _load_env():
    f = ROOT / ".env.lottery"
    for line in f.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip().strip('"').strip("'")


def _trades() -> list[dict]:
    """Traded names from the ledger: symbol, entry_date, entry_avg."""
    if not LEDGER.exists():
        return []
    out = []
    for r in csv.DictReader(open(LEDGER)):
        try:
            out.append({"symbol": r["symbol"], "entry_date": r["entry_date"],
                        "entry": float(r["entry_avg"])})
        except (ValueError, KeyError):
            continue
    return out


def _daily_bars(symbols):
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    dc = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    out = {}
    syms = sorted(set(symbols))
    for i in range(0, len(syms), 50):
        grp = syms[i:i + 50]
        try:
            df = dc.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=grp, timeframe=TimeFrame.Day,
                start=datetime(2026, 6, 10, tzinfo=UTC), end=datetime.now(UTC),
                feed=DataFeed.IEX)).df
        except Exception:
            continue
        for s in grp:
            try:
                sb = df.xs(s, level=0).sort_index()
                out[s] = [(t.tz_convert(ET).date(), float(r.open), float(r.high),
                           float(r.low), float(r.close)) for t, r in sb.iterrows()]
            except KeyError:
                pass
    return out


def sim_one(bars_from_entry, entry, trail_pct, max_days):
    """Return (realized_pct, exit_day_index, reason) for one trade under (trail%, max_days)."""
    hw = entry
    path = bars_from_entry[:max_days + 1]            # day0 (entry) .. max_days
    for i, (_d, _o, h, lo, c) in enumerate(path):
        stop = hw * (1 - trail_pct / 100)
        if i > 0 and lo <= stop:                     # intraday low pierced the trailing stop
            return stop / entry - 1.0, i, "trail"
        hw = max(hw, h)                              # ratchet the high-water up
        if i >= max_days:                            # reached the time-stop
            return c / entry - 1.0, i, "time"
    last = path[-1]
    return last[4] / entry - 1.0, len(path) - 1, "end"


def run_variant(trades, bars, trail_pct, max_days):
    rs, reasons = [], {"trail": 0, "time": 0, "end": 0}
    for t in trades:
        b = bars.get(t["symbol"])
        if not b:
            continue
        frm = [x for x in b if x[0] >= datetime.fromisoformat(t["entry_date"]).date()]
        if len(frm) < 2:
            continue
        r, _i, why = sim_one(frm, t["entry"], trail_pct, max_days)
        rs.append(r * 100)
        reasons[why] += 1
    if not rs:
        return None
    import statistics
    return {"n": len(rs), "avg_%": round(statistics.mean(rs), 2),
            "median_%": round(statistics.median(rs), 2),
            "win_%": round(sum(1 for x in rs if x > 0) / len(rs) * 100),
            "total_%": round(sum(rs), 1),
            "exits": f"{reasons['trail']}tr/{reasons['time']}ti"}


def _print(title, rows):
    print(f"\n=== {title} ===")
    cols = ["n", "avg_%", "median_%", "win_%", "total_%", "exits"]
    print("  " + f"{'variant':>14}" + "".join(f"{c:>10}" for c in cols))
    for name, st in rows:
        if st is None:
            print(f"  {name:>14}   (no data)"); continue
        print("  " + f"{name:>14}" + "".join(f"{str(st.get(c, '')):>10}" for c in cols))


def main():
    _load_env()
    trades = _trades()
    if not trades:
        print("no trades in ledger yet — run trade_ledger.py first."); return 1
    bars = _daily_bars([t["symbol"] for t in trades])
    print("=" * 72)
    print("HYPE EXIT SIMULATOR | DAILY resolution | same traded names | SMALL SAMPLE")
    print(f"{len(trades)} traded names | current config = {CUR_TRAIL:.0f}% trail, T+{CUR_DAYS}")
    print("relative ranking is the signal, not absolute levels")
    print("=" * 72)

    # TIME-STOP horizon sweep at the current 10% trail (answers: T+2/3/4/5?)
    _print(f"TIME-STOP horizon  (trail {CUR_TRAIL:.0f}%)",
           [(f"T+{d}{'  *current' if d == CUR_DAYS else ''}", run_variant(trades, bars, CUR_TRAIL, d))
            for d in (2, 3, 4, 5)])

    # TRAIL-WIDTH sweep at the current T+3 (answers: 10% too tight?)
    _print(f"TRAIL WIDTH  (T+{CUR_DAYS})",
           [(f"{tp:.0f}%{'  *current' if tp == CUR_TRAIL else ''}", run_variant(trades, bars, tp, CUR_DAYS))
            for tp in (8, 10, 12, 15, 20)])

    # joint: a couple of promising-looking combos vs current
    _print("JOINT  (trail% x T+days)",
           [("10% / T+3 *cur", run_variant(trades, bars, 10, 3)),
            ("12% / T+4", run_variant(trades, bars, 12, 4)),
            ("15% / T+5", run_variant(trades, bars, 15, 5))])

    print("\n[reminder] daily-resolution approximation, thin IEX bars on micro-caps, tiny "
          "sample. Re-run at the ~30-day base before trusting any ranking.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
