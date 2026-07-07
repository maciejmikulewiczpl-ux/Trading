"""INTRADAY exit lab for the Hype bot — MINUTE-bar resolution, so it can test exit rules
that the daily-bar exit_sim.py CANNOT see: the entry-day intraday round-trip the bot's
positions do ("up $200 midday, red by close"). Reconstructs the real minute path of every
traded name (from the trade ledger) over the full T+3 hold, then bakes off exit strategies
against the live 10% continuous trailing stop.

WHY THIS EXISTS (2026-07-07 finding — see memory lottery_experiment.md):
  - The intraday give-back is REAL: measured avg entry-day PEAK ~+5.2% vs CLOSE ~-0.8%.
  - But the OBVIOUS fixes all LOSE money because they cap the tail: at the +5% midday peak
    you can't tell the round-tripper from the future +$400 monster — they look identical,
    and a fixed-level sell decapitates the winners (which are ~all the P&L). REJECTED here
    (kept in the bake-off as a guardrail so we don't re-explore): fixed tighter trail,
    profit-targets, scale-outs, EOD-flat.
  - The ONE lead that beat the current trail: TIGHTEN-AFTER-PROVEN-GAIN — keep the wide 10%
    trail normally, but once a name has already printed a big move (+10-12%), clamp the trail
    to ~5-6%. It protects the peak WITHOUT capping the tail (it still trails upward).

STATUS: LEAD, not shipped. Re-run as the sample grows (target the ~late-July verdict). Do
NOT change the live bot on a small/thin-data read.

HONEST LIMITS: IEX minute bars see ~2-3% of volume on micro-caps -> intraday extremes and
absolute levels are UNRELIABLE (the current trail scores far lower here than on cleaner
daily bars because thin prints cause false stop-outs). RELATIVE ranking of rules on the
SAME paths is the signal; SUM% is the tail-aware read (per the profit-only rule). n is small.

Run:  .venv/Scripts/python.exe experiments/lottery/intraday_exit_lab.py
"""
from __future__ import annotations

import csv
import os
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
LEDGER = ROOT / "logs" / "lottery_trade_ledger.csv"
MAX_DAYS = 3                       # T+3 hold (trading days), matches the live bot
TRAIL_SLIP_BPS = 40                # trail/gap exits sell into weakness (aggressor)
TARGET_SLIP_BPS = 10              # profit-target is a resting limit (gentle)
TIME_SLIP_BPS = 25                # time-stop / EOD market at the close


def _load_env():
    """Parse .env.lottery (tolerates CRLF line endings from the Windows dev box)."""
    for line in (ROOT / ".env.lottery").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip().strip('"').strip("'")


def _trades() -> list[dict]:
    if not LEDGER.exists():
        return []
    out = []
    for r in csv.DictReader(open(LEDGER)):
        if r.get("status") != "closed":
            continue
        try:
            out.append({"sym": r["symbol"], "entry_date": r["entry_date"],
                        "entry": float(r["entry_avg"])})
        except (ValueError, KeyError):
            continue
    return out


def _minute_bars(sym, start_date):
    """RTH 1-min bars from the entry date through +6 calendar days (covers T+3 + weekend)."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import DataFeed
    dc = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    d0 = datetime.fromisoformat(start_date).replace(tzinfo=ET)
    try:
        df = dc.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=sym, timeframe=TimeFrame(1, TimeFrameUnit.Minute),
            start=d0.astimezone(UTC), end=(d0 + timedelta(days=6)).astimezone(UTC),
            feed=DataFeed.IEX)).df
    except Exception:
        return []
    try:
        sb = df.xs(sym, level=0).sort_index()
    except (KeyError, AttributeError):
        return []
    bars = []
    for t, r in sb.iterrows():
        et = t.tz_convert(ET)
        if 9 * 60 + 30 <= et.hour * 60 + et.minute < 16 * 60:      # RTH only
            bars.append((et, float(r.open), float(r.high), float(r.low), float(r.close)))
    return bars


def _sessions(bars):
    """Group minute bars into trading days (entry day + up to T+3)."""
    days, cur, curd = [], [], None
    for b in bars:
        d = b[0].date()
        if d != curd:
            if cur:
                days.append(cur)
            cur, curd = [], d
        cur.append(b)
    if cur:
        days.append(cur)
    return days[:MAX_DAYS + 1]


# ---- exit strategies: each returns (fill_price, tag) to exit this bar, or None to hold ----
def make_trail(pct, slip=TRAIL_SLIP_BPS):
    def f(di, bi, o, h, lo, c, hw, entry):
        stop = hw * (1 - pct / 100)
        if (di, bi) != (0, 0) and lo <= stop:
            return (min(o, stop) * (1 - slip / 1e4), "trail")       # gap-through -> fill at open if worse
        return None
    return f


def make_target(trail_pct, tgt_pct):
    tf = make_trail(trail_pct)
    def f(di, bi, o, h, lo, c, hw, entry):
        tgt = entry * (1 + tgt_pct / 100)
        if h >= tgt:
            return (tgt * (1 - TARGET_SLIP_BPS / 1e4), "target")
        return tf(di, bi, o, h, lo, c, hw, entry)
    return f


def make_tighten(base_pct, trigger_pct, tight_pct):
    """Wide trail normally; once the position prints >= trigger% gain, tighten the trail."""
    armed = {"v": False}
    def f(di, bi, o, h, lo, c, hw, entry):
        if h >= entry * (1 + trigger_pct / 100):
            armed["v"] = True
        pct = tight_pct if armed["v"] else base_pct
        stop = hw * (1 - pct / 100)
        if (di, bi) != (0, 0) and lo <= stop:
            return (min(o, stop) * (1 - TRAIL_SLIP_BPS / 1e4), "trail*" if armed["v"] else "trail")
        return None
    return f


def _walk(days, entry, exit_fn):
    hw = entry
    for di, day in enumerate(days):
        last_day = (di >= MAX_DAYS) or (di == len(days) - 1)
        for bi, (ts, o, h, lo, c) in enumerate(day):
            res = exit_fn(di, bi, o, h, lo, c, hw, entry)
            if res is not None:
                return res[0] / entry - 1.0
            hw = max(hw, h)
        if last_day:
            return (day[-1][4] * (1 - TIME_SLIP_BPS / 1e4)) / entry - 1.0
    return (days[-1][-1][4] * (1 - TIME_SLIP_BPS / 1e4)) / entry - 1.0


def _walk_scale(days, entry, targets, trail_pct):
    """Bank a fraction at each (gain%, frac) target (limit), trail the remainder."""
    remaining, realized, hw = 1.0, 0.0, entry
    hit = [False] * len(targets)
    for di, day in enumerate(days):
        last_day = (di >= MAX_DAYS) or (di == len(days) - 1)
        for bi, (ts, o, h, lo, c) in enumerate(day):
            for k, (gp, frac) in enumerate(targets):
                if not hit[k] and remaining > 0 and h >= entry * (1 + gp / 100):
                    f = min(frac, remaining)
                    realized += f * ((entry * (1 + gp / 100) * (1 - TARGET_SLIP_BPS / 1e4)) / entry - 1)
                    remaining -= f; hit[k] = True
            stop = hw * (1 - trail_pct / 100)
            if (di, bi) != (0, 0) and lo <= stop and remaining > 0:
                realized += remaining * ((min(o, stop) * (1 - TRAIL_SLIP_BPS / 1e4)) / entry - 1)
                return realized
            hw = max(hw, h)
        if last_day and remaining > 0:
            realized += remaining * ((day[-1][4] * (1 - TIME_SLIP_BPS / 1e4)) / entry - 1)
            return realized
    return realized


def _row(label, rs):
    if not rs:
        return f"  {label:<28} (no data)"
    return (f"  {label:<28}{len(rs):>4}{statistics.mean(rs) * 100:>8.2f}"
            f"{statistics.median(rs) * 100:>8.2f}{sum(1 for x in rs if x > 0) / len(rs) * 100:>7.0f}"
            f"{sum(rs) * 100:>8.1f}")


def main():
    _load_env()
    trades = _trades()
    if not trades:
        print("no closed trades in ledger — run trade_ledger.py first.")
        return 1
    print(f"loading minute bars for {len(trades)} closed trades...")
    paths = {}
    for t in trades:
        b = _minute_bars(t["sym"], t["entry_date"])
        if b:
            paths[t["sym"]] = _sessions(b)
    trades = [t for t in trades if t["sym"] in paths]
    print(f"got minute data for {len(trades)} names\n")

    # 1. Quantify the observation (robust — uses highs/closes, not stop triggers)
    peaks, closes = [], []
    for t in trades:
        d0 = paths[t["sym"]][0]
        peaks.append(max(h for _, o, h, l, c in d0) / t["entry"] * 100 - 100)
        closes.append(d0[-1][4] / t["entry"] * 100 - 100)
    print("=" * 68)
    print("ENTRY-DAY INTRADAY GIVE-BACK (the observation, in real minutes)")
    print("=" * 68)
    print(f"  n={len(peaks)}  avg intraday PEAK +{statistics.mean(peaks):.2f}%  |  "
          f"avg CLOSE {statistics.mean(closes):+.2f}%  |  give-back {statistics.mean(peaks) - statistics.mean(closes):.2f} pts")

    # 2. Bake-off (SUM% = tail-aware). LEAD first, then the REJECTED guardrail set.
    print("\n" + "=" * 68)
    print(f"EXIT BAKE-OFF  (minute paths, T+{MAX_DAYS}, slippage-aware, gap-through)")
    print("=" * 68)
    print(f"  {'strategy':<28}{'n':>4}{'avg%':>8}{'med%':>8}{'win%':>7}{'SUM%':>8}")
    print("  -- reference --")
    print(_row("10% trail (LIVE)", [_walk(paths[t["sym"]], t["entry"], make_trail(10)) for t in trades]))
    print("  -- LEAD: tighten after a proven gain --")
    for trig, tight in ((8, 4), (10, 5), (12, 6)):
        print(_row(f"tighten @+{trig}% -> {tight}%",
                   [_walk(paths[t["sym"]], t["entry"], make_tighten(10, trig, tight)) for t in trades]))
    print("  -- REJECTED (cap the tail; kept as guardrail -- do not re-explore) --")
    for w in (6, 8):
        print(_row(f"fixed {w}% trail",
                   [_walk(paths[t["sym"]], t["entry"], make_trail(w)) for t in trades]))
    for tg in (10, 15, 20):
        print(_row(f"profit-target +{tg}%",
                   [_walk(paths[t["sym"]], t["entry"], make_target(10, tg)) for t in trades]))
    for tg in (10, 15):
        print(_row(f"scale-out 1/2 @+{tg}%",
                   [_walk_scale(paths[t["sym"]], t["entry"], [(tg, 0.5)], 10) for t in trades]))

    print("\n[caveats] IEX minute bars thin on micro-caps (absolute levels understated by "
          "false stop-outs); n small. Relative ranking on identical paths is the signal; SUM% "
          "is the tail-aware read. LEAD not shipped -- re-run as the sample grows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
