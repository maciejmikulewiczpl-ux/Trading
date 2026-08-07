"""Does the news CATALYST selection work at a MULTI-DAY horizon? (validation before re-structuring
the bot from same-day ORB -> multi-day catalyst hold.)

The news bot failed SAME-DAY (ret_945: claude(+) flat, loses to gappers). Hypothesis (#2): a real
catalyst -- FDA data, M&A, a contract -- drives a MULTI-DAY run, not just a same-day pop, so the same
catalyst signal might have a T+3 edge it never had at 9:45->close. The picks only logged ret_945, so
we RECONSTRUCT ret_1d/ret_3d from daily bars (buy pick-date OPEN, hold N trading days) and test
signal=1 (the tradeable long-only catalyst set) vs the >3%-gapper CONTROL, OOS.
Run: .venv/Scripts/python.exe experiments/news_edge/multiday_test.py"""
from __future__ import annotations
import glob, json, os, statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PICKS = ROOT / "experiments" / "news_edge" / "picks"


def load():
    rows = []
    for f in sorted(glob.glob(str(PICKS / "2026-*.json"))):
        rec = json.load(open(f)); ps = rec if isinstance(rec, list) else rec.get("picks", rec)
        for p in ps:
            if isinstance(p, dict) and p.get("symbol"):
                rows.append({"date": Path(f).stem, "sym": p["symbol"], "sig": p.get("signal"),
                             "ctrl": bool(p.get("control")), "r945": p.get("ret_945_close")})
    return rows


def add_multiday(rows):
    for env in (".env.lottery", ".env"):
        p = ROOT / env
        if p.exists():
            for l in p.read_text().splitlines():
                l = l.strip()
                if l and not l.startswith("#") and "=" in l:
                    k, v = l.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            break
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    from datetime import datetime, timezone
    dc = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    syms = sorted({r["sym"] for r in rows})
    bars: dict = {}
    for i in range(0, len(syms), 100):
        grp = syms[i:i + 100]
        try:
            df = dc.get_stock_bars(StockBarsRequest(symbol_or_symbols=grp, timeframe=TimeFrame.Day,
                 start=datetime(2026, 6, 6, tzinfo=timezone.utc), end=datetime(2026, 8, 12, tzinfo=timezone.utc),
                 feed=DataFeed.IEX)).df
            for s in grp:
                try:
                    seq = [(str(t.date()), float(r.open), float(r.close)) for t, r in df.xs(s, level=0).iterrows()]
                    bars[s] = seq
                except KeyError:
                    pass
        except Exception:
            pass
    for r in rows:
        seq = bars.get(r["sym"])
        if not seq:
            continue
        idx = next((i for i, (d, _o, _c) in enumerate(seq) if d >= r["date"]), None)
        if idx is None:
            continue
        o = seq[idx][1]
        if o and o > 0:
            if idx + 1 < len(seq):
                r["r1d"] = seq[idx + 1][2] / o - 1
            if idx + 3 < len(seq):
                r["r3d"] = seq[idx + 3][2] / o - 1


def stat(rs, key):
    v = [r[key] * 100 for r in rs if r.get(key) is not None]
    if not v:
        return None
    return len(v), statistics.mean(v), statistics.median(v), sum(1 for x in v if x > 0) / len(v) * 100, \
        sum(1 for x in v if x >= 20) / len(v) * 100, max(v)


def show(label, rs):
    for key in ("r945", "r1d", "r3d"):
        s = stat(rs, key)
        if s:
            print(f"    {key:>5}: n={s[0]:>3} mean {s[1]:+.2f}% med {s[2]:+.2f}% win {s[3]:.0f}% "
                  f"win>=20% {s[4]:.0f}% best +{s[5]:.0f}%")


def main():
    rows = load()
    print(f"loaded {len(rows)} news picks; reconstructing ret_1d/ret_3d from bars...")
    add_multiday(rows)
    sig1 = [r for r in rows if r["sig"] in (1, "1") and not r["ctrl"]]
    ctrl = [r for r in rows if r["ctrl"]]
    print(f"signal=1 (tradeable): {len(sig1)} | control gappers: {len(ctrl)}\n")
    print("=== SIGNAL=1 catalyst picks (the bot would trade these) ===")
    show("signal=1", sig1)
    print("\n=== CONTROL (>3% gappers) -- the bar to beat ===")
    show("control", ctrl)

    print("\n=== SIGNAL BUCKETS at ret_3d (does the analyst's direction call order the multi-day move?) ===")
    for sg, lbl in [((1, "1"), "(+) signal=1"), ((0, "0"), "( ) signal=0"), ((-1, "-1"), "(-) signal=-1")]:
        s = stat([r for r in rows if r["sig"] in sg and not r["ctrl"]], "r3d")
        if s:
            print(f"  {lbl:<14} n={s[0]:>3} ret_3d mean {s[1]:+.2f}% med {s[2]:+.2f}% win {s[3]:.0f}%")

    print("\n=== OOS split: signal=1 ret_3d edge vs control, early vs late ===")
    dates = sorted({r["date"] for r in rows}); mid = dates[len(dates) // 2]
    for lbl, cut in [("EARLY", lambda d: d < mid), ("LATE", lambda d: d >= mid)]:
        s = stat([r for r in sig1 if cut(r["date"])], "r3d")
        c = stat([r for r in ctrl if cut(r["date"])], "r3d")
        if s and c:
            print(f"  {lbl}: signal=1 {s[1]:+.2f}% (n={s[0]}) vs control {c[1]:+.2f}% -> edge {s[1]-c[1]:+.2f}pp")

    print("\n[read] catalyst MULTI-DAY edge is real only if signal=1 ret_3d beats control AND survives the "
          "OOS split. If yes -> re-structure the bot to a T+3 catalyst hold. If no -> catalysts don't "
          "predict multi-day either, and news stays retired.")


if __name__ == "__main__":
    main()
