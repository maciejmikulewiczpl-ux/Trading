"""News-edge forward test harness (experiment; logs only, never trades). See README.md.

Subcommands:
  log <picks.json>     record today's analyst picks -> picks/YYYY-MM-DD.json (immutable)
  outcomes YYYY-MM-DD  pull each picked name's 09:45->15:55 ET return, append to that file
  analyze              across all logged days: avg return by signal bucket, (+) vs (-) gap

Picks JSON schema (a list):
  [{"symbol":"NVDA","signal":1,"confidence":0.7,"reason":"...","sources":["..."]}, ...]
  signal: +1 frontrunner / 0 neutral / -1 avoid.

Run:
  .venv/Scripts/python.exe experiments/news_edge/newsedge.py log mypicks.json
  .venv/Scripts/python.exe experiments/news_edge/newsedge.py outcomes 2026-06-09
  .venv/Scripts/python.exe experiments/news_edge/newsedge.py analyze
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
ENTRY_T, EXIT_T = time(9, 45), time(15, 55)   # ORB entry window end -> EOD flatten


def _client():
    import os
    from alpaca.data.historical import StockHistoricalDataClient
    load_env()
    return StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])


def cmd_log(picks_path: str) -> int:
    picks = json.load(open(picks_path))
    assert isinstance(picks, list) and all("symbol" in p and "signal" in p for p in picks), "bad picks schema"
    PICKS_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now(ET).date().isoformat()
    out = PICKS_DIR / f"{today}.json"
    if out.exists():
        print(f"refusing to overwrite existing {out.name} (immutable record). Edit by hand if needed.")
        return 1
    rec = {"date": today, "logged_at": datetime.now(ET).isoformat(timespec="seconds"),
           "picks": [{"symbol": p["symbol"].upper(), "signal": int(p["signal"]),
                      "confidence": float(p.get("confidence", 0.5)),
                      "reason": p.get("reason", ""), "sources": p.get("sources", []),
                      # per-source signals for the head-to-head (e.g. {"stocktitan": 1}).
                      # "claude" = my own `signal` above; sources here are compared against it.
                      "source_signals": {k: int(v) for k, v in (p.get("source_signals") or {}).items()}}
                     for p in picks]}
    json.dump(rec, open(out, "w"), indent=2)
    print(f"logged {len(rec['picks'])} picks -> {out}")
    return 0


def cmd_outcomes(date_str: str) -> int:
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    f = PICKS_DIR / f"{date_str}.json"
    rec = json.load(open(f))
    syms = [p["symbol"] for p in rec["picks"]]
    d = datetime.fromisoformat(date_str).date()
    start = datetime.combine(d, time(9, 30), ET)
    end = datetime.combine(d, time(16, 0), ET)
    dc = _client()
    req = StockBarsRequest(symbol_or_symbols=syms, timeframe=TimeFrame.Minute,
                           start=start.astimezone(UTC), end=end.astimezone(UTC), feed=DataFeed.IEX)
    df = dc.get_stock_bars(req).df
    for p in rec["picks"]:
        try:
            sb = df.xs(p["symbol"], level=0)
            t = sb.index.tz_convert(ET).time
            entry = sb[t >= ENTRY_T]["open"].iloc[0]
            exit_ = sb[t <= EXIT_T]["close"].iloc[-1]
            p["ret_945_close"] = round((exit_ / entry - 1.0) * 100, 3)
        except Exception as e:
            p["ret_945_close"] = None
            p["outcome_error"] = str(e)[:80]
    rec["outcomes_at"] = datetime.now(ET).isoformat(timespec="seconds")
    json.dump(rec, open(f, "w"), indent=2)
    got = [p for p in rec["picks"] if p.get("ret_945_close") is not None]
    print(f"{date_str}: outcomes for {len(got)}/{len(rec['picks'])} names "
          f"(avg {sum(p['ret_945_close'] for p in got)/len(got):+.2f}% if any)" if got else f"{date_str}: no outcomes")
    return 0


def cmd_analyze() -> int:
    # Per-source (signal, ret) pairs. "claude" = my own pick `signal`; other keys come
    # from each pick's source_signals (e.g. "stocktitan"). Lets us rank sources head-to-head.
    by_source: dict[str, list] = {}
    n_days = 0
    for f in sorted(PICKS_DIR.glob("*.json")):
        n_days += 1
        rec = json.load(open(f))
        for p in rec["picks"]:
            ret = p.get("ret_945_close")
            if ret is None:
                continue
            by_source.setdefault("claude", []).append((int(p["signal"]), ret))
            for src, sig in (p.get("source_signals") or {}).items():
                by_source.setdefault(src, []).append((int(sig), ret))
    if not by_source:
        print("no scored picks yet. Log picks + run `outcomes` for a few days first.")
        return 0

    def stats(sel):
        if not sel:
            return "n=  0"
        r = [x[1] for x in sel]
        wins = sum(1 for x in r if x > 0)
        return f"n={len(r):>3}  avg {sum(r)/len(r):+.2f}%  win {wins/len(r)*100:.0f}%"

    print(f"=== news-edge forward test: {n_days} days, sources ranked by (+)-minus-(-) separation ===")
    ranked = []
    for src, pairs in by_source.items():
        pos = [x for x in pairs if x[0] > 0]
        neg = [x for x in pairs if x[0] < 0]
        neu = [x for x in pairs if x[0] == 0]
        sep = (sum(x[1] for x in pos) / len(pos) - sum(x[1] for x in neg) / len(neg)) if pos and neg else None
        ranked.append((sep if sep is not None else -1e9, src, pos, neg, neu, sep))
    ranked.sort(reverse=True)
    for _, src, pos, neg, neu, sep in ranked:
        sep_txt = f"{sep:+.2f}%/name" if sep is not None else "n/a (need both + and - picks)"
        print(f"\n  [{src:<10}] separation: {sep_txt}")
        print(f"    (+) {stats(pos)}")
        print(f"    ( ) {stats(neu)}")
        print(f"    (-) {stats(neg)}")
    print("\n  Separation = avg move of (+) picks minus avg move of (-). Positive AND growing with n = a")
    print("  real edge. Compare 'claude' (my read) vs each source — best separation wins. Needs dozens of days.")
    return 0


def main(argv) -> int:
    if len(argv) < 2:
        print(__doc__); return 1
    cmd = argv[1]
    if cmd == "log" and len(argv) == 3:
        return cmd_log(argv[2])
    if cmd == "outcomes" and len(argv) == 3:
        return cmd_outcomes(argv[2])
    if cmd == "analyze":
        return cmd_analyze()
    print(__doc__); return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
