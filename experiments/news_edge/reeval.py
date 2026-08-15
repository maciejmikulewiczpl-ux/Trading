"""Intraday news re-evaluation — SHADOW layer (logs only, NEVER trades).

The morning news scan picks names at ~9:33 ET; the news-hold bot then holds them mechanically
(T+3 / 10% trail). This module lets the headless LLM re-read fresh news on the CURRENTLY-HELD
names midday and record what it WOULD do (hold / trim / exit) + why. We do NOT act on it — we
log the recommendation and later score it against the mechanical exit, to learn whether an
LLM catalyst-aware exit beats holding. Shadow-first because the -1 same-day-drop-then-bounce
pattern (see analysis 2026-08-14) means naive exits could sell into reversals.

Subcommands:
  positions            print current .env.news holdings as JSON (symbol, qty, avg_px, upl%) — the re-eval universe
  log <reeval.json>    record today's re-eval recommendations -> reeval/YYYY-MM-DD.json (immutable)
  score                across all logged re-evals: forward return (from px_at_reeval) of EXIT vs HOLD
                       calls. If exit-names fall and hold-names hold up, the LLM midday read adds value.

Recommendation JSON schema (a list):
  [{"symbol":"NVDA","action":"hold|trim|exit","catalyst_changed":true,
    "confidence":0.6,"reason":"...","sources":["..."]}, ...]

Run:
  .venv/Scripts/python.exe experiments/news_edge/reeval.py positions
  .venv/Scripts/python.exe experiments/news_edge/reeval.py log myreeval.json
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))   # so `score` can import backtest.run_orb.load_env
REEVAL_DIR = Path(__file__).resolve().parent / "reeval"
NEWS_ENV = ROOT / ".env.news"

VALID_ACTIONS = {"hold", "trim", "exit"}


def _load_news_env() -> None:
    """Load .env.news keys into os.environ (the account the news-hold bot trades)."""
    if not NEWS_ENV.exists():
        print(f"FATAL: {NEWS_ENV} not found.", file=sys.stderr)
        sys.exit(1)
    for line in NEWS_ENV.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip().strip('"').strip("'")


def cmd_positions() -> int:
    _load_news_env()
    from alpaca.trading.client import TradingClient
    tc = TradingClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], paper=True)
    out = []
    for p in tc.get_all_positions():
        try:
            upl = round(float(p.unrealized_plpc) * 100, 2)
        except Exception:
            upl = None
        out.append({"symbol": p.symbol, "qty": p.qty, "avg_px": p.avg_entry_price,
                    "mkt_px": getattr(p, "current_price", None), "upl_pct": upl})
    print(json.dumps(out, indent=2))
    return 0


def cmd_log(path: str) -> int:
    recs = json.load(open(path))
    assert isinstance(recs, list) and all("symbol" in r and "action" in r for r in recs), "bad reeval schema"
    for r in recs:
        assert r["action"] in VALID_ACTIONS, f"bad action {r['action']!r} (must be hold/trim/exit)"
    REEVAL_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(ET)
    today = now.date().isoformat()
    outp = REEVAL_DIR / f"{today}.json"
    if outp.exists():
        print(f"refusing to overwrite existing {outp.name} (immutable). One re-eval per day.")
        return 1
    rec = {
        "date": today,
        "logged_at": now.isoformat(timespec="seconds"),
        "recs": [{
            "symbol": r["symbol"].upper(),
            "action": r["action"],
            "catalyst_changed": bool(r.get("catalyst_changed", False)),
            "confidence": float(r.get("confidence", 0.5)),
            "reason": r.get("reason", ""),
            "sources": r.get("sources", []),
            # snapshot of the position at re-eval time (for later scoring vs mechanical exit)
            "px_at_reeval": r.get("px_at_reeval"),
            "upl_pct_at_reeval": r.get("upl_pct_at_reeval"),
        } for r in recs],
    }
    outp.write_text(json.dumps(rec, indent=2))
    print(f"logged {len(rec['recs'])} re-eval recs -> {outp.relative_to(ROOT)}")
    return 0


def cmd_score() -> int:
    """Measure whether the LLM's midday exit calls discriminate future returns, using only the
    logged px_at_reeval + subsequent daily closes (no entry-date reconstruction needed).
    For an EXIT rec, a NEGATIVE forward return = the call was right (price fell after). For HOLD,
    positive = right. The headline is mean(HOLD) - mean(EXIT): positive => the read adds value."""
    import statistics as st
    from backtest.run_orb import load_env
    load_env()
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    from datetime import timedelta

    files = sorted(REEVAL_DIR.glob("*.json")) if REEVAL_DIR.exists() else []
    recs = []
    for f in files:
        d = json.loads(f.read_text())
        for r in d.get("recs", []):
            r["_date"] = d["date"]
            recs.append(r)
    if not recs:
        print("no re-eval records yet (experiments/news_edge/reeval/ is empty). "
              "The shadow layer needs held positions + a few days of runs first.")
        return 0

    syms = sorted({r["symbol"] for r in recs})
    dc = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    first = min(r["_date"] for r in recs)
    start = datetime.fromisoformat(first).replace(tzinfo=ZoneInfo("UTC")) - timedelta(days=2)
    bars: dict = {}
    for i in range(0, len(syms), 50):
        chunk = syms[i:i + 50]
        try:
            df = dc.get_stock_bars(StockBarsRequest(symbol_or_symbols=chunk, timeframe=TimeFrame.Day,
                                                    start=start, feed=DataFeed.IEX)).df
            for s in chunk:
                try:
                    bars[s] = [(d0.date().isoformat(), float(row.close)) for d0, row in df.xs(s, level=0).iterrows()]
                except KeyError:
                    pass
        except Exception as e:
            print("bar fetch error:", e)

    def fwd(sym, date, px, n):
        b = bars.get(sym)
        if not b or px in (None, 0):
            return None
        try:
            px = float(px)
        except (TypeError, ValueError):
            return None
        dates = [x[0] for x in b]
        closes = [x[1] for x in b]
        # first close on-or-after the re-eval date; n=0 = same-day close, n=1 = next close, ...
        idx = next((i for i, dd in enumerate(dates) if dd >= date), None)
        if idx is None or idx + n >= len(closes):
            return None
        return (closes[idx + n] / px - 1) * 100

    def bucket(action, n, label):
        r = [fwd(x["symbol"], x["_date"], x.get("px_at_reeval"), n) for x in recs if x["action"] == action]
        r = [v for v in r if v is not None]
        if not r:
            print(f"  {label:20s} n=0"); return None
        m = st.mean(r)
        print(f"  {label:20s} n={len(r):3d}  mean_fwd={m:+.3f}%  median={st.median(r):+.3f}%")
        return m

    from collections import Counter
    print(f"re-eval records: {len(recs)} across {len(files)} days | actions: {dict(Counter(r['action'] for r in recs))}")
    for n, hz in [(0, "rest-of-day"), (1, "+1 close"), (2, "+2 close")]:
        print(f"\n=== forward return from px_at_reeval, horizon {hz} ===")
        mh = bucket("hold", n, "HOLD (kept)")
        me = bucket("exit", n, "EXIT (flagged out)")
        bucket("trim", n, "TRIM")
        if mh is not None and me is not None:
            print(f"  --> HOLD minus EXIT = {mh - me:+.3f}%   (positive = LLM exit calls added value)")
    return 0


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in {"positions", "log", "score"}:
        print(__doc__)
        return 2
    if sys.argv[1] == "positions":
        return cmd_positions()
    if sys.argv[1] == "score":
        return cmd_score()
    if len(sys.argv) < 3:
        print("usage: reeval.py log <reeval.json>", file=sys.stderr)
        return 2
    return cmd_log(sys.argv[2])


if __name__ == "__main__":
    raise SystemExit(main())
