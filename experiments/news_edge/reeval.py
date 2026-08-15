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


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in {"positions", "log"}:
        print(__doc__)
        return 2
    if sys.argv[1] == "positions":
        return cmd_positions()
    if len(sys.argv) < 3:
        print("usage: reeval.py log <reeval.json>", file=sys.stderr)
        return 2
    return cmd_log(sys.argv[2])


if __name__ == "__main__":
    raise SystemExit(main())
