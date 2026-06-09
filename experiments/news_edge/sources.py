"""News-edge source connectors: per-ticker sentiment from free APIs (for source_signals).

Alpha Vantage NEWS_SENTIMENT (free, 25 calls/day): per-ticker scored sentiment. ONE call
covers many tickers (tickers=A,B,C). We relevance-weight each ticker's article scores over
the last ~20h and map to +1/0/-1. (Finnhub's news-sentiment is premium/403 on free — only
headlines there, no score — so it's not a ranked source.)

CLI (used by the morning scan to fill source_signals["alphavantage"]):
    .venv/Scripts/python.exe experiments/news_edge/sources.py av NVDA,AMD,TSLA
-> JSON {ticker: {"signal": +1/0/-1, "score": float, "n": articles}}
"""
from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from backtest.run_orb import load_env  # noqa: E402

ET = ZoneInfo("America/New_York")
AV_URL = "https://www.alphavantage.co/query"
POS, NEG = 0.15, -0.15   # AV label bands: >0.15 (Somewhat-)Bullish, <-0.15 (Somewhat-)Bearish


def av_sentiment(tickers: list[str], hours_back: int = 20) -> dict:
    load_env()
    key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not key:
        return {"_error": "ALPHAVANTAGE_API_KEY missing from .env"}
    tf = (datetime.now(ET) - timedelta(hours=hours_back)).strftime("%Y%m%dT%H%M")
    q = urllib.parse.urlencode({"function": "NEWS_SENTIMENT", "tickers": ",".join(tickers),
                                "time_from": tf, "limit": "1000", "apikey": key})
    try:
        with urllib.request.urlopen(f"{AV_URL}?{q}", timeout=25) as r:
            data = json.load(r)
    except Exception as e:
        return {"_error": f"AV fetch failed: {e}"}
    feed = data.get("feed") or []
    agg = {t.upper(): [] for t in tickers}
    for art in feed:
        for ts in art.get("ticker_sentiment", []):
            sym = ts.get("ticker", "").upper()
            if sym in agg:
                try:
                    agg[sym].append((float(ts["ticker_sentiment_score"]),
                                     float(ts.get("relevance_score", 0.5))))
                except (KeyError, ValueError, TypeError):
                    pass
    out = {}
    for sym, lst in agg.items():
        if not lst:
            out[sym] = {"signal": None, "score": None, "n": 0}
            continue
        wsum = sum(r for _, r in lst)
        score = (sum(s * r for s, r in lst) / wsum) if wsum else (sum(s for s, _ in lst) / len(lst))
        out[sym] = {"signal": (1 if score > POS else -1 if score < NEG else 0),
                    "score": round(score, 3), "n": len(lst)}
    note = data.get("Note") or data.get("Information")
    if note and not feed:
        out["_note"] = str(note)[:160]   # rate-limit / key message surfaces here
    return out


def main(argv) -> int:
    if len(argv) < 3 or argv[1].lower() != "av":
        print(__doc__)
        return 1
    tickers = [t.strip().upper() for t in argv[2].split(",") if t.strip()]
    print(json.dumps(av_sentiment(tickers), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
