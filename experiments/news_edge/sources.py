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
import time
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


# ---- StockTwits: free, no key, real-time. Trending tickers + retail crowd sentiment ----
ST_TREND = "https://api.stocktwits.com/api/2/trending/symbols.json"
ST_STREAM = "https://api.stocktwits.com/api/2/streams/symbol/{}.json"
_UA = {"User-Agent": "Mozilla/5.0 (news-edge research)"}


def _st_get(url: str):
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def st_trending(limit: int = 30) -> list:
    """Tickers retail is actively trading right now (broad candidate net). Free, real-time."""
    try:
        d = _st_get(ST_TREND)
    except Exception as e:
        return [f"_error: {e}"]
    return [s.get("symbol") for s in d.get("symbols", []) if s.get("symbol")][:limit]


def st_sentiment(tickers: list) -> dict:
    """Per-ticker retail crowd sentiment from the last ~30 messages: bull/bear ratio -> +1/0/-1.
    NOTE: crowd sentiment is noisy/contrarian — a complementary signal, judged by the head-to-head."""
    out = {}
    for t in [x.upper() for x in tickers]:
        try:
            d = _st_get(ST_STREAM.format(t))
            bull = bear = 0
            for m in d.get("messages", []):
                s = (((m.get("entities") or {}).get("sentiment")) or {}).get("basic")
                if s == "Bullish":
                    bull += 1
                elif s == "Bearish":
                    bear += 1
            tot = bull + bear
            if tot < 3:   # too few tagged msgs to mean anything
                out[t] = {"signal": None, "bull": bull, "bear": bear, "n": tot}
            else:
                ratio = bull / tot
                out[t] = {"signal": (1 if ratio > 0.6 else -1 if ratio < 0.4 else 0),
                          "bull_pct": round(ratio * 100), "bull": bull, "bear": bear, "n": tot}
        except Exception as e:
            out[t] = {"signal": None, "error": str(e)[:60]}
        time.sleep(0.3)   # polite to the public API
    return out


def main(argv) -> int:
    cmd = argv[1].lower() if len(argv) > 1 else ""
    if cmd == "av" and len(argv) >= 3:
        tickers = [t.strip().upper() for t in argv[2].split(",") if t.strip()]
        print(json.dumps(av_sentiment(tickers), indent=2))
        return 0
    if cmd == "st-trending":
        print(json.dumps(st_trending(int(argv[2]) if len(argv) > 2 else 30), indent=2))
        return 0
    if cmd == "st-sentiment" and len(argv) >= 3:
        tickers = [t.strip().upper() for t in argv[2].split(",") if t.strip()]
        print(json.dumps(st_sentiment(tickers), indent=2))
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
