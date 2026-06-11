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


# ---- Reddit (via apewisdom.io): WSB mention counts — a CROWDING gauge, no direction ----
# reddit.com JSON + tradestie are Cloudflare-blocked for scripts (verified 2026-06-11);
# apewisdom's free API serves clean {ticker, mentions, mentions_24h_ago, rank, upvotes}.
# The informative bit is the mention SURGE vs 24h ago (something happened overnight),
# not the absolute count (SPY/NVDA are always top). No sentiment direction is available,
# so this feeds the candidate net + a per-pick reddit_rank context field — never a +/- signal.
APEWISDOM = "https://apewisdom.io/api/v1.0/filter/wallstreetbets/page/1"


def reddit_trending(limit: int = 25) -> list:
    """Top WSB tickers by mention surge (mentions / mentions_24h_ago, min 10 mentions),
    then by mentions. [{ticker, rank, mentions, mentions_24h_ago, surge, upvotes}]."""
    try:
        req = urllib.request.Request(APEWISDOM, headers=_UA)
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.load(r)
    except Exception as e:
        return [f"_error: {e}"]
    out = []
    for it in data.get("results", []):
        try:
            m, m24 = int(it["mentions"]), int(it.get("mentions_24h_ago") or 0)
            out.append({"ticker": it["ticker"], "rank": int(it["rank"]), "mentions": m,
                        "mentions_24h_ago": m24,
                        "surge": round(m / m24, 1) if m24 > 0 else None,
                        "upvotes": int(it.get("upvotes") or 0)})
        except (KeyError, ValueError, TypeError):
            continue
    out.sort(key=lambda x: (-(x["surge"] or 0) if x["mentions"] >= 10 else 0, -x["mentions"]))
    return out[:limit]


# ---- SEC EDGAR: free, official, real-time primary-source catalysts (no key) ----
# Recent material filings mapped to tickers — 8-Ks (material events), 424B5/S-1
# (offerings/dilution), Form 4 (insider). EDGAR requires a descriptive User-Agent
# with contact info per its fair-access policy. CIK->ticker via company_tickers.json.
EDGAR_CURRENT = ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type={typ}"
                 "&company=&dateb=&owner=include&count=100&output=atom")
EDGAR_TICKERS = "https://www.sec.gov/files/company_tickers.json"
_SEC_UA = {"User-Agent": "news-edge research maciej.mikulewicz@gmail.com"}
_CIK2TICK: dict | None = None


def _cik_to_ticker() -> dict:
    global _CIK2TICK
    if _CIK2TICK is not None:
        return _CIK2TICK
    try:
        req = urllib.request.Request(EDGAR_TICKERS, headers=_SEC_UA)
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.load(r)
        _CIK2TICK = {int(v["cik_str"]): v["ticker"] for v in data.values()}
    except Exception:
        _CIK2TICK = {}
    return _CIK2TICK


def edgar_filings(hours_back: int = 16, forms: tuple = ("8-K", "424B5")) -> dict:
    """Recent EDGAR filings of `forms` in the last `hours_back`, mapped to tickers.
    Returns {ticker: [{form, time, title}]}. Catalysts BEFORE the aggregators rewrite them:
    8-K = material event, 424B5/S-1 = offering/dilution. (Form 4 insider filings are noisy
    for a morning scan; pass forms=("4",) explicitly if you want them.) Filings with no
    ticker mapping (private/foreign) are dropped. The reported `form` is parsed from the
    filing title (the requested type can prefix-match, e.g. type=4 also returns 425s)."""
    import re as _re
    import re
    from email.utils import parsedate_to_datetime
    c2t = _cik_to_ticker()
    cutoff = datetime.now(ET) - timedelta(hours=hours_back)
    out: dict[str, list] = {}
    for typ in forms:
        try:
            req = urllib.request.Request(EDGAR_CURRENT.format(typ=urllib.parse.quote(typ)),
                                         headers=_SEC_UA)
            with urllib.request.urlopen(req, timeout=25) as r:
                xml = r.read().decode("utf-8", "replace")
        except Exception:
            continue
        # crude Atom parse (no lxml dependency): split on <entry>
        for ent in re.findall(r"<entry>(.*?)</entry>", xml, re.S):
            title = (re.search(r"<title>(.*?)</title>", ent, re.S) or [None, ""])[1].strip()
            upd = (re.search(r"<updated>(.*?)</updated>", ent, re.S) or [None, ""])[1].strip()
            cikm = re.search(r"\((\d{4,10})\)", title) or re.search(r"CIK=(\d+)", ent)
            if not cikm:
                continue
            tick = c2t.get(int(cikm.group(1)))
            if not tick:
                continue
            try:
                when = datetime.fromisoformat(upd.replace("Z", "+00:00")).astimezone(ET)
            except Exception:
                try:
                    when = parsedate_to_datetime(upd).astimezone(ET)
                except Exception:
                    continue
            if when < cutoff:
                continue
            # actual form is the title prefix ("8-K - COMPANY ..."), not the requested type
            fm = _re.match(r"\s*([\w/.-]+)\s*-", title)
            actual_form = fm.group(1) if fm else typ
            out.setdefault(tick, []).append(
                {"form": actual_form, "time": when.strftime("%Y-%m-%d %H:%M"),
                 "title": re.sub(r"\s+", " ", title)[:120]})
    return out


# ---- Alpaca premarket: relative volume + gap (filter + mechanical control basket) ----
def _alpaca_dc():
    from alpaca.data.historical import StockHistoricalDataClient
    load_env()
    return StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])


def pm_rvol(tickers: list[str], lookback_days: int = 20) -> dict:
    """Premarket (04:00-09:30 ET) dollar-volume today vs the trailing `lookback_days`
    average premarket $-vol, per ticker. >1 = abnormal participation (the filter on a story).
    {ticker: {"pm_rvol": float, "pm_dollar_vol": float, "n_days": int}}."""
    from datetime import time as dtime
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    dc = _alpaca_dc()
    end = datetime.now(ET)
    start = end - timedelta(days=lookback_days + 5)
    out = {}
    PM_S, PM_E = dtime(4, 0), dtime(9, 30)
    try:
        req = StockBarsRequest(symbol_or_symbols=[t.upper() for t in tickers],
                               timeframe=TimeFrame.Minute, start=start, end=end, feed=DataFeed.IEX)
        df = dc.get_stock_bars(req).df
    except Exception as e:
        return {"_error": f"alpaca fetch failed: {e}"}
    today = end.date()
    for t in [x.upper() for x in tickers]:
        try:
            sb = df.xs(t, level=0).copy()
            idx = sb.index.tz_convert(ET)
            sb["dv"] = sb["close"] * sb["volume"]
            tt = idx.time
            pm = sb[(tt >= PM_S) & (tt < PM_E)]
            by_day = pm.groupby(idx[(tt >= PM_S) & (tt < PM_E)].date)["dv"].sum()
            today_pm = float(by_day.get(today, 0.0))
            hist = by_day[[d for d in by_day.index if d < today]]
            avg = float(hist.tail(lookback_days).mean()) if len(hist) else 0.0
            out[t] = {"pm_rvol": round(today_pm / avg, 2) if avg > 0 else None,
                      "pm_dollar_vol": round(today_pm), "n_days": int(len(hist))}
        except Exception:
            out[t] = {"pm_rvol": None, "pm_dollar_vol": 0, "n_days": 0}
    return out


def pm_gappers(min_gap_pct: float = 3.0, limit: int = 40) -> list:
    """Mechanical control basket: liquid names gapping >= min_gap_pct premarket (vs prior
    close), no judgment. Drawn from StockTwits-trending + most-active as the candidate net,
    so it overlaps the analyst's universe. Returns [{symbol, gap_pct, last}] for scoring as
    a signal=0 control — the analyst's (+) picks must beat THIS to earn their keep."""
    from alpaca.data.requests import StockSnapshotRequest
    from alpaca.data.enums import DataFeed
    cand = [s for s in st_trending(40) if isinstance(s, str) and s.isalpha()]
    if not cand:
        return []
    dc = _alpaca_dc()
    try:
        from alpaca.data.requests import StockSnapshotRequest as SSR
        snaps = dc.get_stock_snapshot(SSR(symbol_or_symbols=cand, feed=DataFeed.IEX))
    except Exception as e:
        return [{"_error": f"snapshot failed: {e}"}]
    out = []
    for sym, sn in (snaps or {}).items():
        try:
            prev_c = float(sn.previous_daily_bar.close)
            last = float((sn.minute_bar or sn.daily_bar).close)
            gap = (last / prev_c - 1.0) * 100
            if abs(gap) >= min_gap_pct:
                out.append({"symbol": sym, "gap_pct": round(gap, 2), "last": round(last, 2)})
        except Exception:
            continue
    out.sort(key=lambda x: -abs(x["gap_pct"]))
    return out[:limit]


def main(argv) -> int:
    cmd = argv[1].lower() if len(argv) > 1 else ""
    if cmd == "av" and len(argv) >= 3:
        tickers = [t.strip().upper() for t in argv[2].split(",") if t.strip()]
        print(json.dumps(av_sentiment(tickers), indent=2))
        return 0
    if cmd == "st-trending":
        print(json.dumps(st_trending(int(argv[2]) if len(argv) > 2 else 30), indent=2))
        return 0
    if cmd == "reddit":
        print(json.dumps(reddit_trending(int(argv[2]) if len(argv) > 2 else 25), indent=2))
        return 0
    if cmd == "st-sentiment" and len(argv) >= 3:
        tickers = [t.strip().upper() for t in argv[2].split(",") if t.strip()]
        print(json.dumps(st_sentiment(tickers), indent=2))
        return 0
    if cmd == "edgar":
        print(json.dumps(edgar_filings(int(argv[2]) if len(argv) > 2 else 16), indent=2))
        return 0
    if cmd == "pm-rvol" and len(argv) >= 3:
        tickers = [t.strip().upper() for t in argv[2].split(",") if t.strip()]
        print(json.dumps(pm_rvol(tickers), indent=2))
        return 0
    if cmd == "pm-gappers":
        print(json.dumps(pm_gappers(float(argv[2]) if len(argv) > 2 else 3.0), indent=2))
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
