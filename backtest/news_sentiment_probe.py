"""Backtest probe: does mechanical headline SENTIMENT predict next-session returns?

Validates whether a sentiment signal (VADER now; swap in FinBERT later — the scorer is the
only swappable part) is worth adopting for news-edge, BEFORE the heavy FinBERT install.

Design (lookahead-free):
  - For each (name, session day d): gather headlines timestamped in the OVERNIGHT window
    (prior session close 16:00 ET -> d 09:30 ET), score each, take the mean -> sentiment_d.
  - Forward return = OPEN[d] -> CLOSE[d] (intraday; news is all pre-open, so no lookahead).
  - Among (name,day) pairs WITH news, test: do positive-sentiment days beat negative ones?
    (correlation, pos-vs-neg avg return + hit-rate, vs the all-news-days baseline).

Honest: VADER is general-English (weak on finance) so a NULL is inconclusive for FinBERT,
but a POSITIVE result strongly motivates the FinBERT install. Caches news to avoid re-fetch.

Run:  .venv/Scripts/python.exe backtest/news_sentiment_probe.py
"""
from __future__ import annotations

import pickle
import sys
from datetime import datetime, time as dtime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.run_orb import load_env  # noqa: E402

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
NEWS_CACHE = ROOT / "backtest" / ".news_sentiment_cache.pkl"
START = datetime(2026, 4, 1, tzinfo=UTC)   # ~3 months (dense news, data available)
# liquid names that reliably get news, spanning mega-cap + higher-beta catalyst-y names
UNIVERSE = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "NFLX", "AVGO",
            "INTC", "MU", "PLTR", "COIN", "UBER", "SOFI", "BABA", "DIS", "BA", "PYPL",
            "F", "GM", "PFE", "MRNA", "SHOP", "ROKU", "SNAP", "RIVN", "MARA", "CVNA"]


# ---- swappable sentiment scorer ----
def make_scorer():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    a = SentimentIntensityAnalyzer()
    return lambda text: a.polarity_scores(text)["compound"]  # [-1, +1]


def fetch_news(symbols) -> dict:
    if NEWS_CACHE.exists():
        print(f"using cached news: {NEWS_CACHE.name}")
        return pickle.load(open(NEWS_CACHE, "rb"))
    import os
    from alpaca.data.historical.news import NewsClient
    from alpaca.data.requests import NewsRequest
    c = NewsClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    out = {}
    for i, sym in enumerate(symbols, 1):
        items, tok, pages = [], None, 0
        while pages < 40:
            req = NewsRequest(symbols=sym, start=START, limit=50, page_token=tok)
            r = c.get_news(req)
            arts = r.data.get("news", [])
            for a in arts:
                items.append((a.created_at, a.headline))
            tok = getattr(r, "next_page_token", None)
            pages += 1
            if not tok or not arts:
                break
        out[sym] = items
        print(f"  {i}/{len(symbols)} {sym}: {len(items)} headlines", flush=True)
    pickle.dump(out, open(NEWS_CACHE, "wb"))
    return out


def fetch_daily(symbols) -> pd.DataFrame:
    import os
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    dc = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame.Day,
                           start=START, end=datetime.now(UTC), feed=DataFeed.IEX)
    return dc.get_stock_bars(req).df


def main() -> int:
    load_env()
    score = make_scorer()
    print(f"news-sentiment probe (VADER): {len(UNIVERSE)} names since {START.date()}\n")
    news = fetch_news(UNIVERSE)
    bars = fetch_daily(UNIVERSE)

    rows = []
    for sym in UNIVERSE:
        try:
            sb = bars.xs(sym, level=0).sort_index()
        except KeyError:
            continue
        idx = [t.astimezone(ET) for t in sb.index]
        opens, closes = sb["open"].tolist(), sb["close"].tolist()
        # per-session headlines list, sorted
        heads = sorted(news.get(sym, []), key=lambda x: x[0])
        for k in range(1, len(idx)):
            d = idx[k].date()
            session_open = datetime.combine(d, dtime(9, 30), ET)
            prev_close_t = datetime.combine(idx[k - 1].date(), dtime(16, 0), ET)
            # overnight headlines: after prior close, before this open
            sc = [score(h) for (ts, h) in heads
                  if prev_close_t < ts.astimezone(ET) <= session_open]
            if not sc:
                continue
            sent = sum(sc) / len(sc)
            ret = (closes[k] / opens[k] - 1.0) * 100 if opens[k] else None   # open->close intraday
            if ret is None:
                continue
            rows.append({"sym": sym, "date": d, "n_head": len(sc), "sent": sent, "ret": ret})

    df = pd.DataFrame(rows)
    if df.empty:
        print("no (name,day) pairs with overnight news — widen window/universe."); return 1
    print(f"\n{len(df)} (name,day) pairs WITH overnight news across {df['sym'].nunique()} names")
    corr = df["sent"].corr(df["ret"])
    pos = df[df["sent"] > 0.1]; neg = df[df["sent"] < -0.1]; flat = df[df["sent"].abs() <= 0.1]
    base = df["ret"].mean()
    print(f"\n{'bucket':<18}{'n':>6}{'avg ret%':>10}{'win%':>8}")
    print("-" * 42)
    for name, g in [("positive (>+.1)", pos), ("neutral", flat), ("negative (<-.1)", neg), ("ALL w/ news", df)]:
        if len(g):
            print(f"{name:<18}{len(g):>6}{g['ret'].mean():>+10.3f}{(g['ret'] > 0).mean() * 100:>7.0f}%")
    print(f"\nsentiment<->return correlation: {corr:+.3f}")
    print(f"pos-minus-neg edge: {pos['ret'].mean() - neg['ret'].mean():+.3f}%  "
          f"(pos avg {pos['ret'].mean():+.3f} vs neg avg {neg['ret'].mean():+.3f})")
    print("\nRead: a real signal = positive corr + pos beats neg by a clear margin + holds the")
    print("'pos > ALL-baseline'. VADER null is inconclusive (try FinBERT); VADER positive => "
          "install FinBERT for the finance-grade number. open->close = no overnight-gap lookahead.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
