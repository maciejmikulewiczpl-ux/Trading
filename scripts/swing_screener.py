"""Swing momentum-continuation screener — "already ran, still has room".

Read-only research tool. NOT wired to any bot or money. It mechanizes the idea
from the Instagram "3-step read" post (the part of it that actually has edge:
relative-strength / momentum continuation) into a repeatable daily screen.

A name passes only if ALL of these hold (every filter is plain daily-bar math):

  F1  IT RAN          6-month return >= MIN_6MO_RET, and price within
                      MAX_BELOW_HIGH of its 52-week high.
  F2  STILL LEADING   relative strength vs SPY positive over BOTH the 21-day
                      and 63-day windows (outpacing the market recently AND
                      over the swing trend).
  F3  TREND INTACT    close > 50d SMA, and 50d SMA > 200d SMA (healthy uptrend,
                      not a broken one).
  F4  NOT BLOWN OFF   price <= MAX_EXT_50 above the 50d SMA (avoid buying the
                      exact vertical top — that's mean-reversion risk, not room).

Survivors are then ENRICHED with news + online-hype context (the post's steps
1-2, used as TIEBREAKERS not selectors — they confirm a story, they don't make
one). All from the proven news-edge connectors:
  news    Alpha Vantage per-ticker news sentiment   (+1 bullish / 0 / -1 bearish)
  reddit  WSB mention surge vs 24h ago (apewisdom)   (crowding gauge, no direction)
  stwits  StockTwits retail crowd sentiment          (bull% of tagged messages)

Survivors are RANKED by a composite of 3-month relative strength, proximity to
the 50d (a controlled pullback ranks above an extended name), plus a small tilt
for positive news / fresh crowd attention. The script prints the ranked table
and writes scripts/swing_screen_<date>.csv. Enrichment hits free public APIs;
pass --no-enrich to skip it (pure price/relative-strength screen).

This is the SWING (multi-day) cousin of the intraday ORB momentum edge. The real
edge in any continuation strategy lives in the EXIT (trailing stop), not the
screen — see memory/trailing_exit_finding.md. Treat output as a research
watchlist, not signals.

Run:
    .venv/Scripts/python.exe scripts/swing_screener.py
    .venv/Scripts/python.exe scripts/swing_screener.py --min-6mo 0.40 --top 20
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.run_orb import load_env  # noqa: E402
from backtest.universe_scan import UNIVERSE  # noqa: E402
from backtest.fetch_universe_expanded import EXPANSION  # noqa: E402
# Reuse the proven news-edge / lottery connectors verbatim (never edit those modules).
from experiments.news_edge.sources import (  # noqa: E402
    av_sentiment, reddit_trending, st_trending, st_sentiment,
)

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

BENCH = "SPY"
# --- default filter thresholds (override on the CLI) ---
MIN_6MO_RET = 0.25      # F1: must already be up >=25% over ~6 months
MAX_BELOW_HIGH = 0.10   # F1: price within 10% of its 52-week high
MAX_EXT_50 = 0.20       # F4: price no more than 20% above the 50d SMA


def fetch_daily_ohlc(symbols, start, end) -> pd.DataFrame:
    """Adjusted daily bars, long form (index=[symbol, date], cols=OHLCV), chunked."""
    key, sec = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    if not key or not sec:
        raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY not set in .env")
    dc = StockHistoricalDataClient(key, sec)
    frames = []
    for i in range(0, len(symbols), 20):
        grp = symbols[i:i + 20]
        print(f"  daily bars {i + 1}-{i + len(grp)} of {len(symbols)} ...", flush=True)
        req = StockBarsRequest(
            symbol_or_symbols=grp, timeframe=TimeFrame.Day,
            start=start.astimezone(UTC), end=end.astimezone(UTC),
            feed=DataFeed.IEX, adjustment=Adjustment.ALL,
        )
        df = dc.get_stock_bars(req).df
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames)
    return out


def ret(close: pd.Series, n: int) -> float:
    """Simple return over the last n trading days (None if insufficient history)."""
    if len(close) <= n:
        return None
    return float(close.iloc[-1] / close.iloc[-1 - n] - 1.0)


def screen(close_by_sym: dict[str, pd.Series], high_by_sym: dict[str, pd.Series],
           args) -> pd.DataFrame:
    spy = close_by_sym.get(BENCH)
    if spy is None or len(spy) < 200:
        raise RuntimeError(f"benchmark {BENCH} history missing/short")
    spy_r21, spy_r63 = ret(spy, 21), ret(spy, 63)

    rows = []
    for sym, c in close_by_sym.items():
        if sym == BENCH or len(c) < 200:
            continue
        price = float(c.iloc[-1])
        r126 = ret(c, 126)          # ~6 months
        r21, r63 = ret(c, 21), ret(c, 63)
        if None in (r126, r21, r63):
            continue
        hi = high_by_sym.get(sym)
        high_252 = float(hi.iloc[-252:].max()) if hi is not None and len(hi) >= 1 else float(c.iloc[-252:].max())
        below_high = 1.0 - price / high_252 if high_252 > 0 else 1.0
        sma50 = float(c.iloc[-50:].mean())
        sma200 = float(c.iloc[-200:].mean())
        ext50 = price / sma50 - 1.0
        rs21, rs63 = r21 - spy_r21, r63 - spy_r63

        # --- the four filters ---
        f1 = (r126 >= args.min_6mo) and (below_high <= args.max_below_high)
        f2 = (rs21 > 0) and (rs63 > 0)
        f3 = (price > sma50) and (sma50 > sma200)
        f4 = ext50 <= args.max_ext
        passed = f1 and f2 and f3 and f4

        rows.append({
            "symbol": sym, "price": round(price, 2),
            "ret_6mo": round(r126, 3), "below_52wk_high": round(below_high, 3),
            "rs_1mo": round(rs21, 3), "rs_3mo": round(rs63, 3),
            "ext_vs_50d": round(ext50, 3),
            "above_200d": price > sma200, "trend_up": sma50 > sma200,
            "F1_ran": f1, "F2_leading": f2, "F3_trend": f3, "F4_not_extended": f4,
            "PASS": passed,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Composite rank for survivors: reward 3mo relative strength, penalize
    # extension above the 50d (a controlled pullback ranks above a blow-off).
    df["score"] = df["rs_3mo"] - 0.5 * df["ext_vs_50d"].clip(lower=0)
    return df.sort_values(["PASS", "score"], ascending=[False, False]).reset_index(drop=True)


def alpaca_news_counts(symbols: list[str], days: int = 3) -> dict[str, int]:
    """Fresh-catalyst gauge: # of Alpaca news headlines per symbol over the last
    `days`. Free + reliable (unlike the rate-limited AV sentiment). {sym: count}."""
    from alpaca.data.historical.news import NewsClient
    from alpaca.data.requests import NewsRequest
    key, sec = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    counts = {s: 0 for s in symbols}
    try:
        c = NewsClient(key, sec)
        start = datetime.now(UTC) - timedelta(days=days)
        # page through (50/call); symbols filter is server-side
        req = NewsRequest(symbols=",".join(symbols), start=start, limit=50)
        seen = 0
        while True:
            resp = c.get_news(req)
            arts = resp.data.get("news", [])
            for a in arts:
                for s in a.symbols:
                    if s in counts:
                        counts[s] += 1
            seen += len(arts)
            tok = getattr(resp, "next_page_token", None)
            if not tok or not arts or seen >= 500:
                break
            req = NewsRequest(symbols=",".join(symbols), start=start, limit=50, page_token=tok)
    except Exception as e:
        print(f"    alpaca news fetch failed: {str(e)[:80]}")
    return counts


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add news + online-hype columns for the PASSing names only (keeps API calls
    small) and fold a small tilt into the score. Every source fails soft to None.

    news_3d  = # Alpaca headlines in last 3 days  (PRIMARY catalyst gauge, reliable)
    news     = Alpha Vantage sentiment direction  (+1/0/-1; often None = rate-limited)
    reddit_surge / stwits_bull% = online crowd attention.
    """
    syms = df.loc[df["PASS"], "symbol"].tolist()
    if not syms:
        return df

    # --- fresh-catalyst count (Alpaca news; reliable) ---
    print(f"  enriching {len(syms)} survivors: Alpaca news (catalyst count) ...", flush=True)
    news_3d = alpaca_news_counts(syms, days=3)

    # --- news direction (Alpha Vantage; free tier = 25 req/day, shared w/ the bots) ---
    print("  ... Alpha Vantage sentiment direction ...", flush=True)
    av = av_sentiment(syms)
    av_throttled = isinstance(av, dict) and (
        "_error" in av or "_note" in av
        or all((v.get("signal") is None) for k, v in av.items() if not k.startswith("_")))
    if av_throttled:
        print("    (AV sentiment unavailable/throttled - relying on Alpaca catalyst count; "
              "the 'news' column will be blank. This is normal if the bots already used "
              "today's 25-call AV budget.)")

    # --- online hype: reddit WSB mention surge + StockTwits crowd ---
    print("  ... reddit (WSB mentions) + StockTwits (crowd sentiment) ...", flush=True)
    reddit = {r["ticker"]: r for r in reddit_trending(50) if isinstance(r, dict)}
    st_now = set(s for s in st_trending(30) if isinstance(s, str))
    st_sent = st_sentiment(syms)   # per-survivor bull/bear from recent messages

    rows = []
    for _, row in df.iterrows():
        sym = row["symbol"]
        n3 = news_3d.get(sym)
        a = av.get(sym, {}) if isinstance(av, dict) else {}
        news_sig = a.get("signal")
        r = reddit.get(sym, {})
        reddit_surge = r.get("surge")
        reddit_ment = r.get("mentions")
        s = st_sent.get(sym, {}) if isinstance(st_sent, dict) else {}
        st_bull = s.get("bull_pct")
        st_trending_now = sym in st_now

        # Hype/news tilt — deliberately SMALL: confirms structure, never overrides it.
        tilt = 0.0
        if n3 is not None and n3 >= 3:
            tilt += 0.02          # an active, fresh news flow (rotation candidate)
        if news_sig == 1:
            tilt += 0.04
        elif news_sig == -1:
            tilt -= 0.06          # a bearish news read is a bigger warning than a bullish one helps
        if reddit_surge and reddit_surge >= 2.0:
            tilt += 0.03          # fresh, surging retail attention
        if st_trending_now:
            tilt += 0.01
        if st_bull is not None and st_bull >= 65:
            tilt += 0.01

        rows.append({"symbol": sym, "news_3d": n3, "news": news_sig,
                     "reddit_surge": reddit_surge, "reddit_ment": reddit_ment,
                     "stwits_bull%": st_bull, "stwits_trending": st_trending_now,
                     "hype_tilt": round(tilt, 3)})

    add = pd.DataFrame(rows)
    df = df.merge(add, on="symbol", how="left")
    df["hype_tilt"] = df["hype_tilt"].fillna(0.0)
    df["score"] = df["score"] + df["hype_tilt"]
    return df.sort_values(["PASS", "score"], ascending=[False, False]).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-6mo", type=float, default=MIN_6MO_RET, dest="min_6mo",
                    help="minimum 6-month return (F1), e.g. 0.25 = +25%%")
    ap.add_argument("--max-below-high", type=float, default=MAX_BELOW_HIGH, dest="max_below_high",
                    help="max fraction below 52-week high (F1), e.g. 0.10")
    ap.add_argument("--max-ext", type=float, default=MAX_EXT_50, dest="max_ext",
                    help="max extension above 50d SMA (F4), e.g. 0.20")
    ap.add_argument("--top", type=int, default=25, help="rows to print")
    ap.add_argument("--all", action="store_true", help="print non-passing names too")
    ap.add_argument("--no-enrich", action="store_true",
                    help="skip news/hype enrichment (pure price screen)")
    args = ap.parse_args()

    load_env()
    universe = sorted(set(UNIVERSE) | set(EXPANSION) | {BENCH})
    print(f"Swing screener - {len(universe)} names | "
          f"F1 ret_6mo>={args.min_6mo:.0%}, within {args.max_below_high:.0%} of 52wk high | "
          f"F4 ext<= {args.max_ext:.0%}\n")

    end = datetime.now(ET)
    start = end - timedelta(days=420)   # ~290 sessions: enough for 252d high + 200d SMA
    raw = fetch_daily_ohlc(universe, start, end)
    if raw.empty:
        print("No data returned.")
        return 1

    close_by_sym, high_by_sym = {}, {}
    for sym, sub in raw.groupby(level=0):
        s = sub.droplevel(0).sort_index()
        close_by_sym[sym] = s["close"]
        high_by_sym[sym] = s["high"]

    df = screen(close_by_sym, high_by_sym, args)
    if df.empty:
        print("No names had sufficient history.")
        return 1

    n_pass = int(df["PASS"].sum())
    enriched = not args.no_enrich and n_pass > 0
    if enriched:
        df = enrich(df)

    out_csv = ROOT / "scripts" / f"swing_screen_{end.date()}.csv"
    df.to_csv(out_csv, index=False)

    show = df if args.all else df[df["PASS"]]
    show = show.head(args.top)
    cols = ["symbol", "price", "ret_6mo", "below_52wk_high", "rs_1mo", "rs_3mo",
            "ext_vs_50d"]
    if enriched:
        cols += ["news_3d", "news", "reddit_surge", "stwits_bull%", "hype_tilt"]
    cols += ["PASS"]
    pd.set_option("display.max_rows", None, "display.width", 160)
    print(show[cols].to_string(index=False))
    print(f"\n{n_pass} of {len(df)} names PASS all four filters. "
          f"Full table -> {out_csv.name}")
    if n_pass == 0:
        print("(0 passing usually means a risk-off tape — loosen --min-6mo or "
              "--max-below-high, or wait for leadership to re-form.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
