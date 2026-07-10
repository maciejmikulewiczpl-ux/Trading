"""Lottery-scanner source connectors. Imports the proven news-edge connectors
(reddit_trending, st_trending, pm_rvol, pm_gappers + Alpaca helpers) and ADDS the
lottery-specific signals: squeeze_scores, uoa_snapshot, ignition_scores, random_basket.

NEVER modify experiments/news_edge/* — we only import from it.

Signals come online over days 1-4 (see README):
  - squeeze_scores(): None until .short_interest_cache.json exists (Sunday job fills it).
  - uoa_snapshot(): None per-name until .uoa_state.json has ~20 sessions of history.
  - ignition_scores(): live day 1 (one daily-bars request).

CLI:
  .venv/Scripts/python.exe experiments/lottery/sources.py ignition NVDA,GME,AMC
  .venv/Scripts/python.exe experiments/lottery/sources.py random
  .venv/Scripts/python.exe experiments/lottery/sources.py squeeze NVDA,GME
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, time as dtime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from backtest.run_orb import load_env  # noqa: E402
# Reuse the working news-edge connectors verbatim (never edit that module).
from experiments.news_edge.sources import (  # noqa: E402
    reddit_trending, st_trending, pm_rvol, pm_gappers, _alpaca_dc,
)

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
HERE = Path(__file__).resolve().parent
UNIVERSE_FILE = HERE / "universe.txt"
SHORT_INTEREST_CACHE = HERE / ".short_interest_cache.json"
UOA_STATE = HERE / ".uoa_state.json"


def load_universe() -> list[str]:
    if not UNIVERSE_FILE.exists():
        return []
    return [s.strip().upper() for s in UNIVERSE_FILE.read_text().splitlines() if s.strip()]


# --------------------------------------------------------------- extra subreddits (v1.1)
# Same apewisdom API as the news-edge reddit_trending (WSB), but for OTHER subreddits that
# surface different, more lottery-relevant names — r/pennystocks and r/Shortsqueeze carry
# explosive small/micro-cap + squeeze candidates WSB misses. Added 2026-06-27 as NEW
# MEASURED signals (own scoreboard clock); NOT folded into combined_score until they prove
# out (the bot's traded score is unchanged). Mirrors reddit_trending's parse verbatim.
_APEWISDOM_SUB = "https://apewisdom.io/api/v1.0/filter/{sub}/page/1"
_SUB_UA = {"User-Agent": "Mozilla/5.0 (lottery research)"}


def reddit_sub_trending(subreddit: str, limit: int = 25) -> list:
    """Top tickers by mention SURGE (mentions / mentions_24h_ago, min 10 mentions) on a
    given subreddit. [{ticker, rank, mentions, mentions_24h_ago, surge, upvotes}] — same
    shape as news_edge.reddit_trending so it drops into the board identically."""
    import urllib.request
    try:
        req = urllib.request.Request(_APEWISDOM_SUB.format(sub=subreddit), headers=_SUB_UA)
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


# --------------------------------------------------------------- Google Trends (v1.2)
# Search-interest SPIKE per ticker = recent / trailing-baseline interest (a retail-ATTENTION
# surge, orthogonal to social posts). Uses trendspy (Google's new Trends backend; the old
# pytrends 429s). Batched + sleeps + graceful-None on rate-limit so it can NEVER break the
# board. Added 2026-06-28 as a NEW MEASURED signal (own scoreboard clock; not in combined_score).
def google_trends_spike(tickers: list[str], batch: int = 5, sleep_s: float = 3.0) -> dict:
    """{ticker: spike_ratio} where spike = mean(last 3 obs) / mean(prior obs) over ~1 month.
    >1 = rising search attention. Missing/rate-limited names -> absent (graceful)."""
    try:
        from trendspy import Trends
    except Exception:
        return {}
    import time as _t
    tr = Trends()
    syms = [t.upper() for t in tickers]
    out: dict = {}
    for i in range(0, len(syms), batch):
        grp = syms[i:i + batch]
        terms = [f"{s} stock" for s in grp]
        try:
            df = tr.interest_over_time(terms, timeframe="today 1-m")
            for s, term in zip(grp, terms):
                if term in getattr(df, "columns", []):
                    col = df[term].astype(float)
                    recent = float(col.iloc[-3:].mean())
                    base = max(float(col.iloc[:-3].mean()), 1.0)
                    out[s] = round(recent / base, 2)
        except Exception:
            pass   # rate-limited / failed batch -> those names just have no trends signal today
        _t.sleep(sleep_s)
    return out


# --------------------------------------------------------------- FINRA short volume (v1.3)
# Daily Reg SHO short-sale volume (free public file). Signal = ShortVolume/TotalVolume per
# symbol = how short-skewed the day's tape was = potential squeeze fuel. T+1 (yesterday's
# session; the file for the latest trading day publishes overnight — walk back over
# weekends/holidays). Added 2026-06-29 as a NEW MEASURED signal (own scoreboard clock; NOT
# in combined_score). Graceful-{} so it can never break the board.
_FINRA_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{d}.txt"


def finra_short_volume(max_back: int = 6) -> dict:
    """{ticker: short_ratio} where short_ratio = ShortVolume/TotalVolume from FINRA's most
    recent daily Reg SHO file. High = heavy short-side activity. Graceful-{} on failure."""
    import urllib.request
    for back in range(max_back):
        d = (datetime.now(ET).date() - timedelta(days=back)).strftime("%Y%m%d")
        try:
            req = urllib.request.Request(_FINRA_URL.format(d=d), headers=_SUB_UA)
            with urllib.request.urlopen(req, timeout=20) as r:
                txt = r.read().decode("utf-8", "ignore")
        except Exception:
            continue   # 403 = not published for that date yet -> walk back
        out: dict = {}
        for line in txt.strip().splitlines()[1:]:
            parts = line.split("|")
            if len(parts) < 5:
                continue
            sym = parts[1].strip().upper()
            try:
                sv, tv = float(parts[2]), float(parts[4])
            except ValueError:
                continue
            if tv > 0 and sym.isalpha():
                out[sym] = round(sv / tv, 4)
        if out:
            return out
    return {}


# --------------------------------------------------------------- trading halts (v1.3)
# Nasdaq trade-halt RSS feed. A name in the feed (esp. 'LUDP' LULD volatility pause, or
# news/regulatory halts) is by definition an explosive/volatile name = on-theme for lottery.
# Feed shows recent halts (current/prior session). Added 2026-06-29 as a NEW MEASURED signal
# (own scoreboard clock; NOT in combined_score). Graceful-{} so it can never break the board.
_HALTS_URL = "http://www.nasdaqtrader.com/rss.aspx?feed=tradehalts"


def trading_halts() -> dict:
    """{ticker: reason_code} for symbols currently in the Nasdaq trade-halt feed."""
    import re
    import urllib.request
    try:
        req = urllib.request.Request(_HALTS_URL, headers=_SUB_UA)
        with urllib.request.urlopen(req, timeout=20) as r:
            xml = r.read().decode("utf-8", "ignore")
    except Exception:
        return {}
    out: dict = {}
    for item in re.findall(r"<item>(.*?)</item>", xml, re.S):
        m = re.search(r"<ndaq:IssueSymbol>([^<]+)</ndaq:IssueSymbol>", item)
        rc = re.search(r"<ndaq:ReasonCode>([^<]*)</ndaq:ReasonCode>", item)
        if m and m.group(1).strip().isalpha():
            out[m.group(1).strip().upper()] = (rc.group(1).strip() if rc else "?")
    return out


# --------------------------------------------------------------- random baseline
def random_basket(seed: int | None = None, n: int = 10) -> list[str]:
    """Seeded random basket from the liquid universe. seed defaults to YYYYMMDD (ET)
    so a given day's basket is reproducible / immutable in the picks file."""
    import random
    uni = load_universe()
    if not uni:
        return []
    if seed is None:
        seed = int(datetime.now(ET).strftime("%Y%m%d"))
    rng = random.Random(seed)
    return rng.sample(uni, min(n, len(uni)))


# --------------------------------------------------------------- ignition (signal 6)
def ignition_scores(tickers: list[str]) -> dict:
    """Price/volume ignition for each ticker from ONE daily-bars request (IEX).
    Same lookahead-free composite as backtest/lottery_ignition.py:
      streak (consecutive green closes, capped 5), volaccel (mean5d/mean20d vol),
      high_prox (close / trailing-252 high), prevwin (today up >=2%).
    Returns {ticker: {"ignition": 0..4 int, "streak": int, "volaccel": float,
                      "high_prox": float, "prevwin": 0/1}} or {} on fetch failure.
    All features use bars THROUGH the most recent completed session (no same-day leak;
    at the 6:24am board run only yesterday's daily bar is final)."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    syms = [t.upper() for t in tickers]
    if not syms:
        return {}
    dc = _alpaca_dc()
    end = datetime.now(ET)
    start = end - timedelta(days=400)   # ~252 trading sessions + buffer
    try:
        req = StockBarsRequest(symbol_or_symbols=syms, timeframe=TimeFrame.Day,
                               start=start.astimezone(UTC), end=end.astimezone(UTC),
                               feed=DataFeed.IEX)
        df = dc.get_stock_bars(req).df
    except Exception as e:
        return {"_error": f"ignition fetch failed: {e}"}
    out: dict = {}
    for t in syms:
        try:
            sb = df.xs(t, level=0).sort_index()
            closes = sb["close"].tolist()
            vols = sb["volume"].tolist()
            if len(closes) < 25:
                out[t] = {"ignition": None}
                continue
            # streak of consecutive green closes ending at the last bar
            streak = 0
            for i in range(len(closes) - 1, 0, -1):
                if closes[i] > closes[i - 1]:
                    streak += 1
                else:
                    break
            streak = min(streak, 5)
            v_fast = sum(vols[-5:]) / 5.0
            v_slow = sum(vols[-20:]) / 20.0
            volaccel = (v_fast / v_slow) if v_slow > 0 else None
            high252 = max(closes[-252:]) if len(closes) >= 252 else max(closes)
            high_prox = (closes[-1] / high252) if high252 > 0 else None
            prevwin = 1 if (closes[-2] > 0 and closes[-1] / closes[-2] - 1.0 >= 0.02) else 0
            ig = 0
            ig += 1 if streak >= 3 else 0
            ig += 1 if (volaccel is not None and volaccel >= 1.5) else 0
            ig += 1 if (high_prox is not None and high_prox >= 0.95) else 0
            ig += prevwin
            # realized daily volatility (20d std of close-to-close returns) = the "expected
            # move" -- used by the board's price/vol FILTER variant (measured-only). 2026-06-29.
            import statistics as _stats
            rr = [closes[i] / closes[i - 1] - 1.0 for i in range(len(closes) - 20, len(closes))
                  if closes[i - 1]]
            rvol = round(_stats.pstdev(rr), 4) if len(rr) >= 10 else None
            out[t] = {"ignition": ig, "streak": streak,
                      "volaccel": round(volaccel, 2) if volaccel is not None else None,
                      "high_prox": round(high_prox, 3) if high_prox is not None else None,
                      "prevwin": prevwin, "realized_vol": rvol}
        except Exception:
            out[t] = {"ignition": None}
    return out


# --------------------------------------------------------------- squeeze (signal 4)
def squeeze_scores(tickers: list[str]) -> dict:
    """Short-squeeze score = short%float × days-to-cover, read from the weekly yfinance
    cache (.short_interest_cache.json, filled by update_short_interest.py). Returns
    {ticker: {"squeeze": float|None, "short_pct_float": ..., "days_to_cover": ...}}.
    Returns all-None gracefully if the cache doesn't exist yet (comes online ~day 3)."""
    syms = [t.upper() for t in tickers]
    if not SHORT_INTEREST_CACHE.exists():
        return {t: {"squeeze": None} for t in syms}
    try:
        cache = json.load(open(SHORT_INTEREST_CACHE))
    except Exception:
        return {t: {"squeeze": None} for t in syms}
    rows = cache.get("data", cache) if isinstance(cache, dict) else {}
    out = {}
    for t in syms:
        r = rows.get(t)
        if not r:
            out[t] = {"squeeze": None}
            continue
        spf = r.get("short_pct_float")
        dtc = r.get("days_to_cover")
        sq = (spf * dtc) if (spf is not None and dtc is not None) else None
        out[t] = {"squeeze": round(sq, 2) if sq is not None else None,
                  "short_pct_float": spf, "days_to_cover": dtc,
                  "as_of": r.get("as_of")}
    return out


# --------------------------------------------------------------- UOA (signal 5)
def _load_uoa_state() -> dict:
    if not UOA_STATE.exists():
        return {}
    try:
        return json.load(open(UOA_STATE))
    except Exception:
        return {}


def _save_uoa_state(state: dict) -> None:
    json.dump(state, open(UOA_STATE, "w"), indent=2)


def uoa_snapshot(tickers: list[str], update_state: bool = True) -> dict:
    """Unusual-options call-volume z-score. Pulls today's total CALL daily-bar volume
    across each name's option chain (Alpaca option snapshots), appends to a rolling 20d
    trailing series in .uoa_state.json, and returns the z-score of today's value vs the
    trailing window. Returns {ticker: {"uoa_z": float|None, "call_vol": int, "n_days": int}}.

    uoa_z is None until a name has >= 15 trailing sessions of history (signal comes online
    after ~20 sessions of board runs). On any API failure -> None for that name, no crash.
    """
    syms = [t.upper() for t in tickers]
    state = _load_uoa_state()
    today = datetime.now(ET).date().isoformat()
    out: dict = {}

    # Try the option chain snapshot via alpaca-py; fall back to None gracefully.
    try:
        from alpaca.data.historical.option import OptionHistoricalDataClient
        from alpaca.data.requests import OptionChainRequest
        load_env()
        oc = OptionHistoricalDataClient(os.environ["ALPACA_API_KEY"],
                                        os.environ["ALPACA_SECRET_KEY"])
    except Exception:
        oc = None

    for t in syms:
        call_vol = None
        if oc is not None:
            try:
                from alpaca.data.requests import OptionChainRequest as OCR
                chain = oc.get_option_chain(OCR(underlying_symbol=t))
                tot = 0
                got = False
                for sym, snap in (chain or {}).items():
                    # call options have 'C' in the OCC symbol's option-type slot
                    if "C" not in sym[-9:-8:1] and "C" not in sym:
                        pass
                    db = getattr(snap, "daily_bar", None)
                    if db is not None and getattr(db, "volume", None) is not None:
                        # only count calls
                        if _is_call(sym):
                            tot += int(db.volume)
                            got = True
                call_vol = tot if got else None
            except Exception:
                call_vol = None

        hist = state.get(t, {})           # {date: call_vol}
        series = [v for d, v in sorted(hist.items()) if d < today and v is not None]
        n_days = len(series)
        z = None
        if call_vol is not None and n_days >= 15:
            import statistics
            window = series[-20:]
            mu = statistics.mean(window)
            sd = statistics.pstdev(window)
            z = (call_vol - mu) / sd if sd > 0 else None
        out[t] = {"uoa_z": round(z, 2) if z is not None else None,
                  "call_vol": call_vol, "n_days": n_days}
        if update_state and call_vol is not None:
            hist[today] = call_vol
            # keep only ~40 most recent days
            hist = dict(sorted(hist.items())[-40:])
            state[t] = hist

    if update_state:
        _save_uoa_state(state)
    return out


def _is_call(occ_symbol: str) -> bool:
    """OCC option symbol: ROOT + YYMMDD + C/P + strike. Find the option-type letter."""
    import re
    m = re.search(r"\d{6}([CP])\d{8}$", occ_symbol)
    return bool(m and m.group(1) == "C")


def options_expected_move(tickers: list[str], horizon_days: int = 3, max_dte: int = 45) -> dict:
    """Options-implied EXPECTED MOVE per name = a forward-looking "the market is pricing a big
    move" signal. Value = near-dated ATM straddle mid / strike, sqrt-time normalized to
    `horizon_days` (the bot's T+3 hold) so names with different nearest expiries are comparable.
    ATM is found WITHOUT a spot fetch (the strike where call & put mids are nearest equal ~ ATM).
    Uses QUOTES only (Alpaca free tier has no greeks/IV). Graceful-None per name so it can NEVER
    break the board; ~0.1-1.7s/name so restrict to the candidate net. Measured-only signal."""
    import math
    from datetime import date, timedelta
    try:
        from alpaca.data.historical.option import OptionHistoricalDataClient
        from alpaca.data.requests import OptionChainRequest
        oc = OptionHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    except Exception:
        return {}
    today = date.today()
    out: dict[str, float] = {}
    for t in tickers:
        try:
            try:
                req = OptionChainRequest(underlying_symbol=t, expiration_date_gte=today,
                                         expiration_date_lte=today + timedelta(days=max_dte))
            except Exception:
                req = OptionChainRequest(underlying_symbol=t)
            chain = oc.get_option_chain(req)
            exps: dict = {}
            for k, s in chain.items():
                q = getattr(s, "latest_quote", None)
                if not q or not q.bid_price or not q.ask_price:
                    continue
                mid = (q.bid_price + q.ask_price) / 2
                if mid <= 0:
                    continue
                e, typ, strike = _parse_occ(k)
                if e is None or e < today or (e - today).days > max_dte:
                    continue
                exps.setdefault(e, {}).setdefault(strike, {})[typ] = mid
            if not exps:
                continue
            cand = sorted(exps)
            e = next((x for x in cand if (x - today).days >= 2), cand[0])   # skip 0-1DTE noise
            both = {st: v for st, v in exps[e].items() if "C" in v and "P" in v}
            if not both:
                continue
            atm = min(both, key=lambda st: abs(both[st]["C"] - both[st]["P"]))
            raw = (both[atm]["C"] + both[atm]["P"]) / atm
            days = max((e - today).days, 1)
            out[t] = round(raw * math.sqrt(horizon_days / days) * 100, 3)
        except Exception:
            continue
    return out


def _parse_occ(sym: str):
    """OCC option symbol: ROOT + YYMMDD + C/P + strike(8 digits, thousandths).
    Returns (expiry_date, 'C'/'P', strike_float) or (None, None, None) if unparseable."""
    from datetime import date
    try:
        body = sym[-15:]
        return date(2000 + int(body[0:2]), int(body[2:4]), int(body[4:6])), body[6], int(body[7:]) / 1000.0
    except Exception:
        return None, None, None


def main(argv) -> int:
    cmd = argv[1].lower() if len(argv) > 1 else ""
    if cmd == "ignition" and len(argv) >= 3:
        toks = [t.strip().upper() for t in argv[2].split(",") if t.strip()]
        print(json.dumps(ignition_scores(toks), indent=2))
        return 0
    if cmd == "squeeze" and len(argv) >= 3:
        toks = [t.strip().upper() for t in argv[2].split(",") if t.strip()]
        print(json.dumps(squeeze_scores(toks), indent=2))
        return 0
    if cmd == "uoa" and len(argv) >= 3:
        toks = [t.strip().upper() for t in argv[2].split(",") if t.strip()]
        print(json.dumps(uoa_snapshot(toks, update_state=False), indent=2))
        return 0
    if cmd == "finra":
        sv = finra_short_volume()
        top = sorted(sv.items(), key=lambda kv: -kv[1])[:20]
        print(f"{len(sv)} symbols in latest FINRA daily file; top short-ratio:")
        print(json.dumps(dict(top), indent=2))
        return 0
    if cmd == "halts":
        print(json.dumps(trading_halts(), indent=2))
        return 0
    if cmd == "random":
        print(json.dumps(random_basket(), indent=2))
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
