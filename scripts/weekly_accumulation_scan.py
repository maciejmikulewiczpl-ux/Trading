"""WEEKLY accumulation + online-chatter scan (research watchlist; NOT a trading signal).

Combines two slow-and-fast signals once a week:
  - 13F INSTITUTIONAL ACCUMULATION (slow — quarterly filings, artifact-filtered) via
    institutional_check.check(drop_artifacts=True).
  - ONLINE CHATTER (fast — Reddit WSB surge + StockTwits trending/sentiment) via inline
    urllib (the news-edge connectors import alpaca, which isn't in .venv-openbb).
The OVERLAP = names institutions were net-buying that are ALSO getting fresh retail
attention = the "smart money in + crowd waking up" idea. Pushes the top overlap to the
phone (ntfy) + writes a CSV.

WHY WEEKLY: 13F barely changes between the ~quarterly filing waves; weekly is enough to
catch new filings + building chatter without pretending it's a timing signal.

MUST run under .venv-openbb (yfinance):
    .venv-openbb/Scripts/python.exe scripts/weekly_accumulation_scan.py
    .venv-openbb/Scripts/python.exe scripts/weekly_accumulation_scan.py --limit 30   # quick test
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from institutional_check import check  # noqa: E402  (yfinance-only, no alpaca)
from accumulation_scan import UNIVERSE, EXPANSION, price_since  # noqa: E402

ET = ZoneInfo("America/New_York")
_UA = {"User-Agent": "Mozilla/5.0 (accumulation-scan research)"}
APEWISDOM = "https://apewisdom.io/api/v1.0/filter/wallstreetbets/page/1"
ST_TREND = "https://api.stocktwits.com/api/2/trending/symbols.json"
ST_STREAM = "https://api.stocktwits.com/api/2/streams/symbol/{}.json"
MAX_MOVE = 0.10   # only flag names that haven't already run away (the move would be done)


# ---- chatter (inline urllib — pure HTTP, no alpaca import) ----
def reddit_trending(limit: int = 100) -> dict:
    try:
        with urllib.request.urlopen(urllib.request.Request(APEWISDOM, headers=_UA), timeout=20) as r:
            data = json.load(r)
    except Exception:
        return {}
    out = {}
    for it in data.get("results", []):
        try:
            m, m24 = int(it["mentions"]), int(it.get("mentions_24h_ago") or 0)
            out[it["ticker"]] = {"mentions": m, "surge": round(m / m24, 1) if m24 > 0 else None}
        except (KeyError, ValueError, TypeError):
            continue
        if len(out) >= limit:
            break
    return out


def st_trending(limit: int = 40) -> set:
    try:
        with urllib.request.urlopen(urllib.request.Request(ST_TREND, headers=_UA), timeout=20) as r:
            d = json.load(r)
        return {s.get("symbol") for s in d.get("symbols", [])[:limit] if s.get("symbol")}
    except Exception:
        return set()


def st_sentiment(tickers: list[str]) -> dict:
    out = {}
    for t in [x.upper() for x in tickers]:
        try:
            with urllib.request.urlopen(urllib.request.Request(ST_STREAM.format(t), headers=_UA), timeout=20) as r:
                d = json.load(r)
            bull = bear = 0
            for m in d.get("messages", []):
                s = (((m.get("entities") or {}).get("sentiment")) or {}).get("basic")
                bull += s == "Bullish"; bear += s == "Bearish"
            tot = bull + bear
            out[t] = {"bull_pct": round(bull / tot * 100) if tot else None, "n": tot}
        except Exception:
            out[t] = {"bull_pct": None, "n": 0}
        time.sleep(0.3)
    return out


def _load_ntfy_topic() -> str | None:
    f = ROOT / ".env"
    if f.exists():
        for line in f.read_text().splitlines():
            if line.strip().startswith("NTFY_TOPIC"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("NTFY_TOPIC")


def ntfy(msg: str, title: str) -> bool:
    topic = _load_ntfy_topic()
    if not topic:
        print("NTFY_TOPIC unset — skipping push")
        return False
    try:
        req = urllib.request.Request(f"https://ntfy.sh/{topic}", data=msg.encode("utf-8"),
                                     headers={"Title": title, "Tags": "chart_with_upwards_trend",
                                              "Priority": "3"})
        urllib.request.urlopen(req, timeout=8)
        return True
    except Exception as e:
        print(f"ntfy failed: {e}")
        return False


def main(argv) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0, help="cap universe (for quick tests)")
    args = ap.parse_args(argv[1:])

    universe = sorted(set(UNIVERSE) | set(EXPANSION))
    if args.limit:
        universe = universe[:args.limit]
    date = datetime.now(ET).date().isoformat()
    print(f"weekly accumulation+chatter scan {date} over {len(universe)} names\n")

    df = check(universe, drop_artifacts=True)
    acc = df[(df["verdict"] == "ACCUMULATING") & (~df["own_artifact"])
             & (df["top_adding"] > df["top_trimming"])].copy()
    print(f"\n{len(acc)} real-accumulation names; checking chatter ...", flush=True)

    wsb = reddit_trending(100)
    stt = st_trending(40)
    accset = set(acc["symbol"])
    overlap = sorted(accset & (set(wsb) | stt))
    sent = st_sentiment(overlap) if overlap else {}

    # price since 13f (only need overlap, but batch is cheap) + assemble
    ps = price_since(overlap, dict(zip(acc["symbol"], acc["as_of"]))) if overlap else {}
    acc = acc.set_index("symbol")
    rows = []
    for s in overlap:
        r = wsb.get(s, {}); se = sent.get(s, {}); a = acc.loc[s]
        ret = ps.get(s, (None, None))[1]
        rows.append({"symbol": s, "add": int(a.top_adding), "trim": int(a.top_trimming),
                     "ret_since_13f": round(ret * 100, 1) if ret is not None else None,
                     "wsb_surge": r.get("surge"), "wsb_mentions": r.get("mentions"),
                     "st_trending": s in stt, "st_bull%": se.get("bull_pct"), "st_msgs": se.get("n")})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["fresh"] = out["ret_since_13f"].abs() <= MAX_MOVE * 100  # not already run away
        out = out.sort_values(["fresh", "wsb_surge", "add"], ascending=[False, False, False],
                              na_position="last").reset_index(drop=True)
    csv = ROOT / "scripts" / f"accumulation_chatter_{date}.csv"
    out.to_csv(csv, index=False)

    print(f"\n=== {len(out)} accumulating names ALSO getting retail chatter ===")
    print(out.to_string(index=False) if not out.empty else "  (no overlap this week)")

    # phone push: the freshest combos (accumulation + chatter, not already run away)
    fresh = out[out["fresh"]] if (not out.empty and "fresh" in out) else out
    top = fresh.head(5)
    if not top.empty:
        lines = [f"{len(out)} accumulation+chatter overlaps ({len(fresh)} not-yet-run):"]
        for _, r in top.iterrows():
            tag = "ST" if r["st_trending"] else (f"WSB{r['wsb_surge']}x" if pd.notna(r["wsb_surge"]) else "")
            lines.append(f"{r['symbol']} {int(r['add'])}/{int(r['trim'])} "
                         f"{'' if pd.isna(r['ret_since_13f']) else f'{r.ret_since_13f:+.0f}%'} {tag}")
        msg = "\n".join(lines)
    elif not out.empty:
        msg = (f"{len(out)} accumulation+chatter overlaps, but all already moved "
               f">{MAX_MOVE:.0%} since 13F — nothing fresh: {', '.join(out['symbol'].head(6))}")
    else:
        msg = f"No accumulation+chatter overlap this week ({len(acc)} accumulating, none trending)."
    ntfy(msg, title=f"Accumulation+chatter {date}")
    print("\n" + msg)
    print(f"\n-> {csv.name}.  Research watchlist only — 13F lags ~1.5-3mo, chatter samples thin.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
