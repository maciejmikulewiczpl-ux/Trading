"""biotech_radar.py -- scan the small/mid-cap biotech universe (XBI holdings) for names
HEATING UP: volume building + price momentum + high volatility = the early tell that
"something's brewing" (often pre-catalyst positioning). Read-only watchlist tool.

HONEST FRAMING (important): biotech surges are driven by BINARY catalysts -- FDA decisions,
trial readouts, designations. The *timing* is sometimes knowable; the *direction* is NOT
(good data -> +100-300%, bad data -> -60-90%). So this flags WHERE a surge may be brewing
(volatility + attention), NOT which way it goes. It is a lottery-radar / watchlist, not a
predictor -- same discipline as the hype experiment. A future v2 (clinicaltrials.gov
catalyst calendar) would add the "known event approaching" piece.

Universe: State Street's public XBI daily holdings (equal-weight S&P Biotech -> small/mid-cap
heavy, the surge-prone set). Cached weekly to scripts/.biotech_universe.txt.

MUST run under .venv-openbb (yfinance):
    .venv-openbb/Scripts/python.exe scripts/biotech_radar.py
    .venv-openbb/Scripts/python.exe scripts/biotech_radar.py --no-push
"""
from __future__ import annotations

import io
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ET = ZoneInfo("America/New_York")
UNIV_CACHE = ROOT / "scripts" / ".biotech_universe.txt"
XBI_URL = ("https://www.ssga.com/us/en/intermediary/etfs/library-content/products/"
           "fund-data/etfs/us/holdings-daily-us-en-xbi.xlsx")
_UA = {"User-Agent": "Mozilla/5.0 (biotech-radar research)"}
UNIV_MAX_AGE_DAYS = 7


def fetch_universe() -> list[str]:
    """XBI holdings tickers from State Street's daily file; cache weekly, fall back to cache."""
    if UNIV_CACHE.exists():
        age = (datetime.now().timestamp() - UNIV_CACHE.stat().st_mtime) / 86400
        if age < UNIV_MAX_AGE_DAYS:
            return [s.strip().upper() for s in UNIV_CACHE.read_text().splitlines() if s.strip()]
    try:
        req = urllib.request.Request(XBI_URL, headers=_UA)
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        raw = pd.read_excel(io.BytesIO(data), header=None)
        # the real header row has BOTH "Ticker" and "Weight" (row 1 only has the metadata
        # label "Ticker Symbol:" -- don't match that)
        def _is_hdr(row):
            s = row.astype(str).str.lower()
            return s.str.contains("ticker").any() and s.str.contains("weight").any()
        hdr = next(i for i, row in raw.iterrows() if _is_hdr(row))
        df = pd.read_excel(io.BytesIO(data), skiprows=hdr)
        tcol = next(c for c in df.columns if str(c).strip().lower() == "ticker")
        tickers = sorted({str(t).strip().upper() for t in df[tcol]
                          if isinstance(t, str) and t.strip().isalpha() and 1 <= len(t.strip()) <= 5})
        if tickers:
            UNIV_CACHE.write_text("\n".join(tickers))
            return tickers
    except Exception as e:
        print(f"[warn] XBI holdings fetch failed ({str(e)[:70]}); using cache if present")
    if UNIV_CACHE.exists():
        return [s.strip().upper() for s in UNIV_CACHE.read_text().splitlines() if s.strip()]
    return []


def heat_scan(tickers: list[str]) -> pd.DataFrame:
    """Per name: volume-building, momentum, volatility, 52w-high proximity -> a HEAT score
    (mean of cross-sectional percentile ranks). All lookahead-free (uses completed bars)."""
    import yfinance as yf
    px = yf.download(tickers, period="6mo", auto_adjust=True, progress=False, threads=True)
    if px is None or px.empty:
        return pd.DataFrame()
    close = px["Close"] if isinstance(px.columns, pd.MultiIndex) else px[["Close"]]
    vol = px["Volume"] if isinstance(px.columns, pd.MultiIndex) else px[["Volume"]]
    high = px["High"] if isinstance(px.columns, pd.MultiIndex) else px[["High"]]
    rows = []
    for t in close.columns:
        c = close[t].dropna(); v = vol[t].reindex(c.index); h = high[t].reindex(c.index)
        if len(c) < 30:
            continue
        v20 = v.tail(20).mean()
        rows.append({
            "symbol": t, "price": round(float(c.iloc[-1]), 2),
            "vol_build": round(float(v.tail(5).mean() / v20), 2) if v20 else None,   # 5d vs 20d volume
            "rvol_today": round(float(v.iloc[-1] / v20), 2) if v20 else None,
            "ret_5d": round(float(c.iloc[-1] / c.iloc[-6] - 1) * 100, 1) if len(c) > 6 else None,
            "ret_20d": round(float(c.iloc[-1] / c.iloc[-21] - 1) * 100, 1) if len(c) > 21 else None,
            "realized_vol": round(float(c.pct_change().tail(20).std()), 4),
            "near_high": round(float(c.iloc[-1] / h.tail(252).max()), 3),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # HEAT = blend of percentile ranks (volume-building + short-term momentum + volatility
    # + nearness to highs). Volume-building weighted double -- the clearest "brewing" tell.
    for col in ["vol_build", "rvol_today", "ret_5d", "ret_20d", "realized_vol", "near_high"]:
        df[f"p_{col}"] = df[col].rank(pct=True)
    df["heat"] = (2 * df["p_vol_build"] + df["p_ret_5d"] + df["p_ret_20d"]
                  + df["p_realized_vol"] + df["p_near_high"]) / 6.0
    return df.sort_values("heat", ascending=False).reset_index(drop=True)


def enrich_short(symbols: list[str]) -> dict:
    """Short %float + days-to-cover (squeeze fuel) for the heat leaders. yfinance, slow-ish."""
    import yfinance as yf
    out = {}
    for t in symbols:
        try:
            info = yf.Ticker(t).info
            out[t] = {"short_pct_float": info.get("shortPercentOfFloat"),
                      "short_ratio": info.get("shortRatio")}
        except Exception:
            out[t] = {}
    return out


def company_names(tickers: list[str]) -> dict:
    """{ticker: company name} via yfinance (for clinicaltrials sponsor matching)."""
    import yfinance as yf
    out = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            out[t] = info.get("longName") or info.get("shortName") or t
        except Exception:
            out[t] = t
    return out


_CT_URL = "https://clinicaltrials.gov/api/v2/studies"
_CT_OK = ("RECRUITING", "ACTIVE_NOT_RECRUITING", "ENROLLING_BY_INVITATION", "NOT_YET_RECRUITING")


def upcoming_catalysts(sponsor: str, max_n: int = 3) -> list:
    """Soonest FUTURE Phase 2/3 trial completions for a sponsor (clinicaltrials.gov).
    NOTE: primary-completion date is an APPROXIMATE proxy — it lags the actual data readout,
    and this misses PDUFA/AdCom/conference catalysts. Context, not a precise readout date."""
    import json
    import urllib.parse
    import urllib.request
    from datetime import date
    sp = sponsor                       # trim corporate suffixes -> better sponsor match
    for suf in (", Inc.", " Inc.", " Inc", ", Corp.", " Corporation", " Corp", " Ltd",
                " plc", " Co.", " Company", ", Ltd."):
        sp = sp.replace(suf, "")
    sp = sp.strip().rstrip(",")
    # include Phase 1 too — early-stage data readouts are major catalysts for small biotech
    term = f'AREA[LeadSponsorName]"{sp}" AND (AREA[Phase]PHASE1 OR AREA[Phase]PHASE2 OR AREA[Phase]PHASE3)'
    params = {"query.term": term, "pageSize": "40", "format": "json",
              "fields": "BriefTitle,Phase,OverallStatus,PrimaryCompletionDate"}
    try:
        url = _CT_URL + "?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=20) as r:
            d = json.load(r)
    except Exception:
        return []
    today = date.today().isoformat()
    rows = []
    for s in d.get("studies", []):
        p = s.get("protocolSection", {})
        st = p.get("statusModule", {})
        pcd = st.get("primaryCompletionDateStruct", {}).get("date")
        if pcd and pcd >= today and st.get("overallStatus") in _CT_OK:
            ph = ",".join(p.get("designModule", {}).get("phases", []))
            rows.append((pcd, ph, p.get("identificationModule", {}).get("briefTitle", "")[:46]))
    return sorted(rows)[:max_n]


def render_cards(top: pd.DataFrame, names: dict, n: int = 8) -> None:
    """Per top candidate: why it's flagged, upcoming catalysts, and a SUGGESTED risk structure."""
    print(f"\n{'='*72}\nTRADE CARDS — top {n} heating-up biotechs  (SPECULATIVE — read the risk note)\n{'='*72}")
    for _, r in top.head(n).iterrows():
        s = r["symbol"]; nm = names.get(s, s); px = float(r["price"])
        why = []
        if r["vol_build"] and r["vol_build"] >= 1.3:
            why.append(f"volume building {r['vol_build']:.1f}x")
        if r["ret_5d"] and r["ret_5d"] > 0:
            why.append(f"+{r['ret_5d']:.0f}% 5d")
        if r["ret_20d"] and r["ret_20d"] > 0:
            why.append(f"+{r['ret_20d']:.0f}% 20d")
        if pd.notna(r.get("short%float")) and r.get("short%float"):
            why.append(f"{r['short%float']*100:.0f}% short float")
        extended = r["near_high"] >= 0.95
        print(f"\n{s}  ({nm})  ${px:.2f}   heat {r['heat']:.2f}")
        print(f"  why flagged: {', '.join(why) or '—'}  ·  near 52w-high {r['near_high']:.2f} "
              f"{'(EXTENDED — wait for a pullback, less room)' if extended else '(room to run)'}")
        cats = upcoming_catalysts(nm)
        if cats:
            print("  catalysts (est. trial-completion — LAGS actual readout):")
            for pcd, ph, title in cats:
                print(f"     {pcd}  {ph:12} {title}")
        else:
            print("  catalysts: none via clinicaltrials (may have PDUFA/AdCom/data-update catalysts not listed)")
        print("  SUGGESTED STRUCTURE (speculative capital only — expect most to lose):")
        print(f"     hard stop -25% (~${px*0.75:.2f})  ·  trailing stop 28% (WIDE — let a moonshot run)")
        print(f"     size tiny (~$200-300, <=5 names concurrent)  ·  time-stop ~10-15 sessions if flat")


def _ntfy(msg: str, title: str) -> None:
    f = ROOT / ".env"
    topic = None
    if f.exists():
        for line in f.read_text().splitlines():
            if line.strip().startswith("NTFY_TOPIC"):
                topic = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not topic:
        return
    try:
        urllib.request.urlopen(urllib.request.Request(
            f"https://ntfy.sh/{topic}", data=msg.encode(),
            headers={"Title": title, "Tags": "dna", "Priority": "3"}), timeout=8)
    except Exception:
        pass


def main(argv) -> int:
    push = "--no-push" not in argv
    date = datetime.now(ET).date().isoformat()
    uni = fetch_universe()
    print(f"biotech radar {date}: XBI universe = {len(uni)} names")
    if not uni:
        print("no universe — aborting."); return 1
    df = heat_scan(uni)
    if df.empty:
        print("no bar data — aborting."); return 1

    top = df.head(15).copy()
    sh = enrich_short(list(top["symbol"]))
    top["short%float"] = [(_v.get("short_pct_float") if (_v := sh.get(s)) else None) for s in top["symbol"]]

    print(f"\n=== TOP 15 HEATING-UP biotechs (of {len(df)}) — volume building + momentum ===")
    cols = ["symbol", "price", "heat", "vol_build", "rvol_today", "ret_5d", "ret_20d",
            "realized_vol", "near_high", "short%float"]
    show = top[cols].copy()
    show["heat"] = show["heat"].round(2)
    print(show.to_string(index=False))

    out = ROOT / "scripts" / f"biotech_radar_{date}.csv"
    df.to_csv(out, index=False)
    print(f"\n-> {out.name} (full {len(df)} ranked). HEAT = volume-building + momentum + vol "
          "+ near-high (percentile blend). Surge-PRONE, NOT direction — binary catalyst risk.")

    if "--no-cards" not in argv:
        names = company_names(list(top["symbol"].head(8)))
        render_cards(top, names, n=8)
        print(f"\n{'-'*72}")
        print("RISK NOTE: biotech catalysts are BINARY (good data +100-300%, bad data -60-90%).")
        print("This flags WHERE a surge may brew + the trial timeline — NOT direction. Backtest")
        print("showed positive expectancy ONLY on survivorship-biased data (real edge likely ~0).")
        print("Treat as SPECULATION: size you can lose, hard stop every position, no averaging down.")

    if push and not top.empty:
        lines = [f"Biotech radar {date} — heating up:"]
        for _, r in top.head(6).iterrows():
            sp = f" SI{r['short%float']*100:.0f}%" if pd.notna(r["short%float"]) else ""
            lines.append(f"{r['symbol']} ${r['price']:.0f} vol{r['vol_build']:.1f}x "
                         f"5d{r['ret_5d']:+.0f}%{sp}")
        _ntfy("\n".join(lines), f"Biotech radar {date}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
