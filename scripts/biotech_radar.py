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
import json
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROBS_FILE = ROOT / "backtest" / "biotech_signal_probs.json"
SNAPSHOT = ROOT / "live" / "biotech_radar_latest.json"
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
        w40 = c.tail(40)
        lo40, hi40, mn40 = float(w40.min()), float(w40.max()), float(w40.mean())
        rows.append({
            "symbol": t, "price": round(float(c.iloc[-1]), 2),
            "vol_build": round(float(v.tail(5).mean() / v20), 2) if v20 else None,   # 5d vs 20d volume
            "rvol_today": round(float(v.iloc[-1] / v20), 2) if v20 else None,
            "ret_5d": round(float(c.iloc[-1] / c.iloc[-6] - 1) * 100, 1) if len(c) > 6 else None,
            "ret_20d": round(float(c.iloc[-1] / c.iloc[-21] - 1) * 100, 1) if len(c) > 21 else None,
            "realized_vol": round(float(c.pct_change().tail(20).std()), 4),
            "near_high": round(float(c.iloc[-1] / h.tail(252).max()), 3),
            # consolidation footprint (the "run-up" setup — coiled BEFORE the pop)
            "tightness": round((hi40 - lo40) / mn40, 3) if mn40 else None,   # 40d range / price; lower=tighter
            "pos_in_range": round((float(c.iloc[-1]) - lo40) / (hi40 - lo40), 3) if hi40 > lo40 else None,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # HEAT = already-moving (volume-building x2 + momentum + vol + near-high) percentile blend.
    for col in ["vol_build", "rvol_today", "ret_5d", "ret_20d", "realized_vol", "near_high"]:
        df[f"p_{col}"] = df[col].rank(pct=True)
    df["heat"] = (2 * df["p_vol_build"] + df["p_ret_5d"] + df["p_ret_20d"]
                  + df["p_realized_vol"] + df["p_near_high"]) / 6.0
    # SETUP = pre-breakout "run-up" footprint: coiled (tight 40d range) + at range top +
    # volume just starting to build. De-emphasize names that already popped (ret_20d>40%) —
    # the run-up play wants them BEFORE the move, not after.
    df["p_tight"] = (1.0 / df["tightness"].replace(0, pd.NA)).rank(pct=True)   # tighter -> higher
    df["p_pos"] = df["pos_in_range"].rank(pct=True)
    df["setup"] = (df["p_tight"] + df["p_pos"] + df["p_vol_build"]) / 3.0
    df.loc[df["ret_20d"] > 40, "setup"] = df["setup"] * 0.5   # already extended -> not a fresh setup
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


CAP_LO, CAP_HI = 200e6, 1e9     # the surge "sweet spot" market-cap band ($200M-$1B)


def market_caps(tickers: list[str]) -> dict:
    """{ticker: market cap} via yfinance fast_info (lightweight, no full-info scrape)."""
    import yfinance as yf
    out = {}
    for t in tickers:
        try:
            cap = yf.Ticker(t).fast_info["market_cap"]
            out[t] = float(cap) if cap else None
        except Exception:
            out[t] = None
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
# platform-tech keywords: the ONE case where a Phase 1 readout can re-rate the whole pipeline
# (validates the engine, not one drug). Heuristic match on trial titles + company name.
_PLATFORM_KW = ("mrna", "crispr", "gene edit", "gene-edit", "base edit", "prime edit", "car-t",
                "car t", "cell therapy", "gene therapy", "aav ", "lipid nanoparticle", "sirna",
                "rnai", "oncolytic", "tcr ", "platform", "molecular glue", "degrader")


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


def load_probs() -> dict:
    """Backtest signal-bucket odds (P(+30%)/P(-30%) within 10d) from biotech_backtest.py."""
    try:
        return json.loads(PROBS_FILE.read_text())
    except Exception:
        return {}


def classify_bucket(vol_build, ret_5d_pct, near_high) -> str:
    """Map a live candidate to its backtest odds bucket (ret_5d is in PERCENT here)."""
    vb = vol_build or 0; r5 = ret_5d_pct or 0; nh = near_high or 0
    if vb >= 2 and r5 >= 10:
        return "hot"
    if vb >= 1.5 and r5 >= 5:
        return "building"
    if vb >= 2:
        return "vol2"
    if vb >= 1.5:
        return "vol15"
    if r5 >= 10:
        return "momo"
    if nh >= 0.95:
        return "extended"
    return "base"


def _parse_ct_date(s: str):
    """clinicaltrials dates come as YYYY-MM-DD, YYYY-MM, or YYYY -> a date (assume late)."""
    from datetime import date
    try:
        p = str(s).split("-")
        y = int(p[0]); m = int(p[1]) if len(p) > 1 else 12; d = int(p[2]) if len(p) > 2 else 28
        return date(y, m, d)
    except Exception:
        return None


def build_card(r: dict, nm: str, cap, probs: dict) -> dict:
    """One enriched card: odds, market-cap band, nearest (Phase-2-preferred) catalyst +
    days-away + run-up exit-by date, and the suggested run-up structure."""
    from datetime import date, timedelta
    s = r["symbol"]; px = float(r["price"])
    bkt = classify_bucket(r["vol_build"], r["ret_5d"], r["near_high"])
    bo = (probs.get("buckets", {}).get(bkt) or probs.get("base", {})) if probs else {}
    sp = float(r["short%float"]) if pd.notna(r.get("short%float")) else None
    in_band = bool(cap and CAP_LO <= cap <= CAP_HI)
    why = []
    if r.get("vol_build") and r["vol_build"] >= 1.3:
        why.append(f"vol building {r['vol_build']:.1f}x")
    if r.get("pos_in_range") is not None and r["pos_in_range"] >= 0.8 and (r.get("ret_20d") or 0) < 40:
        why.append("coiled near range-top")
    if r.get("ret_5d") and r["ret_5d"] > 0:
        why.append(f"+{r['ret_5d']:.0f}% 5d")
    if sp:
        why.append(f"{sp*100:.0f}% short")
    # CATALYST HIERARCHY (per the playbook): Phase 2 efficacy = the run-up sweet spot;
    # Phase 1 = safety/dosage (~3% uplift, WEAK) UNLESS a platform tech (then it can re-rate
    # the whole pipeline); Phase 3 = largely priced-in + binary gap-down risk.
    today = date.today()
    dated = []
    for c0, c1, c2 in upcoming_catalysts(nm):
        dt = _parse_ct_date(c0)
        if dt and dt >= today:
            dated.append({"d": (dt - today).days, "date": c0, "phase": c1, "title": c2,
                          "exit_by": (dt - timedelta(days=7)).isoformat()})
    platform = (any(any(k in (x["title"] or "").lower() for k in _PLATFORM_KW) for x in dated)
                or any(k in nm.lower() for k in _PLATFORM_KW))

    def soonest(pred):
        xs = [x for x in dated if pred((x["phase"] or "").upper())]
        return min(xs, key=lambda x: x["d"]) if xs else None
    ph2 = soonest(lambda p: "PHASE2" in p)
    ph1 = soonest(lambda p: "PHASE1" in p and "PHASE2" not in p and "PHASE3" not in p)
    ph3 = soonest(lambda p: "PHASE3" in p and "PHASE2" not in p)
    near = lambda x: x is not None and 14 <= x["d"] <= 150
    days = nearest_phase = exit_by = cat_quality = None
    if near(ph2):
        days, nearest_phase, exit_by, cat_quality = ph2["d"], ph2["phase"], ph2["exit_by"], "phase2"
        structure = (f"RUN-UP PLAY (Phase 2 efficacy ~{days}d = the sweet spot) - buy the run-up, "
                     f"TARGET EXIT ~{exit_by} (~1wk BEFORE data); don't hold the binary.")
    elif near(ph1) and platform:
        days, nearest_phase, cat_quality = ph1["d"], ph1["phase"], "phase1_platform"
        structure = (f"Phase 1 PLATFORM ~{days}d - a platform (mRNA/CRISPR/CAR-T/degrader) Phase 1 "
                     f"CAN re-rate the whole pipeline. Speculative; smaller/earlier than a Phase 2 run-up.")
    elif near(ph1):
        days, nearest_phase, cat_quality = ph1["d"], ph1["phase"], "phase1_weak"
        structure = (f"Phase 1 only ~{days}d - SAFETY/dosage (~3% avg uplift). WEAK run-up - skip "
                     f"unless it's a platform play. NOT the Phase-2 efficacy setup.")
    elif near(ph3):
        days, nearest_phase, cat_quality = ph3["d"], ph3["phase"], "phase3"
        structure = (f"Phase 3 ~{days}d - largely PRICED IN + binary gap-down risk if it misses. "
                     f"Not a clean run-up.")
    else:
        anyc = min(dated, key=lambda x: x["d"]) if dated else None
        if anyc and anyc["d"] < 14:
            days, nearest_phase = anyc["d"], anyc["phase"]
            structure = (f"catalyst IMMINENT (~{days}d, {anyc['phase']}) - run-up priced in; "
                         f"entering now = holding the binary. AVOID.")
        else:
            structure = "no near-term Phase-2 catalyst - momentum/heat play only (higher uncertainty)."
    cats = [{"date": x["date"], "phase": x["phase"], "title": x["title"]} for x in dated]
    return {
        "symbol": s, "name": nm, "price": px,
        "heat": round(float(r["heat"]), 2), "setup": round(float(r.get("setup") or 0), 2),
        "near_high": round(float(r["near_high"]), 3), "extended": bool(r["near_high"] >= 0.95),
        "pos_in_range": r.get("pos_in_range"), "tightness": r.get("tightness"),
        "market_cap": cap, "in_band": in_band, "short_pct": sp,
        "bucket_label": bo.get("label", "baseline"), "p_up": bo.get("p_up"), "p_down": bo.get("p_down"),
        "why": ", ".join(why) or "—",
        "days_to_catalyst": days, "nearest_phase": nearest_phase, "exit_by": exit_by,
        "cat_quality": cat_quality, "platform": platform, "catalysts": cats[:4],
        "stop": round(px * 0.75, 2), "trail_pct": 28, "structure": structure,
    }


def _cap_str(cap):
    if not cap:
        return "cap ?"
    return f"${cap/1e9:.2f}B" if cap >= 1e9 else f"${cap/1e6:.0f}M"


def print_cards(setups: list, heat: list, probs: dict) -> None:
    base = probs.get("base", {}) if probs else {}
    fwd = probs.get("fwd_days", "?")
    print(f"\n{'='*74}\nRUN-UP SETUPS — coiled, $200M-$1B, catalyst approaching (buy BEFORE the pop)")
    print(f"odds = backtest hist. freq within {fwd}d (base +30% = {base.get('p_up',0)*100:.1f}%, "
          f"survivorship-inflated)\n{'='*74}")
    for c in setups:
        _print_one(c, fwd)
    print(f"\n{'-'*74}\nALREADY-HOT (moving now — may be late for a run-up entry):")
    for c in heat[:5]:
        print(f"  {c['symbol']:6} {_cap_str(c['market_cap']):>8} heat {c['heat']:.2f} "
              f"{'BAND' if c['in_band'] else '    '} {('+'+str(int(c['days_to_catalyst']))+'d cat' if c['days_to_catalyst'] is not None else 'no cat')} · {c['why']}")


def _print_one(c, fwd):
    pu = f"{c['p_up']*100:.0f}%" if c.get("p_up") is not None else "?"
    pdn = f"{c['p_down']*100:.0f}%" if c.get("p_down") is not None else "?"
    band = "[in $200M-$1B band]" if c["in_band"] else "[outside band]"
    print(f"\n{c['symbol']}  ({c['name']})  ${c['price']:.2f}  ·  {_cap_str(c['market_cap'])}  {band}")
    print(f"  ODDS [{c['bucket_label']}]: ~{pu} +30% / ~{pdn} -30% within {fwd}d (historical)")
    print(f"  setup {c['setup']:.2f} · heat {c['heat']:.2f} · {c['why']} · range-pos {c.get('pos_in_range')}")
    if c["days_to_catalyst"] is not None:
        print(f"  NEAREST CATALYST: ~{c['days_to_catalyst']}d away ({c['nearest_phase']}), est. readout window")
    if c["catalysts"]:
        for ct in c["catalysts"][:3]:
            print(f"     {ct['date']}  {ct['phase']:12} {ct['title']}")
    print(f"  PLAY: {c['structure']}")
    print(f"        stop -25% (~${c['stop']:.2f}) · trailing {c['trail_pct']}% · tiny size (~$200-300, <=5 names)")


def write_snapshot(date: str, setups: list, heat: list, probs: dict) -> None:
    """Persist the latest radar for the VM status page (committed + pulled by the VM)."""
    snap = {"date": date, "generated": datetime.now(ET).isoformat(timespec="seconds"),
            "probs_asof": probs.get("asof"), "surge_pct": probs.get("surge_pct"),
            "fwd_days": probs.get("fwd_days"), "base": probs.get("base"),
            "cap_band": [CAP_LO, CAP_HI], "setups": setups, "heat": heat}
    SNAPSHOT.parent.mkdir(exist_ok=True)
    SNAPSHOT.write_text(json.dumps(snap, indent=2, default=str))


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

    out = ROOT / "scripts" / f"biotech_radar_{date}.csv"
    df.to_csv(out, index=False)
    print(f"-> {out.name} (full {len(df)} ranked by heat + setup)")

    # candidate pool = top heat UNION top setup (catch both already-moving AND coiling names)
    pool = list(dict.fromkeys(list(df.head(15)["symbol"])
                              + list(df.sort_values("setup", ascending=False).head(15)["symbol"])))
    prows = df[df["symbol"].isin(pool)].copy()
    sh = enrich_short(pool)
    prows["short%float"] = prows["symbol"].map(lambda s: (sh.get(s) or {}).get("short_pct_float"))
    names = company_names(pool)
    caps = market_caps(pool)
    probs = load_probs()
    cards = [build_card(r, names.get(r["symbol"], r["symbol"]), caps.get(r["symbol"]), probs)
             for r in prows.to_dict("records")]

    # RUN-UP SETUPS (the advice's play): coiled pre-breakout, prefer in-band + not extended,
    # ranked so in-band names with a near-term catalyst float to the top.
    def _setup_key(c):
        # the run-up ENGINE is a near-term Phase 2 catalyst -> tier it first, then platform
        # Phase 1, then coiled-no-catalyst; within each tier prefer in-band, then setup score.
        q = c["cat_quality"]
        tier = 0 if q == "phase2" else (1 if q == "phase1_platform" else 2)
        return (tier, not c["in_band"], -c["setup"])
    setups = sorted([c for c in cards if not c["extended"]], key=_setup_key)[:8]
    heat = sorted(cards, key=lambda c: -c["heat"])[:8]

    if "--no-cards" not in argv:
        print_cards(setups, heat, probs)
        write_snapshot(date, setups, heat, probs)
        print(f"\nsnapshot -> {SNAPSHOT.name} (for the VM status page)")
        print(f"\n{'-'*74}")
        print("RISK NOTE: biotech catalysts are BINARY. The RUN-UP play (buy before, EXIT before")
        print("the readout) avoids the coin-flip — but run-ups aren't guaranteed + trial dates are")
        print("approximate (clinicaltrials completion lags actual readout). Position SIZE is the real")
        print("control (a fail gaps -60-90% through any stop). SPECULATION — size you can lose.")

    if push:
        best = setups or heat
        lines = [f"Biotech radar {date} — run-up setups:"]
        for c in best[:6]:
            cat = f" cat~{c['days_to_catalyst']}d" if c["days_to_catalyst"] is not None else ""
            lines.append(f"{c['symbol']} {_cap_str(c['market_cap'])}{'*' if c['in_band'] else ''} "
                         f"setup{c['setup']:.2f}{cat} +{c['p_up']*100:.0f}%odds" if c.get("p_up")
                         else f"{c['symbol']} setup{c['setup']:.2f}{cat}")
        _ntfy("\n".join(lines), f"Biotech radar {date}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
