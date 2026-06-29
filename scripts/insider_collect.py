"""Insider open-market BUY collector (research data — NOT a trading signal).

Builds a persistent, append-only log of every Form 4 open-market purchase (Code 'P')
across the trading universe + expansion, so we can run a proper LIVE forward-return
study later (much bigger / cleaner sample than the one-off backtest in
backtest/insider_cluster_probe.py, which found broad insider buying had a modest
+0.7-1.0% market-adjusted edge at 10-20d, n=274).

Design:
  - Incremental: each run fetches only Form 4s filed in the last `lookback_days`
    (default 14 — a weekly cadence + slack for late filings); dedupe on
    (symbol, filing_date, insider, shares) so overlapping windows don't double-log.
  - Persistent: appends to scripts/insider_buys_log.csv. The buy events themselves are
    permanent public record on EDGAR, so the log is regenerable (hence gitignored).
  - `first_seen` records the scan date we first logged a buy = the live, lookahead-free
    timestamp for the forward-return study (vs filing_date which can lag the trade).

Standalone (full universe, ~3-5 min incremental):
    .venv-openbb/Scripts/python.exe scripts/insider_collect.py
    .venv-openbb/Scripts/python.exe scripts/insider_collect.py --seed     # one-time backfill from backtest cache
    .venv-openbb/Scripts/python.exe scripts/insider_collect.py --limit 20  # quick test

Also importable: collect(universe) appends + returns the fresh events; summary_30d()
gives {sym: (n_buys, usd)} for enrichment columns. Works under .venv OR .venv-openbb
(edgartools only — no alpaca/yfinance needed at collection time).
"""
from __future__ import annotations

import pickle
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

ET = ZoneInfo("America/New_York")
LOG = ROOT / "scripts" / "insider_buys_log.csv"
CLUSTER_CACHE = ROOT / "backtest" / ".insider_cluster_cache.pkl"
COLS = ["symbol", "filing_date", "insider", "shares", "price", "usd", "first_seen"]
KEY = ["symbol", "filing_date", "insider", "shares"]
IDENTITY = "news-edge research maciej.mikulewicz@gmail.com"


def _universe() -> list[str]:
    from accumulation_scan import UNIVERSE, EXPANSION
    return sorted(set(UNIVERSE) | set(EXPANSION))


def load_log() -> pd.DataFrame:
    if LOG.exists():
        df = pd.read_csv(LOG, dtype={"symbol": str, "insider": str})
        for c in COLS:
            if c not in df.columns:
                df[c] = pd.NA
        return df[COLS]
    return pd.DataFrame(columns=COLS)


def _dedupe_append(existing: pd.DataFrame, fresh_rows: list[dict]) -> tuple[pd.DataFrame, int]:
    """Append only rows whose KEY isn't already in the log. Returns (new_log, n_added)."""
    if not fresh_rows:
        return existing, 0

    def _key(df):
        return df[KEY].apply(lambda r: "|".join(str(v) for v in r), axis=1)

    fresh = pd.DataFrame(fresh_rows)[COLS]
    fresh["_k"] = _key(fresh)
    if not existing.empty:
        seen = set(_key(existing))
        fresh = fresh[~fresh["_k"].isin(seen)]
    # also dedupe within this batch
    fresh = fresh.drop_duplicates(subset="_k").drop(columns="_k")
    out = pd.concat([existing, fresh], ignore_index=True) if not fresh.empty else existing
    return out, len(fresh)


def recent_insider_buys(symbols, lookback_days: int = 14, scan_date: str | None = None,
                        progress: bool = True) -> list[dict]:
    """Open-market buys (Code 'P') filed in the last `lookback_days`, one row per buy line."""
    from edgar import set_identity, Company
    set_identity(IDENTITY)
    scan_date = scan_date or datetime.now(ET).date().isoformat()
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=lookback_days)
    events: list[dict] = []
    for i, sym in enumerate(symbols, 1):
        try:
            recent = [f for f in Company(sym).get_filings(form="4")
                      if f.filing_date and pd.Timestamp(f.filing_date) >= cutoff]
            for filing in recent:
                try:
                    obj = filing.obj()
                    insider = str(getattr(obj, "insider_name", "") or "?")
                    mt = obj.market_trades
                    if mt is None or mt.empty:
                        continue
                    for _, t in mt.iterrows():
                        if str(t.get("Code")) == "P":
                            sh = round(float(t.get("Shares") or 0), 2)
                            px = round(float(t.get("Price") or 0), 4)
                            events.append({"symbol": sym,
                                           "filing_date": pd.Timestamp(filing.filing_date).date().isoformat(),
                                           "insider": insider, "shares": sh, "price": px,
                                           "usd": round(sh * px, 2), "first_seen": scan_date})
                except Exception:
                    continue
        except Exception:
            continue
        if progress and i % 40 == 0:
            print(f"  ...{i}/{len(symbols)} scanned, {len(events)} buys so far", flush=True)
    return events


def collect(symbols=None, lookback_days: int = 14, progress: bool = True) -> int:
    """Fetch recent buys, append new ones to the persistent log. Returns n added."""
    symbols = symbols or _universe()
    events = recent_insider_buys(symbols, lookback_days=lookback_days, progress=progress)
    log, n_added = _dedupe_append(load_log(), events)
    if n_added:
        log.to_csv(LOG, index=False)
    return n_added


def summary_30d(as_of: str | None = None) -> dict:
    """{sym: (n_buys, total_usd)} for buys with filing_date in the last 30 calendar days."""
    log = load_log()
    if log.empty:
        return {}
    cut = (pd.Timestamp(as_of) if as_of else pd.Timestamp.now()).normalize() - pd.Timedelta(days=30)
    log["fd"] = pd.to_datetime(log["filing_date"], errors="coerce")
    recent = log[log["fd"] >= cut]
    out = {}
    for sym, g in recent.groupby("symbol"):
        out[str(sym)] = (len(g), round(float(g["usd"].fillna(0).sum()), 0))
    return out


def seed_from_cluster_cache() -> int:
    """One-time backfill from the backtest cluster cache ({sym:[(ts,insider,usd),...]})."""
    if not CLUSTER_CACHE.exists():
        print("no cluster cache to seed from")
        return 0
    data = pickle.load(open(CLUSTER_CACHE, "rb"))
    rows = []
    for sym, blist in data.items():
        for (ts, insider, usd) in (blist or []):
            rows.append({"symbol": sym, "filing_date": pd.Timestamp(ts).date().isoformat(),
                         "insider": str(insider or "?"), "shares": pd.NA, "price": pd.NA,
                         "usd": round(float(usd or 0), 2), "first_seen": "seed"})
    log, n_added = _dedupe_append(load_log(), rows)
    if n_added:
        log.to_csv(LOG, index=False)
    return n_added


def main(argv) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", action="store_true", help="one-time backfill from backtest cluster cache")
    ap.add_argument("--lookback", type=int, default=14, help="days of Form 4s to fetch")
    ap.add_argument("--limit", type=int, default=0, help="cap universe (quick test)")
    args = ap.parse_args(argv[1:])

    if args.seed:
        n = seed_from_cluster_cache()
        print(f"seeded {n} historical buys from cluster cache -> {LOG.name} (total {len(load_log())})")

    universe = _universe()
    if args.limit:
        universe = universe[:args.limit]
    print(f"insider-buy collect: scanning {len(universe)} names (last {args.lookback}d) ...", flush=True)
    n = collect(universe, lookback_days=args.lookback)
    log = load_log()
    print(f"\n+{n} new buys logged.  log now {len(log)} buys across {log['symbol'].nunique()} names "
          f"({LOG.name}).")
    s = summary_30d()
    if s:
        top = sorted(s.items(), key=lambda kv: kv[1][1], reverse=True)[:12]
        print("\ntop names by insider $ bought (last 30d):")
        for sym, (n_b, usd) in top:
            print(f"  {sym:6} {n_b} buys  ${usd:,.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
