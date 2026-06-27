"""Lottery hype board -- morning entrypoint (forward test, NO LLM, logs only).

Builds the daily candidate net (WSB 25 + StockTwits 30 + >3% gappers), filters to
Alpaca-tradable equities priced >= $1, computes all available signals (1-3+6 live day 1;
squeeze + UOA come online days 3-4), and logs the top-K per signal + the seeded random
basket + the gapper control to experiments/lottery/picks/<date>.json (IMMUTABLE).

Schema per README. combined_score = mean percentile rank across non-null signals (fixed,
never tuned). The picks file refuses to overwrite an existing day (like newsedge cmd_log).

Run:
    .venv/Scripts/python.exe experiments/lottery/board.py
    .venv/Scripts/python.exe experiments/lottery/board.py --dry   # print, don't write
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from backtest.run_orb import load_env  # noqa: E402
import experiments.lottery.sources as S  # noqa: E402

ET = ZoneInfo("America/New_York")
HERE = Path(__file__).resolve().parent
PICKS_DIR = HERE / "picks"
TOP_K = 5
MIN_PRICE = 1.0


def _tradable_filter(symbols: list[str]) -> dict:
    """Return {symbol: last_price} for Alpaca-tradable US equities priced >= MIN_PRICE.
    Junk/crypto/warrant tickers from apewisdom get dropped here. On any API failure,
    returns {} (board still logs the signals it has; price filter just won't apply)."""
    import os
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import AssetClass, AssetStatus
    from alpaca.trading.requests import GetAssetsRequest
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockSnapshotRequest
    from alpaca.data.enums import DataFeed
    load_env()
    syms = sorted({s.upper() for s in symbols if s and s.isalpha()})
    if not syms:
        return {}
    try:
        tc = TradingClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"],
                           paper=True)
        assets = tc.get_all_assets(GetAssetsRequest(
            asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE))
        tradable = {a.symbol for a in assets if a.tradable and a.symbol in set(syms)}
    except Exception as e:
        print(f"[warn] asset list fetch failed ({str(e)[:60]}) -- skipping tradable filter")
        tradable = set(syms)
    cand = sorted(tradable)
    if not cand:
        return {}
    prices: dict[str, float] = {}
    try:
        dc = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"],
                                       os.environ["ALPACA_SECRET_KEY"])
        snaps = dc.get_stock_snapshot(StockSnapshotRequest(symbol_or_symbols=cand,
                                                           feed=DataFeed.IEX))
        for sym, sn in (snaps or {}).items():
            try:
                bar = sn.minute_bar or sn.daily_bar or sn.previous_daily_bar
                px = float(bar.close)
                if px >= MIN_PRICE:
                    prices[sym] = round(px, 2)
            except Exception:
                continue
    except Exception as e:
        print(f"[warn] price snapshot failed ({str(e)[:60]}) -- keeping all tradable")
        prices = {s: None for s in cand}
    return prices


def _percentile_rank(values: dict, sym: str) -> float | None:
    """Percentile rank of sym's value within the dict (higher value -> higher rank)."""
    vals = [v for v in values.values() if v is not None]
    v = values.get(sym)
    if v is None or not vals:
        return None
    below = sum(1 for x in vals if x < v)
    return round(below / len(vals), 4)


def build_board() -> dict:
    print("=== lottery board: building candidate net ===")
    # --- 1. candidate net ---
    wsb = S.reddit_trending(25)
    wsb_rows = [r for r in wsb if isinstance(r, dict)]
    wsb_syms = [r["ticker"] for r in wsb_rows]
    st = S.st_trending(30)
    st_syms = [s for s in st if isinstance(s, str) and not s.startswith("_error")]
    gappers = S.pm_gappers(3.0, 40)
    gap_rows = [g for g in gappers if isinstance(g, dict) and "symbol" in g]
    gap_syms = [g["symbol"] for g in gap_rows]
    random_syms = S.random_basket()       # seeded by YYYYMMDD
    # v1.1 (2026-06-27): extra subreddits as NEW MEASURED signals (additive — they get their
    # own scoreboard clock but do NOT enter combined_score, so the bot's traded picks are
    # unchanged until they prove out). r/pennystocks + r/Shortsqueeze surface explosive
    # small-cap/squeeze names WSB misses.
    penny_rows = [r for r in S.reddit_sub_trending("pennystocks", 25) if isinstance(r, dict)]
    squeeze_rows = [r for r in S.reddit_sub_trending("Shortsqueeze", 25) if isinstance(r, dict)]

    candidates = sorted(set(wsb_syms) | set(st_syms) | set(gap_syms))
    print(f"candidates: WSB {len(wsb_syms)}, ST {len(st_syms)}, gappers {len(gap_syms)} "
          f"-> {len(candidates)} unique")

    # --- 2. tradable + price >= $1 filter (random basket already from liquid universe) ---
    prices = _tradable_filter(candidates)
    tradable = sorted(prices.keys())
    print(f"tradable >= ${MIN_PRICE:.0f}: {len(tradable)}")

    # all names we want signals for: tradable candidates + random + gapper-control
    all_syms = sorted(set(tradable) | set(random_syms) | set(gap_syms))

    # v1.1 extra-subreddit signals, tradable-filtered SEPARATELY so they do NOT enter
    # all_syms — keeps existing signals / combined_score / the bot's picks identical.
    penny_surge = {r["ticker"]: r.get("surge") for r in penny_rows}
    squeeze_sub_surge = {r["ticker"]: r.get("surge") for r in squeeze_rows}
    _sub_names = sorted(set(penny_surge) | set(squeeze_sub_surge))
    _sub_extra_prices = _tradable_filter([s for s in _sub_names if s not in prices]) if _sub_names else {}
    sub_eligible = (set(prices) | set(_sub_extra_prices)) & set(_sub_names)   # tradable >= $1

    # --- 3. signals ---
    print("computing signals (ignition + premarket rvol live; squeeze/uoa graceful-None) ...")
    ig = S.ignition_scores(all_syms)
    if "_error" in ig:
        print(f"[warn] ignition: {ig['_error']}")
        ig = {}
    rvol = S.pm_rvol(all_syms) if all_syms else {}
    if "_error" in rvol:
        print(f"[warn] pm_rvol: {rvol['_error']}")
        rvol = {}
    sq = S.squeeze_scores(all_syms)
    uoa = S.uoa_snapshot(all_syms, update_state=True)
    if "_error" in uoa:
        uoa = {}

    wsb_surge = {r["ticker"]: r.get("surge") for r in wsb_rows}
    wsb_rank = {r["ticker"]: r.get("rank") for r in wsb_rows}
    st_rank = {s: i + 1 for i, s in enumerate(st_syms)}   # 1 = top trending
    gap_pct = {g["symbol"]: g.get("gap_pct") for g in gap_rows}

    def signals_for(sym: str) -> dict:
        igv = ig.get(sym, {})
        rv = rvol.get(sym, {})
        sqv = sq.get(sym, {})
        uv = uoa.get(sym, {})
        return {
            "wsb_surge": wsb_surge.get(sym),
            "wsb_rank": wsb_rank.get(sym),
            "st_rank": st_rank.get(sym),
            "pm_rvol": rv.get("pm_rvol"),
            "gap_pct": gap_pct.get(sym),
            "squeeze": sqv.get("squeeze"),
            "uoa_z": uv.get("uoa_z"),
            "ignition": igv.get("ignition"),
            "penny_surge": penny_surge.get(sym),          # v1.1 measured-only (not in combined)
            "squeeze_sub_surge": squeeze_sub_surge.get(sym),
        }

    # --- 4. combined_score: mean percentile rank across non-null signals ---
    # For each scorable signal, build the value-dict over all names so percentile is
    # cross-sectional. st_rank/wsb_rank are "lower is better" -> invert by negating.
    sig_pools: dict[str, dict] = {
        "wsb_surge": {s: signals_for(s)["wsb_surge"] for s in all_syms},
        "wsb_rank": {s: (-r if r is not None else None) for s, r in
                     ((x, signals_for(x)["wsb_rank"]) for x in all_syms)},
        "st_rank": {s: (-r if r is not None else None) for s, r in
                    ((x, signals_for(x)["st_rank"]) for x in all_syms)},
        "pm_rvol": {s: signals_for(s)["pm_rvol"] for s in all_syms},
        "gap_pct": {s: (abs(g) if g is not None else None) for s, g in
                    ((x, signals_for(x)["gap_pct"]) for x in all_syms)},
        "squeeze": {s: signals_for(s)["squeeze"] for s in all_syms},
        "uoa_z": {s: signals_for(s)["uoa_z"] for s in all_syms},
        "ignition": {s: signals_for(s)["ignition"] for s in all_syms},
    }

    def combined(sym: str) -> float | None:
        ranks = []
        for pool in sig_pools.values():
            pr = _percentile_rank(pool, sym)
            if pr is not None:
                ranks.append(pr)
        return round(sum(ranks) / len(ranks), 4) if ranks else None

    # --- 5. assemble baskets: top-K per signal + random + control ---
    def top_k(pool: dict, k: int = TOP_K) -> list[str]:
        scored = [(s, v) for s, v in pool.items() if v is not None and s in set(tradable)]
        scored.sort(key=lambda x: -x[1])
        return [s for s, _ in scored[:k]]

    basket_members: dict[str, set] = {
        "wsb": set(top_k(sig_pools["wsb_surge"])) | set(top_k(sig_pools["wsb_rank"])),
        "stocktwits": set(top_k(sig_pools["st_rank"])),
        "gappers": set(gap_syms),
        "random": set(random_syms),
        "control": set(gap_syms),     # the mechanical >3% gapper control basket
    }
    # which SIGNAL top-K'd each name (for top_k_of)
    signal_topk = {
        "wsb": set(top_k(sig_pools["wsb_surge"])) | set(top_k(sig_pools["wsb_rank"])),
        "stocktwits": set(top_k(sig_pools["st_rank"])),
        "pm_rvol": set(top_k(sig_pools["pm_rvol"])),
        "squeeze": set(top_k(sig_pools["squeeze"])),
        "uoa": set(top_k(sig_pools["uoa_z"])),
        "ignition": set(top_k(sig_pools["ignition"])),
    }
    # v1.1 NEW measured signals (additive): top-K by mention surge over tradable sub names.
    def _topk_surge(surge_map: dict, k: int = TOP_K) -> set:
        scored = [(s, v) for s, v in surge_map.items() if v is not None and s in sub_eligible]
        scored.sort(key=lambda x: -x[1])
        return {s for s, _ in scored[:k]}
    signal_topk["pennystocks"] = _topk_surge(penny_surge)
    signal_topk["shortsqueeze"] = _topk_surge(squeeze_sub_surge)
    basket_members["pennystocks"] = signal_topk["pennystocks"]
    basket_members["shortsqueeze"] = signal_topk["shortsqueeze"]

    # build the pick list. A symbol gets ONE row; basket = its primary basket (priority
    # wsb > stocktwits > gappers > random > control), top_k_of lists ALL signals that
    # flagged it.
    basket_priority = ["wsb", "stocktwits", "gappers", "random", "control",
                       "pennystocks", "shortsqueeze"]   # new baskets lowest priority
    all_picks_syms = sorted(set().union(*basket_members.values()))
    picks = []
    for sym in all_picks_syms:
        primary = next((b for b in basket_priority if sym in basket_members[b]), "random")
        flagged = sorted(name for name, members in signal_topk.items() if sym in members)
        picks.append({
            "symbol": sym,
            "basket": primary,
            "top_k_of": flagged,
            "signals": signals_for(sym),
            "combined_score": combined(sym),
            "ret_945_close": None,
            "ret_1d": None,
            "ret_3d": None,
        })

    today = datetime.now(ET).date().isoformat()
    return {
        "date": today,
        "logged_at": datetime.now(ET).isoformat(timespec="seconds"),
        "n_candidates": len(candidates),
        "n_tradable": len(tradable),
        "picks": picks,
    }


def main(argv) -> int:
    dry = "--dry" in argv
    rec = build_board()
    counts: dict[str, int] = {}
    for p in rec["picks"]:
        counts[p["basket"]] = counts.get(p["basket"], 0) + 1
    print(f"\nboard {rec['date']}: {len(rec['picks'])} picks "
          + ", ".join(f"{b}={n}" for b, n in sorted(counts.items())))
    if dry:
        print(json.dumps(rec, indent=2)[:3000])
        print("\n[--dry] not written.")
        return 0
    PICKS_DIR.mkdir(parents=True, exist_ok=True)
    out = PICKS_DIR / f"{rec['date']}.json"
    if out.exists():
        print(f"refusing to overwrite existing {out.name} (immutable record).")
        return 1
    json.dump(rec, open(out, "w"), indent=2)
    print(f"logged -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
