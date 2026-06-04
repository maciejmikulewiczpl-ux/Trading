"""Does a daily pre-open RANKING of the universe beat scanning the static list?

The live runner scans a fixed ~100-name watchlist every day and fills its 8
concurrent-position slots first-come (whoever breaks out earliest). This asks
the question behind "pick the best frontrunners for scanning": if instead we
rank the universe each morning and only trade the top-K names, do we do better
than trading all 100 — and, crucially, better than picking K names at random?

Selection signals (all computable by 09:45 ET, strictly BEFORE any entry — the
ORB enters on the bar after a breakout, and breakouts are scanned only from
09:45 on, so none of these peek at entry-time information):
  - gap%     : today's 09:30 open vs yesterday's RTH close (pre-open).
  - |gap|    : absolute gap, either direction.
  - atr_pct  : 14-day ATR / prev close, shifted 1 session (pre-open, volatility).
  - or_rvol  : the opening range's (09:30-09:44) volume / its trailing-20-session
               average for that window, shifted 1 session. "Is this name in play
               this morning?" — known at 09:45, before the first possible entry.
  - composite: average cross-sectional percentile rank of or_rvol, |gap|, atr_pct.

Method: generate every broad-universe ORB signal once (shipped long-only params,
11:30 cutoff). Then, per selector and per K, keep only trades whose symbol is in
that day's top-K, and fill an 8-slot portfolio greedily by entry time (as live).
Compare to (a) the full-universe baseline and (b) a RANDOM top-K control averaged
over many seeds. A signal only earns its keep if it beats BOTH at the same K.
The winner is re-checked on a first-half / second-half OOS split.

Bias notes: the universe is the fixed listed-throughout set from universe_scan
(no survivorship / "today's movers" lookahead). Every signal is shifted or uses
only same-morning pre-entry data. $ figures assume the live $100 risk / $10k cap.

Run (first run fetches ~100 names of 1-min bars, a few minutes; then cached):
    .venv/Scripts/python.exe backtest/compare_selection.py
"""
from __future__ import annotations

import os
import pickle
import sys
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from strategies.orb import Params, Trade  # noqa: E402
from backtest.run_orb import (  # noqa: E402
    LOOKBACK_DAYS, STARTING_EQUITY, get_trading_days, load_env, run_backtest, to_et,
)
from backtest.universe_scan import UNIVERSE, fetch_chunked  # noqa: E402
from backtest.universe_portfolio import perf, portfolio  # noqa: E402

ET = ZoneInfo("America/New_York")

CAP = 8                       # live concurrency cap
KS = [10, 15, 20, 30]         # top-K daily watchlist sizes to test
RANDOM_SEEDS = 25             # random-control samples per K
ATR_WINDOW = 14
RVOL_WINDOW = 20
# Window is overridable so the same harness runs a longer, multi-regime check:
#   SELECT_LOOKBACK_DAYS=730 .venv/Scripts/python.exe backtest/compare_selection.py
LOOKBACK = int(os.environ.get("SELECT_LOOKBACK_DAYS", str(LOOKBACK_DAYS)))
CACHE = ROOT / "backtest" / f".bars_cache_univ_{LOOKBACK}d.pkl"

OR_START, OR_END = time(9, 30), time(9, 45)
RTH_START, RTH_END = time(9, 30), time(16, 0)


# --------------------------------------------------------------------------
def load_cached_universe():
    if CACHE.exists():
        print(f"Loading cached universe bars from {CACHE.name} ...")
        with open(CACHE, "rb") as f:
            d = pickle.load(f)
        return d["bars"], d["days"]
    load_env()
    key, sec = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    if not key or not sec:
        raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY not set in .env")
    dc = StockHistoricalDataClient(key, sec)
    tc = TradingClient(key, sec, paper=True)
    end = datetime.now(tz=ET)
    start = end - timedelta(days=LOOKBACK)
    days = get_trading_days(tc, start, end)
    print(f"No cache — fetching {len(UNIVERSE)} names, {len(days)} sessions "
          f"({LOOKBACK}d window, one-time, slow)...")
    bars = to_et(fetch_chunked(dc, UNIVERSE, start, end))
    with open(CACHE, "wb") as f:
        pickle.dump({"bars": bars, "days": days}, f)
    print(f"Cached to {CACHE.name}")
    return bars, days


def build_signals(all_bars: pd.DataFrame, present: list[str]) -> pd.DataFrame:
    """Tidy frame: one row per (date, symbol) with the pre-entry selection signals.

    All signals use only prior sessions (shift 1) or same-morning data available
    by 09:45 ET, so selecting on them is lookahead-free for entries at >= 09:45.
    """
    rows = []
    for sym in present:
        sb = all_bars.xs(sym, level=0)
        tt = sb.index.time
        rth = sb[(tt >= RTH_START) & (tt < RTH_END)]
        if rth.empty:
            continue
        g = rth.groupby(rth.index.date)
        daily = pd.DataFrame({
            "open": g["open"].first(),
            "high": g["high"].max(),
            "low": g["low"].min(),
            "close": g["close"].last(),
        }).sort_index()
        orm = (rth.index.time >= OR_START) & (rth.index.time < OR_END)
        orr = rth[orm]
        daily["or_vol"] = orr.groupby(orr.index.date)["volume"].sum()

        prev_close = daily["close"].shift(1)
        gap = daily["open"] / prev_close - 1.0
        tr = pd.concat([
            daily["high"] - daily["low"],
            (daily["high"] - prev_close).abs(),
            (daily["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr_pct = tr.rolling(ATR_WINDOW, min_periods=ATR_WINDOW // 2).mean().shift(1) / prev_close
        or_rvol = daily["or_vol"] / daily["or_vol"].rolling(
            RVOL_WINDOW, min_periods=10).mean().shift(1)

        for d in daily.index:
            gp = gap.get(d)
            rows.append({
                "date": d, "symbol": sym,
                "gap": gp,
                "abs_gap": abs(gp) if pd.notna(gp) else np.nan,
                "atr_pct": atr_pct.get(d),
                "or_rvol": or_rvol.get(d),
            })
    return pd.DataFrame(rows)


def _tday(t: Trade):
    return pd.Timestamp(t.date).date()


def filter_to_sets(trades: list[Trade], day_sets: dict) -> list[Trade]:
    return [t for t in trades if t.symbol in day_sets.get(_tday(t), ())]


SIGNAL_COLS = ["or_rvol", "abs_gap", "atr_pct"]


def topk_by_column(sig: pd.DataFrame, col: str, k: int) -> dict:
    """Per day: the K symbols with the highest `col` (NaN dropped)."""
    out = {}
    for d, grp in sig.groupby("date"):
        s = grp.dropna(subset=[col])
        out[d] = set(s.nlargest(k, col)["symbol"])
    return out


def topk_composite(sig: pd.DataFrame, k: int) -> dict:
    """Per day: top-K by the mean cross-sectional percentile rank of the 3 signals."""
    out = {}
    for d, grp in sig.groupby("date"):
        g = grp.dropna(subset=SIGNAL_COLS).copy()
        if g.empty:
            out[d] = set()
            continue
        score = sum(g[c].rank(pct=True) for c in SIGNAL_COLS) / len(SIGNAL_COLS)
        g = g.assign(score=score)
        out[d] = set(g.nlargest(k, "score")["symbol"])
    return out


def composite_score_map(sig: pd.DataFrame) -> dict:
    """(date, symbol) -> composite percentile-rank score (mean of the 3 signal
    ranks). Names missing any signal that day are absent (treated as lowest)."""
    out = {}
    for d, grp in sig.groupby("date"):
        g = grp.dropna(subset=SIGNAL_COLS)
        if g.empty:
            continue
        score = sum(g[c].rank(pct=True) for c in SIGNAL_COLS) / len(SIGNAL_COLS)
        for sym, sc in zip(g["symbol"], score):
            out[(d, sym)] = float(sc)
    return out


def daily_cap(trades: list[Trade], cap: int, keyfn, reverse: bool) -> list[Trade]:
    """Per session, keep the `cap` breakouts that sort first by `keyfn`.

    With keyfn = entry_time this is first-come (time order); with keyfn = rank
    score it allocates the slots by RANK instead. Comparing the two — holding
    this per-day-cap concurrency model fixed — isolates the value of *ordering
    the slot allocation by rank*, separate from shrinking the universe.

    Per-day-cap (vs the live cap-on-concurrent-overlap) is a faithful proxy
    here because ORB entries cluster after the open and ride to the EOD flatten,
    so the day's positions overlap heavily. The rank-ordered version is an UPPER
    BOUND: live you can't know in advance which names will break out.
    """
    by_day: dict = {}
    for t in trades:
        by_day.setdefault(_tday(t), []).append(t)
    taken = []
    for d, ts in by_day.items():
        ts.sort(key=lambda t: keyfn(d, t), reverse=reverse)
        taken.extend(ts[:cap])
    return taken


def eligible_pool(sig: pd.DataFrame) -> dict:
    """Per day: symbols with all 3 signals present (fair pool for the random control)."""
    out = {}
    for d, grp in sig.groupby("date"):
        out[d] = grp.dropna(subset=SIGNAL_COLS)["symbol"].tolist()
    return out


def random_perf(trades, pool: dict, k: int, trading_days) -> dict:
    """Mean perf of trading K RANDOM names/day from the eligible pool, over seeds."""
    acc = {m: [] for m in ("n", "win", "sum_r", "pnl", "ret_pct", "max_dd", "sharpe")}
    for seed in range(RANDOM_SEEDS):
        rng = np.random.default_rng(seed)
        day_sets = {}
        for d, syms in pool.items():
            day_sets[d] = set(syms) if len(syms) <= k else set(
                rng.choice(syms, size=k, replace=False))
        taken = portfolio(filter_to_sets(trades, day_sets), CAP)
        s = perf(taken, trading_days)
        if s.get("n", 0) == 0:
            continue
        for m in acc:
            acc[m].append(s[m])
    return {m: (np.mean(v) if v else float("nan")) for m, v in acc.items()}


# --------------------------------------------------------------------------
HEAD = (f"{'config':<26}{'n':>5}{'win%':>7}{'sumR':>8}{'PnL$':>11}"
        f"{'ret%':>8}{'maxDD$':>11}{'Sharpe':>8}")


def prow(label: str, s: dict):
    if s.get("n", 0) == 0:
        print(f"{label:<26}{'(no trades)':>20}")
        return
    print(f"{label:<26}{s['n']:>5.0f}{s['win']:>6.1f}%{s['sum_r']:>+8.1f}"
          f"{s['pnl']:>+11,.0f}{s['ret_pct']:>+7.2f}%{s['max_dd']:>+11,.0f}{s['sharpe']:>8.2f}")


def main() -> int:
    all_bars, trading_days = load_cached_universe()
    present = sorted(all_bars.index.get_level_values(0).unique())
    print(f"Universe: {len(present)} names with data, {len(trading_days)} sessions.\n")

    params = Params(or_minutes=15, target_r=2.0, risk_per_trade=100.0,
                    max_position_pct=0.25, max_position_dollars=10_000.0,
                    no_entry_after_time=time(11, 30))
    all_trades, _ = run_backtest(all_bars, trading_days, present, params, STARTING_EQUITY)
    print(f"Broad-universe ORB signals: {len(all_trades)}")

    sig = build_signals(all_bars, present)
    pool = eligible_pool(sig)
    score = composite_score_map(sig)

    # ---- baselines + decomposition (does ranking the slots beat first-come?) ----
    print("\n" + HEAD)
    print("-" * len(HEAD))
    base8 = perf(portfolio(all_trades, CAP), trading_days)
    prow(f"static-100, cap={CAP} (LIVE)", base8)
    prow("static-100, UNCAPPED", perf(portfolio(all_trades, None), trading_days))
    # Same 100 names, same per-day cap — only the slot ORDER differs (time vs rank).
    fc_daily = perf(daily_cap(all_trades, CAP, lambda d, t: t.entry_time, False), trading_days)
    rp_daily = perf(daily_cap(all_trades, CAP, lambda d, t: score.get((d, t.symbol), float("-inf")), True), trading_days)
    prow(f"static-100, first-come {CAP}/day", fc_daily)
    prow(f"static-100, RANK-priority {CAP}/day*", rp_daily)

    # ---- selectors x K ----
    selectors = {
        "or_rvol": lambda k: topk_by_column(sig, "or_rvol", k),
        "abs_gap": lambda k: topk_by_column(sig, "abs_gap", k),
        "atr_pct": lambda k: topk_by_column(sig, "atr_pct", k),
        "gap_up": lambda k: topk_by_column(sig, "gap", k),     # signed: favor gap-ups
        "composite": lambda k: topk_composite(sig, k),
    }

    best = None  # (pnl, name, k, perf_dict)
    for k in KS:
        print("-" * len(HEAD))
        rnd = random_perf(all_trades, pool, k, trading_days)
        prow(f"RANDOM top-{k} (x{RANDOM_SEEDS})", rnd)
        for name, fn in selectors.items():
            s = perf(portfolio(filter_to_sets(all_trades, fn(k)), CAP), trading_days)
            prow(f"  {name} top-{k}", s)
            if s.get("n", 0) and (best is None or s["pnl"] > best[0]):
                best = (s["pnl"], name, k, s, rnd)

    # ---- verdict + OOS check on the best selector ----
    print("\n" + "=" * 64)
    if best is None:
        print("No selector produced trades.")
        return 0
    _, bname, bk, bperf, brnd = best
    print(f"Best selector by PnL: {bname} top-{bk}")
    print(f"  vs static-100 cap{CAP}: PnL {bperf['pnl'] - base8['pnl']:+,.0f}, "
          f"Sharpe {bperf['sharpe'] - base8['sharpe']:+.2f}")
    print(f"  vs RANDOM top-{bk}:    PnL {bperf['pnl'] - brnd['pnl']:+,.0f}, "
          f"Sharpe {bperf['sharpe'] - brnd['sharpe']:+.2f}")
    beats_static = bperf["pnl"] > base8["pnl"] and bperf["sharpe"] >= base8["sharpe"]
    beats_random = bperf["pnl"] > brnd["pnl"]
    if beats_static and beats_random:
        verdict = "PROMISING — beats both the static list and random selection."
    elif beats_random and not beats_static:
        verdict = "WEAK — beats random but not the full static list (breadth ~ helps)."
    elif not beats_random:
        verdict = "NO SELECTION EDGE — does not beat random top-K. The signal is noise."
    else:
        verdict = "MIXED — read the table."
    print(f"  verdict: {verdict}")

    # OOS: split sessions in half; fill once over the full window, then score halves.
    days_sorted = sorted(trading_days)
    mid = days_sorted[len(days_sorted) // 2]
    sel_sets = selectors[bname](bk)
    taken = portfolio(filter_to_sets(all_trades, sel_sets), CAP)
    h1 = [t for t in taken if _tday(t) < mid]
    h2 = [t for t in taken if _tday(t) >= mid]
    d1 = [d for d in trading_days if d < mid]
    d2 = [d for d in trading_days if d >= mid]
    print(f"\nOOS split at {mid} ({bname} top-{bk}):")
    print("  " + HEAD)
    prow("  first half", perf(h1, d1))
    prow("  second half", perf(h2, d2))
    print("\nNotes:")
    print(f"- * RANK-priority {CAP}/day is an idealized UPPER BOUND (needs foreknowledge of")
    print("  which names break out). Gap between it and first-come {CAP}/day = the value of")
    print("  ranking slots; gap between top-K and it = what a pre-open watchlist leaves behind.")
    print(f"- cap={CAP} concurrent, $100 risk / $10k per trade, 11:30 ET entry cutoff (live).")
    print("- A selector must beat BOTH static-100 and RANDOM top-K, in both halves,")
    print("  before it's worth shipping. Sharpe is annualized from daily PnL (indicative).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
