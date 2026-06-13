"""fetch_swing_data.py -- step 1 of the swing-engine backtest.

Builds `.swing_daily_cache.pkl` used by run_swing.py / compare_swing_variants.py.

Source of candidates: pit_daily_730d.pkl (top 500 by mean dollar-vol, already
live in the repo, skips BLOCK + ETFS junk). Then yfinance fetches 2014-01-01 ->
today so run_swing has full lookback for 2016 signals (ATR14, 55d high, 252d
percentile). SPY fetched separately for regime + benchmark.

Survivorship note (spec-documented): Alpaca active-only pool + yfinance = truly
delisted names absent. The dollar-vol mechanical ranking removes the bigger bias.
Dropout rate is printed and checked against the 15% gate.

Run:
    .venv-openbb\\Scripts\\python.exe backtest\\fetch_swing_data.py
"""
from __future__ import annotations

import pickle
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Inlined from pit_expand to avoid alpaca import chain in .venv-openbb
BLOCK = {"TQQQ", "SQQQ", "SOXL", "SOXS", "TZA", "TNA", "SPXL", "SPXS", "UPRO",
         "UVXY", "SVXY", "TMF", "TMV", "YINN", "FNGU", "BOIL", "UCO",
         "MSTR", "IBIT", "ETHA", "BITO", "BMNR", "CRCL", "CRWV", "MARA", "RIOT"}
ETFS = {"EEM", "EFA", "EWY", "EWZ", "FXI", "GDX", "GLD", "IEFA", "IVV", "KRE", "KWEB",
        "RSP", "SLV", "SMH", "SOXX", "XBI", "XLE", "XLF", "XLI", "XLK", "XLU", "XLV",
        "AGG", "TLT", "IEF", "LQD", "HYG", "JNK", "EMB", "VCIT", "VCLT", "USHY",
        "SPY", "QQQ", "IWM", "DIA", "VOO", "IGV"}

CACHE = ROOT / "backtest" / ".swing_daily_cache.pkl"
FETCH_START = "2014-01-01"
TOP_N = 500          # candidates to attempt (top by recent dollar-vol)
MIN_ROWS = 400       # need at least this many trading-day bars to be usable
BATCH = 100          # yfinance batch size


def top_candidates(n: int) -> list[str]:
    """Top N by mean dollar-vol from pit_daily, excluding junk."""
    d = pickle.load(open(ROOT / "backtest" / ".pit_daily_730d.pkl", "rb"))
    dvol: pd.DataFrame = d["dvol"]
    exclude = BLOCK | ETFS | {"SPY"}
    mean_dv = dvol.mean()
    mean_dv = mean_dv[[s for s in mean_dv.index if s not in exclude]]
    return list(mean_dv.nlargest(n).index)


def fetch_batch(syms: list[str], start: str) -> dict[str, pd.DataFrame]:
    """Download yfinance OHLCV for a batch of symbols, return per-symbol dicts."""
    raw = yf.download(
        tickers=syms,
        start=start,
        end=date.today().isoformat(),
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    out: dict[str, pd.DataFrame] = {}
    if raw.empty:
        return out
    # multi-ticker: columns are (Price, Ticker)
    if isinstance(raw.columns, pd.MultiIndex):
        for sym in syms:
            try:
                df = raw.xs(sym, axis=1, level=1)[["Open", "High", "Low", "Close", "Volume"]]
                df = df.dropna(how="all")
                if len(df) >= MIN_ROWS:
                    out[sym] = df
            except (KeyError, Exception):
                pass
    else:
        # single-ticker fallback
        sym = syms[0]
        df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")
        if len(df) >= MIN_ROWS:
            out[sym] = df
    return out


def main() -> None:
    print("=== fetch_swing_data.py ===")
    candidates = top_candidates(TOP_N)
    print(f"Candidates: {len(candidates)} symbols to fetch from {FETCH_START}")

    # fetch SPY first (small, fast)
    print("Fetching SPY...")
    spy_raw = yf.download("SPY", start=FETCH_START, auto_adjust=True, progress=False)
    # yfinance may return multi-level columns; flatten to plain OHLCV
    if isinstance(spy_raw.columns, pd.MultiIndex):
        spy_raw = spy_raw.xs("SPY", axis=1, level=1)
    spy_df = spy_raw[["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")
    print(f"  SPY: {len(spy_df)} bars  {spy_df.index[0].date()} -> {spy_df.index[-1].date()}")

    # fetch in batches
    symbols: dict[str, pd.DataFrame] = {}
    failed: list[str] = []
    for i in range(0, len(candidates), BATCH):
        batch = candidates[i : i + BATCH]
        t0 = time.time()
        result = fetch_batch(batch, FETCH_START)
        elapsed = time.time() - t0
        symbols.update(result)
        got = len(result)
        miss = len(batch) - got
        if miss:
            failed.extend([s for s in batch if s not in result])
        print(f"  batch {i//BATCH+1:>2}/{(len(candidates)-1)//BATCH+1}: "
              f"{got}/{len(batch)} ok  ({elapsed:.1f}s)  total so far: {len(symbols)}")

    print(f"\nFetch complete: {len(symbols)} usable / {len(candidates)} attempted "
          f"({len(failed)} failed/thin)")

    # dropout gate (from spec: if >15% unfetchable among PIT top-100 each year slice,
    # flag it -- here we just check overall top-500 coverage as a proxy)
    dropout_pct = 100.0 * len(failed) / len(candidates)
    print(f"Dropout rate: {dropout_pct:.1f}%  (spec gate: flag if >15%)")
    if dropout_pct > 15:
        print("  WARNING: high dropout -- survivorship bias note applies, proceed with caution")

    # date coverage report
    min_dates = {sym: df.index[0].date() for sym, df in symbols.items()}
    has_2016 = sum(1 for d in min_dates.values() if d.year <= 2015)
    print(f"Symbols with data reaching <=2015: {has_2016}/{len(symbols)} "
          f"(needed for 2016 signals)")

    cache = {
        "symbols": symbols,
        "spy": spy_df,
        "candidates": candidates,
        "fetched": list(symbols.keys()),
        "run_date": date.today(),
    }
    pickle.dump(cache, open(CACHE, "wb"), protocol=4)
    print(f"\nSaved -> {CACHE.name}  ({CACHE.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
