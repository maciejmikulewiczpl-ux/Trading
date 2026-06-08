"""Dig #2 (data step): fetch minute + daily bars for an EXPANSION universe.

The live universe is ~100 S&P-100-tier names, kept upper-liquidity on purpose. To test
whether MORE names = more profitable tight-OR volume, we add ~120 liquid (but tier-2)
S&P 400/500 names not already covered. Saved to separate 'EXP' caches; the backtest
(compare_tightOR_universe.py) concatenates them with the existing cache and charges the
new names HIGHER slippage (they're less liquid). Network-only; run alongside CPU tests.

Run:
    .venv/Scripts/python.exe backtest/fetch_universe_expanded.py
"""
from __future__ import annotations

import pickle
import sys
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.run_orb import load_env  # noqa: E402
from backtest.universe_scan import UNIVERSE, fetch_chunked  # noqa: E402
from backtest.compare_norefill_trend import fetch_daily_closes, DAILY_BUFFER_DAYS  # noqa: E402

import os  # noqa: E402
from alpaca.data.historical import StockHistoricalDataClient  # noqa: E402

ET = ZoneInfo("America/New_York")

# ~120 liquid names NOT in the live universe (tier-2: liquid mid/large caps, popular
# high-volume names). Curated to stay reasonably tradeable; sub-$5 / thinnest names avoided.
EXPANSION = [
    # tech / semis
    "MRVL", "ADI", "KLAC", "SNPS", "CDNS", "MCHP", "ON", "MPWR", "FTNT", "DDOG",
    "NET", "SNOW", "ZS", "TEAM", "WDAY", "DELL", "HPQ", "HPE", "WDC", "STX",
    "SMCI", "ARM", "TSM", "ASML", "SONY", "FI", "GPN", "FIS",
    # internet / media
    "SPOT", "RBLX", "PINS", "SNAP", "DASH", "ROKU", "TTD", "MTCH", "EA", "TTWO",
    "WBD", "PARA", "FOXA", "OMC", "EBAY", "ETSY", "DKNG",
    # financials
    "PNC", "TFC", "COF", "BK", "STT", "AIG", "MET", "PRU", "ALL", "TRV",
    "AFL", "MMC", "AON", "PGR", "CB", "DFS", "SYF", "ALLY", "HOOD", "SOFI",
    "NDAQ", "MSCI", "MCO", "KKR", "BX", "APO",
    # healthcare
    "GILD", "VRTX", "REGN", "MRNA", "BIIB", "ZTS", "CI", "HUM", "MCK", "BDX",
    "SYK", "BSX", "MDT", "EW", "DXCM", "IDXX", "HCA", "CAH",
    # consumer staples / discretionary
    "PG", "CL", "KMB", "GIS", "MDLZ", "MO", "PM", "STZ", "KDP", "MNST",
    "EL", "CLX", "KR", "DG", "DLTR", "ULTA", "TJX", "ORLY", "AZO", "YUM",
    "MAR", "HLT", "RCL", "CCL", "DPZ", "DRI", "CVNA", "DECK",
    # industrials / energy / materials / utilities / RE
    "MMM", "EMR", "ETN", "ITW", "PH", "CMI", "PCAR", "GD", "NOC", "FDX",
    "UPS", "CSX", "NSC", "WM", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY",
    "KMI", "WMB", "OKE", "LIN", "APD", "SHW", "FCX", "NEM", "NUE", "DOW",
    "DUK", "SO", "AMT", "PLD", "CCI", "EQIX", "PSA", "O", "MARA", "FSLR",
]
WINDOWS = [730, 180]


def main():
    load_env()
    key, sec = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    dc = StockHistoricalDataClient(key, sec)
    new = [s for s in EXPANSION if s not in set(UNIVERSE)]
    print(f"expansion: {len(new)} new names (of {len(EXPANSION)}; {len(EXPANSION)-len(new)} already in universe)")

    for w in WINDOWS:
        univ_cache = ROOT / "backtest" / f".bars_cache_univ_{w}d.pkl"
        exp_min = ROOT / "backtest" / f".bars_cache_univ_EXP_{w}d.pkl"
        exp_day = ROOT / "backtest" / f".bars_cache_daily_EXP_{w}d.pkl"
        if exp_min.exists() and exp_day.exists():
            print(f"{w}d: EXP caches already present, skipping.")
            continue
        base = pickle.load(open(univ_cache, "rb"))
        days = base["days"]
        start = datetime.combine(min(days), time(9, 0), ET)
        end = datetime.combine(max(days), time(16, 0), ET) + timedelta(days=1)
        print(f"{w}d: fetching MINUTE bars for {len(new)} names {min(days)}..{max(days)} ...", flush=True)
        mins = fetch_chunked(dc, new, start, end)
        pickle.dump({"bars": mins, "days": days}, open(exp_min, "wb"))
        print(f"{w}d: minute rows {len(mins):,}; fetching DAILY closes ...", flush=True)
        dstart = datetime.combine(min(days), time(0, 0), ET) - timedelta(days=DAILY_BUFFER_DAYS)
        dend = datetime.combine(max(days), time(0, 0), ET) + timedelta(days=1)
        closes = fetch_daily_closes(new, dstart, dend)
        pickle.dump(closes, open(exp_day, "wb"))
        print(f"{w}d: DONE — minute {len(mins):,} rows, daily {closes.shape}.")
    print("Fetch complete. Next: compare_tightOR_universe.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
