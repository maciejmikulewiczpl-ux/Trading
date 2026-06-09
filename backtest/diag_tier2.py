"""Why did tier-2 names yield 0 trades? Isolate data-coverage vs trend-filter."""
from __future__ import annotations
import sys
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import os
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from strategies.orb import Params
from backtest.run_orb import run_backtest, load_env, to_et, STARTING_EQUITY
from backtest.universe_scan import fetch_chunked
from backtest.compare_norefill_trend import fetch_daily_closes, trend_eligibility, DAILY_BUFFER_DAYS
import pickle
from alpaca.data.historical import StockHistoricalDataClient

ET = ZoneInfo("America/New_York")
NAMES = ["MRVL", "ADI", "GILD", "PG", "MMM"]


def main():
    load_env()
    dc = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    base = pickle.load(open(ROOT / "backtest" / ".bars_cache_univ_730d.pkl", "rb"))
    days = base["days"]
    spy_daily = pickle.load(open(ROOT / "backtest" / ".bars_cache_daily_730d.pkl", "rb"))[["SPY"]]
    start = datetime.combine(min(days), time(9, 0), ET)
    end = datetime.combine(max(days), time(16, 0), ET) + timedelta(days=1)
    bars = to_et(fetch_chunked(dc, NAMES, start, end))   # <-- the fix: UTC -> ET before backtest
    print(f"minute rows: {len(bars):,}")
    if not bars.empty:
        for s in NAMES:
            try:
                n = len(bars.xs(s, level=0))
            except Exception:
                n = 0
            print(f"  {s}: {n:,} minute bars")
    pres = sorted(bars.index.get_level_values(0).unique())
    dstart = datetime.combine(min(days), time(0, 0), ET) - timedelta(days=DAILY_BUFFER_DAYS)
    dend = datetime.combine(max(days), time(0, 0), ET) + timedelta(days=1)
    closes = fetch_daily_closes(NAMES, dstart, dend).join(spy_daily, how="outer")
    print(f"daily cols: {list(closes.columns)}  rows: {len(closes)}")
    p = Params(or_minutes=15, target_r=2.0, risk_per_trade=100.0, max_position_pct=0.25,
               max_position_dollars=10_000.0, no_entry_after_time=time(11, 30))
    raw, _ = run_backtest(bars, days, pres, p, STARTING_EQUITY)
    print(f"raw backtest trades: {len(raw)}  (long: {sum(1 for t in raw if t.side=='long')})")
    elig = trend_eligibility(closes, pres, days)
    nelig = {s: (len(v) if v else 0) for s, v in elig.items()}
    print(f"trend-eligible day-counts per name: {nelig}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
