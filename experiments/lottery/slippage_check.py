"""Live fill-slippage check (DeepSeek's one surviving kernel: is smart-limit execution worth it?).

For every REAL entry in logs/lottery_trade_ledger.csv, compare the actual fill (entry_avg) to the
09:45 ET minute bar as the arrival reference (open and VWAP, IEX -- same reference outcomes.py uses).
Reports slippage in bps and, critically, whether slippage is concentrated in the NON-runners (where a
limit order is safe) or the runners/tail-makers (where waiting for a limit risks missing the move).

    slip_open_bps = (entry_avg / open_945  - 1) * 1e4     (buy: >0 = paid above the 09:45 open)
    slip_vwap_bps = (entry_avg / vwap_945  - 1) * 1e4     (vs the bar's traded VWAP)

Run:
    .venv/Scripts/python.exe experiments/lottery/slippage_check.py
"""
from __future__ import annotations

import csv
import statistics as st
import sys
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from backtest.run_orb import load_env  # noqa: E402

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
LEDGER = ROOT / "logs" / "lottery_trade_ledger.csv"
ENTRY_T = time(9, 45)


def _client():
    import os
    from alpaca.data.historical import StockHistoricalDataClient
    load_env()
    return StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return (len(vals), st.mean(vals), st.median(vals),
            100 * sum(1 for v in vals if v > 0) / len(vals), max(vals), min(vals))


def main() -> int:
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    rows = list(csv.DictReader(open(LEDGER)))
    dc = _client()
    out = []  # (sym, date, slip_open, slip_vwap, mfe, realized_pct)
    for r in rows:
        sym, date, fill = r["symbol"], r["entry_date"], _f(r["entry_avg"])
        if fill is None:
            continue
        d = datetime.fromisoformat(date).date()
        start = datetime.combine(d, time(9, 30), ET)
        end = datetime.combine(d, time(10, 0), ET)
        try:
            req = StockBarsRequest(symbol_or_symbols=[sym], timeframe=TimeFrame.Minute,
                                   start=start.astimezone(UTC), end=end.astimezone(UTC),
                                   feed=DataFeed.IEX)
            sb = dc.get_stock_bars(req).df.xs(sym, level=0)
            t = sb.index.tz_convert(ET).time
            bar = sb[t >= ENTRY_T].iloc[0]
            o = float(bar["open"])
            vw = float(bar["vwap"]) if "vwap" in bar and bar["vwap"] else o
            out.append((sym, date, (fill / o - 1) * 1e4, (fill / vw - 1) * 1e4,
                        _f(r.get("mfe_pct")), _f(r.get("realized_pct"))))
        except Exception:
            continue

    if not out:
        print("no fills matched to bars."); return 0
    so, sv = _agg([r[2] for r in out]), _agg([r[3] for r in out])
    print(f"=== LIVE FILL SLIPPAGE: {len(out)} entries (vs 09:45 IEX bar) ===\n")
    print(f"{'ref':<22}{'n':>4}{'mean bps':>10}{'med bps':>9}{'>0%':>6}{'worst':>8}{'best':>8}")
    print(f"{'vs 09:45 open':<22}{so[0]:>4}{so[1]:>+10.1f}{so[2]:>+9.1f}{so[3]:>6.0f}{so[4]:>+8.0f}{so[5]:>+8.0f}")
    print(f"{'vs 09:45 VWAP':<22}{sv[0]:>4}{sv[1]:>+10.1f}{sv[2]:>+9.1f}{sv[3]:>6.0f}{sv[4]:>+8.0f}{sv[5]:>+8.0f}")

    # DeepSeek's safety question: is slippage concentrated in RUNNERS (limit risky) or non-runners?
    mfes = sorted([r[4] for r in out if r[4] is not None])
    if mfes:
        med_mfe = st.median(mfes)
        run = _agg([r[2] for r in out if r[4] is not None and r[4] >= med_mfe])
        non = _agg([r[2] for r in out if r[4] is not None and r[4] < med_mfe])
        print("\n  Slippage (vs 09:45 open) split by outcome (MFE = max favorable excursion):")
        if run:
            print(f"    runners  (MFE>=med {med_mfe:.1f}%)  n={run[0]:>2} slip mean={run[1]:+.1f} bps")
        if non:
            print(f"    non-run  (MFE< med {med_mfe:.1f}%)  n={non[0]:>2} slip mean={non[1]:+.1f} bps")

    avg = so[1]
    print(f"\nRead: +bps = we pay ABOVE the 09:45 open (spread-cross cost). Recoverable ceiling from a")
    print(f"mid-seeking limit ~= this figure ({avg:+.0f} bps/entry). If slippage is bigger on the")
    print("RUNNERS than non-runners, a limit-order ladder risks worsening the fills that matter most.")
    print("Judge vs the ~+50%/moonshot tail: a few bps of entry edge is noise next to the tail. n small.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
