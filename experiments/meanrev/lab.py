"""Mean-reversion LIVE forward-test lab (cost-free, no trading account needed).

Runs daily after the close. Maintains a paper book of the Connors mean-reversion strategy validated
in backtest.py: enter a QUALITY name (close > 200d SMA) when OVERSOLD (RSI(2) < RSI_BUY) at today's
close; exit at the close when RSI(2) > RSI_EXIT (bounce) or after MAX_H days. Records every closed
trade to logs/meanrev_trades.csv -> a genuine OUT-OF-SAMPLE track record to check the backtest edge
holds forward. Data-only (StockHistoricalDataClient); no orders, no account. Slightly different from
the backtest (enter at signal-day CLOSE, not next open) -- simpler + stateless-friendly for a live lab.

  scan   (default) : one daily pass -- process exits, then entries; persist book; append closed trades
  report           : summarise the forward track record (win%, mean, tail) vs the backtest
Run: .venv/bin/python experiments/meanrev/lab.py scan
"""
from __future__ import annotations
import os, sys, csv, json, datetime as dt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.meanrev.backtest import UNIV, rsi2, RSI_BUY, RSI_EXIT, MAX_H
from backtest.run_orb import load_env

ROOT = Path(__file__).resolve().parents[2]
POS_FILE = ROOT / "logs" / "meanrev_positions.json"
TRADES = ROOT / "logs" / "meanrev_trades.csv"


def _recent_bars():
    load_env()
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    dc = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    start = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=420)   # >200 trading days for SMA200
    data = {}
    for i in range(0, len(UNIV), 50):
        grp = UNIV[i:i + 50]
        try:
            df = dc.get_stock_bars(StockBarsRequest(symbol_or_symbols=grp, timeframe=TimeFrame.Day,
                 start=start, feed=DataFeed.IEX)).df
            for s in grp:
                try:
                    sb = df.xs(s, level=0).sort_index()
                    data[s] = [(t.date().isoformat(), float(r.close)) for t, r in sb.iterrows()]
                except KeyError:
                    pass
        except Exception:
            pass
    return data


def _load_pos():
    return json.load(open(POS_FILE)) if POS_FILE.exists() else {}


def _save_pos(p):
    POS_FILE.parent.mkdir(exist_ok=True)
    json.dump(p, open(POS_FILE, "w"), indent=2)


def _append_trade(row):
    new = not TRADES.exists()
    with open(TRADES, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["entry_date", "exit_date", "symbol", "entry_px", "exit_px", "ret_pct", "days", "exit_reason"])
        w.writerow(row)


def scan():
    data = _recent_bars()
    pos = _load_pos()
    meta = pos.pop("_meta", {})
    today = max((bars[-1][0] for bars in data.values() if bars), default=None)
    if today and meta.get("last_date") == today:
        print(f"meanrev-lab: session {today} already processed -- skip (no new daily bar yet).")
        pos["_meta"] = meta; _save_pos(pos); return
    closed = entered = 0
    for s, bars in data.items():
        if len(bars) < 210:
            continue
        closes = [c for _, c in bars]
        rs = rsi2(closes)
        d, close = bars[-1]
        today = d
        r = rs[-1]
        sma200 = sum(closes[-200:]) / 200
        if s in pos:                                   # manage an open position
            p = pos[s]; p["days"] = p.get("days", 0) + 1
            reason = "bounce" if (r is not None and r > RSI_EXIT) else ("time-stop" if p["days"] >= MAX_H else None)
            if reason:
                ret = close / p["entry_px"] - 1.0
                _append_trade([p["entry_date"], d, s, round(p["entry_px"], 2), round(close, 2),
                               round(ret * 100, 2), p["days"], reason])
                del pos[s]; closed += 1
        elif r is not None and r < RSI_BUY and close > sma200:   # new entry (oversold in uptrend)
            pos[s] = {"entry_date": d, "entry_px": close, "days": 0}
            entered += 1
    meta["last_date"] = today
    pos["_meta"] = meta
    _save_pos(pos)
    nopen = len([k for k in pos if not k.startswith("_")])
    print(f"meanrev-lab {today}: {entered} new entries, {closed} exits, {nopen} open positions")


def report():
    if not TRADES.exists():
        print("no closed trades yet."); return
    rows = list(csv.DictReader(open(TRADES)))
    rets = sorted(float(x["ret_pct"]) for x in rows)
    import statistics
    n = len(rets)
    win = sum(1 for x in rets if x > 0) / n * 100
    worst5 = statistics.mean(rets[:max(1, n // 20)])
    print(f"=== MEAN-REV FORWARD TRACK RECORD ({n} closed trades) ===")
    print(f"  mean {statistics.mean(rets):+.2f}%  median {statistics.median(rets):+.2f}%  win {win:.0f}%  "
          f"best {rets[-1]:+.1f}%  worst {rets[0]:+.1f}%  worst5% {worst5:+.1f}%  SUM {sum(rets):+.0f}")
    nopen = len([k for k in _load_pos() if not k.startswith("_")])
    print(f"  open positions: {nopen}  |  backtest bar to beat: +0.47%/trade, 66% win")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "scan"
    (report if cmd == "report" else scan)()
