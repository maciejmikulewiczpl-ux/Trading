"""Mean-reversion (oversold bounce) BACKTEST -- the 'steady income' candidate.

Classic Connors-style setup: buy a QUALITY name (above its 200d SMA = uptrend) when it gets
OVERSOLD (RSI(2) < threshold), exit on the bounce (RSI recovers) or after H days. High win-rate by
design -- so the real question isn't the win rate, it's the TAIL: do the rare big losses eat the
many small wins? We report worst trades + the strategy's own drawdown, not just the average.

vs a RANDOM-day entry baseline on the same names (does 'oversold' add anything over just buying?).
Frictionless daily bars (Alpaca IEX 2023-); a real sleeve nets ~5-10bps/trade. Relative read.
Run: .venv/Scripts/python.exe experiments/meanrev/backtest.py"""
from __future__ import annotations
import os, sys, datetime as dt, statistics, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backtest.run_orb import load_env

# diversified liquid large-caps (mean-reversion wants QUALITY, not the high-vol lottery universe)
UNIV = ["AAPL","MSFT","GOOGL","AMZN","META","NVDA","AVGO","TSLA","JPM","V","MA","UNH","HD","PG","JNJ",
        "COST","WMT","BAC","XOM","CVX","KO","PEP","MRK","ABBV","LLY","ORCL","CRM","ADBE","AMD","NFLX",
        "DIS","CSCO","INTC","QCOM","TXN","IBM","GE","CAT","BA","NKE","MCD","SBUX","LOW","GS","MS","C",
        "WFC","T","VZ","PFE"]
RSI_BUY = 10.0        # RSI(2) below this = oversold
RSI_EXIT = 60.0       # RSI(2) above this = bounce done
MAX_H = 5             # hard time-stop (days) if no bounce


def rsi2(closes):
    """Wilder RSI period-2, returned per-index (None until warmed)."""
    out = [None] * len(closes)
    if len(closes) < 3:
        return out
    ag = al = 0.0
    for i in range(1, len(closes)):
        ch = closes[i] - closes[i - 1]
        g, l = max(ch, 0), max(-ch, 0)
        if i <= 2:
            ag += g / 2; al += l / 2
        else:
            ag = (ag * 1 + g) / 2; al = (al * 1 + l) / 2   # 2-period Wilder smoothing
        if i >= 2:
            rs = ag / al if al else 99
            out[i] = 100 - 100 / (1 + rs)
    return out


def load():
    load_env()
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    dc = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    data = {}
    for i in range(0, len(UNIV), 50):
        grp = UNIV[i:i + 50]
        df = dc.get_stock_bars(StockBarsRequest(symbol_or_symbols=grp, timeframe=TimeFrame.Day,
             start=dt.datetime(2023, 1, 1), feed=DataFeed.IEX)).df
        for s in grp:
            try:
                sb = df.xs(s, level=0).sort_index()
                data[s] = [(t.date(), float(r.open), float(r.close)) for t, r in sb.iterrows()]
            except KeyError:
                pass
    return data


def trades_for(bars):
    closes = [c for _, _, c in bars]
    rs = rsi2(closes)
    sma = [None] * len(closes)
    for i in range(len(closes)):
        if i >= 200:
            sma[i] = sum(closes[i - 199:i + 1]) / 200
    out = []
    i = 201
    while i < len(bars) - 1:
        # signal known at close of day i -> enter next open
        if rs[i] is not None and sma[i] and rs[i] < RSI_BUY and closes[i] > sma[i]:
            entry = bars[i + 1][1]                      # next open
            exit_px = None
            for h in range(1, MAX_H + 1):
                j = i + h
                if j >= len(bars):
                    break
                if rs[j] is not None and rs[j] > RSI_EXIT:
                    exit_px = bars[j][2]; break          # bounce -> exit at that close
                if h == MAX_H:
                    exit_px = bars[j][2]                  # time-stop
            if entry and exit_px:
                out.append((bars[i + 1][0], exit_px / entry - 1.0))
            i += 2
        else:
            i += 1
    return out


def rand_baseline(bars, n):
    import random
    closes = [c for _, _, c in bars]
    idx = [i for i in range(201, len(bars) - MAX_H - 1)]
    random.seed(len(bars)); random.shuffle(idx)
    out = []
    for i in idx[:n]:
        entry = bars[i + 1][1]; exit_px = bars[i + MAX_H][2]
        if entry:
            out.append(exit_px / entry - 1.0)
    return out


def report(name, rets):
    if not rets:
        print(f"  {name}: no trades"); return
    v = sorted(r * 100 for r in rets)
    mean = statistics.mean(v); med = statistics.median(v)
    win = sum(1 for x in v if x > 0) / len(v) * 100
    worst5 = statistics.mean(v[:max(1, len(v) // 20)])       # avg of worst 5%
    print(f"  {name:<26} n={len(v):>4}  mean {mean:+.2f}%  med {med:+.2f}%  win {win:.0f}%  "
          f"best {v[-1]:+.0f}%  WORST {v[0]:+.0f}%  worst5%avg {worst5:+.1f}%  SUM {sum(v):+.0f}")


def main():
    data = load()
    print(f"loaded {len(data)} names\n=== MEAN-REVERSION (RSI(2)<{RSI_BUY:.0f} & >200dSMA, exit RSI>{RSI_EXIT:.0f} or T+{MAX_H}) ===")
    all_tr = []
    dated = []
    for s, bars in data.items():
        tr = trades_for(bars)
        all_tr += [r for _, r in tr]
        dated += tr
    report("mean-rev (all)", all_tr)
    # random baseline (same count)
    rb = []
    for s, bars in data.items():
        rb += rand_baseline(bars, max(1, len(trades_for(bars))))
    report("random-day baseline", rb)
    # OOS split by entry date
    dated.sort()
    mid = dated[len(dated) // 2][0]
    report("mean-rev EARLY half", [r for d, r in dated if d < mid])
    report("mean-rev LATE half", [r for d, r in dated if d >= mid])
    # the income lens: equity curve of sequential trades (are the wins steady or lumpy?)
    eq = 1.0; peak = 1.0; mdd = 0.0
    for _, r in dated:
        eq *= (1 + r); peak = max(peak, eq); mdd = min(mdd, eq / peak - 1)
    print(f"\n  sequential-compounding: total {(eq-1)*100:+.0f}%  maxDD {mdd*100:.0f}%  over {len(dated)} trades")
    print("\n[caveats] frictionless; ~5-10bps/trade real cost matters at this win-rate/edge size; 2023- "
          "(bull-heavy) so the TAIL is under-sampled -- the real risk shows in a bear. Relative read.")


if __name__ == "__main__":
    main()
