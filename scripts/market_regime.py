"""Market regime gauge — is the tape BULLISH, STABLE, a buyable PULLBACK, or a DOWNTREND?

A descriptive weather report for the user, combining the classic chart-room reads
(50/120/200-day moving averages, golden/death cross, RSI-14, MACD 12-26-9, distance
from the 52-week high) with the two measures this project has actually validated:
the SPY realized-vol regime (the live bot's vol-dial: 20d vol vs 126d median,
compare_volpause.prior_vol_flags) and watchlist BREADTH (% of our own tradable
names above their 200d/50d MA — breadth leads the index at turns).

HONEST FRAMING: indicators DESCRIBE the tape, they don't predict it — none of these
is a validated entry signal (our backtests keep rejecting indicator gates). The
composite verdict is a weight-of-evidence read for the human, NOT a bot input. The
bot's regime inputs remain the shipped vol-dial + trend filter, unchanged.

The buy-the-dip logic encodes the one regime fact with broad historical support:
pullbacks in an INTACT long-term uptrend (price above a rising 200d MA) have
historically resolved up, while "dips" below a falling 200d MA are downtrend
rallies. The TURN CHECKLIST tracks classic bottoming markers for the latter case.

Run:
    .venv/Scripts/python.exe scripts/market_regime.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.run_orb import load_env  # noqa: E402
from live import config as orb_config  # noqa: E402

ET = ZoneInfo("America/New_York")
INDEXES = ["SPY", "QQQ", "IWM"]
LOOKBACK_CAL_DAYS = 560          # ~380 trading days: 200d MA + slope + 52wk high
MA_WINDOWS = (50, 120, 200)


# ---------- data ----------
def fetch_daily(symbols: list[str]) -> dict[str, pd.Series]:
    """Split-adjusted daily closes per symbol (IEX feed, free)."""
    import os
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed, Adjustment
    load_env()
    dc = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    end = datetime.now(ET)
    req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame.Day,
                           start=end - timedelta(days=LOOKBACK_CAL_DAYS), end=end,
                           feed=DataFeed.IEX, adjustment=Adjustment.SPLIT)
    df = dc.get_stock_bars(req).df
    out = {}
    for sym in symbols:
        try:
            s = df.xs(sym, level=0)["close"]
            s.index = pd.DatetimeIndex([i.date() for i in s.index])
            out[sym] = s.astype(float).sort_index()
        except KeyError:
            continue
    return out


# ---------- indicators ----------
def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI."""
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, pd.NA)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def macd(close: pd.Series):
    line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    sig = line.ewm(span=9, adjust=False).mean()
    return line, sig, line - sig


def vol_regime(close: pd.Series):
    """The live bot's validated dial: 20d realized vol vs its 126d rolling median."""
    vol = close.pct_change().rolling(20).std()
    med = vol.rolling(126, min_periods=40).median()
    return vol, bool(vol.iloc[-1] > med.iloc[-1])


def analyze(close: pd.Series) -> dict:
    px = float(close.iloc[-1])
    ma = {n: close.rolling(n).mean() for n in MA_WINDOWS}
    r = rsi(close)
    line, sig, hist = macd(close)
    hi52 = float(close.tail(252).max())
    vol, vol_hot = vol_regime(close)
    a = {
        "px": px,
        "ma": {n: float(ma[n].iloc[-1]) for n in MA_WINDOWS},
        "ma200_rising": bool(ma[200].iloc[-1] > ma[200].iloc[-21]),
        "ma50_rising": bool(ma[50].iloc[-1] > ma[50].iloc[-11]),
        "golden": bool(ma[50].iloc[-1] > ma[200].iloc[-1]),
        "rsi": float(r.iloc[-1]),
        "rsi_10d_ago": float(r.iloc[-11]),
        "rsi_low60": float(r.tail(60).min()),
        "macd_above_sig": bool(line.iloc[-1] > sig.iloc[-1]),
        "macd_hist": float(hist.iloc[-1]),
        "macd_hist_rising": bool((hist.diff().tail(5) > 0).sum() >= 4),
        "macd_cross_up_10d": bool(((line > sig) & ~(line > sig).shift(1).fillna(False)).tail(10).any()),
        "dd52": (px / hi52 - 1.0) * 100,
        "vol20_ann": float(vol.iloc[-1]) * (252 ** 0.5) * 100,
        "vol_hot": vol_hot,
        "vol_falling": bool(vol.iloc[-1] < vol.iloc[-11]),
    }
    return a


def breadth(watch: dict[str, pd.Series]) -> dict:
    n200 = a200 = n50 = a50 = 0
    for s in watch.values():
        if len(s) >= 210:
            n200 += 1
            if s.iloc[-1] > s.rolling(200).mean().iloc[-1]:
                a200 += 1
        if len(s) >= 60:
            n50 += 1
            if s.iloc[-1] > s.rolling(50).mean().iloc[-1]:
                a50 += 1
    return {"pct200": 100.0 * a200 / max(n200, 1), "pct50": 100.0 * a50 / max(n50, 1),
            "n": n200}


# ---------- verdict ----------
def composite(spy: dict, br: dict) -> tuple[int, list[str]]:
    """Weight-of-evidence score in ~[-10, +10]. Long-term structure and breadth
    carry double weight; momentum oscillators single."""
    votes = []
    score = 0

    def vote(cond_pos, cond_neg, w, name, detail):
        nonlocal score
        s = w if cond_pos else (-w if cond_neg else 0)
        score += s
        votes.append(f"  {'+'+str(s) if s > 0 else s if s < 0 else ' 0':>3}  {name:<26} {detail}")

    vote(spy["px"] > spy["ma"][200], spy["px"] <= spy["ma"][200], 2,
         "price vs 200d MA", f"{spy['px']:.0f} vs {spy['ma'][200]:.0f}")
    vote(spy["ma200_rising"], not spy["ma200_rising"], 1,
         "200d MA slope (1mo)", "rising" if spy["ma200_rising"] else "falling")
    vote(spy["golden"], not spy["golden"], 1,
         "50d vs 200d MA", "golden cross" if spy["golden"] else "death cross")
    vote(spy["px"] > spy["ma"][50], spy["px"] <= spy["ma"][50], 1,
         "price vs 50d MA", f"{spy['px']:.0f} vs {spy['ma'][50]:.0f}")
    vote(spy["macd_above_sig"] and spy["macd_hist"] > 0,
         not spy["macd_above_sig"] and spy["macd_hist"] < 0, 1,
         "MACD (12-26-9)", f"hist {spy['macd_hist']:+.2f}")
    vote(spy["rsi"] > 55, spy["rsi"] < 45, 1, "RSI-14 zone", f"{spy['rsi']:.0f}")
    vote(not spy["vol_hot"], spy["vol_hot"], 1,
         "vol regime (live dial)", f"20d {spy['vol20_ann']:.0f}%ann "
         f"{'> 126d median (ELEVATED)' if spy['vol_hot'] else '<= median (calm)'}")
    vote(br["pct200"] > 55, br["pct200"] < 45, 2,
         "breadth >200d MA", f"{br['pct200']:.0f}% of {br['n']} watchlist names")
    vote(spy["dd52"] > -5, spy["dd52"] < -15, 1,
         "off 52-week high", f"{spy['dd52']:+.1f}%")
    return score, votes


def label(score: int) -> str:
    if score >= 5:
        return "BULLISH — uptrend in force"
    if score >= 1:
        return "STABLE / mildly bullish"
    if score >= -4:
        return "WEAKENING / correction zone"
    return "BEARISH — downtrend"


def dip_read(spy: dict) -> list[str]:
    structure = spy["px"] > spy["ma"][200] and (spy["ma200_rising"] or spy["golden"])
    oversold = spy["rsi"] < 35 or (spy["px"] < spy["ma"][50] and -12 < spy["dd52"] < -3)
    out = [f"  long-term structure : {'INTACT (above ~rising 200d MA)' if structure else 'BROKEN (below/falling 200d MA)'}",
           f"  short-term stretch  : {'OVERSOLD' if oversold else 'not oversold'} "
           f"(RSI {spy['rsi']:.0f}, {spy['dd52']:+.1f}% off high)"]
    if structure and oversold:
        out.append("  => BUYABLE-DIP ZONE: pullback inside an intact uptrend — the kind that has"
                   "\n     historically resolved up. (Not a validated signal; size accordingly.)")
    elif structure:
        out.append("  => No dip on offer — trend intact and not stretched. Nothing to time.")
    elif oversold:
        out.append("  => CAUTION: oversold BELOW a broken 200d MA = downtrend rally risk, not a dip."
                   "\n     Wait for the turn checklist below before buying weakness.")
    else:
        out.append("  => Downtrend, not yet washed out. Watch the turn checklist.")
    return out


def turn_checklist(spy: dict, br: dict) -> list[str]:
    checks = [
        ("RSI washed out (<35 in last 60d) then recovered >40",
         spy["rsi_low60"] < 35 and spy["rsi"] > 40 and spy["rsi"] > spy["rsi_10d_ago"]),
        ("MACD crossed up (10d) or histogram rising 4 of 5 days",
         spy["macd_cross_up_10d"] or spy["macd_hist_rising"]),
        ("price reclaimed the 50d MA", spy["px"] > spy["ma"][50]),
        ("50d MA slope turned up (2wk)", spy["ma50_rising"]),
        ("volatility compressing (20d vol < 2wk ago)", spy["vol_falling"]),
        ("breadth repair: >50% of watchlist above 50d MA", br["pct50"] > 50),
    ]
    n_on = sum(1 for _, ok in checks if ok)
    out = [f"  [{'x' if ok else ' '}] {name}" for name, ok in checks]
    out.append(f"  {n_on}/6 bottoming markers ON"
               + (" — turn forming" if n_on >= 4 else " — no confirmed turn yet" if n_on <= 2 else ""))
    return out


def main() -> int:
    watchlist = list(orb_config.load_config()["watchlist"])
    closes = fetch_daily(sorted(set(INDEXES + watchlist)))
    missing = [s for s in INDEXES if s not in closes]
    if missing:
        print(f"missing index data: {missing}")
        return 1
    asof = closes["SPY"].index[-1].date()
    print(f"=== MARKET REGIME GAUGE — data through {asof} (daily closes, IEX) ===")

    rows = {}
    for sym in INDEXES:
        rows[sym] = analyze(closes[sym])
    hdr = (f"{'':<6}{'last':>8}{'MA50':>8}{'MA120':>8}{'MA200':>8}{'200d':>7}"
           f"{'RSI':>6}{'MACDh':>7}{'off-hi':>8}{'vol20':>7}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for sym, a in rows.items():
        print(f"{sym:<6}{a['px']:>8.0f}{a['ma'][50]:>8.0f}{a['ma'][120]:>8.0f}"
              f"{a['ma'][200]:>8.0f}{'rise' if a['ma200_rising'] else 'FALL':>7}"
              f"{a['rsi']:>6.0f}{a['macd_hist']:>+7.2f}{a['dd52']:>+8.1f}%"
              f"{a['vol20_ann']:>6.0f}%")

    watch = {s: c for s, c in closes.items() if s in watchlist}
    br = breadth(watch)
    print(f"\nBREADTH (our {br['n']}-name watchlist): {br['pct200']:.0f}% above 200d MA, "
          f"{br['pct50']:.0f}% above 50d MA")

    spy = rows["SPY"]
    score, votes = composite(spy, br)
    print(f"\nWEIGHT OF EVIDENCE (SPY + breadth) — score {score:+d} of +/-10:")
    print("\n".join(votes))
    print(f"\nVERDICT: {label(score)}")

    print("\nBUY-THE-DIP READ:")
    print("\n".join(dip_read(spy)))

    print("\nDOWNTREND-ENDING (turn) CHECKLIST:")
    print("\n".join(turn_checklist(spy, br)))

    print("\nNote: descriptive read for the human, not a bot input — the live bot's regime")
    print("logic (vol-dial + trend filter) is unchanged. Re-run any morning: ")
    print("  .venv/Scripts/python.exe scripts/market_regime.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
