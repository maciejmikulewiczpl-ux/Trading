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

`snapshot()` returns the whole read as a JSON-safe dict — consumed by both the
CLI below and the status page's Market tab (live/status_server.py /api/regime).

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
    return {
        "px": px,
        "ret5": (px / float(close.iloc[-6]) - 1.0) * 100 if len(close) > 6 else 0.0,
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
# TWO-AXIS verdict (reworked 2026-06-11 after the v1 label failure: with SPY -4.5%
# in 4 sessions the single blended score still said "BULLISH" because long-term
# structure outvoted a sharp short-term fall). STRUCTURE (where price is vs the
# slow averages + breadth, +/-7) and MOMENTUM (which way it's moving right now,
# +/-6) are scored separately and BOTH are shown — "uptrend under pressure" and
# "bullish" are different states and the label must say which one we're in.
def composite(spy: dict, br: dict) -> dict:
    """{votes, struct_score, mom_score, struct_state, mom_state}.
    votes = [{pts, grp: structure|momentum, name, detail}]."""
    votes: list[dict] = []
    scores = {"structure": 0, "momentum": 0}

    def vote(grp, cond_pos, cond_neg, w, name, detail):
        s = w if cond_pos else (-w if cond_neg else 0)
        scores[grp] += s
        votes.append({"pts": s, "grp": grp, "name": name, "detail": detail})

    # STRUCTURE — the slow axis (+/-7)
    vote("structure", spy["px"] > spy["ma"][200], spy["px"] <= spy["ma"][200], 2,
         "price vs 200d MA", f"{spy['px']:.0f} vs {spy['ma'][200]:.0f}")
    vote("structure", spy["ma200_rising"], not spy["ma200_rising"], 1,
         "200d MA slope (1mo)", "rising" if spy["ma200_rising"] else "falling")
    vote("structure", spy["golden"], not spy["golden"], 1,
         "50d vs 200d MA", "golden cross" if spy["golden"] else "death cross")
    vote("structure", br["pct200"] > 55, br["pct200"] < 45, 2,
         "breadth >200d MA", f"{br['pct200']:.0f}% of {br['n']} watchlist names")
    vote("structure", spy["dd52"] > -5, spy["dd52"] < -15, 1,
         "off 52-week high", f"{spy['dd52']:+.1f}%")

    # MOMENTUM — the fast axis (+/-6)
    vote("momentum", spy["ret5"] > 1.0, spy["ret5"] < -1.0, 1,
         "5-day return", f"{spy['ret5']:+.1f}%")
    vote("momentum", spy["px"] > spy["ma"][50], spy["px"] <= spy["ma"][50], 1,
         "price vs 50d MA", f"{spy['px']:.0f} vs {spy['ma'][50]:.0f}")
    vote("momentum", spy["macd_above_sig"] and spy["macd_hist"] > 0,
         not spy["macd_above_sig"] and spy["macd_hist"] < 0, 1,
         "MACD (12-26-9)", f"hist {spy['macd_hist']:+.2f}")
    vote("momentum", spy["rsi"] > 55, spy["rsi"] < 45, 1,
         "RSI-14 zone", f"{spy['rsi']:.0f}")
    vote("momentum", not spy["vol_hot"], spy["vol_hot"], 1,
         "vol regime (live dial)", f"20d {spy['vol20_ann']:.0f}% ann, "
         f"{'> 126d median (ELEVATED)' if spy['vol_hot'] else '<= median (calm)'}")
    vote("momentum", br["pct50"] > 55, br["pct50"] < 45, 1,
         "breadth >50d MA", f"{br['pct50']:.0f}% of watchlist names")

    ss, ms = scores["structure"], scores["momentum"]
    return {"votes": votes, "struct_score": ss, "mom_score": ms,
            "struct_state": "UP" if ss >= 3 else "BROKEN" if ss <= -3 else "MIXED",
            "mom_state": "RISING" if ms >= 2 else "FALLING" if ms <= -2 else "FLAT"}


# (structure, momentum) -> (verdict label, tone for the banner color)
VERDICTS = {
    ("UP", "RISING"):     ("BULLISH — uptrend with momentum", "good"),
    ("UP", "FLAT"):       ("BULLISH — uptrend, momentum cooling", "good"),
    ("UP", "FALLING"):    ("UPTREND UNDER PRESSURE — correction in progress", "caution"),
    ("MIXED", "RISING"):  ("REPAIRING — structure mixed, momentum improving", "neutral"),
    ("MIXED", "FLAT"):    ("NEUTRAL / range-bound", "neutral"),
    ("MIXED", "FALLING"): ("WEAKENING — structure cracking, momentum down", "caution"),
    ("BROKEN", "RISING"): ("DOWNTREND BOUNCE — watch the turn checklist", "caution"),
    ("BROKEN", "FLAT"):   ("BEARISH — downtrend", "bad"),
    ("BROKEN", "FALLING"): ("BEARISH — downtrend in force", "bad"),
}


def dip_assess(spy: dict, mom_state: str) -> dict:
    """{structure, oversold, verdict, note} — the buy-the-dip read."""
    structure = spy["px"] > spy["ma"][200] and (spy["ma200_rising"] or spy["golden"])
    oversold = spy["rsi"] < 35 or (spy["px"] < spy["ma"][50] and -12 < spy["dd52"] < -3)
    if structure and oversold:
        verdict, note = "BUYABLE-DIP ZONE", \
            ("Pullback inside an intact uptrend — the kind that has historically "
             "resolved up. (Not a validated signal; size accordingly.)")
    elif structure and mom_state == "FALLING":
        verdict, note = "DIP FORMING — not at trigger yet", \
            (f"Correction inside an intact uptrend, but not yet at the classic dip "
             f"triggers (RSI<35 — now {spy['rsi']:.0f} — or below the 50d MA with a "
             f"3-12% drawdown — now {spy['dd52']:+.1f}%). Falling momentum says don't "
             f"front-run it; let it reach a trigger or stabilize.")
    elif structure:
        verdict, note = "NO DIP ON OFFER", \
            "Trend intact and not stretched. Nothing to time."
    elif oversold:
        verdict, note = "KNIFE — NOT A DIP", \
            ("Oversold BELOW a broken 200d MA = downtrend rally risk. "
             "Wait for the turn checklist before buying weakness.")
    else:
        verdict, note = "DOWNTREND", "Not yet washed out. Watch the turn checklist."
    return {"structure": structure, "oversold": oversold, "verdict": verdict, "note": note}


def turn_checks(spy: dict, br: dict) -> list[dict]:
    """Classic bottoming markers — [{name, on}]. 4+ on = a turn forming."""
    return [{"name": n, "on": bool(ok)} for n, ok in [
        ("RSI washed out (<35 in last 60d) then recovered >40",
         spy["rsi_low60"] < 35 and spy["rsi"] > 40 and spy["rsi"] > spy["rsi_10d_ago"]),
        ("MACD crossed up (10d) or histogram rising 4 of 5 days",
         spy["macd_cross_up_10d"] or spy["macd_hist_rising"]),
        ("price reclaimed the 50d MA", spy["px"] > spy["ma"][50]),
        ("50d MA slope turned up (2wk)", spy["ma50_rising"]),
        ("volatility compressing (20d vol < 2wk ago)", spy["vol_falling"]),
        ("breadth repair: >50% of watchlist above 50d MA", br["pct50"] > 50),
    ]]


def snapshot() -> dict:
    """The full regime read as a JSON-safe dict (CLI + status-page Market tab)."""
    watchlist = list(orb_config.load_config()["watchlist"])
    closes = fetch_daily(sorted(set(INDEXES + watchlist)))
    missing = [s for s in INDEXES if s not in closes]
    if missing:
        return {"error": f"missing index data: {missing}"}
    idx = {sym: analyze(closes[sym]) for sym in INDEXES}
    br = breadth({s: c for s, c in closes.items() if s in watchlist})
    spy = idx["SPY"]
    comp = composite(spy, br)
    verdict, tone = VERDICTS[(comp["struct_state"], comp["mom_state"])]
    checks = turn_checks(spy, br)
    return {
        "generated": datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "asof": closes["SPY"].index[-1].date().isoformat(),
        "indexes": idx,
        "breadth": br,
        **comp,                      # votes, struct_score, mom_score, *_state
        "verdict": verdict,
        "tone": tone,
        "dip": dip_assess(spy, comp["mom_state"]),
        "turn": checks,
        "n_turn_on": sum(1 for c in checks if c["on"]),
    }


def main() -> int:
    snap = snapshot()
    if snap.get("error"):
        print(snap["error"])
        return 1
    print(f"=== MARKET REGIME GAUGE — data through {snap['asof']} (daily closes, IEX) ===")

    hdr = (f"{'':<6}{'last':>8}{'MA50':>8}{'MA120':>8}{'MA200':>8}{'200d':>7}"
           f"{'RSI':>6}{'MACDh':>7}{'off-hi':>8}{'vol20':>7}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for sym, a in snap["indexes"].items():
        print(f"{sym:<6}{a['px']:>8.0f}{a['ma'][50]:>8.0f}{a['ma'][120]:>8.0f}"
              f"{a['ma'][200]:>8.0f}{'rise' if a['ma200_rising'] else 'FALL':>7}"
              f"{a['rsi']:>6.0f}{a['macd_hist']:>+7.2f}{a['dd52']:>+8.1f}%"
              f"{a['vol20_ann']:>6.0f}%")

    br = snap["breadth"]
    print(f"\nBREADTH (our {br['n']}-name watchlist): {br['pct200']:.0f}% above 200d MA, "
          f"{br['pct50']:.0f}% above 50d MA")

    for grp, score, lim, state in [
            ("structure", snap["struct_score"], 7, snap["struct_state"]),
            ("momentum", snap["mom_score"], 6, snap["mom_state"])]:
        print(f"\n{grp.upper()} — score {score:+d} of +/-{lim} -> {state}:")
        for v in snap["votes"]:
            if v["grp"] != grp:
                continue
            pts = f"+{v['pts']}" if v["pts"] > 0 else (str(v["pts"]) if v["pts"] < 0 else " 0")
            print(f"  {pts:>3}  {v['name']:<26} {v['detail']}")
    print(f"\nVERDICT: {snap['verdict']}")

    d = snap["dip"]
    spy = snap["indexes"]["SPY"]
    print("\nBUY-THE-DIP READ:")
    print(f"  long-term structure : {'INTACT (above ~rising 200d MA)' if d['structure'] else 'BROKEN (below/falling 200d MA)'}")
    print(f"  short-term stretch  : {'OVERSOLD' if d['oversold'] else 'not oversold'} "
          f"(RSI {spy['rsi']:.0f}, {spy['dd52']:+.1f}% off high)")
    print(f"  => {d['verdict']}: {d['note']}")

    print("\nDOWNTREND-ENDING (turn) CHECKLIST:")
    for c in snap["turn"]:
        print(f"  [{'x' if c['on'] else ' '}] {c['name']}")
    n_on = snap["n_turn_on"]
    print(f"  {n_on}/6 bottoming markers ON"
          + (" — turn forming" if n_on >= 4 else " — no confirmed turn yet" if n_on <= 2 else ""))

    print("\nNote: descriptive read for the human, not a bot input — the live bot's regime")
    print("logic (vol-dial + trend filter) is unchanged. Re-run any morning: ")
    print("  .venv/Scripts/python.exe scripts/market_regime.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
