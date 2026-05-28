"""PEAD research backtest — does post-earnings drift exist, and which filters predict it?

Research-first (per plans/put-yourself-as-an-majestic-cupcake.md): the earlier
n=5 test showed naive "buy any beat" trails SPY. This collects ALL earnings
events for a broad universe over a MULTI-YEAR window (earnings are quarterly, so
a 6-month window is far too few events), measures market-relative drift, attaches
candidate filter factors, and asks:
  1. Baseline: is there any drift at all? (expect ~flat/negative = decayed)
  2. Per-factor sorts: does conditioning on a factor produce monotonic drift?
  3. OOS split: does any promising factor hold in BOTH halves?

Self-contained: yfinance + pandas only (NO alpaca import), so it runs in the
.venv-openbb environment that has yfinance.

  .venv-openbb\\Scripts\\python.exe backtest\\run_pead.py

Writes the event table to backtest/pead_events.csv so re-analysis is free.

Caveats baked in / flagged:
- yfinance "EPS Estimate" is a current snapshot; surprise% may not be perfectly
  point-in-time. Treat as indicative.
- Entry strictly AFTER the announcement is public (AMC -> next open, BMO -> same
  open), so no lookahead.
- Fixed modern universe over history = survivorship bias; fine for a first read,
  not for sizing.
"""
from __future__ import annotations

import io
import sys
from datetime import timedelta

import numpy as np
import pandas as pd
import yfinance as yf

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Broad liquid single-name universe (ETFs excluded — no earnings). Fixed list =
# survivorship bias; acceptable for a first read.
UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "ORCL",
    "AMD", "NFLX", "ADBE", "CRM", "INTC", "CSCO", "QCOM", "TXN", "MU",
    "JPM", "BAC", "WFC", "GS", "MS", "C", "V", "MA", "AXP",
    "UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV", "TMO", "ABT",
    "WMT", "HD", "COST", "NKE", "MCD", "SBUX", "DIS", "KO", "PEP", "PG", "TGT",
    "XOM", "CVX", "COP", "CAT", "BA", "GE", "HON", "UPS",
    "PLTR", "UBER", "BABA", "QCOM",
]
UNIVERSE = sorted(set(UNIVERSE))

WINDOW_START = "2021-01-01"
HOLD_PERIODS = [1, 3, 5, 10, 21]
PRIMARY_HOLD = 10           # the hold used for factor sorts
MIN_SURPRISE = 0.0          # baseline keeps all *beats* (surprise > 0)
EVENTS_CSV = __file__.replace("run_pead.py", "pead_events.csv")


def collect_events(tickers):
    """Per ticker, pull all reported earnings with surprise%, keep those in-window."""
    start = pd.Timestamp(WINDOW_START, tz="America/New_York")
    cutoff = pd.Timestamp.now(tz="America/New_York") - pd.Timedelta(days=35)  # need post-data
    events = []
    for i, sym in enumerate(tickers):
        try:
            ed = yf.Ticker(sym).get_earnings_dates(limit=48)
        except Exception as e:
            print(f"  {sym}: earnings fetch failed ({e})")
            continue
        if ed is None or ed.empty:
            continue
        ed = ed.dropna(subset=["Reported EPS", "Surprise(%)"])
        for edate, row in ed.iterrows():
            if edate < start or edate > cutoff:
                continue
            events.append({
                "symbol": sym,
                "edate": edate,
                "surprise_pct": float(row["Surprise(%)"]) * (100.0 if abs(float(row["Surprise(%)"])) < 1.5 else 1.0),
                "eps_est": float(row["EPS Estimate"]),
                "eps_act": float(row["Reported EPS"]),
            })
        print(f"  [{i+1}/{len(tickers)}] {sym}: {sum(1 for e in events if e['symbol']==sym)} events")
    return events


def download_daily(tickers):
    """Batch daily OHLC for all tickers + SPY. Returns dict[sym] -> DataFrame."""
    start = (pd.Timestamp(WINDOW_START) - pd.Timedelta(days=400)).date()  # pad for SMA/pre-drift
    syms = sorted(set(tickers) | {"SPY"})
    raw = yf.download(syms, start=start, auto_adjust=True, progress=False, group_by="ticker")
    out = {}
    for s in syms:
        try:
            df = raw[s].copy()
        except Exception:
            continue
        df = df.dropna(subset=["Open", "Close"])
        if df.empty:
            continue
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        out[s] = df
    return out


def build_table(events, daily):
    spy = daily.get("SPY")
    if spy is None:
        raise RuntimeError("No SPY data")
    spy_sma200 = spy["Close"].rolling(200).mean()
    rows = []
    for ev in events:
        sym = ev["symbol"]
        px = daily.get(sym)
        if px is None:
            continue
        edate = ev["edate"]
        edate_naive = pd.Timestamp(edate.date())
        amc = edate.hour >= 16
        entry_target = edate_naive + pd.Timedelta(days=1) if amc else edate_naive

        future = px[px.index >= entry_target]
        pre = px[px.index < entry_target]
        if len(future) < max(HOLD_PERIODS) + 1 or pre.empty:
            continue
        entry_day = future.index[0]
        entry_price = float(future.iloc[0]["Open"])
        pre_close = float(pre.iloc[-1]["Close"])
        gap_pct = (entry_price / pre_close - 1) * 100

        # pre-announcement 20d drift (momentum into the print)
        pre20 = pre.iloc[-21:]
        pre_drift = (pre_close / float(pre20.iloc[0]["Close"]) - 1) * 100 if len(pre20) >= 2 else np.nan

        # liquidity: avg dollar volume over prior 20 sessions
        if "Volume" in pre.columns:
            dv = (pre["Close"] * pre["Volume"]).iloc[-20:].mean()
        else:
            dv = np.nan

        # market regime at the event: SPY above its 200d SMA?
        try:
            spy_at = spy.index.asof(entry_target)
            spy_close = float(spy.loc[spy_at, "Close"])
            spy_sma = float(spy_sma200.loc[spy_at])
            regime_up = spy_close > spy_sma if not np.isnan(spy_sma) else np.nan
        except Exception:
            regime_up = np.nan

        rec = {**{k: ev[k] for k in ("symbol", "surprise_pct", "eps_est", "eps_act")},
               "edate": edate_naive, "entry_day": entry_day, "amc": amc,
               "gap_pct": gap_pct, "pre_drift_pct": pre_drift,
               "dollar_vol": dv, "regime_up": regime_up}

        # market-relative drift at each hold N (entry open -> exit close N bars out)
        try:
            spy_fut = spy[spy.index >= entry_target]
            spy_entry = float(spy_fut.iloc[0]["Open"])
        except Exception:
            spy_entry = None
        for n in HOLD_PERIODS:
            exit_price = float(future.iloc[n]["Close"])
            raw_r = (exit_price / entry_price - 1) * 100
            if spy_entry is not None and len(spy_fut) > n:
                spy_r = (float(spy_fut.iloc[n]["Close"]) / spy_entry - 1) * 100
                rec[f"rel_{n}"] = raw_r - spy_r
            else:
                rec[f"rel_{n}"] = np.nan
            rec[f"raw_{n}"] = raw_r
        rows.append(rec)
    return pd.DataFrame(rows)


def _stat(series):
    s = series.dropna()
    if len(s) == 0:
        return (0, np.nan, np.nan, np.nan)
    mean = s.mean()
    se = s.std() / np.sqrt(len(s)) if len(s) > 1 else np.nan
    t = mean / se if se and se > 0 else np.nan
    pos = (s > 0).mean() * 100
    return (len(s), mean, t, pos)


def baseline(df):
    print("\n" + "=" * 70)
    print("BASELINE: market-relative drift, all beats (surprise > 0)")
    print("=" * 70)
    beats = df[df["surprise_pct"] > MIN_SURPRISE]
    print(f"  events: {len(beats)} (of {len(df)} total)")
    print(f"  {'hold':>5}{'n':>6}{'mean_rel%':>11}{'t':>7}{'win%':>7}")
    for n in HOLD_PERIODS:
        cnt, mean, t, pos = _stat(beats[f"rel_{n}"])
        print(f"  {n:>5}{cnt:>6}{mean:>+11.3f}{t:>+7.2f}{pos:>6.1f}%")
    print("  (PEAD claim: positive & rising mean_rel with hold. Decay -> ~flat/neg.)")


def factor_sorts(df):
    print("\n" + "=" * 70)
    print(f"FACTOR SORTS on market-relative drift at hold={PRIMARY_HOLD}d (terciles)")
    print("=" * 70)
    beats = df[df["surprise_pct"] > MIN_SURPRISE].copy()
    col = f"rel_{PRIMARY_HOLD}"
    factors = ["surprise_pct", "gap_pct", "pre_drift_pct", "dollar_vol"]
    for f in factors:
        sub = beats.dropna(subset=[f, col])
        if len(sub) < 30:
            print(f"\n  {f}: too few ({len(sub)})")
            continue
        try:
            sub["bucket"] = pd.qcut(sub[f], 3, labels=["low", "mid", "high"])
        except Exception:
            print(f"\n  {f}: cannot bucket")
            continue
        print(f"\n  by {f}:")
        print(f"    {'bucket':<6}{'n':>5}{'mean_rel%':>11}{'t':>7}{'win%':>7}")
        for b in ["low", "mid", "high"]:
            cnt, mean, t, pos = _stat(sub[sub["bucket"] == b][col])
            print(f"    {b:<6}{cnt:>5}{mean:>+11.3f}{t:>+7.2f}{pos:>6.1f}%")
    # regime (binary)
    print(f"\n  by regime_up (SPY > 200d SMA at event):")
    print(f"    {'regime':<6}{'n':>5}{'mean_rel%':>11}{'t':>7}{'win%':>7}")
    for val, lab in [(True, "up"), (False, "down")]:
        cnt, mean, t, pos = _stat(beats[beats["regime_up"] == val][col])
        print(f"    {lab:<6}{cnt:>5}{mean:>+11.3f}{t:>+7.2f}{pos:>6.1f}%")


def oos_split(df):
    print("\n" + "=" * 70)
    print(f"OOS SPLIT (hold={PRIMARY_HOLD}d): does the top surprise-tercile hold in BOTH halves?")
    print("=" * 70)
    beats = df[df["surprise_pct"] > MIN_SURPRISE].dropna(subset=[f"rel_{PRIMARY_HOLD}"]).copy()
    beats = beats.sort_values("edate")
    mid = beats["edate"].iloc[len(beats) // 2]
    col = f"rel_{PRIMARY_HOLD}"
    for half_name, part in [("first half", beats[beats["edate"] < mid]),
                            ("second half", beats[beats["edate"] >= mid])]:
        if len(part) < 30:
            print(f"  {half_name}: too few ({len(part)})")
            continue
        try:
            part = part.copy()
            part["b"] = pd.qcut(part["surprise_pct"], 3, labels=["low", "mid", "high"])
            cnt, mean, t, pos = _stat(part[part["b"] == "high"][col])
            allc, allm, allt, allp = _stat(part[col])
            print(f"  {half_name}: all beats mean_rel {allm:+.3f} (n={allc}); "
                  f"top-surprise tercile {mean:+.3f} (t={t:+.2f}, n={cnt})")
        except Exception as e:
            print(f"  {half_name}: {e}")


def main():
    print(f"PEAD research: {len(UNIVERSE)} names, window {WINDOW_START} -> ~now")
    print("Collecting earnings events (yfinance, per-ticker)...")
    events = collect_events(UNIVERSE)
    print(f"\nTotal in-window reported events: {len(events)}")
    if not events:
        print("No events; aborting.")
        return 1
    print("Downloading daily bars (batch)...")
    daily = download_daily(UNIVERSE)
    print(f"Daily frames: {len(daily)} symbols")
    df = build_table(events, daily)
    print(f"Usable events (full post-window): {len(df)}")
    if df.empty:
        print("No usable events; aborting.")
        return 1
    try:
        df.to_csv(EVENTS_CSV, index=False)
        print(f"Saved event table -> {EVENTS_CSV}")
    except Exception as e:
        print(f"(could not save CSV: {e})")

    baseline(df)
    factor_sorts(df)
    oos_split(df)

    print("\nGATE: ship a live scanner ONLY if a factor gives POSITIVE market-relative")
    print("drift that holds in BOTH OOS halves with a sane event count. Otherwise stop.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
