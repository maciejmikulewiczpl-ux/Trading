"""Single-event PEAD simulation on the most recent earnings call of a non-AI stock.

Two phases:
  1) Scan a small list of non-AI tickers, find the one with the most recent
     positive EPS surprise that we have at least HOLD_DAYS of price data after.
  2) Simulate: enter at the OPEN of the first trading day after the announcement,
     hold HOLD_DAYS trading days (or until today if earlier), exit at close.
     Report raw and SPY-adjusted returns, plus the day-by-day equity curve.

Run with:
    .\\.venv-openbb\\Scripts\\python.exe scripts\\pead_sim.py
"""
from __future__ import annotations

import io
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Non-AI, large-cap, US-listed candidates that report quarterly on roughly the
# Jan/Apr/Jul/Oct cadence. Mixed sectors so we have options.
CANDIDATES = [
    "WMT",   # Walmart - retail
    "HD",    # Home Depot - retail/home improvement
    "COST",  # Costco - retail
    "KO",    # Coca-Cola - consumer staples
    "PG",    # Procter & Gamble - consumer staples
    "JPM",   # JPMorgan - financials
    "V",     # Visa - financials/payments
    "JNJ",   # J&J - healthcare
    "XOM",   # Exxon - energy
    "DIS",   # Disney - media
]

POSITION_DOLLARS = 10_000.0
MIN_SURPRISE_PCT = 3.0      # require a meaningful beat
MIN_DAYS_SINCE = 7          # need at least a week of post-earnings price action
HOLD_DAYS = 21              # ~1 calendar month of trading days

# -----------------------------------------------------------
# Phase 1: scan
# -----------------------------------------------------------
def find_candidate():
    today = pd.Timestamp.now(tz="America/New_York")
    print(f"Scanning non-AI candidates for a recent EPS beat (surprise > {MIN_SURPRISE_PCT}%)...")
    print()
    rows = []
    for sym in CANDIDATES:
        t = yf.Ticker(sym)
        try:
            ed = t.earnings_dates
        except Exception as e:
            print(f"  {sym}: earnings_dates fetch failed ({e})")
            continue
        if ed is None or ed.empty:
            continue
        reported = ed.dropna(subset=["Reported EPS"]).sort_index(ascending=False)
        if reported.empty:
            continue
        last = reported.iloc[0]
        edate = last.name  # tz-aware timestamp
        days_since = (today - edate).days
        surprise = float(last["Surprise(%)"])
        eps_est = float(last["EPS Estimate"])
        eps_act = float(last["Reported EPS"])
        rows.append({
            "symbol": sym, "edate": edate, "days_since": days_since,
            "eps_est": eps_est, "eps_act": eps_act, "surprise_pct": surprise,
        })
        print(f"  {sym:5s}  {edate.strftime('%Y-%m-%d %H:%M %Z')}  "
              f"d_ago={days_since:3d}  est=${eps_est:>6.3f}  act=${eps_act:>6.3f}  "
              f"surprise={surprise:+6.2f}%")
    print()
    # Filter: positive beat, enough days post-earnings, then prefer the most recent
    eligible = [r for r in rows
                if r["surprise_pct"] >= MIN_SURPRISE_PCT and r["days_since"] >= MIN_DAYS_SINCE]
    if not eligible:
        return None
    eligible.sort(key=lambda r: r["edate"], reverse=True)
    return eligible[0]


# -----------------------------------------------------------
# Phase 2: simulate
# -----------------------------------------------------------
def simulate(pick: dict):
    sym = pick["symbol"]
    edate = pick["edate"]

    print("=" * 72)
    print(f"PEAD simulation: {sym}")
    print("=" * 72)
    print(f"Earnings announcement: {edate.strftime('%Y-%m-%d %H:%M %Z')}")
    print(f"  EPS estimate (consensus): ${pick['eps_est']:.3f}")
    print(f"  EPS actual (reported):    ${pick['eps_act']:.3f}")
    print(f"  Surprise:                 {pick['surprise_pct']:+.2f}%")
    print()

    # Pull a wide price window: 5 trading days before to ~30 trading days after.
    start = (edate - timedelta(days=15)).date()
    end = (edate + timedelta(days=HOLD_DAYS * 2 + 5)).date()
    px = yf.download(sym, start=start, end=end, auto_adjust=True, progress=False, group_by="column")
    spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False, group_by="column")
    if isinstance(px.columns, pd.MultiIndex):
        px.columns = [c[0] for c in px.columns]
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = [c[0] for c in spy.columns]
    if px.empty:
        print(f"No price data for {sym}; aborting.")
        return
    # Normalize index to date-only naive Timestamps for clean comparison
    px.index = pd.to_datetime(px.index).tz_localize(None).normalize()
    spy.index = pd.to_datetime(spy.index).tz_localize(None).normalize()

    edate_naive = pd.Timestamp(edate.date())
    # If announcement was AMC (after 16:00), next trading day is the entry day.
    # If BMO (before market open), the same day is the entry day (open after the news).
    if edate.hour >= 16:
        entry_day_target = edate_naive + pd.Timedelta(days=1)
        timing = "AMC (after market close)"
    else:
        entry_day_target = edate_naive
        timing = "BMO (before market open)"
    print(f"Announcement timing: {timing}")

    future = px[px.index >= entry_day_target]
    if future.empty:
        print("Not enough post-earnings price data.")
        return
    entry_day = future.index[0]
    entry_price = float(future.iloc[0]["Open"])
    last_close = float(future.iloc[-1]["Close"])

    hold_days_actual = min(HOLD_DAYS, len(future) - 1)
    if hold_days_actual <= 0:
        print("Not enough trading days post-earnings to simulate.")
        return
    exit_idx = hold_days_actual  # 0=entry day; we exit at close of bar `hold_days_actual`
    exit_day = future.index[exit_idx]
    exit_price = float(future.iloc[exit_idx]["Close"])

    # Reference: close of previous trading day (pre-announcement)
    pre = px[px.index < entry_day_target]
    pre_close = float(pre.iloc[-1]["Close"]) if not pre.empty else None
    gap_open_pct = ((entry_price / pre_close) - 1) * 100 if pre_close else None

    # SPY market-relative
    try:
        spy_entry = float(spy.loc[entry_day, "Open"])
        spy_exit = float(spy.loc[exit_day, "Close"])
        spy_return_pct = ((spy_exit / spy_entry) - 1) * 100
    except KeyError:
        spy_entry = spy_exit = spy_return_pct = None

    shares = int(POSITION_DOLLARS // entry_price)
    cost_basis = shares * entry_price
    market_value = shares * exit_price
    pnl_dollars = market_value - cost_basis
    raw_return_pct = ((exit_price / entry_price) - 1) * 100
    mkt_relative_pct = (raw_return_pct - spy_return_pct) if spy_return_pct is not None else None

    print()
    print(f"Pre-earnings close:    ${pre_close:.2f}" if pre_close else "Pre-earnings close: (n/a)")
    print(f"Entry (open of {entry_day.strftime('%Y-%m-%d')}): ${entry_price:.2f}"
          + (f"   [gap from prev close: {gap_open_pct:+.2f}%]" if gap_open_pct is not None else ""))
    print(f"Exit  (close of {exit_day.strftime('%Y-%m-%d')}): ${exit_price:.2f}   "
          f"[held {hold_days_actual} trading days]")
    print(f"Last available close:  ${last_close:.2f}  (for reference)")
    print()
    print(f"Position:    {shares} shares of {sym} @ ${entry_price:.2f}  =  ${cost_basis:,.2f} cost basis")
    print(f"P&L:         ${pnl_dollars:+,.2f}")
    print(f"Return:      {raw_return_pct:+.2f}% raw"
          + (f"   |   SPY over same window: {spy_return_pct:+.2f}%   |   market-relative: {mkt_relative_pct:+.2f}%"
             if spy_return_pct is not None else ""))
    print()
    # Day-by-day equity curve relative to entry
    eq = (future.iloc[:exit_idx+1]["Close"] / entry_price - 1) * 100
    spy_window = spy.loc[entry_day:exit_day, "Close"]
    if not spy_window.empty:
        spy_eq = (spy_window / float(spy.loc[entry_day, "Open"]) - 1) * 100
    else:
        spy_eq = pd.Series(dtype=float)

    print(f"Day-by-day (% return from entry):")
    print(f"  {'date':<12} {sym+' close':>10}  {sym+' %':>8}    {'SPY %':>8}    {'relative %':>10}")
    for i, (dt, row) in enumerate(future.iloc[:exit_idx+1].iterrows()):
        s_pct = (float(row["Close"]) / entry_price - 1) * 100
        spy_pct = None
        try:
            spy_pct = (float(spy.loc[dt, "Close"]) / spy_entry - 1) * 100
        except (KeyError, TypeError):
            pass
        rel = (s_pct - spy_pct) if spy_pct is not None else None
        spy_str = f"{spy_pct:+7.2f}%" if spy_pct is not None else "    n/a"
        rel_str = f"{rel:+9.2f}%" if rel is not None else "      n/a"
        print(f"  {dt.strftime('%Y-%m-%d'):<12} ${float(row['Close']):>9.2f}  {s_pct:+7.2f}%    {spy_str}    {rel_str}")

    print()
    print("Interpretation:")
    print(f"  - The 'gap from prev close' is the IMMEDIATE reaction to the news (already in entry price).")
    print(f"  - PEAD claims further drift AFTER the gap, in the direction of the surprise.")
    print(f"  - Market-relative return is the better signal: it strips out whatever SPY did in the window.")


# -----------------------------------------------------------
def find_all_candidates() -> list[dict]:
    """Same scan as find_candidate, but return EVERY eligible row, newest first."""
    today = pd.Timestamp.now(tz="America/New_York")
    rows = []
    for sym in CANDIDATES:
        t = yf.Ticker(sym)
        try:
            ed = t.earnings_dates
        except Exception:
            continue
        if ed is None or ed.empty:
            continue
        reported = ed.dropna(subset=["Reported EPS"]).sort_index(ascending=False)
        if reported.empty:
            continue
        last = reported.iloc[0]
        edate = last.name
        days_since = (today - edate).days
        surprise = float(last["Surprise(%)"])
        if surprise < MIN_SURPRISE_PCT or days_since < MIN_DAYS_SINCE:
            continue
        rows.append({
            "symbol": sym, "edate": edate, "days_since": days_since,
            "eps_est": float(last["EPS Estimate"]),
            "eps_act": float(last["Reported EPS"]),
            "surprise_pct": surprise,
        })
    rows.sort(key=lambda r: r["edate"], reverse=True)
    return rows


def simulate_quiet(pick: dict) -> dict | None:
    """Same simulation as simulate() but returns a result dict, no printing."""
    sym = pick["symbol"]
    edate = pick["edate"]
    start = (edate - timedelta(days=15)).date()
    end = (edate + timedelta(days=HOLD_DAYS * 2 + 5)).date()
    px = yf.download(sym, start=start, end=end, auto_adjust=True, progress=False, group_by="column")
    spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False, group_by="column")
    if isinstance(px.columns, pd.MultiIndex):
        px.columns = [c[0] for c in px.columns]
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = [c[0] for c in spy.columns]
    if px.empty:
        return None
    px.index = pd.to_datetime(px.index).tz_localize(None).normalize()
    spy.index = pd.to_datetime(spy.index).tz_localize(None).normalize()
    edate_naive = pd.Timestamp(edate.date())
    if edate.hour >= 16:
        entry_day_target = edate_naive + pd.Timedelta(days=1)
        timing = "AMC"
    else:
        entry_day_target = edate_naive
        timing = "BMO"
    future = px[px.index >= entry_day_target]
    if future.empty:
        return None
    entry_day = future.index[0]
    entry_price = float(future.iloc[0]["Open"])
    hold = min(HOLD_DAYS, len(future) - 1)
    if hold <= 0:
        return None
    exit_day = future.index[hold]
    exit_price = float(future.iloc[hold]["Close"])
    pre = px[px.index < entry_day_target]
    pre_close = float(pre.iloc[-1]["Close"]) if not pre.empty else None
    gap_pct = ((entry_price / pre_close) - 1) * 100 if pre_close else None
    try:
        spy_entry = float(spy.loc[entry_day, "Open"])
        spy_exit = float(spy.loc[exit_day, "Close"])
        spy_return_pct = ((spy_exit / spy_entry) - 1) * 100
    except KeyError:
        spy_return_pct = None
    raw_return = ((exit_price / entry_price) - 1) * 100
    mkt_rel = (raw_return - spy_return_pct) if spy_return_pct is not None else None
    shares = int(POSITION_DOLLARS // entry_price)
    pnl = shares * (exit_price - entry_price)
    return {
        **pick,
        "timing": timing,
        "pre_close": pre_close,
        "entry_day": entry_day,
        "entry_price": entry_price,
        "gap_pct": gap_pct,
        "exit_day": exit_day,
        "exit_price": exit_price,
        "hold_days": hold,
        "raw_return_pct": raw_return,
        "spy_return_pct": spy_return_pct,
        "mkt_relative_pct": mkt_rel,
        "shares": shares,
        "pnl": pnl,
    }


def main():
    eligible = find_all_candidates()
    if not eligible:
        print(f"No eligible candidate (need surprise>{MIN_SURPRISE_PCT}%, >={MIN_DAYS_SINCE}d post-earnings).")
        return

    print()
    print(f"Eligible non-AI beats with >= {MIN_DAYS_SINCE}d of post-earnings data:")
    print()
    print(f"  {'symbol':<7}{'edate':<14}{'d_ago':>6}  {'surprise':>10}")
    for r in eligible:
        print(f"  {r['symbol']:<7}{r['edate'].strftime('%Y-%m-%d'):<14}{r['days_since']:>6}  "
              f"{r['surprise_pct']:>+9.2f}%")
    print()
    print(f"Simulating PEAD on ALL {len(eligible)} of them: enter at next-session open,")
    print(f"hold up to {HOLD_DAYS} trading days, exit at close. $10,000 notional per trade.")
    print()

    results = []
    for r in eligible:
        sim = simulate_quiet(r)
        if sim is not None:
            results.append(sim)

    print("=" * 90)
    print(f"{'sym':<5}{'surprise':>10}  {'gap%':>7}  {'hold':>4}  {'raw%':>7}  {'SPY%':>7}  {'mkt_rel%':>9}  {'P&L $':>10}")
    print("-" * 90)
    for s in results:
        print(f"{s['symbol']:<5}{s['surprise_pct']:>+9.2f}%  "
              f"{s['gap_pct']:>+6.2f}%  "
              f"{s['hold_days']:>4}  "
              f"{s['raw_return_pct']:>+6.2f}%  "
              f"{s['spy_return_pct']:>+6.2f}%  "
              f"{s['mkt_relative_pct']:>+8.2f}%  "
              f"${s['pnl']:>+9.2f}")
    print("-" * 90)
    avg_raw = sum(s["raw_return_pct"] for s in results) / len(results)
    avg_rel = sum(s["mkt_relative_pct"] for s in results) / len(results)
    total_pnl = sum(s["pnl"] for s in results)
    print(f"{'AVG':<5}{'':<10}  {'':<7}  {'':<4}  {avg_raw:>+6.2f}%  {'':<7}  "
          f"{avg_rel:>+8.2f}%  ${total_pnl:>+9.2f}  (sum across all {len(results)} trades)")
    print()

    # Show the day-by-day curve for the textbook biggest-surprise name so the user
    # sees what the holding period actually looked like.
    biggest = max(results, key=lambda s: s["surprise_pct"])
    print()
    print(f"Daily detail for biggest beat: {biggest['symbol']} (surprise {biggest['surprise_pct']:+.2f}%)")
    # Re-pull for daily detail
    sym = biggest["symbol"]
    edate = biggest["edate"]
    start = (edate - timedelta(days=5)).date()
    end = (edate + timedelta(days=HOLD_DAYS * 2 + 5)).date()
    px = yf.download(sym, start=start, end=end, auto_adjust=True, progress=False, group_by="column")
    spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False, group_by="column")
    if isinstance(px.columns, pd.MultiIndex):
        px.columns = [c[0] for c in px.columns]
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = [c[0] for c in spy.columns]
    px.index = pd.to_datetime(px.index).tz_localize(None).normalize()
    spy.index = pd.to_datetime(spy.index).tz_localize(None).normalize()
    e_day = biggest["entry_day"]
    e_price = biggest["entry_price"]
    x_day = biggest["exit_day"]
    window = px.loc[e_day:x_day]
    spy_entry = float(spy.loc[e_day, "Open"])
    print(f"  {'date':<12} {'close':>9}  {'sym %':>7}    {'SPY %':>7}    {'rel %':>7}")
    for dt, row in window.iterrows():
        sp = (float(row["Close"]) / e_price - 1) * 100
        try:
            ms = (float(spy.loc[dt, "Close"]) / spy_entry - 1) * 100
        except KeyError:
            ms = None
        rel = (sp - ms) if ms is not None else None
        print(f"  {dt.strftime('%Y-%m-%d')}  ${float(row['Close']):>8.2f}  {sp:>+6.2f}%    "
              f"{ms:>+6.2f}%    {rel:>+6.2f}%" if ms is not None else
              f"  {dt.strftime('%Y-%m-%d')}  ${float(row['Close']):>8.2f}  {sp:>+6.2f}%       n/a       n/a")


if __name__ == "__main__":
    main()
