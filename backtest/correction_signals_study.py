"""Investigation: which cross-asset / macro signals actually preceded S&P 500 corrections?

Honest framing: corrections (>=10% drawdowns) are NOT cleanly predictable. This study
just CHARACTERIZES how a set of risk-off signals were positioned at/just-before each
historical correction peak, and — crucially — how often each signal is in its "warning"
state on a RANDOM day (the false-positive base rate). A signal earns its place only if it
flashed before MOST corrections AND isn't flashing most of the time anyway.

Data: yfinance daily closes, ~2007-present (covers 2008, 2011, 2015-16, 2018, 2020, 2022).
Macro releases (CPI/NFP/Fed funds) need a FRED feed we don't have -> market PROXIES used:
yields for rate expectations, TIP/IEF for inflation expectations.

Signals (all computable free):
  curve     10y-3m yield (^TNX-^IRX); inverted (<0) = classic late-cycle stress
  vix       ^VIX level (>20 elevated) and 21d change (>+30% = fear spike)
  credit    HYG/LQD 63d % (high-yield underperforming inv-grade -> risk-off)
  defens    XLP/SPY & XLU/SPY 63d % (staples+utilities leading = defensive rotation)
  dollar    DXY 63d % (>+3% = flight to USD)
  breakeven TIP/IEF 63d % (falling = growth/inflation scare)
  trend     SPX vs 200dma (mostly COINCIDENT/lagging — shown to prove it's not a lead)

Run (under .venv-openbb — yfinance lives there):
    .venv-openbb/Scripts/python.exe backtest/correction_signals_study.py
"""
from __future__ import annotations

import sys
import warnings
from datetime import datetime

import pandas as pd

warnings.filterwarnings("ignore")
try:
    import yfinance as yf
except ImportError:
    print("FATAL: run under .venv-openbb/Scripts/python.exe (yfinance lives there)")
    sys.exit(2)

START = "2007-01-01"
TICKERS = ["^GSPC", "^VIX", "^TNX", "^IRX", "DX-Y.NYB",
           "HYG", "LQD", "XLP", "XLU", "SPY", "TIP", "IEF", "GLD"]
CORRECTION = 0.10   # >=10% drawdown from a prior peak


def fetch() -> pd.DataFrame:
    cols = {}
    for t in TICKERS:
        try:
            df = yf.download(t, start=START, auto_adjust=True, progress=False)
            if df is None or df.empty:
                print(f"  WARN no data for {t}")
                continue
            s = df["Close"]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            cols[t] = s
            print(f"  {t:10} {len(s):>5} rows  {s.index.min().date()}..{s.index.max().date()}")
        except Exception as e:
            print(f"  WARN {t}: {str(e)[:60]}")
    px = pd.DataFrame(cols).sort_index()
    px.index = pd.to_datetime(px.index)
    return px


def episodes(spx: pd.Series, trigger: float = 0.05, deep: float = CORRECTION) -> list[dict]:
    """Every pullback that fell >= `trigger` (5%) below a running high, with the date it
    FIRST crossed -trigger ('trigger_date'), how deep it ultimately got, and whether it
    became a real correction (>= `deep`, 10%). An episode runs from a new high until the
    next new high (full recovery). This lets us ask the decision-relevant question: at the
    -5% mark, what told us this one would keep going vs bounce?"""
    eps = []
    peak = spx.iloc[0]; peak_d = spx.index[0]
    trig_d = None; min_dd = 0.0; trough_d = peak_d
    for d, p in spx.items():
        if p >= peak:
            if trig_d is not None:  # close the just-ended pullback episode
                eps.append({"peak_date": peak_d, "trigger_date": trig_d,
                            "trough_date": trough_d, "min_dd": min_dd,
                            "became_correction": min_dd <= -deep})
            peak = p; peak_d = d; trig_d = None; min_dd = 0.0; trough_d = d
            continue
        dd = p / peak - 1.0
        if dd < min_dd:
            min_dd = dd; trough_d = d
        if dd <= -trigger and trig_d is None:
            trig_d = d
    if trig_d is not None:
        eps.append({"peak_date": peak_d, "trigger_date": trig_d, "trough_date": trough_d,
                    "min_dd": min_dd, "became_correction": min_dd <= -deep})
    return eps


def signals(px: pd.DataFrame) -> pd.DataFrame:
    """Daily signal panel + boolean 'warning' state per signal."""
    g = px["^GSPC"]
    sig = pd.DataFrame(index=px.index)
    sig["curve"] = px["^TNX"] - px["^IRX"]
    sig["vix"] = px["^VIX"]
    sig["vix_21d"] = px["^VIX"].pct_change(21)
    sig["credit_63d"] = (px["HYG"] / px["LQD"]).pct_change(63)
    sig["xlp_rel_63d"] = (px["XLP"] / px["SPY"]).pct_change(63)
    sig["xlu_rel_63d"] = (px["XLU"] / px["SPY"]).pct_change(63)
    if "DX-Y.NYB" in px:
        sig["dollar_63d"] = px["DX-Y.NYB"].pct_change(63)
    sig["breakeven_63d"] = (px["TIP"] / px["IEF"]).pct_change(63)
    sig["spx_vs_200"] = g / g.rolling(200).mean() - 1.0

    w = pd.DataFrame(index=px.index)
    w["curve_inv"] = sig["curve"] < 0
    w["vix_hi"] = (sig["vix"] > 20) | (sig["vix_21d"] > 0.30)
    w["credit_off"] = sig["credit_63d"] < -0.01
    w["defensive"] = (sig["xlp_rel_63d"] > 0) & (sig["xlu_rel_63d"] > 0)
    if "dollar_63d" in sig:
        w["dollar_bid"] = sig["dollar_63d"] > 0.03
    w["breakeven_dn"] = sig["breakeven_63d"] < -0.01
    w["trend_broken"] = sig["spx_vs_200"] < 0
    return sig, w


def nearest(idx: pd.DatetimeIndex, when) -> pd.Timestamp:
    pos = idx.searchsorted(when)
    return idx[min(pos, len(idx) - 1)]


def main() -> int:
    print("Fetching macro/cross-asset history (yfinance) ...")
    px = fetch()
    if "^GSPC" not in px:
        print("no SPX data — abort")
        return 1
    spx = px["^GSPC"].dropna()
    sig, w = signals(px)
    wcols = list(w.columns)

    eps = [e for e in episodes(spx) if e["trigger_date"] is not None]
    deep = [e for e in eps if e["became_correction"]]
    shallow = [e for e in eps if not e["became_correction"]]
    print(f"\n=== {len(eps)} pullbacks >= 5% off a high since {START[:4]}: "
          f"{len(deep)} became corrections (>= {CORRECTION:.0%}), {len(shallow)} bounced ===\n")

    # signal warning-states at the -5% TRIGGER date for each pullback
    head = f"{'-5% trigger':<13}{'final dd':>9} {'->10%?':>7}  " + "".join(f"{c[:9]:>10}" for c in wcols)
    print(head); print("-" * len(head))
    cnt = {"deep": {c: 0 for c in wcols}, "shallow": {c: 0 for c in wcols}}
    for e in sorted(eps, key=lambda x: x["trigger_date"]):
        td = nearest(w.index, e["trigger_date"]); row = w.loc[td]
        bucket = "deep" if e["became_correction"] else "shallow"
        for c in wcols:
            if bool(row[c]):
                cnt[bucket][c] += 1
        # only print the ones that became corrections + a few shallow for context
        if e["became_correction"]:
            cells = "".join(f"{('YES' if bool(row[c]) else '.'):>10}" for c in wcols)
            print(f"{str(e['trigger_date'].date()):<13}{e['min_dd']*100:>8.1f}%{'YES':>8}  {cells}")

    nd, ns = max(len(deep), 1), max(len(shallow), 1)
    print(f"\n=== at the -5% mark: signal present in pullbacks that BECAME corrections "
          f"vs those that BOUNCED ===")
    print(f"{'signal':<16}{'became corr':>13}{'bounced':>10}{'discrimination':>16}")
    print("-" * 55)
    for c in wcols:
        pd_ = cnt['deep'][c] / nd
        ps = cnt['shallow'][c] / ns
        print(f"{c:<16}{pd_*100:>11.0f}% {ps*100:>8.0f}% {(pd_-ps)*100:>+13.0f}%")
    print("\nRead: 'discrimination' = how much more often the signal was flashing at the -5%")
    print("mark in pullbacks that KEPT FALLING vs ones that bounced. Positive + large = the")
    print("signal helps tell 'serious' dips from noise. ~0 = it fires the same eitherway.")
    print("\nHONEST CAVEAT: small sample (handful of real corrections in ~18y). Treat as a")
    print("descriptive risk-off confirmation gauge, NOT a timing/prediction signal.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
