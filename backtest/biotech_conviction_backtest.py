"""biotech_conviction_backtest.py — does the SLS-type CONVICTION STACK actually beat the
generic setup? Tests whether "coil + recent INSIDER BUYING" out-performs "coil alone" on the
XBI universe over years (the catalyst layer needs historical dates we don't have cleanly, so
this validates 2 of the 3 stack layers). Answers: is the +2% generic number an UNDERstatement
for the smart-money subset?

For every coil setup event (vol_build>=1.5 & 40d range-top & not-already-popped), flag whether
an open-market insider BUY (Form 4 Code P) landed in the PRIOR 90 days, then compare the
trailing-stop harvest (buy next open, 25% trail, hold 20d, 50bps) WITH vs WITHOUT insider.

SAME caveats as biotech_backtest: survivorship-biased (no delisted failures) = OPTIMISTIC;
relative comparison (insider vs not, same universe) is the robust read. n=1 anecdote (SLS)
motivated this; this is the actual measurement.

Run under .venv-openbb (yfinance + edgartools, SLOW ~30-60min for the Form-4 fetch, cached):
    .venv-openbb/Scripts/python.exe backtest/biotech_conviction_backtest.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backtest"))
from biotech_backtest import build, _harvest, universe  # noqa: E402

INS_CACHE = ROOT / "backtest" / ".biotech_insider_hist.pkl"
SINCE = "2022-01-01"
MAX_F4 = 80


def fetch_insider_history(symbols) -> dict:
    """{sym: [Timestamp,...]} dates of open-market insider BUYS (Form 4 Code P) since SINCE."""
    if INS_CACHE.exists():
        print(f"using cached insider history: {INS_CACHE.name}")
        return pickle.load(open(INS_CACHE, "rb"))
    from edgar import set_identity, Company
    set_identity("research maciej.mikulewicz@gmail.com")
    out = {}
    for i, sym in enumerate(symbols, 1):
        dates = []
        try:
            fs = [f for f in Company(sym).get_filings(form="4")
                  if f.filing_date and pd.Timestamp(f.filing_date) >= pd.Timestamp(SINCE)]
            for fl in fs[:MAX_F4]:
                try:
                    mt = fl.obj().market_trades
                    if mt is not None and not mt.empty and any(
                            str(t.get("Code")) == "P" for _, t in mt.iterrows()):
                        dates.append(pd.Timestamp(fl.filing_date))
                except Exception:
                    continue
        except Exception:
            pass
        out[sym] = sorted(set(dates))
        if i % 20 == 0:
            print(f"  ...{i}/{len(symbols)} ({sum(len(v) for v in out.values())} buys so far)", flush=True)
    pickle.dump(out, open(INS_CACHE, "wb"))
    return out


def main() -> int:
    uni = universe()
    if not uni:
        print("no universe — run scripts/biotech_radar.py first."); return 1
    print(f"=== biotech CONVICTION backtest: coil + insider vs coil alone ({len(uni)} names) ===")
    print("*** survivorship-biased; relative (insider vs not) is the robust read ***\n")
    panel, paths = build(uni)
    ins = fetch_insider_history(uni)
    nbuys = sum(len(v) for v in ins.values())
    print(f"insider buy-dates: {nbuys} across {sum(1 for v in ins.values() if v)} names\n")

    coil = panel[(panel["vol_build"] >= 1.5) & (panel["pos_in_range"] >= 0.85)
                 & (panel["ret_20d"] < 0.40)].copy()

    def has_insider_prior(sym, dt):
        return any(0 <= (dt - d).days <= 90 for d in ins.get(sym, []))
    coil["ins90"] = [has_insider_prior(r["sym"], idx) for idx, r in coil.iterrows()]

    with_ins = coil[coil["ins90"]]
    without = coil[~coil["ins90"]]
    print(f"coil setups: {len(coil):,}  |  WITH insider-prior-90d: {len(with_ins):,}  |  "
          f"without: {len(without):,}\n")

    print(f"  {'cohort':>24}{'n':>7}{'avg/trade':>11}{'median':>9}{'win%':>7}{'P90':>8}{'best':>8}")
    for name, trig in [("coil ALONE", coil), ("coil + INSIDER (prior 90d)", with_ins),
                       ("coil, NO insider", without)]:
        st = _harvest(trig, paths, 25.0, 20, 50.0)
        if st:
            print(f"  {name:>24}{st['n']:>7,}{st['avg']:>+10.2f}%{st['median']:>+8.2f}%"
                  f"{st['win']:>6.0f}%{st['p90']:>+7.1f}%{st['best']:>+7.0f}%")
        else:
            print(f"  {name:>24}   (no trades)")
    print("\nVERDICT: insider layer adds edge IFF (coil+insider) avg/win materially BEAT coil-alone")
    print("AND (coil, no insider). If similar, the SLS insider buy was coincidence, not signal.")
    print("Survivorship-biased; the relative gap is the read. Catalyst layer still untested.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
