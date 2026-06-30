"""biotech_backtest.py -- backtest the biotech surge radar's HEAT signal:
  (1) PRECURSOR: do surges have detectable pre-surge heat? i.e. is P(surge | heating up)
      meaningfully above the base rate -> can we see them coming at all?
  (2) STRATEGY: does buying on a heat trigger and harvesting with a trailing stop make
      money net of costs, and with what trail width? (the "how to invest" part)
On ~5yr daily history of the CURRENT XBI universe (yfinance).

*** SURVIVORSHIP WARNING (read first) *** The universe is TODAY's XBI holdings, so it
EXCLUDES every biotech that failed a trial and delisted -- a large fraction in biotech.
Results here are an OPTIMISTIC UPPER BOUND; the real edge is LOWER. A clean test needs
point-in-time constituents (incl. delisted), which we don't have. Directional only.
Also: surges are BINARY catalysts -- this tests whether price/volume gives ANY pre-warning,
not whether we can predict the outcome (we can't).

Run under .venv-openbb:  .venv-openbb/Scripts/python.exe backtest/biotech_backtest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UNIV_CACHE = ROOT / "scripts" / ".biotech_universe.txt"
SURGE = 0.30        # "surge" = max close over next FWD days >= +30%
FWD = 10            # forward window (trading days)


def universe() -> list[str]:
    if UNIV_CACHE.exists():
        return [s.strip().upper() for s in UNIV_CACHE.read_text().splitlines() if s.strip()]
    return []


def build(tickers: list[str]):
    import yfinance as yf
    px = yf.download(tickers, period="5y", auto_adjust=True, progress=False, threads=True)
    close = px["Close"]; vol = px["Volume"]; high = px["High"]; low = px["Low"]; opn = px["Open"]
    frames = []
    paths = {}    # sym -> DataFrame(open,high,low,close) for the strategy sim
    for t in close.columns:
        c = close[t].dropna()
        if len(c) < 60:
            continue
        v = vol[t].reindex(c.index); h = high[t].reindex(c.index)
        lo = low[t].reindex(c.index); o = opn[t].reindex(c.index)
        v20 = v.rolling(20).mean()
        rng40 = (c.rolling(40).max() - c.rolling(40).min())
        f = pd.DataFrame({
            "sym": t, "close": c,
            "vol_build": v.rolling(5).mean() / v20,            # 5d vs 20d volume (through t)
            "ret_5d": c / c.shift(5) - 1.0,
            "ret_20d": c / c.shift(20) - 1.0,
            "near_high": c / c.rolling(252, min_periods=60).max(),
            "pos_in_range": (c - c.rolling(40).min()) / rng40.replace(0, pd.NA),   # 40d range pos
        }, index=c.index)
        # forward max close over t+1..t+FWD (the surge label), lookahead only in the LABEL
        fwd_max = pd.concat([c.shift(-k) for k in range(1, FWD + 1)], axis=1).max(axis=1)
        fwd_min = pd.concat([c.shift(-k) for k in range(1, FWD + 1)], axis=1).min(axis=1)
        f["fwd_surge"] = (fwd_max / c - 1.0)
        f["fwd_crash"] = (fwd_min / c - 1.0)
        frames.append(f.dropna(subset=["vol_build", "ret_5d", "ret_20d", "near_high"]))
        paths[t] = pd.DataFrame({"open": o, "high": h, "low": lo, "close": c})
    return pd.concat(frames, ignore_index=False), paths


PROBS_OUT = ROOT / "backtest" / "biotech_signal_probs.json"


def signal_probs(panel: pd.DataFrame):
    """Per signal bucket: historical P(+30% up) AND P(-30% down) within FWD days. Saved to
    JSON for the live radar to attach 'odds of a good shot' to each candidate. NOTE both
    biases: survivorship OVERstates p_up and UNDERstates p_down (worst crashes delisted)."""
    import json
    p = panel.dropna(subset=["fwd_surge", "fwd_crash"]).copy()
    base_up = float((p["fwd_surge"] >= SURGE).mean())
    base_dn = float((p["fwd_crash"] <= -SURGE).mean())
    defs = [  # (key, label, mask) — strongest first; the live radar classifies the same way
        ("hot", "vol>=2x & +10%/5d", (p["vol_build"] >= 2) & (p["ret_5d"] >= 0.10)),
        ("building", "vol>=1.5x & +5%/5d", (p["vol_build"] >= 1.5) & (p["ret_5d"] >= 0.05)),
        ("vol2", "vol>=2x", p["vol_build"] >= 2),
        ("vol15", "vol>=1.5x", p["vol_build"] >= 1.5),
        ("momo", "+10%/5d", p["ret_5d"] >= 0.10),
        ("extended", "near 52wk high", p["near_high"] >= 0.95),
    ]
    out = {"surge_pct": SURGE, "fwd_days": FWD, "asof": str(pd.Timestamp.now().date()),
           "base": {"p_up": round(base_up, 4), "p_down": round(base_dn, 4), "n": int(len(p))},
           "buckets": {}}
    for key, label, mask in defs:
        sub = p[mask]
        if len(sub) == 0:
            continue
        out["buckets"][key] = {"label": label, "n": int(len(sub)),
                               "p_up": round(float((sub["fwd_surge"] >= SURGE).mean()), 4),
                               "p_down": round(float((sub["fwd_crash"] <= -SURGE).mean()), 4)}
    PROBS_OUT.write_text(json.dumps(out, indent=2))
    print(f"\n=== signal-bucket odds (+/-{SURGE*100:.0f}% within {FWD}d) -> {PROBS_OUT.name} ===")
    print(f"  {'bucket':22}{'n':>8}{'P(+30%)':>10}{'P(-30%)':>10}")
    print(f"  {'(base/any day)':22}{out['base']['n']:>8,}{base_up*100:>9.1f}%{base_dn*100:>9.1f}%")
    for k, v in out["buckets"].items():
        print(f"  {v['label']:22}{v['n']:>8,}{v['p_up']*100:>9.1f}%{v['p_down']*100:>9.1f}%")
    print("  (survivorship: p_up is an OPTIMISTIC ceiling, p_down a FLOOR — real odds worse both ways)")
    return out


def precursor(panel: pd.DataFrame):
    p = panel.dropna(subset=["fwd_surge"]).copy()
    p["surge"] = (p["fwd_surge"] >= SURGE).astype(int)
    base = p["surge"].mean()
    print(f"\n=== (1) PRECURSOR: can we SEE surges coming? ===")
    print(f"base rate: P(+{SURGE*100:.0f}% within {FWD}d on any day) = {base*100:.2f}%\n")
    print(f"  {'heat trigger':34}{'n days':>9}{'P(surge)':>10}{'lift':>7}")
    triggers = [
        ("vol_build >= 1.5", p["vol_build"] >= 1.5),
        ("vol_build >= 2.0", p["vol_build"] >= 2.0),
        ("ret_5d >= +10%", p["ret_5d"] >= 0.10),
        ("near 52w-high (>=0.95)", p["near_high"] >= 0.95),
        ("vol_build>=1.5 AND ret_5d>=+5%", (p["vol_build"] >= 1.5) & (p["ret_5d"] >= 0.05)),
        ("vol_build>=2 AND ret_5d>=+10%", (p["vol_build"] >= 2.0) & (p["ret_5d"] >= 0.10)),
    ]
    for name, mask in triggers:
        sub = p[mask]
        if len(sub) == 0:
            print(f"  {name:34}{'0':>9}"); continue
        ps = sub["surge"].mean()
        print(f"  {name:34}{len(sub):>9,}{ps*100:>9.2f}%{ps/base:>6.2f}x")
    print("  READ: a usable pre-warning needs lift meaningfully >1 (heating-up days surge more")
    print("  often than random). Lift ~1 = surges come out of nowhere (no price/vol pre-warning).")


def _harvest(trig: pd.DataFrame, paths: dict, trail: float, max_days: int, cost_bps: float):
    """Buy next open for each trigger row; 'trail'% trailing stop + time-stop at max_days."""
    rets = []
    for sym, sub in trig.groupby("sym"):
        df = paths.get(sym)
        if df is None:
            continue
        idx = df.index
        for t in sub.index:
            pos = idx.get_indexer([t])[0]
            if pos < 0 or pos + 1 >= len(idx):
                continue
            fut = df.iloc[pos + 1: pos + 2 + max_days]
            if len(fut) < 2:
                continue
            o = fut["open"].values; h = fut["high"].values
            loo = fut["low"].values; cl = fut["close"].values
            entry = float(o[0])
            if entry <= 0:
                continue
            hw, exitp = entry, float(cl[-1])
            for i in range(len(fut)):
                stop = hw * (1 - trail / 100)
                if i > 0 and loo[i] <= stop:
                    exitp = stop; break
                hw = max(hw, h[i])
                if i >= max_days:
                    exitp = float(cl[i]); break
            rets.append(exitp / entry - 1.0 - cost_bps / 10000.0)
    r = np.array(rets)
    if r.size == 0:
        return None
    return {"n": r.size, "avg": r.mean() * 100, "median": np.median(r) * 100,
            "win": np.mean(r > 0) * 100, "best": r.max() * 100, "worst": r.min() * 100,
            "total": r.sum() * 100, "p90": np.percentile(r, 90) * 100, "ret": r}


def strategy(panel, paths, trail, max_days, cost_bps):
    return _harvest(panel[(panel["vol_build"] >= 1.5) & (panel["ret_5d"] >= 0.05)],
                    paths, trail, max_days, cost_bps)


def runup(panel, paths, trail, hold, cost_bps):
    """The RUN-UP setup: coiled (40d range-top) + volume building + NOT already popped,
    held ~`hold` sessions with a wide trailing stop = 'buy the run-up, exit before data'."""
    trig = panel[(panel["vol_build"] >= 1.5) & (panel["pos_in_range"] >= 0.85)
                 & (panel["ret_20d"] < 0.40)]
    return _harvest(trig, paths, trail, hold, cost_bps)


def main() -> int:
    uni = universe()
    if not uni:
        print("no universe — run scripts/biotech_radar.py first to cache it."); return 1
    print("=== biotech_backtest: heat signal on the CURRENT XBI universe ===")
    print("*** SURVIVORSHIP-BIASED (no delisted names) -> OPTIMISTIC UPPER BOUND ***")
    print(f"universe: {len(uni)} names, 5y daily\n")
    panel, paths = build(uni)
    print(f"panel: {len(panel):,} (name,day) rows, {panel['sym'].nunique()} names")

    precursor(panel)
    signal_probs(panel)

    print(f"\n=== (2) STRATEGY: heat-entry (vol_build>=1.5 & ret_5d>=+5%) + trailing stop ===")
    print(f"buy next open, time-stop {FWD}d, cost 50bps round-trip. trail-width sweep:")
    print(f"  {'trail %':>10}{'n':>8}{'avg/trade':>11}{'median':>9}{'win%':>7}{'best':>9}{'worst':>9}")
    for trail in (15, 20, 25, 30):
        st = strategy(panel, paths, trail, FWD, 50.0)
        if st:
            print(f"  {trail:>9}%{st['n']:>8,}{st['avg']:>+10.2f}%{st['median']:>+8.2f}%"
                  f"{st['win']:>6.0f}%{st['best']:>+8.0f}%{st['worst']:>+8.0f}%")
    print("\n  READ: these are lottery-like (a few huge winners, many small losers). A POSITIVE")
    print("  avg/trade after cost = the trailing stop harvests the up-tail faster than the")
    print("  down-tail bleeds. REMEMBER survivorship inflates this -- haircut it hard.")

    print(f"\n=== (3) THE RUN-UP SETUP (what the tool recommends): coiled + vol-building, ===")
    print("buy next open, 25% trailing stop, EXIT at `hold` sessions (before the binary), 50bps:")
    print(f"  {'hold (sessions)':>16}{'n':>7}{'avg/trade':>11}{'median':>9}{'win%':>7}"
          f"{'P90':>8}{'best':>8}{'$/trade on $300':>17}")
    for hold in (10, 20, 30):
        st = runup(panel, paths, 25.0, hold, 50.0)
        if st:
            print(f"  {str(hold)+'d (~'+str(hold//5)+'wk)':>16}{st['n']:>7,}{st['avg']:>+10.2f}%"
                  f"{st['median']:>+8.2f}%{st['win']:>6.0f}%{st['p90']:>+7.1f}%{st['best']:>+7.0f}%"
                  f"{300*st['avg']/100:>+16.2f}")
    print("\n  EXPECTATION READ (honest): avg/trade is the per-position % over the hold; on a $300")
    print("  position that's the $ column. Median NEGATIVE + win<50% = MOST positions lose a little;")
    print("  the average is carried by rare winners (see P90/best). HAIRCUT HARD: survivorship")
    print("  excludes delisted failures (the -70% trial flops), so the REAL avg is materially lower")
    print("  — plausibly near zero after that + real small-cap spreads. This is SPECULATION, not")
    print("  expected income: size to lose, profit (if any) comes from catching the occasional moonshot.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
