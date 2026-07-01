"""Hype SELECTION lab (review #4 + Gemini/DeepSeek combined_score critique).

Tests the SELECTION ALGORITHM (not individual signals) retroactively from the logged per-pick
signals -- no board change, no forward wait. Compares competing top-3 baskets, the Top-N curve,
a score->expectancy calibration curve, and NEGATIVE-CONTROL junk baskets (false-positive canaries).

Baskets (top-3/day):
  current      : the bot's live basket = top-3 by the LOGGED combined_score (mean of 8 signal
                 percentile ranks). Suffers the composition bias (single-signal mega-caps top it)
                 + WSB double-count (wsb_surge + wsb_rank).
  clean        : drop wsb_rank (dedup WSB) AND require >=3 active signals -> the Gemini/DeepSeek fix.
  minsig3      : the SAME logged score, but only among names with >=3 active signals (isolates the
                 min-signals gate from the drop-wsb_rank change).
  confluence   : rank by COUNT of signals in the top quintile (>=0.8 pct) -- "exceptional somewhere",
                 not "pretty good everywhere" (DeepSeek Alt A).
  ignition_only: top-3 by the ignition signal alone -- the "is it just momentum?" baseline.
  NEG len4     : junk control = tickers of length 4 (should have NO edge).
  NEG revalpha : junk control = top-3 by reverse-alphabetical ticker (should have NO edge).

Metric = ret_945_close (same-day 09:45->close, the bot's now-traded horizon) + ret_1d, vs the
random basket. MEASUREMENT ONLY; does not touch the live bot or combined_score.

Run:
    .venv/Scripts/python.exe experiments/lottery/selection_lab.py
"""
from __future__ import annotations

import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.lottery.analyze import load_days, PICKS_DIR  # noqa: E402

# (signal, transform): -1 = "lower is better" (negate), 'abs' = magnitude, 1 = as-is. Matches board.py.
SCORING = [("wsb_surge", 1), ("wsb_rank", -1), ("st_rank", -1), ("pm_rvol", 1),
           ("gap_pct", "abs"), ("squeeze", 1), ("uoa_z", 1), ("ignition", 1)]
METRIC = "ret_945_close"


def _sigval(p, name, tf):
    v = (p.get("signals") or {}).get(name)
    if v is None:
        return None
    if tf == -1:
        return -v
    if tf == "abs":
        return abs(v)
    return v


def _scores(picks, signals):
    """{sym: (combined_pctrank_mean, n_active, {sig: pct})} recomputed cross-sectionally per day."""
    pools = {name: {p["symbol"]: _sigval(p, name, tf) for p in picks} for name, tf in signals}
    out = {}
    for p in picks:
        s = p["symbol"]
        ranks, pcts = [], {}
        for name, _ in signals:
            v = pools[name][s]
            if v is None:
                continue
            vals = [x for x in pools[name].values() if x is not None]
            pct = sum(1 for x in vals if x < v) / len(vals)
            ranks.append(pct)
            pcts[name] = pct
        out[s] = (sum(ranks) / len(ranks) if ranks else None, len(ranks), pcts)
    return out


def _agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return len(vals), st.mean(vals), st.median(vals), 100 * sum(1 for v in vals if v > 0) / len(vals)


def _baskets_for_day(picks):
    """Return {basket_name: [top-3 symbols]} for one day."""
    scored = [p for p in picks if p.get("combined_score") is not None]
    b = {}
    # current: logged combined_score
    cur = sorted(scored, key=lambda x: -x["combined_score"])
    b["current"] = [p["symbol"] for p in cur[:3]]
    # minsig3: logged score, restricted to >=3 active scoring signals
    sc_all = _scores(scored, SCORING)
    ms = [p for p in cur if sc_all[p["symbol"]][1] >= 3]
    b["minsig3"] = [p["symbol"] for p in ms[:3]]
    # clean: drop wsb_rank + require >=3 active signals, recompute score
    sig_clean = [(n, tf) for n, tf in SCORING if n != "wsb_rank"]
    sc_clean = _scores(scored, sig_clean)
    elig = [(s, v[0]) for s, v in sc_clean.items() if v[0] is not None and v[1] >= 3]
    b["clean"] = [s for s, _ in sorted(elig, key=lambda x: -x[1])[:3]]
    # confluence: count of signals in the top quintile (>=0.8)
    conf = [(s, sum(1 for pct in v[2].values() if pct >= 0.8)) for s, v in sc_all.items()]
    b["confluence"] = [s for s, c in sorted(conf, key=lambda x: -x[1])[:3] if c > 0]
    # ignition_only
    ig = [(p["symbol"], _sigval(p, "ignition", 1)) for p in scored]
    ig = [(s, v) for s, v in ig if v is not None]
    b["ignition_only"] = [s for s, _ in sorted(ig, key=lambda x: -x[1])[:3]]
    # negative controls
    b["NEG len4"] = [p["symbol"] for p in scored if len(p["symbol"]) == 4][:3]
    b["NEG revalpha"] = [p["symbol"] for p in sorted(scored, key=lambda x: x["symbol"], reverse=True)][:3]
    return b


def main():
    days = load_days(PICKS_DIR)
    if not days:
        print("no picks."); return 0
    # random baseline (same-day)
    rand = [p.get(METRIC) for rec in days for p in rec["picks"] if p.get("basket") == "random"]
    ra = _agg(rand)
    print(f"=== SELECTION LAB: {len(days)} days | metric = {METRIC} (same-day 09:45->close) ===")
    if ra:
        print(f"RANDOM basket baseline: n={ra[0]} mean={ra[1]:+.2f}% median={ra[2]:+.2f}% win={ra[3]:.0f}%\n")

    # gather per-basket returns
    by_ret = {}
    by_ret1d = {}
    lookup = {rec["date"]: {p["symbol"]: p for p in rec["picks"]} for rec in days}
    for rec in days:
        b = _baskets_for_day([p for p in rec["picks"] if p.get("combined_score") is not None])
        L = lookup[rec["date"]]
        for name, syms in b.items():
            for s in syms:
                by_ret.setdefault(name, []).append(L.get(s, {}).get(METRIC))
                by_ret1d.setdefault(name, []).append(L.get(s, {}).get("ret_1d"))

    print(f"  {'basket':<14}{'n':>4}{'mean%':>8}{'median%':>9}{'win%':>6}{'edge vs rand':>13}{'1d mean%':>10}")
    print("  " + "-" * 66)
    order = ["current", "minsig3", "clean", "confluence", "ignition_only", "NEG len4", "NEG revalpha"]
    for name in order:
        a = _agg(by_ret.get(name, []))
        a1 = _agg(by_ret1d.get(name, []))
        if not a:
            continue
        edge = (a[1] - ra[1]) if ra else float("nan")
        m1 = f"{a1[1]:+.2f}" if a1 else "n/a"
        print(f"  {name:<14}{a[0]:>4}{a[1]:>+8.2f}{a[2]:>+9.2f}{a[3]:>6.0f}{edge:>+12.2f}pp{m1:>10}")

    # Top-N curve (logged combined_score)
    print("\n  Top-N curve (by logged combined_score), same-day:")
    for N in (1, 3, 5, 10, 20):
        pool = []
        for rec in days:
            sc = sorted([p for p in rec["picks"] if p.get("combined_score") is not None],
                        key=lambda x: -x["combined_score"])
            pool += [p.get(METRIC) for p in sc[:N]]
        a = _agg(pool)
        if a:
            print(f"    Top-{N:<2} n={a[0]:>3} mean={a[1]:+.2f}% median={a[2]:+.2f}% win={a[3]:.0f}%")

    # calibration: expectancy by combined_score quantile
    print("\n  Calibration (same-day mean by combined_score quintile, all scored picks):")
    allp = [(p["combined_score"], p.get(METRIC)) for rec in days for p in rec["picks"]
            if p.get("combined_score") is not None and p.get(METRIC) is not None]
    allp.sort(key=lambda x: x[0])
    n = len(allp)
    for q in range(5):
        lo, hi = q * n // 5, (q + 1) * n // 5
        seg = [r for _, r in allp[lo:hi]]
        a = _agg(seg)
        if a:
            csr = f"{allp[lo][0]:.2f}-{allp[min(hi, n - 1)][0]:.2f}"
            print(f"    Q{q+1} (cs {csr:<11}) n={a[0]:>3} mean={a[1]:+.2f}% win={a[3]:.0f}%")

    print("\nRead: 'clean'/'minsig3' beating 'current' = the composition-bias fix helps. ignition_only")
    print("~= current => it's largely a momentum ranker. NEG baskets should be ~0 (flat); if a NEG")
    print("basket shows edge, the pipeline leaks false positives. Calibration should rise with cs.")
    print("Directional, small n. Same-day metric (the bot's now-traded horizon).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
