"""Build D (Fable): a VWAP-reversion second stream — is there a DECORRELATED edge?

Thesis: ORB harvests trend/expansion days and bleeds on chop. VWAP-reversion does the
opposite — on range days price oscillates around session VWAP, so fading stretches BELOW
VWAP captures the snap-back; on trend days it stops out. So it should be anti-correlated
with ORB BY CONSTRUCTION, and two decorrelated Sharpe-~1 streams combine into a smoother
curve that earns leverage more cheaply than 3.5x on one engine.

This validates BEFORE any production module:
  1. MR standalone: avgR, win%, Sharpe, BOTH OOS halves, BOTH windows, cents-slippage.
  2. THE GATE: correlation of MR daily PnL vs ORB (tight-OR trailing) daily PnL. Need <= ~0.3.
  3. Combined 50/50 portfolio Sharpe/maxDD vs ORB-alone — must improve.

MR sim (long-only, intraday, lookahead-free): session VWAP = cum(typical*vol)/cum(vol);
sigma = expanding std of (close - vwap). After WARMUP, first bar closing <= vwap - K*sigma
-> enter NEXT bar open; stop = entry - KSTOP*sigma_at_entry; target = vwap_at_entry; EOD
flat; one setup/day; no entry after CUTOFF. Same-bar stop+target -> stop first (conservative).

Run:
    .venv/Scripts/python.exe backtest/compare_meanrev.py
"""
from __future__ import annotations

import math
import statistics
import sys
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.universe_portfolio import portfolio  # noqa: E402
from backtest.compare_selection import _tday  # noqa: E402
from backtest.compare_norefill_trend import trend_eligibility, apply_filter  # noqa: E402
from backtest.compare_exits import load, bucket, reexit, POLICIES, EOD  # noqa: E402
from backtest.compare_volpause import prior_vol_flags, perf, RISK, CAP  # noqa: E402
from backtest.compare_or_range_realcost import or_pct, risk_ps  # noqa: E402

WINDOWS = [730, 180]
WARMUP = time(10, 0)       # let VWAP stabilize
CUTOFF = time(14, 30)      # last new entry (need time to revert before EOD flat)
EOD_FLAT = time(15, 55)
RTH_S = time(9, 30)
K_ENTRY = 2.0              # enter when price K_ENTRY*sigma below VWAP
K_STOP = 1.5               # stop K_STOP*sigma below entry
TARGET_MEDIAN_R = 0.042
NOTIONAL_CAP = 10_000.0
SLIP = [1.0, 1.5]


class MRTrade:
    __slots__ = ("symbol", "date", "entry_price", "stop_price", "exit_price", "pnl_r",
                 "entry_time", "exit_time")

    def __init__(self, **k):
        for key, v in k.items():
            setattr(self, key, v)


def mr_session(sym, day, o, h, l, c, v, ns, idx, k_entry, k_stop):
    """One VWAP-reversion long for a symbol-day; numpy arrays in. Returns MRTrade or None."""
    n = len(c)
    tp = (h + l + c) / 3.0
    cum_pv = np.cumsum(tp * v)
    cum_v = np.cumsum(v)
    vwap = np.where(cum_v > 0, cum_pv / np.maximum(cum_v, 1e-9), c)
    dev = c - vwap
    tt = idx.time
    warm = np.array([t >= WARMUP for t in tt])
    cut = np.array([t <= CUTOFF for t in tt])
    for i in range(n):
        if not warm[i] or not cut[i]:
            continue
        if i < 20:
            continue
        sigma = dev[:i + 1].std()
        if sigma <= 1e-6:
            continue
        if c[i] <= vwap[i] - k_entry * sigma:        # stretched below VWAP
            if i + 1 >= n:
                return None
            entry = o[i + 1]
            stop = entry - k_stop * sigma
            target = vwap[i]                          # revert to the mean (fixed level)
            if target <= entry or entry <= stop:
                return None
            risk = entry - stop
            for j in range(i + 1, n):
                if ns[j] >= EOD_NS:
                    return _mk(sym, day, entry, stop, c[j], idx[i + 1], idx[j], risk)
                if l[j] <= stop:
                    return _mk(sym, day, entry, stop, stop, idx[i + 1], idx[j], risk)
                if h[j] >= target:
                    return _mk(sym, day, entry, stop, target, idx[i + 1], idx[j], risk)
            return _mk(sym, day, entry, stop, c[-1], idx[i + 1], idx[-1], risk)
    return None


def _mk(sym, day, entry, stop, exitp, et, xt, risk):
    return MRTrade(symbol=sym, date=pd.Timestamp(day), entry_price=float(entry),
                   stop_price=float(stop), exit_price=float(exitp),
                   pnl_r=float((exitp - entry) / risk), entry_time=et, exit_time=xt)


EOD_NS = None


def efficiency_ratio(closes, days, n=10):
    """Kaufman Efficiency Ratio on SPY per trading day (lookahead-free: uses closes
    STRICTLY before the day). ER = |net move over n| / sum(|daily moves|). Low ER (<0.30)
    = choppy/ranging market (mean-reversion's home); high ER = trending."""
    spy = closes["SPY"].dropna().sort_index()
    vals = list(spy.values)
    dts = [pd.Timestamp(ts).date() for ts in spy.index]
    out = {}
    for d in days:
        prior = [v for v, dd in zip(vals, dts) if dd < d]
        if len(prior) < n + 1:
            out[d] = None
            continue
        seg = prior[-(n + 1):]
        net = abs(seg[-1] - seg[0])
        path = sum(abs(seg[k] - seg[k - 1]) for k in range(1, len(seg)))
        out[d] = (net / path) if path > 0 else None
    return out


def build_arrays(all_bars, present):
    out = {}
    for sym in present:
        sb = all_bars.xs(sym, level=0)
        t = sb.index.time
        sb = sb[(t >= RTH_S) & (t < time(16, 0))]
        d = {}
        for day, g in sb.groupby(sb.index.date):
            d[day] = (g["open"].to_numpy(float), g["high"].to_numpy(float),
                      g["low"].to_numpy(float), g["close"].to_numpy(float),
                      g["volume"].to_numpy(float), g.index.asi8, g.index)
        out[sym] = d
    return out


def daily_dollars(taken, days, mult, cents):
    by = {}
    for t in taken:
        rps = max(t.entry_price - t.stop_price, 1e-6)
        sh = min(math.floor(RISK * mult.get(_tday(t), 1.0) / rps), math.floor(NOTIONAL_CAP / t.entry_price))
        if sh <= 0:
            continue
        by[_tday(t)] = by.get(_tday(t), 0.0) + (t.exit_price - t.entry_price) * sh - 2 * cents * sh
    return pd.Series({d: by.get(d, 0.0) for d in sorted(days)})


def run_window(w):
    global EOD_NS
    all_bars, days, present, trades, closes = load(w)
    tz = all_bars.index.get_level_values(1).tz
    eod_ns_by = {d: pd.Timestamp.combine(d, EOD).tz_localize(tz).value for d in days}
    EOD_NS = pd.Timestamp.combine(days[0], EOD_FLAT).tz_localize(tz).value  # same clock daily
    elig = trend_eligibility(closes, present, days)
    mid = sorted(days)[len(days) // 2]
    prior = prior_vol_flags(closes, days)
    mult = {d: (0.5 if prior[d] else 1.0) for d in days}

    # ---- ORB tight-OR trailing daily $ (the engine we want to decorrelate from) ----
    buckets = bucket(all_bars, present)
    orb = [t for t in apply_filter(reexit(trades, buckets, POLICIES["trail_1R"], eod_ns_by), elig)
           if t.side == "long" and or_pct(t) <= 0.5]
    orb_taken = portfolio(orb, CAP)
    orb_cents = TARGET_MEDIAN_R * statistics.median(risk_ps(t) for t in orb) / 2.0

    # ---- MR sim across the universe ----
    arrays = build_arrays(all_bars, present)
    raw = []
    for sym, dmap in arrays.items():
        for day, (o, h, l, c, v, ns, idx) in dmap.items():
            tr = mr_session(sym, day, o, h, l, c, v, ns, idx, K_ENTRY, K_STOP)
            if tr is not None:
                raw.append(tr)
    # trend filter ON (test) — MR long fades only in uptrending names
    mr_tf = apply_filter(raw, elig)
    mr_cents = TARGET_MEDIAN_R * statistics.median(max(t.entry_price - t.stop_price, 1e-6) for t in raw) / 2.0

    print(f"\n{'='*78}\n=== {w}d MEAN-REVERSION (VWAP fade) | {len(raw)} raw, {len(mr_tf)} trend-filtered | OOS {mid} ===")
    for label, mrset in (("MR no-trend-filter", raw), ("MR trend-filter", mr_tf)):
        taken = portfolio(mrset, CAP)
        rs = [t.pnl_r for t in taken]
        win = 100 * sum(1 for r in rs if r > 0) / len(rs) if rs else 0
        s = daily_dollars(taken, days, mult, mr_cents)
        f = perf(s)
        h1 = s[[d for d in s.index if d < mid]].sum(); h2 = s[[d for d in s.index if d >= mid]].sum()
        print(f"  {label:<20} n={len(taken):>4} win {win:>4.0f}% avgR {statistics.mean(rs):>+6.3f} "
              f"PnL ${f['pnl']:>+7,.0f} Sharpe {f['sharpe']:>5.2f} maxDD ${f['maxdd']:>+7,.0f}  h2 ${h2:>+6,.0f}")

    # ---- THE GATE: correlation MR vs ORB daily $, and combined portfolio ----
    mr_taken = portfolio(mr_tf, CAP)
    mr_s = daily_dollars(mr_taken, days, mult, mr_cents)
    orb_s = daily_dollars(orb_taken, days, mult, orb_cents)
    both = pd.concat([orb_s.rename("orb"), mr_s.rename("mr")], axis=1).fillna(0.0)
    corr = both["orb"].corr(both["mr"])
    comb = both["orb"] + both["mr"]
    fo, fm, fc = perf(orb_s), perf(mr_s), perf(comb)
    print(f"\n  --- THE GATE (trend-filtered MR vs ORB tight-OR trailing) ---")
    print(f"    daily-PnL CORRELATION (ORB, MR): {corr:+.3f}   (need <= ~0.30 for a real 2nd engine)")
    print(f"    {'stream':<16}{'PnL$':>10}{'Sharpe':>8}{'maxDD$':>10}")
    print(f"    {'ORB alone':<16}{fo['pnl']:>+10,.0f}{fo['sharpe']:>8.2f}{fo['maxdd']:>10,.0f}")
    print(f"    {'MR alone':<16}{fm['pnl']:>+10,.0f}{fm['sharpe']:>8.2f}{fm['maxdd']:>10,.0f}")
    print(f"    {'ORB + MR':<16}{fc['pnl']:>+10,.0f}{fc['sharpe']:>8.2f}{fc['maxdd']:>10,.0f}")
    verdict = ("PASS" if (corr <= 0.35 and fm["sharpe"] > 0.5 and fc["sharpe"] > fo["sharpe"]) else "FAIL")
    print(f"    VERDICT: {verdict}  (combined Sharpe {fc['sharpe']:.2f} vs ORB-alone {fo['sharpe']:.2f})")

    # ---- TIER-1 REGIME TEST: restrict MR to RANGE days (SPY ER < 0.30) ----
    er = efficiency_ratio(closes, days)
    ervals = [er[d] for d in days if er.get(d) is not None]
    ranging = {d: (er.get(d) is not None and er[d] < 0.30) for d in days}
    n_range = sum(1 for d in days if ranging[d])
    # diagnostic: avgR of trend-filtered MR by ER tercile (is the regime axis monotone?)
    tv = sorted(ervals)
    lo_c, hi_c = tv[len(tv) // 3], tv[2 * len(tv) // 3]
    terc = {"low-ER (range)": [], "mid": [], "high-ER (trend)": []}
    for t in mr_tf:
        e = er.get(_tday(t))
        if e is None:
            continue
        (terc["low-ER (range)"] if e <= lo_c else terc["high-ER (trend)"] if e > hi_c else terc["mid"]).append(t.pnl_r)
    print(f"\n  --- TIER-1: regime-conditioned MR (SPY ER<0.30 = range; {n_range}/{len(days)} days) ---")
    print(f"    ER terciles cut at {lo_c:.2f}/{hi_c:.2f}; MR avgR by regime (diagnostic):")
    for k, rs in terc.items():
        if rs:
            print(f"      {k:<18} n={len(rs):>4}  avgR {statistics.mean(rs):>+6.3f}  win {100*sum(1 for r in rs if r>0)/len(rs):.0f}%")
    mr_range = portfolio([t for t in mr_tf if ranging.get(_tday(t))], CAP)
    mrr_s = daily_dollars(mr_range, days, mult, mr_cents)
    comb_r = (orb_s + mrr_s)
    fmr, fcr = perf(mrr_s), perf(comb_r)
    corr_r = pd.concat([orb_s.rename("o"), mrr_s.rename("m")], axis=1).fillna(0.0).corr().iloc[0, 1]
    print(f"    range-day MR standalone: PnL ${fmr['pnl']:>+7,.0f}  Sharpe {fmr['sharpe']:>5.2f}  maxDD ${fmr['maxdd']:>+7,.0f}")
    print(f"    correlation w/ ORB: {corr_r:+.3f}   combined Sharpe {fcr['sharpe']:.2f} vs ORB-alone {fo['sharpe']:.2f}")
    gate = (fmr["sharpe"] > 0.5 and corr_r <= 0.35 and fcr["sharpe"] > fo["sharpe"])
    print(f"    >>> PRE-REGISTERED GATE: {'PASS' if gate else 'FAIL'}  "
          f"(need Sharpe>0.5 [{fmr['sharpe']:.2f}], corr<=0.35 [{corr_r:.2f}], combined>{fo['sharpe']:.2f} [{fcr['sharpe']:.2f}])")


def main():
    for w in WINDOWS:
        run_window(w)
    print("\nGate = (corr<=~0.3) AND (MR Sharpe>0.5 net) AND (combined Sharpe > ORB-alone). All three")
    print("=> promote to strategies/meanrev.py + a live runner on the 2nd account. Any fail => not a")
    print("real 2nd engine yet; iterate K_ENTRY/K_STOP/filters or drop it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
