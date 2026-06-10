"""Does a PRICE / MIN-RISK floor on top of the tight-OR cut add real-$ edge?

Fable's review (2026-06-09) flagged: the cents-based slippage cost scales 1/risk_
per_share, so a low-priced name on a tight OR pays a brutal R-haircut. Example: F
~$12, a 0.5%-of-price OR = ~$0.06 risk/share -> ~0.04c round-trip slippage is ~0.66R.
The current live `min_risk_per_share` floor is only $0.05, so these survive. This
dig asks, on the SHIPPED config (tight-OR <=0.5%, trailing, vol-dial, trend filter,
$50 risk / $10k cap, cents-based slippage that scales 1/risk):

  (1) STRATIFY the tight-OR trade set by entry price and by risk/share, and print
      avg$/trade per bucket — so we SEE whether the cheap/tight-risk names bleed.
  (2) Re-price the same cap-aware portfolio with an added PRICE floor and an added
      MIN-RISK/SHARE floor, and check Sharpe / maxDD / OOS-half PnL.

If a floor lifts Sharpe + cuts maxDD in BOTH windows without gutting trade count,
it's a one-line live-config change (min_risk_per_share, or a new price floor). If
it doesn't, the cents model already handled it and we leave the config alone.

Cheap: cached trades + minute bars (same as compare_or_range_capaware.py).
Run:
    .venv/Scripts/python.exe backtest/compare_tightOR_pricefloor.py
"""
from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.universe_portfolio import portfolio  # noqa: E402
from backtest.compare_selection import _tday  # noqa: E402
from backtest.compare_norefill_trend import trend_eligibility, apply_filter  # noqa: E402
from backtest.compare_exits import load, bucket, reexit, POLICIES, EOD  # noqa: E402
from backtest.compare_volpause import prior_vol_flags, perf, RISK, CAP  # noqa: E402
from backtest.compare_or_range_realcost import or_pct, risk_ps  # noqa: E402

import pandas as pd  # noqa: E402

WINDOWS = [730, 180]
OR_THR = 0.5                 # the shipped tight-OR cut
TARGET_MEDIAN_R = 0.042
NOTIONAL_CAP = 10_000.0
SLIP_MULT = [1.0, 1.5]

PRICE_FLOORS = [0.0, 20.0, 50.0, 100.0]      # min entry price ($)
RISK_FLOORS = [0.05, 0.10, 0.15, 0.25]       # min risk/share ($); 0.05 = current live


def cap_shares(t, mult, days_mult):
    """Cap-aware share count: min(risk-based, notional-cap-based), vol-dial applied."""
    rps = risk_ps(t)
    target = RISK * days_mult.get(_tday(t), 1.0)
    return min(math.floor(target / rps), math.floor(NOTIONAL_CAP / t.entry_price))


def net_pnl(t, shares, cents):
    return (t.exit_price - t.entry_price) * shares - 2.0 * cents * shares


def dollar_series(taken, days, days_mult, cents):
    by, tot = {}, 0.0
    for t in taken:
        shares = cap_shares(t, RISK, days_mult)
        if shares <= 0:
            continue
        pnl = net_pnl(t, shares, cents)
        by[_tday(t)] = by.get(_tday(t), 0.0) + pnl
        tot += pnl
    s = pd.Series({d: by.get(d, 0.0) for d in sorted(days)})
    return s, (tot / len(taken) if taken else 0.0)


def stratify(tight, days_mult, cents):
    """Print avg net-$/trade by price bucket and by risk/share bucket (cap-aware)."""
    price_buckets = [(0, 20), (20, 50), (50, 100), (100, 250), (250, 1e9)]
    risk_buckets = [(0, 0.10), (0.10, 0.20), (0.20, 0.40), (0.40, 0.80), (0.80, 1e9)]

    def show(name, getter, buckets):
        print(f"    {name:<16}{'n':>6}{'tot$':>10}{'avg$/tr':>9}{'win%':>7}")
        for lo, hi in buckets:
            grp = [t for t in tight if lo <= getter(t) < hi]
            if not grp:
                continue
            tot = 0.0
            wins = 0
            for t in grp:
                sh = cap_shares(t, RISK, days_mult)
                if sh <= 0:
                    continue
                p = net_pnl(t, sh, cents)
                tot += p
                wins += (p > 0)
            lab = f"{lo:g}-{hi:g}" if hi < 1e9 else f"{lo:g}+"
            print(f"    {lab:<16}{len(grp):>6}{tot:>+10,.0f}{tot/len(grp):>+9.1f}"
                  f"{100*wins/len(grp):>6.0f}%")

    show("price ($)", lambda t: t.entry_price, price_buckets)
    show("risk/share ($)", risk_ps, risk_buckets)


HEAD = (f"{'config':<26}{'trades':>7}{'PnL$':>10}{'Sharpe':>8}{'maxDD$':>9}"
        f"   {'h1 PnL':>9}{'h2 PnL':>9}{'avg$/tr':>8}")


def run_window(w):
    all_bars, days, present, trades, closes = load(w)
    elig = trend_eligibility(closes, present, days)
    buckets = bucket(all_bars, present)
    tz = all_bars.index.get_level_values(1).tz
    eod_ns = {d: pd.Timestamp.combine(d, EOD).tz_localize(tz).value for d in days}
    trail = [t for t in apply_filter(reexit(trades, buckets, POLICIES["trail_1R"], eod_ns), elig)
             if t.side == "long"]
    mid = sorted(days)[len(days) // 2]
    prior = prior_vol_flags(closes, days)
    days_mult = {d: (0.5 if prior[d] else 1.0) for d in days}
    base_cents = TARGET_MEDIAN_R * statistics.median(risk_ps(t) for t in trail) / 2.0

    tight = [t for t in trail if or_pct(t) <= OR_THR]
    print(f"\n{'='*78}\n=== {w}d | tight-OR<={OR_THR}% set: {len(tight)} trades | OOS {mid} ===")

    for sm in SLIP_MULT:
        cents = base_cents * sm
        print(f"\n  --- slip {sm:.1f}x (${cents:.3f}/share) ---")
        print(f"\n  STRATIFY tight-OR trades (cap-aware net $):")
        stratify(tight, days_mult, cents)

        print(f"\n  PRICE FLOOR on top of tight-OR<={OR_THR}%:")
        print("  " + HEAD); print("  " + "-" * len(HEAD))
        for pf in PRICE_FLOORS:
            kept = [t for t in tight if t.entry_price >= pf]
            taken = portfolio(kept, CAP)
            s, avgtr = dollar_series(taken, days, days_mult, cents)
            f = perf(s)
            h1 = s[[d for d in s.index if d < mid]].sum()
            h2 = s[[d for d in s.index if d >= mid]].sum()
            label = "no floor" if pf == 0 else f"price >= ${pf:.0f}"
            print(f"  {label:<26}{len(taken):>7}{f['pnl']:>+10,.0f}{f['sharpe']:>8.2f}"
                  f"{f['maxdd']:>9,.0f}   {h1:>+9,.0f}{h2:>+9,.0f}{avgtr:>+8.1f}")

        print(f"\n  MIN RISK/SHARE floor on top of tight-OR<={OR_THR}%:")
        print("  " + HEAD); print("  " + "-" * len(HEAD))
        for rf in RISK_FLOORS:
            kept = [t for t in tight if risk_ps(t) >= rf]
            taken = portfolio(kept, CAP)
            s, avgtr = dollar_series(taken, days, days_mult, cents)
            f = perf(s)
            h1 = s[[d for d in s.index if d < mid]].sum()
            h2 = s[[d for d in s.index if d >= mid]].sum()
            label = f"risk/sh >= ${rf:.2f}" + ("  (live)" if rf == 0.05 else "")
            print(f"  {label:<26}{len(taken):>7}{f['pnl']:>+10,.0f}{f['sharpe']:>8.2f}"
                  f"{f['maxdd']:>9,.0f}   {h1:>+9,.0f}{h2:>+9,.0f}{avgtr:>+8.1f}")


def main():
    for w in WINDOWS:
        run_window(w)
    print("\nReads: in the STRATIFY block, a price/risk bucket with negative avg$/tr is")
    print("a bleeder the cents model is NOT pricing out. A floor is worth shipping ONLY if it")
    print("lifts Sharpe AND cuts maxDD in BOTH windows without gutting the (already small) count.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
