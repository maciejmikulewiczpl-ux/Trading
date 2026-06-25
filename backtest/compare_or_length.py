"""Does the OPENING-RANGE WINDOW LENGTH matter? Fixed sweep: 5 / 10 / 15 / 30-min OR.

Everything is built on the 15-min OR; the tightness *threshold* has been swept but the
window *duration* never has (only a regime-CONDITIONAL longer-OR test exists,
compare_or_byregime.py). Hypothesis: a SHORTER OR (5-10 min) locks tighter ranges earlier
-> more names clear the tight-OR filter AND more runway to EOD -> potentially more
profitable breakouts; a LONGER OR is more selective/later. This sweeps fixed OR lengths on
the live edge filters (tight-OR <=0.5% of price + trend filter + cap 16 + half-risk vol
dial), 730d, OOS halved.

CAVEAT: fixed-2R exits here (the OR-length backtest re-runs the engine; re-pricing each
length with the live TRAILING exit is the expensive follow-up — only worth it if a length
beats OR15 on this cheaper test first). So read this as a DIRECTIONAL screen of OR length,
not the final live number. Gate to advance an alternative: PnL up AND Sharpe >= OR15 AND
maxDD not worse, in BOTH OOS halves.

Run (~15-20 min, re-runs the backtest at each OR length):
    .venv/Scripts/python.exe backtest/compare_or_length.py
"""
from __future__ import annotations

import sys
from datetime import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from strategies.orb import Params  # noqa: E402
from backtest.run_orb import run_backtest, STARTING_EQUITY  # noqa: E402
from backtest.universe_portfolio import portfolio  # noqa: E402
from backtest.compare_norefill_trend import trend_eligibility, apply_filter  # noqa: E402
from backtest.compare_exits import load  # noqa: E402
from backtest.compare_volpause import prior_vol_flags, series, perf, CAP  # noqa: E402
from backtest.compare_selection import _tday  # noqa: E402

WINDOW = sys.argv[1] if len(sys.argv) > 1 else "730"   # pass 180 for a fast directional screen
ORS = [10, 15, 30]   # 5-min dropped: floods the book with marginal breakouts (pathologically slow + noisy)
TIGHT = 0.005   # tight-OR: risk/share (~OR range) <= 0.5% of entry price (shipped edge)


def tight_or(trades):
    return [t for t in trades if abs(t.entry_price - t.stop_price) / t.entry_price <= TIGHT]


def trades_for_or(all_bars, days, present, cached15, or_min):
    if or_min == 15:
        return cached15
    p = Params(or_minutes=or_min, target_r=2.0, risk_per_trade=100.0, max_position_pct=0.25,
               max_position_dollars=10_000.0, no_entry_after_time=time(11, 30))
    t, _ = run_backtest(all_bars, days, present, p, STARTING_EQUITY)
    return t


def main():
    all_bars, days, present, cached15, closes = load(int(WINDOW))
    elig = trend_eligibility(closes, present, days)
    hv = prior_vol_flags(closes, days)
    mid = sorted(days)[len(days) // 2]
    mult = {d: (0.5 if hv[d] else 1.0) for d in days}   # live half-risk dial
    d1 = [d for d in days if d < mid]
    d2 = [d for d in days if d >= mid]

    print(f"=== {WINDOW}d OR-LENGTH sweep (tight-OR<=0.5% + trend + cap{CAP} + vol dial, fixed-2R) ===")
    print(f"{'OR length':<14}{'trades':>8}{'PnL$':>11}{'Sharpe':>8}{'maxDD$':>10}{'h1 Sh':>7}{'h2 Sh':>7}")
    print("-" * 65)
    for o in ORS:
        raw = trades_for_or(all_bars, days, present, cached15, o)
        taken = portfolio(tight_or(apply_filter(raw, elig)), CAP)
        s = perf(series(taken, days, mult))
        s1 = perf(series([t for t in taken if _tday(t) < mid], d1, mult))
        s2 = perf(series([t for t in taken if _tday(t) >= mid], d2, mult))
        tag = "  <- baseline" if o == 15 else ""
        print(f"{str(o)+'-min':<14}{len(taken):>8}{s['pnl']:>+11,.0f}{s['sharpe']:>8.2f}"
              f"{s['maxdd']:>10,.0f}{s1['sharpe']:>7.2f}{s2['sharpe']:>7.2f}{tag}")
    print("\nAdvance a length only if it beats 15-min on PnL AND Sharpe AND maxDD in BOTH halves;")
    print("then confirm with the live TRAILING exit before believing the $ number.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
