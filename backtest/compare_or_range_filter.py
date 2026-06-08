"""Cut the wide-OR breakouts? A max-opening-range filter, at the PORTFOLIO level.

trade_anatomy.py found that breakouts whose 15-min opening range is WIDE (>0.8% of
price) are net-NEGATIVE (-0.045 / -0.025R), ~half of all trades, and they're LOW-
slippage (big R) so the negative is real — not a cost artifact. The tight-OR bucket
(<0.3%) looks great (+0.145R) but is the slippage trap (tiny R inflates avgR; real
slippage eats it). So the clean hypothesis is: CUT THE WIDE TAIL, keep the rest.

Per-trade avgR isn't the whole story: the cap fills greedily, so dropping wide-OR
trades FREES slots for other breakouts the same day. This tests it where it counts —
at the portfolio level (cap 16, $50, vol-dial half-risk on high-vol days, as live),
across both windows + OOS halves, net of the measured ~0.042R cost. Cheap: cached
fixed-2R trades only, no minute bars / re-backtest (won't fight a running test).

A pass = the filter RAISES Sharpe and CUTS drawdown vs no-filter while keeping PnL
roughly intact, in BOTH windows and BOTH OOS halves. Anything window-specific or
that only helps one half = an overfit threshold; ignore it.

Run:
    .venv/Scripts/python.exe backtest/compare_or_range_filter.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.universe_portfolio import portfolio  # noqa: E402
from backtest.compare_selection import _tday  # noqa: E402
from backtest.compare_norefill_trend import trend_eligibility, apply_filter  # noqa: E402
from backtest.compare_volpause import prior_vol_flags, three, RISK, CAP, COST  # noqa: E402

WINDOWS = [730, 180]
THRESHOLDS = [None, 1.2, 0.8, 0.5]      # max OR as % of price; None = no filter (baseline)

HEAD = f"{'config':<22}{'trades':>7}{'PnL$':>10}{'Sharpe':>8}{'maxDD$':>10}   {'h1 Sh':>6}{'h2 Sh':>6}{'h2 PnL':>9}"


def or_pct(t) -> float:
    return (t.or_high - t.or_low) / t.entry_price * 100 if t.entry_price else 0.0


def load_cached(w):
    trades = pickle.load(open(ROOT / "backtest" / f".bars_cache_trades_{w}d.pkl", "rb"))
    closes = pickle.load(open(ROOT / "backtest" / f".bars_cache_daily_{w}d.pkl", "rb"))
    present = sorted({t.symbol for t in trades})
    days = sorted({_tday(t) for t in trades})
    trades = apply_filter(trades, trend_eligibility(closes, present, days))
    trades = [t for t in trades if t.side == "long"]
    return trades, closes, days


def prow(label, n, f, h1, h2):
    print(f"{label:<22}{n:>7}{f['pnl']:>+10,.0f}{f['sharpe']:>8.2f}{f['maxdd']:>10,.0f}   "
          f"{h1['sharpe']:>6.2f}{h2['sharpe']:>6.2f}{h2['pnl']:>+9,.0f}")


def run_window(w):
    trades, closes, days = load_cached(w)
    mid = sorted(days)[len(days) // 2]
    prior = prior_vol_flags(closes, days)
    mult = {d: (0.5 if prior[d] else 1.0) for d in days}      # live half-risk vol dial

    print(f"\n=== {w}d: {len(days)} sessions, OOS split {mid}  (fixed-2R, vol-dial, cap {CAP}, ${RISK:.0f}, net {COST}R) ===")
    print(HEAD); print("-" * len(HEAD))
    for thr in THRESHOLDS:
        kept = trades if thr is None else [t for t in trades if or_pct(t) <= thr]
        taken = portfolio(kept, CAP)
        label = "no filter (baseline)" if thr is None else f"max OR <= {thr:.1f}%"
        prow(label, len(taken), *three(taken, days, mid, mult))


def main():
    for w in WINDOWS:
        run_window(w)
    print("\nReads: the filter EARNS its place only if it lifts Sharpe + cuts maxDD vs baseline")
    print("in BOTH windows AND keeps h1/h2 healthy (no half carrying it). Fewer trades is fine")
    print("if risk-adjusted return improves. If it's a wash or window-specific, drop the idea.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
