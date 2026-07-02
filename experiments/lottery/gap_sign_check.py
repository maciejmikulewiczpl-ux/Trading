"""Gap-SIGN diagnostic (Fable PnL #5): combined_score currently scores |gap_pct| (abs), so a -16%
gap-DOWN ranks as bullish as a +16% gap-UP. Fable's claim: down-gappers in small caps continue down/
chop and are "your worst trades" -- dropping them is a free left-tail removal.

Direct test: bucket every scored pick by gap sign and compare forward returns at each horizon. If
down-gappers systematically underperform up-gappers (and drag the traded top-3), signing the gap helps.
MEASUREMENT ONLY; does not touch the live bot.

Run:
    .venv/Scripts/python.exe experiments/lottery/gap_sign_check.py
"""
from __future__ import annotations

import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.lottery.analyze import load_days, PICKS_DIR  # noqa: E402

HORIZONS = [("ret_945_close", "same-day"), ("ret_1d", "1-day"), ("ret_3d", "3-day")]
UP, DOWN = 1.0, -1.0   # gap_pct thresholds (%), names between are "flat/none"


def _gap(p):
    return (p.get("signals") or {}).get("gap_pct")


def _bucket(g):
    if g is None:
        return "no-gap"
    if g >= UP:
        return "gap-UP"
    if g <= DOWN:
        return "gap-DOWN"
    return "flat"


def _agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return (len(vals), st.mean(vals), st.median(vals),
            100 * sum(1 for v in vals if v > 0) / len(vals), sum(vals), max(vals), min(vals))


def main():
    days = load_days(PICKS_DIR)
    if not days:
        print("no picks."); return 0
    allp = [p for rec in days for p in rec["picks"] if p.get("combined_score") is not None]
    print(f"=== GAP-SIGN diagnostic: {len(days)} days, {len(allp)} scored picks ===")
    print(f"buckets: gap-UP >= +{UP:.0f}% | flat | gap-DOWN <= {DOWN:.0f}% | no-gap (gap_pct null)\n")

    for fld, label in HORIZONS:
        print(f"  [{label}]  {'bucket':<10}{'n':>4}{'mean%':>8}{'SUM%':>9}{'best%':>8}{'worst%':>8}{'win%':>6}")
        for bk in ["gap-UP", "flat", "gap-DOWN", "no-gap"]:
            a = _agg([p.get(fld) for p in allp if _bucket(_gap(p)) == bk])
            if a:
                print(f"           {bk:<10}{a[0]:>4}{a[1]:>+8.2f}{a[4]:>+9.1f}{a[5]:>+8.1f}{a[6]:>+8.1f}{a[3]:>6.0f}")
        print()

    # How many of the TRADED top-3 (by combined_score) were down-gappers, and how did they do?
    traded = []
    for rec in days:
        sc = sorted([p for p in rec["picks"] if p.get("combined_score") is not None],
                    key=lambda x: -x["combined_score"])[:3]
        traded += sc
    n_down = sum(1 for p in traded if _bucket(_gap(p)) == "gap-DOWN")
    print(f"  TRADED top-3 basket: {len(traded)} names, {n_down} were gap-DOWN "
          f"({100*n_down/len(traded):.0f}%).")
    for fld, label in HORIZONS:
        dn = _agg([p.get(fld) for p in traded if _bucket(_gap(p)) == "gap-DOWN"])
        rest = _agg([p.get(fld) for p in traded if _bucket(_gap(p)) != "gap-DOWN"])
        if dn and rest:
            print(f"    {label:9s}: down-gappers SUM {dn[4]:+.1f} (n={dn[0]}, mean {dn[1]:+.2f}%)  "
                  f"vs rest SUM {rest[4]:+.1f} (n={rest[0]}, mean {rest[1]:+.2f}%)")
    print("\nRead: if gap-DOWN underperforms gap-UP AND down-gappers in the traded basket drag SUM,")
    print("signing the gap (drop down-gaps from the score) removes them = free left-tail cut. Small n.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
