"""Lottery analyze -- the VERDICT engine. Across all logged picks files:

For each signal basket (and combined-score top-3), compute W1/W2/W3 hit-rates and the
LIFT vs the RANDOM basket's base rate, with a binomial p-value and a day-resampled
bootstrap CI (when n_days >= 15). Then print the pre-registered verdict line per signal.

Winners (pre-registered, immutable):
  W1 = ret_945_close >= +5%   W2 = ret_1d >= +10%   W3 = ret_3d >= +20%
Success bar: a signal "works" iff W1 hit-rate >= 2x the random base rate, p < 0.05,
n_days >= 30.

Run:
    .venv/Scripts/python.exe experiments/lottery/analyze.py
    .venv/Scripts/python.exe experiments/lottery/analyze.py --dir /path/to/picks
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PICKS_DIR = HERE / "picks"

# (key, return-field, threshold-pct)
WINDEFS = [("W1", "ret_945_close", 5.0), ("W2", "ret_1d", 10.0), ("W3", "ret_3d", 20.0)]
SUCCESS_LIFT = 2.0
SUCCESS_NDAYS = 30


def _binom_p(k: int, n: int, p0: float) -> float:
    """One-sided binomial P(X >= k | n, p0). Exact for small n; normal approx otherwise."""
    if n == 0:
        return float("nan")
    if p0 <= 0:
        return 0.0 if k > 0 else 1.0
    if n <= 1000:
        # exact upper tail
        from math import comb
        return float(sum(comb(n, i) * p0**i * (1 - p0)**(n - i) for i in range(k, n + 1)))
    mu = n * p0
    sd = math.sqrt(n * p0 * (1 - p0))
    if sd == 0:
        return 1.0
    z = (k - 0.5 - mu) / sd
    return 0.5 * math.erfc(z / math.sqrt(2))


def load_days(picks_dir: Path) -> list[dict]:
    return [json.load(open(f)) for f in sorted(picks_dir.glob("*.json"))]


def _is_win(pick: dict, field: str, thr: float) -> bool | None:
    v = pick.get(field)
    if v is None:
        return None
    return v >= thr


def _basket_of(pick: dict) -> str:
    return pick.get("basket", "?")


def hit_stats(picks: list[dict], field: str, thr: float) -> tuple[int, int]:
    """(#winners, #scored) for the given window over a list of picks."""
    wins = scored = 0
    for p in picks:
        w = _is_win(p, field, thr)
        if w is None:
            continue
        scored += 1
        wins += 1 if w else 0
    return wins, scored


def bootstrap_lift_ci(days: list[dict], select, field: str, thr: float,
                      base_rate: float, n_boot: int = 2000) -> tuple:
    """Resample DAYS with replacement; recompute hit-rate / base_rate each draw -> 95% CI."""
    import random
    rng = random.Random(20260612)
    per_day = []   # (wins, scored) for the selected basket, per day
    for rec in days:
        sel = [p for p in rec["picks"] if select(p)]
        per_day.append(hit_stats(sel, field, thr))
    lifts = []
    n = len(per_day)
    if n == 0 or base_rate <= 0:
        return float("nan"), float("nan")
    for _ in range(n_boot):
        w = s = 0
        for _ in range(n):
            ww, ss = per_day[rng.randrange(n)]
            w += ww
            s += ss
        if s == 0:
            continue
        lifts.append((w / s) / base_rate)
    if not lifts:
        return float("nan"), float("nan")
    lifts.sort()
    lo = lifts[int(0.025 * len(lifts))]
    hi = lifts[int(0.975 * len(lifts))]
    return lo, hi


def analyze(picks_dir: Path) -> int:
    days = load_days(picks_dir)
    n_days = len(days)
    if n_days == 0:
        print("no picks files yet. Run board.py for a few days first.")
        return 0

    print(f"=== lottery forward test: {n_days} logged day(s) ===")
    print("Winners: W1 ret_945_close>=+5%  W2 ret_1d>=+10%  W3 ret_3d>=+20%")
    print(f"Bar: W1 hit-rate >= {SUCCESS_LIFT:.0f}x random base, p<0.05, n_days>={SUCCESS_NDAYS}\n")

    # gather picks by basket and by signal-flag (top_k_of)
    by_basket: dict[str, list] = defaultdict(list)
    by_signal: dict[str, list] = defaultdict(list)
    combined_top3: list = []
    for rec in days:
        ranked = sorted([p for p in rec["picks"] if p.get("combined_score") is not None],
                        key=lambda x: -x["combined_score"])
        combined_top3.extend(ranked[:3])
        for p in rec["picks"]:
            by_basket[_basket_of(p)].append(p)
            for sig in p.get("top_k_of", []):
                by_signal[sig].append(p)

    # --- RANDOM base rate per window (the luck baseline) ---
    random_picks = by_basket.get("random", [])
    base = {}
    for key, field, thr in WINDEFS:
        w, s = hit_stats(random_picks, field, thr)
        base[key] = (w / s) if s else float("nan")
    print("RANDOM-BASKET base rates (the luck baseline):")
    for key, field, thr in WINDEFS:
        w, s = hit_stats(random_picks, field, thr)
        br = base[key]
        print(f"  {key} ({field}>=+{thr:.0f}%): {w}/{s} = "
              + (f"{br*100:.1f}%" if br == br else "n/a"))
    print()

    def report(name: str, picks: list[dict], select=None):
        print(f"  [{name}]")
        for key, field, thr in WINDEFS:
            w, s = hit_stats(picks, field, thr)
            if s == 0:
                print(f"    {key}: n=0 scored")
                continue
            rate = w / s
            br = base.get(key, float("nan"))
            lift = (rate / br) if (br and br == br and br > 0) else float("nan")
            p = _binom_p(w, s, br) if (br == br and br > 0) else float("nan")
            ci_txt = ""
            if n_days >= 15 and select is not None:
                lo, hi = bootstrap_lift_ci(days, select, field, thr, br)
                if lo == lo:
                    ci_txt = f"  lift95%CI[{lo:.2f},{hi:.2f}]"
            lift_s = f"{lift:.2f}x" if lift == lift else "n/a"
            p_s = f"p={p:.3f}" if p == p else "p=n/a"
            verdict = ""
            if key == "W1" and s > 0 and lift == lift:
                ok = (lift >= SUCCESS_LIFT and p < 0.05 and n_days >= SUCCESS_NDAYS)
                verdict = "  >>> PASSES BAR" if ok else (
                    "  (lift ok, need more days)" if (lift >= SUCCESS_LIFT and p < 0.05)
                    else "")
            print(f"    {key}: {w}/{s} = {rate*100:5.1f}%  lift {lift_s}  {p_s}{ci_txt}{verdict}")

    print("=== per-signal baskets (top_k_of) ===")
    for sig in sorted(by_signal.keys()):
        report(sig, by_signal[sig],
               select=lambda p, _s=sig: _s in p.get("top_k_of", []))
        print()

    print("=== per-primary-basket ===")
    for b in ["wsb", "stocktwits", "gappers", "control", "random"]:
        if by_basket.get(b):
            report(b, by_basket[b], select=lambda p, _b=b: p.get("basket") == _b)
            print()

    print("=== combined_score TOP-3 / day ===")
    report("combined_top3", combined_top3)
    print()

    if n_days < SUCCESS_NDAYS:
        print(f"NOTE: {n_days}/{SUCCESS_NDAYS} days. Verdict is informational until n_days>=30.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(PICKS_DIR))
    args = ap.parse_args()
    return analyze(Path(args.dir))


if __name__ == "__main__":
    raise SystemExit(main())
