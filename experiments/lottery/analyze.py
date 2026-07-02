"""Lottery analyze -- the VERDICT engine. Across all logged picks files:

For each signal basket (and combined-score top-3), compute W1/W2/W3 hit-rates and the
LIFT vs the RANDOM basket's base rate, with a binomial p-value and a day-resampled
bootstrap CI (when n_days >= 15). Then print the pre-registered verdict line per signal.

Winners (pre-registered, immutable):
  W1 = ret_945_close >= +5%   W2 = ret_1d >= +10%   W3 = ret_3d >= +20%
Success bar (v2, 2026-06-30, DeepSeek/ChatGPT review #4a): a signal "works" iff EITHER
  (a) W1 hit-rate >= 2x the random base rate AND the day-resampled bootstrap lift-CI lower
      bound > 1.0                                         [reliability path], OR
  (b) EXPECTANCY edge on the traded horizon (ret_3d mean) beats random by a bootstrap-CI
      lower bound > 0                                     [lumpy-moonshot path],
  with n_days >= 30. The (b) path exists because a signal can have a mediocre hit-rate but
  a great average return driven by fat right-tail winners (the whole lottery premise) — a
  hit-rate-only bar would WRONGLY REJECT it.
  NOTE (Fable review #3, 2026-07-01): path (a) uses the day-CLUSTERED bootstrap CI, NOT the
  plain binomial p (which assumes independent name-days -- invalid when hype names co-move and
  overstates significance). The binomial p is still printed for reference, tagged "(naive)".

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


def _rvol(pick: dict):
    """realized_vol logged per pick (board v1.4, 2026-06-29+); None on older picks."""
    s = pick.get("signals") or {}
    return s.get("realized_vol")


def vol_edges(picks: list[dict], nbins: int = 3) -> list[float] | None:
    """Quantile edges (nbins-1 of them) over all picks that carry realized_vol. None if too few."""
    vs = sorted(v for v in (_rvol(p) for p in picks) if v is not None)
    if len(vs) < nbins * 4:            # need a few per bucket to be meaningful
        return None
    return [vs[int(q * (len(vs) - 1))] for q in [i / nbins for i in range(1, nbins)]]


def _vol_bucket(v, edges) -> int | None:
    if v is None or edges is None:
        return None
    b = 0
    for e in edges:
        if v > e:
            b += 1
    return b


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


def ret_stats(picks: list[dict], field: str) -> tuple[float, float, int]:
    """(mean-return, profit-factor, n) over scored picks. Expectancy = mean; profit-factor
    = gross gains / gross losses (inf if no losers). Captures the lumpy right-tail that a
    hit-rate misses."""
    vals = [p.get(field) for p in picks]
    vals = [v for v in vals if v is not None]
    if not vals:
        return float("nan"), float("nan"), 0
    n = len(vals)
    mean = sum(vals) / n
    gains = sum(v for v in vals if v > 0)
    losses = -sum(v for v in vals if v < 0)
    pf = (gains / losses) if losses > 0 else float("inf")
    return mean, pf, n


def bootstrap_edge_ci(days: list[dict], select, field: str, base_mean: float,
                      n_boot: int = 2000) -> tuple:
    """Day-resampled bootstrap 95% CI on the EXPECTANCY EDGE (signal mean - random mean, in
    the return field's units). base_mean is held fixed (the random baseline). CI lower bound
    > 0 => the signal's average return beats luck. Handles negative baselines cleanly
    (a difference, not a ratio)."""
    import random
    rng = random.Random(20260630)
    per_day = []  # (sum, count) of the selected picks' returns, per day
    for rec in days:
        sel = [p.get(field) for p in rec["picks"] if select(p)]
        sel = [v for v in sel if v is not None]
        per_day.append((sum(sel), len(sel)))
    n = len(per_day)
    if n == 0 or base_mean != base_mean:
        return float("nan"), float("nan")
    edges = []
    for _ in range(n_boot):
        tot = cnt = 0
        for _ in range(n):
            s, c = per_day[rng.randrange(n)]
            tot += s
            cnt += c
        if cnt == 0:
            continue
        edges.append((tot / cnt) - base_mean)
    if not edges:
        return float("nan"), float("nan")
    edges.sort()
    return edges[int(0.025 * len(edges))], edges[int(0.975 * len(edges))]


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


def _day_top3(rec) -> list[dict]:
    return sorted([p for p in rec["picks"] if p.get("combined_score") is not None],
                  key=lambda x: -x["combined_score"])[:3]


def bootstrap_top3_lift_ci(days, field, thr, base_rate, n_boot: int = 2000) -> tuple:
    """Day-resampled 95% CI on the combined-score TOP-3 basket's hit-rate lift (each drawn day
    contributes its OWN top-3). Clustering-aware verdict for the bot's real basket, which has no
    per-pick select predicate. Mirrors bootstrap_lift_ci at the day/top-3 level (Fable #3)."""
    import random
    rng = random.Random(20260701)
    per_day = [hit_stats(_day_top3(rec), field, thr) for rec in days]
    n = len(per_day)
    if n == 0 or base_rate <= 0:
        return float("nan"), float("nan")
    lifts = []
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
    return lifts[int(0.025 * len(lifts))], lifts[int(0.975 * len(lifts))]


def bootstrap_top3_edge_ci(days, field, base_mean, n_boot: int = 2000) -> tuple:
    """Day-resampled 95% CI on the top-3 basket's EXPECTANCY edge (mean return - base_mean)."""
    import random
    rng = random.Random(20260701)
    per_day = []
    for rec in days:
        sel = [v for v in (p.get(field) for p in _day_top3(rec)) if v is not None]
        per_day.append((sum(sel), len(sel)))
    n = len(per_day)
    if n == 0 or base_mean != base_mean:
        return float("nan"), float("nan")
    edges = []
    for _ in range(n_boot):
        tot = cnt = 0
        for _ in range(n):
            s, c = per_day[rng.randrange(n)]
            tot += s
            cnt += c
        if cnt == 0:
            continue
        edges.append((tot / cnt) - base_mean)
    if not edges:
        return float("nan"), float("nan")
    edges.sort()
    return edges[int(0.025 * len(edges))], edges[int(0.975 * len(edges))]


def analyze(picks_dir: Path) -> int:
    days = load_days(picks_dir)
    n_days = len(days)
    if n_days == 0:
        print("no picks files yet. Run board.py for a few days first.")
        return 0

    print(f"=== lottery forward test: {n_days} logged day(s) ===")
    print("Winners: W1 ret_945_close>=+5%  W2 ret_1d>=+10%  W3 ret_3d>=+20%")
    print(f"Bar v2: PASS iff (hit-rate >= {SUCCESS_LIFT:.0f}x random AND day-clustered lift-CI lo>1) "
          f"OR (ret_3d expectancy edge CI>0), n_days>={SUCCESS_NDAYS}.  [binomial p shown but NOT used]")
    print("  expectancy path catches lumpy signals: mediocre hit-rate but fat-tail avg return.\n")

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
    # RANDOM expectancy baseline (mean return) per return field -- the luck baseline for path (b)
    base_exp = {}
    for _, field, _ in WINDEFS:
        m, _pf, _n = ret_stats(random_picks, field)
        base_exp[field] = m
    print("RANDOM-BASKET base rates (the luck baseline):")
    for key, field, thr in WINDEFS:
        w, s = hit_stats(random_picks, field, thr)
        br = base[key]
        m = base_exp[field]
        print(f"  {key} ({field}>=+{thr:.0f}%): {w}/{s} = "
              + (f"{br*100:.1f}%" if br == br else "n/a")
              + (f"   | mean {m:+.2f}%" if m == m else "   | mean n/a"))
    print()

    # --- VOL-MATCHED control (review #4c): compare a signal to random names of the SAME realized
    # vol, so a lift isn't just "the signal picks higher-vol names" (higher vol => more +5% days by
    # luck). realized_vol logged per pick from 2026-06-29 (board v1.4); activates as coverage grows. ---
    all_picks = [p for rec in days for p in rec["picks"]]
    v_edges = vol_edges(all_picks, nbins=3)
    n_rand_vol = sum(1 for p in random_picks if _rvol(p) is not None)
    # random per-vol-bucket stats: hit (W1) and mean-return (ret_3d)
    W1F = ("ret_945_close", 5.0)
    EXPF = "ret_3d"
    rb_hit: dict[int, list] = defaultdict(lambda: [0, 0])   # bucket -> [wins, scored]
    rb_mean: dict[int, list] = defaultdict(lambda: [0.0, 0])  # bucket -> [sum, n]
    for p in random_picks:
        b = _vol_bucket(_rvol(p), v_edges)
        if b is None:
            continue
        hw = _is_win(p, W1F[0], W1F[1])
        if hw is not None:
            rb_hit[b][1] += 1
            rb_hit[b][0] += 1 if hw else 0
        rv = p.get(EXPF)
        if rv is not None:
            rb_mean[b][0] += rv
            rb_mean[b][1] += 1

    def vol_matched(picks):
        """(W1 rate, W1 vol-matched-base, ret_3d mean, ret_3d vol-matched-base-mean, n_cov) for a
        signal's picks that carry realized_vol; reweights the RANDOM baseline to the signal's vol mix.
        nan where a bucket lacks random coverage or too few picks."""
        wt_hit: dict[int, int] = defaultdict(int)
        wt_mean: dict[int, int] = defaultdict(int)
        sw = sm = sn = 0
        hw_tot = hs_tot = 0
        rm_sum = rm_n = 0
        for p in picks:
            b = _vol_bucket(_rvol(p), v_edges)
            if b is None:
                continue
            sn += 1
            hw = _is_win(p, W1F[0], W1F[1])
            if hw is not None:
                wt_hit[b] += 1
                hs_tot += 1
                hw_tot += 1 if hw else 0
            rv = p.get(EXPF)
            if rv is not None:
                wt_mean[b] += 1
                rm_sum += rv
                rm_n += 1
        # vol-matched random base = signal-vol-weighted average of random per-bucket stat
        def _wavg(weights, table, idx_val, idx_den):
            tot = sum(weights.values())
            if tot == 0:
                return float("nan")
            acc = 0.0
            for b, wcnt in weights.items():
                den = table[b][idx_den]
                if den == 0:
                    return float("nan")   # a bucket the signal uses has no random comparison
                acc += (wcnt / tot) * (table[b][idx_val] / den)
            return acc
        w1_rate = (hw_tot / hs_tot) if hs_tot else float("nan")
        w1_base = _wavg(wt_hit, rb_hit, 0, 1)
        exp_mean = (rm_sum / rm_n) if rm_n else float("nan")
        exp_base = _wavg(wt_mean, rb_mean, 0, 1)
        return w1_rate, w1_base, exp_mean, exp_base, sn

    if v_edges is None:
        print(f"VOL-MATCHED control: not enough realized_vol coverage yet "
              f"({n_rand_vol} random picks carry it; logging began 2026-06-29). "
              f"Activates automatically as the sample grows.\n")
    else:
        print(f"VOL-MATCHED control ACTIVE: vol terciles at {[round(e,3) for e in v_edges]} "
              f"({n_rand_vol} random picks carry realized_vol). Compares each signal to random "
              f"names of the SAME vol.\n")

    def report(name: str, picks: list[dict], select=None, top3=False):
        print(f"  [{name}]")
        w1_pass = False   # path (a): hit-rate reliability (day-CLUSTERED bootstrap, not naive binomial)
        for key, field, thr in WINDEFS:
            w, s = hit_stats(picks, field, thr)
            if s == 0:
                print(f"    {key}: n=0 scored")
                continue
            rate = w / s
            br = base.get(key, float("nan"))
            lift = (rate / br) if (br and br == br and br > 0) else float("nan")
            # The binomial p assumes INDEPENDENT name-days, but hype names co-move (one meme day =
            # several correlated W1 hits), so it OVERSTATES significance. It is shown for reference
            # only -- the PASS decision now uses the day-resampled bootstrap lift CI, which is
            # clustering-aware (Fable review #3, 2026-07-01).
            p = _binom_p(w, s, br) if (br == br and br > 0) else float("nan")
            ci_lo = ci_hi = float("nan")
            ci_txt = ""
            if n_days >= 15 and (select is not None or top3):
                ci_lo, ci_hi = (bootstrap_top3_lift_ci(days, field, thr, br) if top3
                                else bootstrap_lift_ci(days, select, field, thr, br))
                if ci_lo == ci_lo:
                    ci_txt = f"  lift95%CI[{ci_lo:.2f},{ci_hi:.2f}]"
            lift_s = f"{lift:.2f}x" if lift == lift else "n/a"
            p_s = f"p~{p:.3f}(naive)" if p == p else "p=n/a"
            if key == "W1" and s > 0 and lift == lift:
                # clustering-aware pass: the day-resampled lift-CI lower bound must clear 1.0 (beats
                # random once day-level correlation is accounted for) AND the point lift must meet the
                # 2x effect-size bar. Needs n_days>=15 for the CI to exist.
                w1_pass = (lift >= SUCCESS_LIFT) and (ci_lo == ci_lo and ci_lo > 1.0)
            print(f"    {key}: {w}/{s} = {rate*100:5.1f}%  lift {lift_s}  {p_s}{ci_txt}")

        # --- EXPECTANCY block: mean return + profit factor per horizon (path b) ---
        exp_pass = False   # path (b): traded-horizon (ret_3d) expectancy edge CI > 0
        for _, field, _ in WINDEFS:
            mean, pf, n = ret_stats(picks, field)
            if n == 0:
                continue
            bm = base_exp.get(field, float("nan"))
            edge = (mean - bm) if bm == bm else float("nan")
            pf_s = "inf" if pf == float("inf") else (f"{pf:.2f}" if pf == pf else "n/a")
            ci_txt = ""
            edge_lo = float("nan")
            if n_days >= 15 and (select is not None or top3) and bm == bm:
                edge_lo, edge_hi = (bootstrap_top3_edge_ci(days, field, bm) if top3
                                    else bootstrap_edge_ci(days, select, field, bm))
                if edge_lo == edge_lo:
                    ci_txt = f"  edge95%CI[{edge_lo:+.2f},{edge_hi:+.2f}]pp"
            edge_s = f"{edge:+.2f}pp" if edge == edge else "n/a"
            if field == "ret_3d" and edge_lo == edge_lo:
                exp_pass = edge_lo > 0   # traded horizon: avg return beats luck (CI>0)
            print(f"    exp {field:13s}: mean {mean:+6.2f}%  vs rand {edge_s:>8s}  "
                  f"PF {pf_s:>5s}{ci_txt}")

        # --- VOL-MATCHED line (review #4c): signal vs SAME-vol random names ---
        vm_checked = False
        vm_pass = False
        if v_edges is not None and select is not None:
            w1r, w1b, em, eb, ncov = vol_matched(picks)
            if ncov >= 3:
                vm_checked = True
                vlift = (w1r / w1b) if (w1b == w1b and w1b > 0) else float("nan")
                vedge = (em - eb) if (em == em and eb == eb) else float("nan")
                vlift_s = f"{vlift:.2f}x" if vlift == vlift else "n/a"
                vedge_s = f"{vedge:+.2f}pp" if vedge == vedge else "n/a"
                vm_pass = (vlift == vlift and vlift >= SUCCESS_LIFT) or (vedge == vedge and vedge > 0)
                print(f"    vol-matched (n={ncov}): W1 {w1r*100:.0f}% vs same-vol rand "
                      f"{w1b*100:.0f}% = lift {vlift_s}  |  ret_3d edge {vedge_s}")

        # --- combined verdict: pass on EITHER path, gated on n_days ---
        if select is not None or top3:
            ndays_ok = n_days >= SUCCESS_NDAYS
            if w1_pass or exp_pass:
                paths = "+".join([p for p, ok in (("hit-rate", w1_pass), ("expectancy", exp_pass)) if ok])
                # vol-matched confirmation: is the raw lift real, or a higher-vol artifact?
                vm_note = ""
                if vm_checked:
                    vm_note = " [vol-matched OK]" if vm_pass else " [WARN: vol-matched lift weak -> may be a vol artifact]"
                tag = (f"  >>> PASSES BAR ({paths}){vm_note}" if ndays_ok
                       else f"  ({paths} ok, need more days){vm_note}")
                print(f"   {tag}")

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

    print("=== combined_score TOP-3 / day (the bot's real basket) ===")
    report("combined_top3", combined_top3, top3=True)
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
