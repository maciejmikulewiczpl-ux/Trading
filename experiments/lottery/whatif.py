"""What-if policy simulator for the Hype bot — sweeps SELECTION / HORIZON / SIZING
variants over the logged candidate returns, so we can ask "would a different rule have
done better?" without having traded it.

Built on the picks files, which log EVERY candidate each day with combined_score and
returns at three horizons: ret_945_close (9:45->EOD), ret_1d, ret_3d. That makes them a
(small) backtest set for policy questions:
  - HORIZON: close at EOD vs hold 1d vs hold 3d  (the recycle-vs-hold question)
  - COUNT:   trade top-1 / top-3 / top-5 by combined_score  (concentrate vs spread)
  - WEIGHT:  equal vs combined_score-weighted
plus a random-pick baseline. Reports return + a capital-EFFICIENCY view (return per
capital-day, since a 3-day hold ties up 3x the capital a daily recycle does).

HONEST LIMITS (printed in the banner too):
  - FRICTIONLESS: no slippage/spread. EOD-recycle has ~3x the turnover, so it is FLATTERED
    here until we net out costs from logs/lottery_execution.csv (collecting now).
  - BUY-AND-HOLD horizons: this sim uses raw ret_1d/ret_3d, NOT the bot's path-dependent
    trailing stop. The trailing stop can beat buy-and-hold (rode BIRD/SLS up, exited before
    the reversal) — see the ledger cross-check section for the traded names.
  - SMALL SAMPLE: at ~a dozen days every number is noise-dominated. Directional only until
    the ~30-day base (and these slices want MORE sample, not less).

Run:  .venv/Scripts/python.exe experiments/lottery/whatif.py
"""
from __future__ import annotations

import csv
import glob
import json
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PICKS = ROOT / "experiments" / "lottery" / "picks"
LEDGER = ROOT / "logs" / "lottery_trade_ledger.csv"
# horizon -> (candidate-dict key set in _load_days, capital-days the hold ties up)
HORIZONS = {"EOD": ("EOD", 1), "1d": ("1d", 1), "3d": ("3d", 3)}


def _load_days() -> list[dict]:
    """[{date, picks:[{symbol, cs, EOD, 1d, 3d}]}] for days with at least one scored pick."""
    days = []
    for fp in sorted(glob.glob(str(PICKS / "*.json"))):
        rec = json.load(open(fp))
        cands = []
        for p in rec.get("picks", []):
            cs = p.get("combined_score")
            if cs is None:
                continue
            cands.append({"symbol": p["symbol"], "cs": cs,
                          "EOD": p.get("ret_945_close"), "1d": p.get("ret_1d"),
                          "3d": p.get("ret_3d")})
        if any(c["EOD"] is not None for c in cands):
            days.append({"date": rec.get("date", Path(fp).stem), "picks": cands})
    return days


def _select(cands, n, weighting, rng=None):
    """Return [(symbol, weight)] for the chosen policy."""
    pool = [c for c in cands if c["cs"] is not None]
    if weighting == "random" and rng is not None:
        chosen = rng.sample(pool, min(n, len(pool))) if pool else []
    else:
        chosen = sorted(pool, key=lambda c: -c["cs"])[:n]
    if not chosen:
        return []
    if weighting == "score":
        tot = sum(c["cs"] for c in chosen) or 1.0
        return [(c["symbol"], c["cs"] / tot, c) for c in chosen]
    w = 1.0 / len(chosen)
    return [(c["symbol"], w, c) for c in chosen]


def simulate(days, horizon, n, weighting, seed=7):
    """Daily portfolio return under a policy, using the buy-and-hold horizon return."""
    field, hdays = HORIZONS[horizon]
    rng = random.Random(seed)
    daily, trade_rets = [], []
    for d in days:
        picks = _select(d["picks"], n, weighting, rng)
        rs = [(w, c[field]) for (_s, w, c) in picks if c[field] is not None]
        if not rs:
            continue
        # renormalize weights over names that have this horizon's return
        wsum = sum(w for w, _ in rs) or 1.0
        day_ret = sum((w / wsum) * r for w, r in rs)
        daily.append(day_ret)
        trade_rets.extend(r for _w, r in rs)
    if not daily:
        return None
    mean_trade = statistics.mean(trade_rets)
    return {
        "days": len(daily), "trades": len(trade_rets),
        "mean_trade_%": round(mean_trade, 2),
        "ret_per_capital_day_%": round(mean_trade / hdays, 2),
        "win_rate_%": round(sum(1 for r in trade_rets if r > 0) / len(trade_rets) * 100),
        "mean_day_%": round(statistics.mean(daily), 2),
        "vol_day_%": round(statistics.pstdev(daily), 2) if len(daily) > 1 else 0.0,
        "sharpe_day": round(statistics.mean(daily) / statistics.pstdev(daily), 2)
                      if len(daily) > 1 and statistics.pstdev(daily) else None,
        "worst_day_%": round(min(daily), 2),
        "cum_sum_%": round(sum(daily), 2),
    }


def _table(title, rows, cols):
    print(f"\n=== {title} ===")
    print("  " + "".join(f"{c:>16}" for c in ["variant"] + cols))
    for name, st in rows:
        if st is None:
            print(f"  {name:>16}  (no data)"); continue
        print("  " + f"{name:>16}" + "".join(f"{st.get(c, ''):>16}" for c in cols))


def ledger_trailing_check():
    """For TRADED names: actual trailing-stop realized vs the buy-and-hold 3d return — shows
    what the path-dependent stop adds/subtracts vs the sim's buy-and-hold assumption."""
    if not LEDGER.exists():
        return
    rows = list(csv.DictReader(open(LEDGER)))
    pairs = []
    for r in rows:
        rp = r.get("realized_pct")
        # match to that pick's 3d buy-hold return from the picks file
        try:
            rec = json.load(open(PICKS / f"{r['entry_date']}.json"))
            h3 = next((p.get("ret_3d") for p in rec["picks"] if p["symbol"] == r["symbol"]), None)
        except Exception:
            h3 = None
        if rp not in ("", None) and h3 is not None:
            pairs.append((r["symbol"], float(rp), float(h3)))
    if not pairs:
        return
    real = statistics.mean(p[1] for p in pairs)
    bh = statistics.mean(p[2] for p in pairs)
    print(f"\n=== trailing-stop reality check (traded names, n={len(pairs)}) ===")
    print(f"  actual trailing-stop realized avg: {real:+.2f}%   vs   buy-and-hold 3d avg: {bh:+.2f}%")
    print(f"  => the trailing stop {'ADDED' if real > bh else 'COST'} ~{abs(real-bh):.2f}% vs naive 3d hold")


def main():
    days = _load_days()
    print("=" * 74)
    print("HYPE BOT WHAT-IF SIMULATOR  |  FRICTIONLESS | buy-and-hold horizons | SMALL SAMPLE")
    print(f"{len(days)} scored days  |  numbers are DIRECTIONAL ONLY until the ~30-day base")
    print("=" * 74)

    cols = ["days", "trades", "mean_trade_%", "ret_per_capital_day_%", "win_rate_%", "cum_sum_%"]
    # Q1 — horizon: close at EOD vs hold 1d vs 3d (top-3 by combined_score, equal weight)
    _table("HORIZON  (top-3, equal weight)  - recycle-daily vs hold",
           [(h, simulate(days, h, 3, "equal")) for h in HORIZONS], cols)

    # Q2 — count: concentrate vs spread (EOD horizon, equal weight)
    pcols = ["days", "trades", "mean_day_%", "vol_day_%", "sharpe_day", "worst_day_%", "cum_sum_%"]
    _table("COUNT  (EOD, equal weight)  - concentrate (top-1) vs spread (top-5)",
           [(f"top-{n}", simulate(days, "EOD", n, "equal")) for n in (1, 3, 5)], pcols)

    # weighting: equal vs score (top-3, EOD)
    _table("WEIGHTING  (top-3, EOD)",
           [(w, simulate(days, "EOD", 3, w)) for w in ("equal", "score")], pcols)

    # baseline: random top-3 vs the score-selected top-3 (EOD)
    _table("vs RANDOM baseline  (top-3, EOD)",
           [("score-top3", simulate(days, "EOD", 3, "equal")),
            ("random-3", simulate(days, "EOD", 3, "random"))], cols)

    ledger_trailing_check()
    print("\n[reminder] frictionless + buy-and-hold + tiny sample. EOD-recycle is flattered "
          "(no turnover cost); the bot's real 3d hold uses a trailing stop (see check above).")


if __name__ == "__main__":
    main()
