"""PRE-REGISTERED multi-day selector study for the Hype bot.

THE PROBLEM (established 2026-07-10): the bot holds T+3, but combined_score predicts the
SAME-DAY move (ret_945). At the 3-day horizon combined_score's top-3 is NEGATIVE (-2.17%,
worse than random). So the selector is mis-matched to the hold. QUESTION: does ANY logged
signal, used as the day's selector, predict the 3-DAY move better -- robustly out-of-sample?

PRE-REGISTRATION (fixed before looking at results):
- Universe: every logged candidate with a non-null ret_3d (immutable picks files).
- Selector test: for each signal, take the day's TOP-3 by that signal; measure that basket's
  ret_3d vs two benchmarks -- combined_score top-3 (the LIVE bot) and the net average (= the
  expected value of RANDOM selection from the net).
- Metric = TAIL-AWARE (bot PnL is tail-driven, per profit_only_changes): mean ret_3d, SUM,
  win@W3 (ret_3d>=+20%), and BEST single pick. Mean alone is misleading.
- OOS DISCIPLINE: split boards into EARLY vs LATE halves by date. A signal is a LEAD only if
  its ret_3d edge (vs net) is POSITIVE IN BOTH HALVES. One-half wins are rejected as noise.
- MULTIPLE COMPARISONS: ~10 signals x 2 directions tested; the bar is both-halves robustness,
  NOT a single p-value. Report the count so we don't fool ourselves.
- SAME-DAY CONTRAST: print each signal's ret_945 edge beside its ret_3d edge -> shows which
  signals are same-day-only (the combined_score trap) vs genuinely multi-day.

NOT A SHIP GATE. n~566 candidate-days / ~17 scored boards = SMALL. This GENERATES hypotheses
for the pre-registered 30-day+ decision; it does not authorize a live change. Re-run as the
sample grows.  Run: .venv/Scripts/python.exe experiments/lottery/multiday_selector.py
"""
from __future__ import annotations

import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PICKS = ROOT / "experiments" / "lottery" / "picks"
W3 = 20.0                      # ret_3d >= +20% == the pre-registered W3 explosive winner
K = 3                          # the bot selects top-3/day

# signals with usable coverage; True = higher is "more hype" (sort desc), False = rank (sort asc)
SIGNALS = {
    "combined_score": True,    # the live bot's selector (benchmark)
    "ignition": True, "pm_rvol": True, "gap_pct": True, "gtrends_spike": True,
    "wsb_surge": True, "squeeze": True, "realized_vol": True, "finra_short_ratio": True,
    "st_rank": False, "wsb_rank": False,   # ranks: 1 = most-mentioned -> lower is stronger
}


def load_boards():
    boards = defaultdict(list)
    for f in sorted(glob.glob(str(PICKS / "2026-*.json"))):
        date = Path(f).stem
        d = json.load(open(f))
        d = d if isinstance(d, list) else d.get("picks", d)
        for c in d:
            if isinstance(c, dict) and c.get("ret_3d") is not None:
                sig = dict(c.get("signals") or {})
                sig["combined_score"] = c.get("combined_score")
                boards[date].append({"sym": c["symbol"], "r3": float(c["ret_3d"]),
                                     "r945": c.get("ret_945_close"), "sig": sig})
    return boards


def topk(board, signal, higher):
    cands = [c for c in board if c["sig"].get(signal) is not None]
    cands.sort(key=lambda c: c["sig"][signal], reverse=higher)
    return cands[:K]


def basket(boards, dates, signal=None, higher=True):
    """Return metrics for the selector (top-K by signal); signal=None => the whole net (random)."""
    r3, r945 = [], []
    for date in dates:
        sel = boards[date] if signal is None else topk(boards[date], signal, higher)
        for c in sel:
            r3.append(c["r3"])
            if c["r945"] is not None:
                r945.append(c["r945"])
    if not r3:
        return None
    return {"n": len(r3), "mean": statistics.mean(r3), "sum": sum(r3),
            "winW3": sum(1 for x in r3 if x >= W3) / len(r3) * 100,
            "best": max(r3), "worst": min(r3),
            "mean945": statistics.mean(r945) if r945 else float("nan")}


def main():
    boards = load_boards()
    dates = sorted(boards)
    mid = len(dates) // 2
    early, late = dates[:mid], dates[mid:]
    ncand = sum(len(v) for v in boards.values())
    print("=" * 92)
    print(f"MULTI-DAY SELECTOR STUDY | {len(dates)} scored boards, {ncand} candidate-days | "
          f"metric = ret_3d (the bot's hold)")
    print(f"OOS split: EARLY {early[0]}..{early[-1]} ({len(early)}d) | LATE {late[0]}..{late[-1]} ({len(late)}d)")
    print("=" * 92)

    net_full = basket(boards, dates)
    net_e = basket(boards, early); net_l = basket(boards, late)
    print(f"\nBENCHMARK  net/random (all candidates): mean ret_3d {net_full['mean']:+.2f}%  "
          f"SUM {net_full['sum']:+.0f}  win@W3 {net_full['winW3']:.0f}%  best +{net_full['best']:.0f}%")

    print("\n--- SELECTOR = TOP-3/day by each signal | ret_3d edge vs net, both halves ---")
    print(f"  {'signal':<18}{'mean':>7}{'SUM':>7}{'win%':>6}{'best':>7} | "
          f"{'edgeFULL':>9}{'edgeEARLY':>10}{'edgeLATE':>9}  {'ret945edge':>11}  verdict")
    rows = []
    for sig, higher in SIGNALS.items():
        full = basket(boards, dates, sig, higher)
        e = basket(boards, early, sig, higher)
        l = basket(boards, late, sig, higher)
        if not full or not e or not l:
            continue
        edge_f = full["mean"] - net_full["mean"]
        edge_e = e["mean"] - net_e["mean"]
        edge_l = l["mean"] - net_l["mean"]
        edge945 = (full["mean945"] - net_full["mean945"]) if net_full["mean945"] == net_full["mean945"] else float("nan")
        robust = edge_e > 0 and edge_l > 0
        flip = (edge945 > 0) != (edge_f > 0)          # same-day and multi-day disagree
        verdict = "LEAD (both halves +)" if robust else ("one-half" if (edge_e > 0 or edge_l > 0) else "no")
        if sig != "combined_score" and flip and edge945 > 0:
            verdict += " [same-day trap]"
        tag = " *bot" if sig == "combined_score" else ""
        rows.append((edge_f, robust, sig, full, edge_f, edge_e, edge_l, edge945, verdict, tag))
    # print benchmark (combined_score) first, then signals ranked by full-sample ret_3d edge
    bench = [r for r in rows if r[2] == "combined_score"]
    sigs = sorted([r for r in rows if r[2] != "combined_score"], key=lambda r: -r[0])
    for _, robust, sig, full, ef, ee, el, e945, verdict, tag in bench + sigs:
        print(f"  {sig+tag:<18}{full['mean']:>+7.1f}{full['sum']:>+7.0f}{full['winW3']:>6.0f}"
              f"{full['best']:>+7.0f} | {ef:>+9.2f}{ee:>+10.2f}{el:>+9.2f}  {e945:>+11.2f}  {verdict}")

    print("\n[reads] edge = selector mean ret_3d MINUS net(random) mean. LEAD = positive in BOTH halves.")
    print("  'ret945edge' = same-day edge; if it disagrees in sign with the ret_3d edge, the signal is")
    print("  a same-day selector (the combined_score trap). ~10 signals tested -> both-halves is the bar,")
    print(f"  not a single p-value. n small (~{ncand} candidate-days); hypotheses only, re-run at 30d+.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
