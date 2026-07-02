"""What if the bot had picked a DIFFERENT selector but kept its EXECUTION (10% trail + T+3)?

Isolates SELECTION from EXECUTION: takes each selector's top-3/day (current, bottom3, prepeak,
relsurprise) from the logged picks, reconstructs the ~09:45 entry (from the logged ret_945_close +
the entry-day close), then runs the bot's real exit sim (exit_sim.sim_one: 10% trailing stop, T+3
time-stop) on the daily price path. Reports realized % and $ at the bot's $2,000/name size.

Sanity: 'current' here should ~match the live account (it IS the bot's selection under its execution).
HONEST LIMITS (inherited from exit_sim): daily-resolution stop approximation; thin IEX bars on some
micro-caps (those names drop out); SMALL SAMPLE. Apples-to-apples across selectors -> RELATIVE ranking
is the signal, not the absolute level.

Run:  .venv/Scripts/python.exe experiments/lottery/bottom3_sim.py
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.lottery.analyze import PICKS_DIR, load_days  # noqa: E402
from experiments.lottery.selection_lab import _baskets_for_day  # noqa: E402
from experiments.lottery.exit_sim import CUR_DAYS, CUR_TRAIL, _daily_bars, _load_env, sim_one  # noqa: E402

SIZE = 2000.0   # $ per name (the bot's notional)
SELECTORS = ["current", "bottom3", "prepeak", "relsurprise", "gap_signed"]


def _picks_for(days) -> dict:
    """{selector: [(symbol, entry_date, ret_945_close), ...]} from each day's top-3 for that selector."""
    out = {s: [] for s in SELECTORS}
    for rec in days:
        scored = [p for p in rec["picks"] if p.get("combined_score") is not None]
        bk = _baskets_for_day(scored)
        by_sym = {p["symbol"]: p for p in rec["picks"]}
        for sel in SELECTORS:
            for s in bk.get(sel, []):
                p = by_sym.get(s, {})
                out[sel].append((s, rec["date"], p.get("ret_945_close")))
    return out


def sim_fixed(frm, entry, stop_pct, max_days):
    """A FIXED (non-trailing) stop at entry*(1-stop_pct/100), else exit at the T+max_days close."""
    stop = entry * (1 - stop_pct / 100)
    path = frm[:max_days + 1]
    for i, (_d, _o, _h, lo, c) in enumerate(path):
        if i > 0 and lo <= stop:
            return stop / entry - 1.0
        if i >= max_days:
            return c / entry - 1.0
    return path[-1][4] / entry - 1.0


# exit policies: fn(frm, entry) -> realized fraction
EXITS = {
    "trail 10% (bot)": lambda frm, e: sim_one(frm, e, CUR_TRAIL, CUR_DAYS)[0],
    "no stop (hold T+3)": lambda frm, e: sim_one(frm, e, 999, CUR_DAYS)[0],
    "fixed -15% (T+3)": lambda frm, e: sim_fixed(frm, e, 15, CUR_DAYS),
}


def _run(trades, bars, exitfn) -> dict | None:
    """Run a given exit policy on a selector's picks; entry reconstructed at ~09:45."""
    import statistics
    rs = []
    for sym, edate_s, r945 in trades:
        b = bars.get(sym)
        if not b:
            continue
        edate = datetime.fromisoformat(edate_s).date()
        frm = [x for x in b if x[0] >= edate]
        if len(frm) < 2:
            continue
        close0 = frm[0][4]                                   # entry-day close
        entry = close0 / (1 + r945 / 100) if r945 is not None else frm[0][1]   # 09:45 px, else day open
        rs.append(exitfn(frm, entry) * 100)
    if not rs:
        return None
    tot = sum(rs)
    return {"n": len(rs), "avg_%": round(statistics.mean(rs), 2),
            "win_%": round(sum(1 for x in rs if x > 0) / len(rs) * 100),
            "total_%": round(tot, 1), "total_$": round(tot / 100 * SIZE)}


def main() -> int:
    _load_env()
    days = load_days(PICKS_DIR)
    picks = _picks_for(days)
    allsyms = [t[0] for sel in SELECTORS for t in picks[sel]]
    bars = _daily_bars(allsyms)
    print("=" * 78)
    print(f"SELECTION x EXIT matrix | total $ at ${SIZE:.0f}/name | {len(days)} logged days")
    print("daily-res stop approximation, thin micro-cap bars, SMALL SAMPLE (relative ranking = signal)")
    print("=" * 78)
    exits = list(EXITS)
    print(f"  {'selector':>14}" + "".join(f"{e:>20}" for e in exits))
    best = None
    for sel in SELECTORS:
        cells = {e: _run(picks[sel], bars, EXITS[e]) for e in exits}
        line = f"  {sel:>14}"
        for e in exits:
            st = cells[e]
            line += f"{('$'+format(st['total_$'],'+')+' /'+str(st['win_%'])+'%w') if st else '(no data)':>20}"
            if st and (best is None or st["total_$"] > best[2]):
                best = (sel, e, st["total_$"])
        print(line + ("  <- live bot" if sel == "current" else ""))
    print(f"\n  BEST cell: {best[0]} + [{best[1]}] = ${best[2]:+}")
    print("\nRead: down a column = which SELECTION wins for that exit. Across a row = which EXIT suits that")
    print("selection. 'no stop' lets the tail run but eats full drawdowns; 'fixed -15%' is looser than the")
    print("10% trail. Idealized (~2x live) -> trust the RANKING, not the $. 30-day verdict still rules.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
