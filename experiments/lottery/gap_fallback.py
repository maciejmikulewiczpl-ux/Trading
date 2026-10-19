"""VM-side fallback board for the Hype bot: gap-ups only, used ONLY when the Surface board is missing.

Why (2026-10-18): the full board is built on the Surface (6:24am PT). When the laptop sleeps
through it, the bot sat IDLE -- 10 straight trading days 10-05..10-16. Since the bot now selects
purely on premarket gap_pct (top-3, gap >= 10%), the only signal it needs is the board's own
`pm_gappers()` -- one StockTwits call + one Alpaca snapshot, cheap enough for the 1 GB VM (the
FULL board is not: its pm_rvol step pulls weeks of minute bars).

Writes picks/<date>.fallback.json (NOT <date>.json) so it can never collide with the Surface's
immutable board or its git push, and never enters the scoreboard (outcomes/analyze glob the
board files; this file stays untracked on the VM). run_lottery_bot.py reads it only when the
real board is absent.

Run:  .venv/bin/python experiments/lottery/gap_fallback.py [YYYY-MM-DD]
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.news_edge.sources import pm_gappers  # noqa: E402

ET = ZoneInfo("America/New_York")
PICKS = ROOT / "experiments" / "lottery" / "picks"
MIN_PRICE = 1.0   # same floor as board.py's tradable filter


def main(argv) -> int:
    date = argv[1] if len(argv) > 1 else datetime.now(ET).date().isoformat()
    if (PICKS / f"{date}.json").exists():
        print(f"real board for {date} present -- fallback not needed.")
        return 0
    out = PICKS / f"{date}.fallback.json"
    if out.exists():
        print(f"fallback for {date} already written ({out.name}).")
        return 0
    raw = pm_gappers(3.0, 40)
    for g in raw:
        if isinstance(g, dict) and "_error" in g:
            print(f"pm_gappers error: {g['_error']}")
    rows = [g for g in raw if isinstance(g, dict) and "symbol" in g]
    picks = [{"symbol": g["symbol"], "price": g["last"], "basket": "gappers",
              "top_k_of": [], "signals": {"gap_pct": g["gap_pct"]}}
             for g in rows if (g.get("last") or 0) >= MIN_PRICE]
    rec = {"date": date, "source": "vm_gap_fallback",
           "logged_at": datetime.now(ET).isoformat(timespec="seconds"), "picks": picks}
    PICKS.mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(out, "w"), indent=2)
    top = sorted(picks, key=lambda p: -p["signals"]["gap_pct"])[:5]
    print(f"wrote {out.name}: {len(picks)} gappers; top: "
          + ", ".join(f"{p['symbol']} {p['signals']['gap_pct']:+.1f}%" for p in top))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
