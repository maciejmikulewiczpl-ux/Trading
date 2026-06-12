"""Morning: build today's lottery board and push a short summary to the phone via ntfy.
Run by the LotteryBoard Windows task ~6:24am PT. Plain python (no LLM/agent).

Run:
    .venv/Scripts/python.exe experiments/lottery/notify_board.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from backtest.run_orb import load_env  # noqa: E402
from live.notify import notify  # noqa: E402
import experiments.lottery.board as board  # noqa: E402

ET = ZoneInfo("America/New_York")
PICKS_DIR = Path(__file__).resolve().parent / "picks"


def main() -> int:
    load_env()
    rc = board.main([])   # builds + writes today's picks (refuses overwrite)
    date = datetime.now(ET).date().isoformat()
    f = PICKS_DIR / f"{date}.json"
    if not f.exists():
        notify(f"Lottery {date}: no board file.", title="Lottery board")
        return rc
    rec = json.load(open(f))
    picks = rec.get("picks", [])
    # top-3 by combined_score = the names the bot will buy
    ranked = sorted([p for p in picks if p.get("combined_score") is not None],
                    key=lambda x: -x["combined_score"])[:3]
    lines = [f"{len(picks)} board picks "
             f"({rec.get('n_tradable', '?')} tradable / {rec.get('n_candidates', '?')} cand)"]
    lines.append("Bot top-3 (combined_score):")
    for p in ranked:
        sig = p["signals"]
        bits = []
        if sig.get("ignition") is not None:
            bits.append(f"ig{sig['ignition']}")
        if sig.get("wsb_surge") is not None:
            bits.append(f"wsb{sig['wsb_surge']}x")
        if sig.get("gap_pct") is not None:
            bits.append(f"gap{sig['gap_pct']:+.0f}%")
        lines.append(f"{p['symbol']} cs={p['combined_score']:.2f} {' '.join(bits)}")
    msg = "\n".join(lines)
    ok = notify(msg, title=f"Lottery board {date}", priority=3, tags=["game_die"])
    print(msg)
    print("ntfy push sent" if ok else "ntfy NOT sent (NTFY_TOPIC unset?)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
