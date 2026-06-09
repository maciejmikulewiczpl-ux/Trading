"""After the close: score today's news-edge picks (9:45->close) and push the head-to-head
(my read vs the StockTwits crowd) to your phone via ntfy. Run by the NewsEdgeScore task
~13:05 PT (just after the 12:55 PT measurement close = 15:55 ET; small buffer for the free IEX minute feed to settle).

Run:
    .venv/Scripts/python.exe experiments/news_edge/notify_score.py [YYYY-MM-DD]
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
import experiments.news_edge.newsedge as ne  # noqa: E402

ET = ZoneInfo("America/New_York")


def _sep(pairs):
    pos = [r for s, r in pairs if s and s > 0]
    neg = [r for s, r in pairs if s and s < 0]
    avg = lambda x: (sum(x) / len(x)) if x else None
    s = (avg(pos) - avg(neg)) if pos and neg else None
    return avg(pos), avg(neg), s, len(pos), len(neg)


def main() -> int:
    load_env()
    date = sys.argv[1] if len(sys.argv) > 1 else datetime.now(ET).date().isoformat()
    try:
        ne.cmd_outcomes(date)   # fetch 9:45->close, write ret_945_close into the picks file
    except Exception as e:
        print(f"outcomes failed: {e}")
    f = ROOT / "experiments" / "news_edge" / "picks" / f"{date}.json"
    if not f.exists():
        notify(f"News-edge {date}: no picks file.", title="News-edge score")
        return 0
    picks = json.load(open(f)).get("picks", [])
    scored = [p for p in picks if p.get("ret_945_close") is not None]
    if not scored:
        notify(f"News-edge {date}: outcomes not available yet.", title="News-edge score")
        return 0

    mine = [(p["signal"], p["ret_945_close"]) for p in scored]
    crowd = [(p.get("source_signals", {}).get("stocktwits"), p["ret_945_close"]) for p in scored]
    _, _, ms, mpos, mneg = _sep(mine)
    _, _, cs, cpos, cneg = _sep(crowd)

    lines = [f"{len(scored)} picks scored (9:45->close):"]
    for p in sorted(scored, key=lambda x: -x["signal"]):
        s, r = p["signal"], p["ret_945_close"]
        hit = "OK" if ((s > 0 and r > 0) or (s < 0 and r < 0)) else ("x" if s != 0 else "-")
        tag = "+" if s > 0 else "-" if s < 0 else "0"
        lines.append(f"{tag}{p['symbol']} {r:+.2f}% {hit}")
    lines.append("")
    lines.append(f"MY edge (+ vs -): {ms:+.2f}%" if ms is not None else "MY edge: n/a")
    lines.append(f"CROWD edge: {cs:+.2f}%" if cs is not None else "CROWD edge: n/a")
    msg = "\n".join(lines)
    notify(msg, title=f"News-edge {date} scored", priority=3, tags=["bar_chart"])
    print(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
