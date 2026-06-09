"""Launch the NEWS-EDGE ORB bot — the parallel "catalyst-selected ORB" experiment.

Same validated ORB engine (live/paper_orb.py), but:
  - SECOND paper account (keys from .env.news) — isolated P&L, no collision with the
    VM baseline bot,
  - universe = today's POSITIVE (+) news picks (experiments/news_edge/picks/<date>.json),
    not the full ~100-name watchlist,
  - NO trend filter (ORB_TREND_FILTER=false) — the morning news catalyst is the screen,
    not a technical test. (Tight-OR isn't in the live engine anyway.)
  - trailing exit + $50 risk stay the same, so it's apples-to-apples per trade vs baseline.

Env + the trend-filter flag are set BEFORE importing paper_orb (TREND_FILTER_ENABLED is
read at import). Passthrough args (e.g. --dry-run, --preflight-only, --ignore-clock,
--watchlist SYMS) go straight to paper_orb.main().

Run:
    .venv/Scripts/python.exe scripts/run_news_orb.py                 # trade today's +picks
    .venv/Scripts/python.exe scripts/run_news_orb.py --preflight-only
    .venv/Scripts/python.exe scripts/run_news_orb.py --watchlist NVDA,AMD --dry-run --ignore-clock
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
ET = ZoneInfo("America/New_York")


def _load_news_env() -> None:
    f = ROOT / ".env.news"
    if not f.exists():
        print("FATAL: .env.news not found (second-account keys). Aborting.", file=sys.stderr)
        sys.exit(2)
    for line in f.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip().strip('"').strip("'")   # override (this IS the news account)
    # Catalyst selects the names — turn OFF both technical screens (trend + tight-OR width).
    os.environ["ORB_TREND_FILTER"] = "false"
    os.environ["ORB_TIGHT_OR"] = "false"
    # Isolation when sharing a machine with the baseline bot (the VM): separate heartbeat +
    # log file so we don't clobber its liveness/logs. And a later late-start cutoff because
    # we start after the morning scan (~9:43 ET) — give entries room to ~10:10 ET.
    os.environ.setdefault("ORB_HEARTBEAT_FILE", str(ROOT / "logs" / "heartbeat_news.json"))
    os.environ.setdefault("ORB_LOG_TAG", "news_")
    os.environ.setdefault("ORB_LATE_START_CUTOFF", "25")


def _todays_positive_picks(date_str: str) -> list[str]:
    f = ROOT / "experiments" / "news_edge" / "picks" / f"{date_str}.json"
    if not f.exists():
        return []
    rec = json.load(open(f))
    return [p["symbol"].upper() for p in rec.get("picks", []) if p.get("signal", 0) > 0]


def main() -> int:
    _load_news_env()
    passthrough = sys.argv[1:]

    # If caller didn't pin a watchlist, derive it from today's positive news picks.
    if not any(a == "--watchlist" or a.startswith("--watchlist=") for a in passthrough):
        if "--preflight-only" not in passthrough:
            date_str = datetime.now(ET).date().isoformat()
            syms = _todays_positive_picks(date_str)
            if not syms:
                print(f"News bot: no positive (+) picks for {date_str} — idle today, nothing to trade.")
                return 0
            passthrough = ["--watchlist", ",".join(syms)] + passthrough
            print(f"News bot: trading today's {len(syms)} positive picks: {','.join(syms)}")

    # Hand off to the validated engine. Import AFTER env is set (TREND_FILTER_ENABLED reads it).
    sys.path.insert(0, str(ROOT))
    import live.paper_orb as P  # noqa: E402
    sys.argv = ["news_orb"] + passthrough
    return P.main()


if __name__ == "__main__":
    sys.exit(main())
