"""Weekly short-interest cache builder for the squeeze signal (signal 4).

MUST run under .venv-openbb (yfinance lives there only; .venv has no yfinance).
Writes experiments/lottery/.short_interest_cache.json:
  {"updated": "...ISO...", "data": {SYM: {"short_pct_float": float, "days_to_cover": float,
                                          "as_of": "YYYY-MM-DD"}}}

squeeze score (computed in sources.squeeze_scores) = short_pct_float * days_to_cover.
yfinance short interest is 2-4 weeks stale by nature -- a STRUCTURAL feature, not a
morning signal. Scheduled Sundays.

Run:
    .venv-openbb/Scripts/python.exe experiments/lottery/update_short_interest.py
    .venv-openbb/Scripts/python.exe experiments/lottery/update_short_interest.py NVDA,GME
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
HERE = Path(__file__).resolve().parent
UNIVERSE_FILE = HERE / "universe.txt"
CACHE = HERE / ".short_interest_cache.json"


def load_universe() -> list[str]:
    if not UNIVERSE_FILE.exists():
        return []
    return [s.strip().upper() for s in UNIVERSE_FILE.read_text().splitlines() if s.strip()]


def fetch_one(yf, sym: str) -> dict | None:
    """Pull short%float + days-to-cover for one ticker. Returns None if unavailable."""
    try:
        info = yf.Ticker(sym).get_info()
    except Exception:
        try:
            info = yf.Ticker(sym).info
        except Exception:
            return None
    spf = info.get("shortPercentOfFloat")
    dtc = info.get("shortRatio")          # days-to-cover
    si_date = info.get("dateShortInterest")
    if spf is None and dtc is None:
        return None
    as_of = None
    if si_date:
        try:
            as_of = datetime.fromtimestamp(int(si_date)).strftime("%Y-%m-%d")
        except Exception:
            as_of = str(si_date)
    # shortPercentOfFloat is a fraction (0.12 = 12%); store as percent for the score
    return {"short_pct_float": round(spf * 100, 2) if spf is not None else None,
            "days_to_cover": round(float(dtc), 2) if dtc is not None else None,
            "as_of": as_of}


def main(argv) -> int:
    try:
        import yfinance as yf
    except ImportError:
        print("FATAL: yfinance not found. Run under .venv-openbb/Scripts/python.exe")
        return 2
    syms = ([s.strip().upper() for s in argv[1].split(",") if s.strip()]
            if len(argv) > 1 else load_universe())
    if not syms:
        print("no symbols (universe.txt missing?)")
        return 1
    print(f"fetching short interest for {len(syms)} symbols (yfinance, ~stale 2-4 wks) ...")
    data: dict = {}
    ok = 0
    for i, sym in enumerate(syms):
        r = fetch_one(yf, sym)
        if r is not None:
            data[sym] = r
            ok += 1
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(syms)}] {ok} with data")
    out = {"updated": datetime.now(ET).isoformat(timespec="seconds"), "data": data}
    json.dump(out, open(CACHE, "w"), indent=2)
    print(f"wrote {ok}/{len(syms)} short-interest rows -> {CACHE.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
