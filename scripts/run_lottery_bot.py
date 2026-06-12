"""Lottery paper bot (Track C) — the BOLD day-1 forward-test trader.

At ~09:45 ET it reads today's lottery board (experiments/lottery/picks/<ET-date>.json),
takes the TOP-3 by combined_score (the hype basket), and for each:
  - submits a $500-notional market BUY (fractional ok),
  - attaches a 10% NATIVE Alpaca trailing stop on fill,
  - records the entry so a later run can time-stop close at T+3 if still open.

Runs on the REPURPOSED dual-momentum paper account (keys in .env.lottery). Isolation
mirrors scripts/run_news_orb.py: loads its own .env.lottery (override), writes its own
heartbeat (logs/heartbeat_lottery.json), uses log tag "lottery_". Bot PnL is COLOR, not
the verdict — Track B's hit-rate stats are the verdict.

State for the T+3 time-stop lives in logs/lottery_positions.json (entry date per symbol).

Run:
    .venv/Scripts/python.exe scripts/run_lottery_bot.py --dry-run      # print intended orders
    .venv/Scripts/python.exe scripts/run_lottery_bot.py                # live paper buys
    .venv/Scripts/python.exe scripts/run_lottery_bot.py --time-stops-only  # only close T+3
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
ET = ZoneInfo("America/New_York")

NOTIONAL = 500.0            # $ per pick
TRAIL_PCT = 10.0           # native trailing stop %
TOP_N = 3                  # top-3 by combined_score
TIME_STOP_DAYS = 3         # close at T+3 trading-ish days if still open
STATE_FILE = ROOT / "logs" / "lottery_positions.json"

log = logging.getLogger("lottery_bot")


def _load_lottery_env() -> None:
    f = ROOT / ".env.lottery"
    if not f.exists():
        print("FATAL: .env.lottery not found (repurposed dual-mom account keys). Aborting.",
              file=sys.stderr)
        sys.exit(2)
    for line in f.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip().strip('"').strip("'")   # override: THIS is the account


def _todays_picks(date_str: str) -> list[dict]:
    f = ROOT / "experiments" / "lottery" / "picks" / f"{date_str}.json"
    if not f.exists():
        return []
    rec = json.load(open(f))
    picks = [p for p in rec.get("picks", []) if p.get("combined_score") is not None]
    picks.sort(key=lambda x: -x["combined_score"])
    return picks[:TOP_N]


def _load_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.load(open(STATE_FILE))
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(exist_ok=True)
    json.dump(state, open(STATE_FILE, "w"), indent=2)


def _trading_client():
    from alpaca.trading.client import TradingClient
    return TradingClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"],
                         paper=True)


def _trading_days_since(entry_iso: str) -> int:
    """Rough trading-day count since entry (weekdays only; holidays ignored — close on
    the safe side). Good enough for a T+3 time stop."""
    entry = datetime.fromisoformat(entry_iso).date()
    today = datetime.now(ET).date()
    n = 0
    d = entry
    while d < today:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n += 1
    return n


def run_time_stops(tc, dry_run: bool) -> int:
    """Close any tracked position that's reached T+TIME_STOP_DAYS."""
    from alpaca.trading.requests import GetOrdersRequest
    state = _load_state()
    if not state:
        print("time-stops: no tracked positions.")
        return 0
    try:
        positions = {p.symbol: p for p in tc.get_all_positions()}
    except Exception as e:
        print(f"time-stops: get_all_positions failed: {e}")
        positions = {}
    closed = 0
    for sym in list(state.keys()):
        info = state[sym]
        age = _trading_days_since(info["entry_date"])
        if sym not in positions:
            # position gone (trailing stop hit or manually closed) -> drop tracking
            print(f"time-stops: {sym} no longer held (age {age}) -> untracked.")
            del state[sym]
            continue
        if age >= TIME_STOP_DAYS:
            if dry_run:
                print(f"[DRY-RUN] WOULD time-stop close {sym} (age {age} >= {TIME_STOP_DAYS})")
            else:
                try:
                    tc.close_position(sym)          # cancels related orders + flattens
                    print(f"time-stop CLOSED {sym} (age {age}).")
                    del state[sym]
                    closed += 1
                except Exception as e:
                    print(f"time-stop close {sym} FAILED: {e}")
        else:
            print(f"time-stops: {sym} age {age} < {TIME_STOP_DAYS}, hold.")
    if not dry_run:
        _save_state(state)
    return closed


def run_entries(tc, dry_run: bool) -> int:
    from alpaca.trading.requests import MarketOrderRequest, TrailingStopOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    date_str = datetime.now(ET).date().isoformat()
    picks = _todays_picks(date_str)
    if not picks:
        print(f"lottery bot: no board picks for {date_str} -- idle, nothing to buy.")
        return 0
    print(f"lottery bot: top-{len(picks)} by combined_score for {date_str}: "
          + ", ".join(f"{p['symbol']}(cs={p['combined_score']:.2f})" for p in picks))

    state = _load_state()
    try:
        held = {p.symbol for p in tc.get_all_positions()}
    except Exception:
        held = set()

    placed = 0
    for p in picks:
        sym = p["symbol"]
        if sym in held or sym in state:
            print(f"  {sym}: already held/tracked, skip.")
            continue
        if dry_run:
            print(f"  [DRY-RUN] WOULD BUY {sym} ${NOTIONAL:.0f} notional (market) "
                  f"+ {TRAIL_PCT:.0f}% trailing stop on fill")
            continue
        try:
            order = tc.submit_order(MarketOrderRequest(
                symbol=sym, notional=NOTIONAL, side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY))
            print(f"  BUY submitted {sym} ${NOTIONAL:.0f} notional (order {order.id}).")
            state[sym] = {"entry_date": date_str, "notional": NOTIONAL,
                          "combined_score": p["combined_score"], "trail_attached": False}
            placed += 1
        except Exception as e:
            print(f"  {sym} BUY FAILED: {e}")
    if not dry_run:
        _save_state(state)
    return placed


def attach_trailing_stops(tc, dry_run: bool) -> int:
    """For any tracked position that's filled but has no trailing stop yet, attach a
    10% native trailing stop. Idempotent (skips ones already attached)."""
    from alpaca.trading.requests import TrailingStopOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    if dry_run:
        return 0
    state = _load_state()
    try:
        positions = {p.symbol: p for p in tc.get_all_positions()}
    except Exception as e:
        print(f"trailing: get_all_positions failed: {e}")
        return 0
    attached = 0
    for sym, info in state.items():
        if info.get("trail_attached"):
            continue
        pos = positions.get(sym)
        if pos is None:
            continue
        try:
            qty = abs(float(pos.qty))
            if qty <= 0:
                continue
            to = tc.submit_order(TrailingStopOrderRequest(
                symbol=sym, qty=qty, side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY, trail_percent=TRAIL_PCT))
            info["trail_attached"] = True
            info["trail_order_id"] = str(to.id)
            print(f"  trailing stop attached {sym}: {TRAIL_PCT:.0f}% (order {to.id}).")
            attached += 1
        except Exception as e:
            print(f"  {sym} trailing attach FAILED: {e}")
    _save_state(state)
    return attached


def _emit_heartbeat(phase: str) -> None:
    try:
        sys.path.insert(0, str(ROOT))
        from live import heartbeat
        hb = ROOT / "logs" / "heartbeat_lottery.json"
        heartbeat.write(datetime.now().timestamp() + 86400, path=hb, phase=phase,
                        tag="lottery_")
    except Exception:
        pass


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s lottery_ %(message)s")
    _load_lottery_env()
    args = sys.argv[1:]
    dry_run = "--dry-run" in args
    time_stops_only = "--time-stops-only" in args

    tc = _trading_client()
    try:
        acct = tc.get_account()
        print(f"lottery bot account: equity ${float(acct.equity):,.2f}  "
              f"cash ${float(acct.cash):,.2f}  (paper)")
    except Exception as e:
        print(f"account fetch failed: {e}")

    _emit_heartbeat("start")

    # 1. time-stops first (close anything at T+3)
    run_time_stops(tc, dry_run)

    if not time_stops_only:
        # 2. new entries (top-3 by combined_score)
        run_entries(tc, dry_run)
        # 3. attach trailing stops to newly-filled positions (give fills a moment in live)
        if not dry_run:
            import time as _t
            _t.sleep(3)
            attach_trailing_stops(tc, dry_run)

    _emit_heartbeat("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
