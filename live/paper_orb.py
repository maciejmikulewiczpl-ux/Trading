"""ORB live paper-trading runner.

Run on a US-equity trading day, ideally a few minutes before 9:30 ET.

  1. Pre-flight: load .env, verify account is healthy, verify market is open today,
     resume any in-flight state from existing orders.
  2. Wait for 9:30 ET. From 9:30 -> 9:30+OR_MINUTES, build the opening range
     (high/low of 1-min bars) per symbol.
  3. From 9:45 ET onward, on every newly-closed 1-min bar, check each not-yet-entered
     symbol: if bar close > OR high, submit a BRACKET market buy with take_profit at
     +target_R and stop_loss at OR low.
  4. At 15:55 ET, cancel any unfilled bracket legs and market-close remaining positions.

Usage (PowerShell):
    .\\.venv\\Scripts\\python.exe live\\paper_orb.py --dry-run
    .\\.venv\\Scripts\\python.exe live\\paper_orb.py        # live (paper) submission

Safety:
  - --dry-run logs every decision but submits no orders.
  - Per-trade risk capped at $100; per-position notional capped at $10,000.
  - Daily-loss circuit breaker: no new entries after -$500 of realized PnL.
  - Idempotent: client_order_id = "orb-YYYYMMDD-SYMBOL-entry". Restarts won't double-fire.
  - Refuses to run on a calendar day where the US equity market is closed.
"""
from __future__ import annotations

import argparse
import atexit
import logging
import math
import os
import sys
import time as time_mod
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
from alpaca.common.exceptions import APIError
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetCalendarRequest,
    GetOrdersRequest,
    MarketOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from strategies.orb import Params  # noqa: E402
from live.notify import notify  # noqa: E402

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

# === Hard-coded guardrails (intentionally not configurable via CLI) ===
WATCHLIST_DEFAULT = ["SPY", "QQQ", "AAPL", "NVDA", "TSLA"]
PARAMS = Params(or_minutes=15, target_r=2.0,
                risk_per_trade=100.0, max_position_dollars=10_000.0)
DAILY_LOSS_CAP = 500.0   # absolute dollars; halts NEW entries after this much realized loss
MIN_RISK_PER_SHARE = 0.05
MAX_RISK_PER_SHARE = 10.00
LATE_START_CUTOFF_MINUTES = 10  # if script starts > N min after OR window closes, halt new entries
POLL_SECONDS = 10
RTH_OPEN = time(9, 30)
RTH_CLOSE = time(16, 0)
EOD_FLAT_TIME = time(15, 55)

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

log = logging.getLogger("orb_paper")


@dataclass
class SymbolState:
    or_high: Optional[float] = None
    or_low: Optional[float] = None
    or_locked: bool = False
    entered: bool = False
    entry_order_id: Optional[str] = None
    last_processed_bar_ts: Optional[pd.Timestamp] = None
    # Trade details (set when we submit a bracket — purely for status display)
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    target_price: Optional[float] = None
    shares: Optional[int] = None
    last_close: Optional[float] = None
    reject_reason: Optional[str] = None
    exit_notified: bool = False


@dataclass
class RunState:
    states: dict[str, SymbolState] = field(default_factory=dict)
    halted: bool = False
    starting_equity: float = 0.0


# ---------- env / clients ----------
def load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def build_clients() -> tuple[TradingClient, StockHistoricalDataClient]:
    api_key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret:
        raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY missing from .env")
    return (
        TradingClient(api_key, secret, paper=True),
        StockHistoricalDataClient(api_key, secret),
    )


# ---------- logging ----------
def setup_logging(dry_run: bool) -> None:
    today = datetime.now(ET).strftime("%Y-%m-%d")
    log_path = LOG_DIR / f"orb_{today}{'_dryrun' if dry_run else ''}.log"
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S")
    log.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    log.info(f"Log file: {log_path}")


# ---------- pre-flight ----------
def is_today_a_trading_day(tc: TradingClient, today: date) -> bool:
    cal = tc.get_calendar(GetCalendarRequest(start=today, end=today))
    return len(cal) > 0 and cal[0].date == today


def preflight(tc: TradingClient, today: date) -> bool:
    if not is_today_a_trading_day(tc, today):
        log.error(f"Market is closed today ({today}). Nothing to do.")
        return False
    acct = tc.get_account()
    log.info(f"Account {acct.account_number} status={acct.status} equity=${acct.equity} cash=${acct.cash}")
    if acct.trading_blocked or acct.account_blocked or acct.transfers_blocked:
        log.error(f"Account is blocked (trading_blocked={acct.trading_blocked}, "
                  f"account_blocked={acct.account_blocked}). Aborting.")
        return False
    status_str = getattr(acct.status, "value", str(acct.status)).rsplit(".", 1)[-1].upper()
    if status_str != "ACTIVE":
        log.error(f"Account status is {acct.status}, expected ACTIVE. Aborting.")
        return False
    return True


def smoke_test(tc: TradingClient, dc: StockHistoricalDataClient, today: date) -> int:
    """Validate everything except wait-for-open and the trading-day gate.

    Designed to be run any time (including overnight or on holidays) to catch
    bugs that would only surface at the 06:15 PDT live fire. Returns 0 on
    pass, 1 on any hard fail. Notifications and status-UI import are soft
    checks that warn but don't fail the run.
    """
    ok = True
    log.info("=" * 56)
    log.info("Smoke test: account / data / notifications / status_ui")
    log.info("=" * 56)

    acct = None
    try:
        acct = tc.get_account()
        log.info(f"Account {acct.account_number} status={acct.status} "
                 f"equity=${acct.equity} cash=${acct.cash}")
        if acct.trading_blocked or acct.account_blocked or acct.transfers_blocked:
            log.error("FAIL  account is blocked")
            ok = False
        status_str = getattr(acct.status, "value", str(acct.status)).rsplit(".", 1)[-1].upper()
        if status_str != "ACTIVE":
            log.error(f"FAIL  account status is {acct.status}, expected ACTIVE")
            ok = False
        else:
            log.info("PASS  account ACTIVE, no blocks")
    except Exception as e:
        log.error(f"FAIL  account check raised: {e}")
        ok = False

    try:
        is_open = is_today_a_trading_day(tc, today)
        if is_open:
            log.info(f"PASS  calendar: {today} is a trading day")
        else:
            log.info(f"INFO  {today} is not a trading day (live runner would exit 0)")
    except Exception as e:
        log.error(f"FAIL  calendar lookup raised: {e}")
        ok = False

    try:
        bars = fetch_today_bars(dc, ["SPY"], today)
        log.info(f"PASS  data feed reachable (SPY bars returned: {len(bars)})")
    except Exception as e:
        log.error(f"FAIL  bar fetch raised: {e}")
        ok = False

    try:
        eq = f"${acct.equity}" if acct is not None else "?"
        sent = notify(
            f"Smoke test push from ORB runner.\n"
            f"Account {eq}, today {today.isoformat()}.\n"
            "If you see this, notifications are wired up.",
            title="ORB smoke test",
            tags=["test_tube"],
        )
        if sent:
            log.info("PASS  ntfy push accepted")
        else:
            log.warning("WARN  ntfy not configured (NTFY_TOPIC unset?) — push not sent")
    except Exception as e:
        log.warning(f"WARN  ntfy push raised: {e}")

    try:
        from live.status_ui import _AVAILABLE  # noqa: F401
        from live import status_ui  # noqa: F401
        log.info(f"PASS  status_ui imports (pystray available: {_AVAILABLE})")
    except Exception as e:
        log.warning(f"WARN  status_ui import failed: {e}")

    log.info("=" * 56)
    log.info(f"Smoke test result: {'PASS' if ok else 'FAIL'}")
    log.info("=" * 56)
    return 0 if ok else 1


def sync_existing_orders_today(tc: TradingClient, run: RunState, today: date) -> None:
    """Find any orb-* orders already submitted today and mark those symbols as entered."""
    today_start = datetime.combine(today, time(0, 0, tzinfo=ET))
    req = GetOrdersRequest(
        status=QueryOrderStatus.ALL,
        after=today_start.astimezone(UTC),
        limit=500,
    )
    try:
        orders = tc.get_orders(filter=req)
    except TypeError:
        orders = tc.get_orders(req)  # SDK signature variants
    coid_prefix = f"orb-{today.strftime('%Y%m%d')}-"
    resumed = 0
    for o in orders:
        if not getattr(o, "client_order_id", None):
            continue
        if not o.client_order_id.startswith(coid_prefix):
            continue
        sym = o.symbol
        if sym in run.states:
            run.states[sym].entered = True
            run.states[sym].entry_order_id = str(o.id)
            resumed += 1
    if resumed:
        log.info(f"Resumed {resumed} entry state(s) from existing orders today.")


# ---------- bars ----------
def fetch_today_bars(dc: StockHistoricalDataClient, symbols: list[str], today: date) -> pd.DataFrame:
    start_et = datetime.combine(today, RTH_OPEN, tzinfo=ET)
    end_et = datetime.now(ET)
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Minute,
        start=start_et.astimezone(UTC),
        end=end_et.astimezone(UTC),
        feed=DataFeed.IEX,
    )
    bars = dc.get_stock_bars(req).df
    if bars.empty:
        return bars
    sym = bars.index.get_level_values(0)
    ts = bars.index.get_level_values(1).tz_convert(ET)
    bars = bars.copy()
    bars.index = pd.MultiIndex.from_arrays([sym, ts], names=["symbol", "timestamp"])
    # Keep only bars whose minute has fully closed (start_ts + 60s <= now).
    cutoff = pd.Timestamp(datetime.now(ET)) - pd.Timedelta(seconds=60)
    bars = bars[bars.index.get_level_values(1) <= cutoff]
    return bars


# ---------- sizing / guardrails ----------
def size_position(entry: float, stop: float, equity: float) -> tuple[int, str]:
    """Returns (shares, reason). shares==0 means reject; reason explains."""
    risk_per_share = entry - stop
    if risk_per_share < MIN_RISK_PER_SHARE:
        return 0, f"risk_per_share ${risk_per_share:.4f} < min ${MIN_RISK_PER_SHARE}"
    if risk_per_share > MAX_RISK_PER_SHARE:
        return 0, f"risk_per_share ${risk_per_share:.2f} > max ${MAX_RISK_PER_SHARE}"
    shares_by_risk = math.floor(PARAMS.risk_per_trade / risk_per_share)
    cap_dollars = equity * PARAMS.max_position_pct
    if PARAMS.max_position_dollars is not None:
        cap_dollars = min(cap_dollars, PARAMS.max_position_dollars)
    shares_by_cap = math.floor(cap_dollars / entry)
    shares = max(0, min(shares_by_risk, shares_by_cap))
    if shares <= 0:
        return 0, f"sizing produced 0 shares (by_risk={shares_by_risk}, by_cap={shares_by_cap})"
    return shares, "ok"


def realized_pnl_today(tc: TradingClient, today: date) -> float:
    """Sum of today's closed trade activities. Used by the daily-loss circuit breaker."""
    # Best-effort: use account portfolio_history or rely on equity delta. Simpler: equity - starting.
    try:
        acct = tc.get_account()
        return float(acct.equity) - float(acct.last_equity)
    except Exception as e:
        log.warning(f"Could not compute realized PnL today: {e}; assuming 0")
        return 0.0


# ---------- order submission ----------
def submit_bracket(tc: TradingClient, sym: str, qty: int, entry_est: float,
                   stop: float, target: float,
                   today: date, dry_run: bool) -> Optional[str]:
    coid = f"orb-{today.strftime('%Y%m%d')}-{sym}-entry"
    req = MarketOrderRequest(
        symbol=sym,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
        order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(limit_price=round(target, 2)),
        stop_loss=StopLossRequest(stop_price=round(stop, 2)),
        client_order_id=coid,
    )
    if dry_run:
        log.info(f"[DRY-RUN] WOULD SUBMIT bracket BUY {qty} {sym} "
                 f"target=${target:.2f} stop=${stop:.2f} coid={coid}")
        return f"dryrun-{coid}"
    try:
        o = tc.submit_order(req)
        log.info(f"SUBMITTED {sym}: id={o.id} status={o.status} coid={coid}")
        try:
            notify(
                f"BUY {qty} {sym} @ ~${entry_est:.2f}  "
                f"stop ${stop:.2f}  target ${target:.2f}  "
                f"risk ${(entry_est - stop) * qty:.0f}",
                title=f"ORB entry: {sym}",
                tags=["chart_with_upwards_trend"],
            )
        except Exception as e:
            log.warning(f"Entry notification failed: {e}")
        return str(o.id)
    except APIError as e:
        log.error(f"{sym} submission FAILED: {e}")
        return None


def _status_str(s) -> str:
    """Normalize an alpaca-py enum or string to its bare upper-case name (e.g. 'FILLED')."""
    return getattr(s, "value", str(s)).rsplit(".", 1)[-1].upper()


def poll_exits(tc: TradingClient, run: RunState) -> None:
    """For each entered symbol, check if a bracket leg has filled and notify once.

    Pure read-only: never modifies orders/positions, only updates exit_notified.
    Swallows all errors — notifications are decoration, not load-bearing.
    """
    for sym, st in run.states.items():
        if not (st.entered and st.entry_order_id) or st.exit_notified or st.reject_reason:
            continue
        if str(st.entry_order_id).startswith("dryrun-"):
            continue
        try:
            parent = tc.get_order_by_id(st.entry_order_id)
        except Exception as e:
            log.debug(f"poll_exits: get_order_by_id({sym}) failed: {e}")
            continue
        entry_fill = getattr(parent, "filled_avg_price", None)
        if entry_fill is None:
            continue  # entry hasn't filled yet — wait
        for leg in (getattr(parent, "legs", None) or []):
            if _status_str(getattr(leg, "status", "")) != "FILLED":
                continue
            try:
                exit_fill = float(leg.filled_avg_price)
                entry_px = float(entry_fill)
                qty = int(getattr(leg, "filled_qty", None) or st.shares or 0)
                # Take-profit leg has a limit_price; stop-loss has stop_price.
                is_target = getattr(leg, "limit_price", None) is not None
                reason = "target" if is_target else "stop"
                pnl = (exit_fill - entry_px) * qty
                notify(
                    f"{sym} {reason} @ ${exit_fill:.2f}  "
                    f"(entry ${entry_px:.2f}, qty {qty})  "
                    f"PnL ${pnl:+,.0f}",
                    title=f"ORB exit: {sym} {reason}",
                    tags=["dart"] if is_target else ["octagonal_sign"],
                )
            except Exception as e:
                log.warning(f"poll_exits: notify failed for {sym}: {e}")
            st.exit_notified = True
            break


CLOSE_MAX_ATTEMPTS = 4
CLOSE_RETRY_DELAY_SEC = 2.0
CANCEL_PROPAGATION_DELAY_SEC = 1.5


def _close_position_with_retry(tc: TradingClient, sym: str) -> bool:
    """Call close_position with retries on the cancel-then-close race.

    When EOD-flatten cancels bracket legs and immediately tries to close the
    position, Alpaca may briefly report `insufficient qty available` because
    the cancellations haven't propagated. Retry with a short delay until
    Alpaca releases the held qty. Returns True on success.
    """
    for attempt in range(1, CLOSE_MAX_ATTEMPTS + 1):
        try:
            tc.close_position(sym)
            if attempt > 1:
                log.info(f"Close {sym} succeeded on attempt {attempt}")
            return True
        except APIError as e:
            msg = str(e).lower()
            is_race = "insufficient qty" in msg or '"available":"0"' in msg
            if is_race and attempt < CLOSE_MAX_ATTEMPTS:
                log.info(f"Close {sym} attempt {attempt}: held by pending cancels, "
                         f"retrying in {CLOSE_RETRY_DELAY_SEC}s")
                time_mod.sleep(CLOSE_RETRY_DELAY_SEC)
                continue
            log.warning(f"Close {sym} failed on attempt {attempt}: {e}")
            return False
        except Exception as e:
            log.warning(f"Close {sym} unexpected error on attempt {attempt}: {e}")
            return False
    return False


def flatten_all(tc: TradingClient, watchlist: list[str], dry_run: bool,
                run: Optional[RunState] = None) -> None:
    """Cancel any open orders for our symbols, then market-close any open positions in them.

    If `run` is provided, push an ntfy exit notification per position closed
    (using the position's unrealized PnL just before close).

    Handles the cancel-then-close race: after cancelling bracket legs, Alpaca
    needs a moment to release the held qty. We sleep briefly, then use a
    retry-with-backoff close to recover if `insufficient qty` is still seen.
    """
    try:
        open_orders = tc.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500))
    except TypeError:
        open_orders = tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500))
    cancelled_any = False
    for o in open_orders:
        if o.symbol not in watchlist:
            continue
        log.info(f"EOD cancel: order {o.id} ({o.symbol} {o.side} qty={o.qty})")
        if not dry_run:
            try:
                tc.cancel_order_by_id(o.id)
                cancelled_any = True
            except APIError as e:
                log.warning(f"Cancel {o.id} failed: {e}")

    # Give Alpaca a moment to release qty held by cancelled bracket legs before
    # we try to close the underlying positions.
    if cancelled_any and not dry_run:
        log.info(f"Waiting {CANCEL_PROPAGATION_DELAY_SEC}s for cancellations to propagate.")
        time_mod.sleep(CANCEL_PROPAGATION_DELAY_SEC)

    try:
        positions = tc.get_all_positions()
    except APIError as e:
        log.warning(f"get_all_positions failed: {e}")
        return
    for p in positions:
        if p.symbol not in watchlist:
            continue
        log.info(f"EOD close: position {p.symbol} qty={p.qty} mv=${p.market_value}")
        if run is not None:
            st = run.states.get(p.symbol)
            if st is not None and not st.exit_notified:
                try:
                    pnl = float(getattr(p, "unrealized_pl", 0) or 0)
                    px  = float(getattr(p, "current_price", 0) or 0)
                    notify(
                        f"{p.symbol} EOD-flat @ ${px:.2f}  qty {p.qty}  "
                        f"PnL ${pnl:+,.0f}",
                        title=f"ORB exit: {p.symbol} EOD",
                        tags=["checkered_flag"],
                    )
                except Exception as e:
                    log.warning(f"EOD exit notification failed for {p.symbol}: {e}")
                st.exit_notified = True
        if not dry_run:
            _close_position_with_retry(tc, p.symbol)


# ---------- Windows sleep prevention ----------
# Windows API constants for SetThreadExecutionState.
_ES_CONTINUOUS       = 0x80000000
_ES_SYSTEM_REQUIRED  = 0x00000001


def prevent_system_sleep() -> None:
    """Ask Windows not to enter sleep while this process is alive.

    Sets ES_CONTINUOUS | ES_SYSTEM_REQUIRED on the calling thread. The flag is
    process-scoped and Windows clears it automatically when the process exits.
    We also register `allow_system_sleep` with atexit as a belt-and-braces release.
    No-op on non-Windows platforms.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        prev = ctypes.windll.kernel32.SetThreadExecutionState(
            _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED
        )
        if prev == 0:
            log.warning("SetThreadExecutionState returned 0 — could not lock against sleep.")
        else:
            log.info("Sleep-prevention ACTIVE: system will not auto-sleep until this script exits.")
            atexit.register(allow_system_sleep)
    except Exception as e:
        log.warning(f"Could not request sleep prevention: {e}. System sleep may interrupt the session.")


def allow_system_sleep() -> None:
    """Release the sleep lock. Safe to call even if no lock was set."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
        log.info("Sleep-prevention released.")
    except Exception:
        pass


# ---------- time helpers ----------
def combine_et(today: date, t: time) -> datetime:
    return datetime.combine(today, t, tzinfo=ET)


def wait_until(target: datetime) -> None:
    while True:
        now = datetime.now(ET)
        delta = (target - now).total_seconds()
        if delta <= 0:
            return
        log.info(f"Waiting {int(delta)}s until {target.strftime('%H:%M:%S %Z')}")
        time_mod.sleep(min(delta, 30))


# ---------- main loop ----------
def _phase_for(now: datetime, open_dt: datetime, or_end_dt: datetime, eod_dt: datetime) -> str:
    if now < open_dt:
        return "pre-market (waiting for 9:30 ET)"
    if now < or_end_dt:
        return "building opening range"
    if now < eod_dt:
        return "hunting for breakouts"
    return "EOD flatten / done"


def _build_snapshot(tc: TradingClient, run: "RunState", watchlist: list[str],
                    open_dt: datetime, or_end_dt: datetime, eod_dt: datetime) -> dict:
    now = datetime.now(ET)
    snap = {
        "phase": _phase_for(now, open_dt, or_end_dt, eod_dt),
        "halted": run.halted,
        "last_update": now.strftime("%H:%M:%S %Z"),
        "symbols": {},
    }
    try:
        acct = tc.get_account()
        snap["equity"] = float(acct.equity)
        snap["day_pnl"] = float(acct.equity) - float(acct.last_equity)
    except Exception:
        pass
    for sym in watchlist:
        st = run.states[sym]
        if st.entered and st.entry_order_id and not st.reject_reason:
            status = "ENTERED (bracket live on Alpaca)"
        elif st.entered and st.reject_reason:
            status = f"skipped ({st.reject_reason[:18]})"
        elif st.or_locked:
            status = "watching for breakout"
        elif now < open_dt:
            status = "(market not open yet)"
        else:
            status = "building OR..."
        snap["symbols"][sym] = {
            "or_high": st.or_high, "or_low": st.or_low, "or_locked": st.or_locked,
            "entered": st.entered, "status": status,
            "entry_price": st.entry_price, "stop_price": st.stop_price,
            "target_price": st.target_price, "shares": st.shares,
        }
    return snap


def _send_eod_notification(tc: TradingClient, run: "RunState", watchlist: list[str], today: date) -> None:
    """Best-effort end-of-session summary push. Never raises."""
    try:
        try:
            final_eq = float(tc.get_account().equity)
        except Exception:
            final_eq = run.starting_equity
        pnl = final_eq - run.starting_equity
        submitted = [s for s in watchlist if run.states[s].entered and run.states[s].entry_order_id]
        rejected = [s for s in watchlist if run.states[s].reject_reason]
        lines = [
            f"PnL: ${pnl:+,.2f}  (equity ${final_eq:,.2f})",
            f"Entries submitted: {len(submitted)}" + (f" — {', '.join(submitted)}" if submitted else ""),
        ]
        if rejected:
            lines.append(f"Rejected: {', '.join(rejected)}")
        if run.halted:
            lines.append("Note: NEW entries were halted today (loss cap or late start).")
        notify(
            "\n".join(lines),
            title=f"ORB done ({today.isoformat()})",
            tags=["checkered_flag"],
        )
    except Exception as e:
        log.warning(f"EOD notification failed: {e}")


def run_session(tc: TradingClient, dc: StockHistoricalDataClient,
                watchlist: list[str], today: date, dry_run: bool,
                skip_wait: bool = False) -> None:
    prevent_system_sleep()
    run = RunState(states={s: SymbolState() for s in watchlist})
    acct = tc.get_account()
    run.starting_equity = float(acct.equity)

    sync_existing_orders_today(tc, run, today)

    open_dt = combine_et(today, RTH_OPEN)
    or_end_dt = open_dt + timedelta(minutes=PARAMS.or_minutes)
    eod_dt = combine_et(today, EOD_FLAT_TIME)
    close_dt = combine_et(today, RTH_CLOSE)

    log.info(f"Today: open={open_dt.strftime('%H:%M %Z')}  "
             f"OR ends={or_end_dt.strftime('%H:%M')}  "
             f"EOD flat={eod_dt.strftime('%H:%M')}  "
             f"close={close_dt.strftime('%H:%M')}")

    # Phone notification: session is up. Sent once per run, before waiting for open.
    notify(
        f"Account ${run.starting_equity:,.0f}. Waiting for 09:30 ET market open. "
        f"Watchlist: {', '.join(watchlist)}.",
        title=f"ORB started ({today.isoformat()})",
        tags=["green_circle"],
    )

    # Optional tray icon + status window. Failures here never stop the trader.
    ui = None
    try:
        from live.status_ui import StatusController
        ui = StatusController(get_status=lambda: _build_snapshot(tc, run, watchlist, open_dt, or_end_dt, eod_dt))
        if not ui.start():
            ui = None
        else:
            atexit.register(ui.stop)
    except Exception as e:
        log.info(f"Status UI not loaded ({e}). Trader continues without it.")
        ui = None

    if not skip_wait:
        wait_until(open_dt)

    # Late-start guardrail: if we missed the OR window by more than the cutoff, do not
    # take any new entries today. Stale OR data + a multi-minute-old breakout = chase risk.
    started_at = datetime.now(ET)
    late_threshold = or_end_dt + timedelta(minutes=LATE_START_CUTOFF_MINUTES)
    if started_at > late_threshold:
        late_by_min = int((started_at - or_end_dt).total_seconds() / 60)
        log.warning(f"Started {late_by_min} min after OR window closed (threshold: "
                    f"{LATE_START_CUTOFF_MINUTES} min). Halting NEW entries for today. "
                    f"Will still EOD-flatten any positions that already exist.")
        run.halted = True
        if ui is not None:
            ui.set_state("warning")

    while True:
        now = datetime.now(ET)
        if now >= eod_dt:
            log.info("EOD flat time reached. Flattening.")
            flatten_all(tc, watchlist, dry_run, run=run)
            _send_eod_notification(tc, run, watchlist, today)
            return
        if now >= close_dt:
            log.info("Past market close. Exiting.")
            _send_eod_notification(tc, run, watchlist, today)
            return

        # Daily-loss circuit breaker (halts NEW entries only)
        if not run.halted:
            pnl = realized_pnl_today(tc, today)
            if pnl <= -DAILY_LOSS_CAP:
                log.warning(f"Daily loss cap hit: realized PnL ${pnl:+,.2f}. Halting new entries.")
                run.halted = True
                if ui is not None:
                    ui.set_state("halted")

        # Push per-trade exit notifications when target/stop legs fill (best-effort).
        try:
            poll_exits(tc, run)
        except Exception as e:
            log.warning(f"poll_exits cycle failed: {e}")

        try:
            bars = fetch_today_bars(dc, watchlist, today)
        except Exception as e:
            log.warning(f"Bar fetch failed: {e}; retrying in {POLL_SECONDS}s")
            time_mod.sleep(POLL_SECONDS)
            continue

        symbols_in_data = set(bars.index.get_level_values(0).unique()) if not bars.empty else set()

        for sym in watchlist:
            state = run.states[sym]
            if sym not in symbols_in_data:
                continue
            sym_bars = bars.xs(sym, level=0)

            # OR construction (only while OR window is open or until first lock)
            if not state.or_locked:
                or_bars = sym_bars[sym_bars.index < or_end_dt]
                if not or_bars.empty:
                    state.or_high = float(or_bars["high"].max())
                    state.or_low = float(or_bars["low"].min())
                if now >= or_end_dt and state.or_high is not None and state.or_low is not None:
                    state.or_locked = True
                    log.info(f"{sym} OR locked: high=${state.or_high:.2f} low=${state.or_low:.2f}")
                continue  # nothing more for this symbol until OR locks

            if state.entered or run.halted:
                continue

            post_or = sym_bars[sym_bars.index >= or_end_dt]
            if post_or.empty:
                continue
            last_bar_ts = post_or.index[-1]
            if state.last_processed_bar_ts is not None and last_bar_ts <= state.last_processed_bar_ts:
                continue
            state.last_processed_bar_ts = last_bar_ts
            last_close = float(post_or.iloc[-1]["close"])

            if last_close <= state.or_high:
                continue

            # Breakout. Use last_close as entry estimate (true fill will be next-bar market).
            entry_estimate = last_close
            stop = state.or_low
            target = entry_estimate + PARAMS.target_r * (entry_estimate - stop)
            equity = float(tc.get_account().equity)
            qty, reason = size_position(entry_estimate, stop, equity)

            log.info(f"BREAKOUT {sym}: close=${last_close:.2f} > OR_high=${state.or_high:.2f}  "
                     f"-> entry~=${entry_estimate:.2f} stop=${stop:.2f} target=${target:.2f}  "
                     f"qty={qty} (sizing: {reason})")

            if qty == 0:
                log.warning(f"{sym} REJECTED: {reason}")
                state.entered = True  # don't keep retrying a rejected setup
                state.reject_reason = reason
                continue

            oid = submit_bracket(tc, sym, qty, entry_estimate, stop, target, today, dry_run)
            state.entered = True
            state.entry_order_id = oid
            state.entry_price = entry_estimate
            state.stop_price = stop
            state.target_price = target
            state.shares = qty
            state.last_close = last_close

        time_mod.sleep(POLL_SECONDS)


def main() -> int:
    ap = argparse.ArgumentParser(description="ORB live paper runner")
    ap.add_argument("--dry-run", action="store_true",
                    help="Log every decision but submit no orders.")
    ap.add_argument("--watchlist", default=",".join(WATCHLIST_DEFAULT),
                    help=f"Comma-separated symbols (default: {','.join(WATCHLIST_DEFAULT)})")
    ap.add_argument("--ignore-clock", action="store_true",
                    help="TESTING: skip the 'market open today' check and the wait-for-9:30 step. "
                         "Use ONLY with --dry-run to exercise the loop on a closed day.")
    ap.add_argument("--preflight-only", action="store_true",
                    help="Run smoke test (account, data feed, ntfy, imports) and exit. "
                         "Safe to run any time, including overnight or on holidays.")
    args = ap.parse_args()
    if args.ignore_clock and not args.dry_run:
        print("Refusing to run with --ignore-clock without --dry-run.", file=sys.stderr)
        return 2

    setup_logging(args.dry_run)
    watchlist = [s.strip().upper() for s in args.watchlist.split(",") if s.strip()]
    log.info(f"Starting ORB paper runner (dry_run={args.dry_run})")
    log.info(f"Watchlist: {watchlist}")
    log.info(f"Params: or_min={PARAMS.or_minutes} target_r={PARAMS.target_r} "
             f"risk=${PARAMS.risk_per_trade} max_pos=${PARAMS.max_position_dollars} "
             f"loss_cap=${DAILY_LOSS_CAP}")

    load_env()
    try:
        tc, dc = build_clients()
    except RuntimeError as e:
        log.error(str(e))
        return 1

    today = datetime.now(ET).date()
    if args.preflight_only:
        return smoke_test(tc, dc, today)

    if args.ignore_clock:
        log.warning("--ignore-clock set: skipping market-open pre-flight (TESTING MODE)")
        acct = tc.get_account()
        log.info(f"Account {acct.account_number} status={acct.status} equity=${acct.equity}")
    else:
        if not preflight(tc, today):
            return 0  # not an error — market is just closed

    try:
        run_session(tc, dc, watchlist, today, args.dry_run, skip_wait=args.ignore_clock)
    except KeyboardInterrupt:
        log.warning("Interrupted by user (Ctrl-C). Flattening before exit.")
        flatten_all(tc, watchlist, args.dry_run)
        return 130
    except Exception as e:
        log.exception(f"Unhandled error in run_session: {e}. Flattening before exit.")
        flatten_all(tc, watchlist, args.dry_run)
        try:
            notify(
                f"Script crashed: {type(e).__name__}: {str(e)[:200]}\n"
                f"Check logs at C:\\Users\\macie\\VSC\\Trading\\logs\\",
                title=f"ORB CRASHED ({today.isoformat()})",
                priority=5,
                tags=["rotating_light"],
            )
        except Exception:
            pass
        return 1

    log.info("Session complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
