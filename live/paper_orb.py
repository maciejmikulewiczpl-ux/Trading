"""ORB live paper-trading runner.

Run on a US-equity trading day, ideally a few minutes before 9:30 ET.

  1. Pre-flight: load .env, verify account is healthy, verify market is open today,
     resume any in-flight state from existing orders.
  2. Wait for 9:30 ET. From 9:30 -> 9:30+OR_MINUTES, build the opening range
     (high/low of 1-min bars) per symbol.
  3. From 9:45 ET onward, on every newly-closed 1-min bar, check each not-yet-entered
     symbol:
       - LONG: bar close > OR high -> BRACKET market BUY, take_profit +target_R, stop OR low.
       - SHORT: bar close < OR low -> BRACKET market SELL, take_profit -target_R, stop OR high.
         Shorts are regime-gated (only when SPY is in a confirmed downtrend) and limited
         to SHORT_SYMBOLS (index/large-cap; TSLA excluded). See compute_short_regime.
  4. At 15:55 ET, cancel any unfilled bracket legs and market-close remaining positions.

Usage:
    # Windows (PowerShell)
    .\\.venv\\Scripts\\python.exe live\\paper_orb.py --dry-run
    .\\.venv\\Scripts\\python.exe live\\paper_orb.py        # live (paper) submission
    # Linux (bash)
    ./.venv/bin/python live/paper_orb.py --dry-run
    ./.venv/bin/python live/paper_orb.py

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
from alpaca.data.enums import Adjustment, DataFeed
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
    StopOrderRequest,
    TakeProfitRequest,
    TrailingStopOrderRequest,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from strategies.orb import Params  # noqa: E402
from live.notify import notify  # noqa: E402
from live import config as orb_config  # noqa: E402
from live import heartbeat  # noqa: E402

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

# === Tunable parameters ===
# Loaded from live/orb_config.json if present, else the validated defaults in
# live/config.py (see each Setting.help there for the basis). Edit them via the
# GUI: `python live/config_ui.py`. Changes apply on the NEXT session (a running
# bot keeps the params it started with). When no config file exists these equal
# the project's backtested defaults, so behavior is unchanged.
_CFG = orb_config.load_config()
WATCHLIST_DEFAULT = list(_CFG["watchlist"])
PARAMS = orb_config.build_params(_CFG)
DAILY_LOSS_CAP = float(_CFG["daily_loss_cap"])   # halts NEW entries after this much realized loss
MIN_RISK_PER_SHARE = float(_CFG["min_risk_per_share"])
MAX_RISK_PER_SHARE = float(_CFG["max_risk_per_share"])
# Max simultaneously-open positions (broad-watchlist guardrail). 0 = unlimited.
_mcp = int(_CFG.get("max_concurrent_positions", 0))
MAX_CONCURRENT_POSITIONS = _mcp if _mcp > 0 else None
# Open-risk cap: hard rail on TOTAL initial $ risk across open positions (the
# daily loss cap only counts REALIZED losses — it never sees what the open book
# could lose). 0 = AUTO: risk_per_trade x max_concurrent (the theoretical normal
# max, so it only trips on sizing bugs / config mistakes). None = disabled
# (auto with unlimited concurrency has no defined ceiling).
_mor = float(_CFG.get("max_open_risk", 0.0) or 0.0)
if _mor <= 0:
    _mor = float(_CFG["risk_per_trade"]) * _mcp if _mcp > 0 else 0.0
MAX_OPEN_RISK = _mor if _mor > 0 else None
# Default from config, but overridable per-run via the ORB_TREND_FILTER env var so a
# second instance (the news-edge bot) can run with NO technical screen — letting the
# morning news catalyst, not a trend test, select the names. Read at import, so the
# launcher must set it in the environment BEFORE python starts. Baseline VM bot never
# sets it -> uses the config default, behaviour unchanged.
_tf_env = os.environ.get("ORB_TREND_FILTER")
TREND_FILTER_ENABLED = (
    _tf_env.strip().lower() not in ("false", "0", "no", "off")
    if _tf_env is not None and _tf_env.strip() != ""
    else bool(_CFG.get("trend_filter_enabled", True))
)
TREND_SMA_DAYS = 200
TREND_RET_DAYS = 20
# Tight-OR filter: only take breakouts whose opening range is narrow (<= TIGHT_OR_MAX_PCT
# of entry price). The biggest validated edge (see memory tightOR_finding / backtest/
# compare_or_range_*.py): tight ranges + the trailing exit give the asymmetric payoff.
# Config default ON, overridable off per-run via ORB_TIGHT_OR=false (the news-edge bot
# sets it off so the catalyst, not OR width, selects its names).
_to_env = os.environ.get("ORB_TIGHT_OR")
TIGHT_OR_FILTER_ENABLED = (
    _to_env.strip().lower() not in ("false", "0", "no", "off")
    if _to_env is not None and _to_env.strip() != ""
    else bool(_CFG.get("tight_or_filter_enabled", True))
)
TIGHT_OR_MAX_PCT = float(_CFG.get("tight_or_max_pct", 0.5))
# Trailing-stop exit (let winners run): replaces the fixed 2R target with a
# native Alpaca trailing stop at 1R below the high-water mark. Long-only; OFF by
# default. See backtest/compare_exits.py.
TRAILING_EXIT_ENABLED = bool(_CFG.get("trailing_exit_enabled", False))
# Vol-regime risk dial: halve (or pause) risk in high-vol regimes. Reactive — sits
# out the turbulent aftermath of a shock. See backtest/compare_volpause.py.
# Config default (ON for the validated baseline ORB), overridable off per-run via
# ORB_VOL_REGIME_FILTER=false — the news-edge bot turns it OFF because its whole point is
# to act on the morning catalyst EVERY day, choppy or calm (the dial would otherwise halve
# its notional on high-vol days and drop the priciest catalyst names). Baseline never sets
# it -> config default, validated vol-dial behaviour unchanged.
_vr_env = os.environ.get("ORB_VOL_REGIME_FILTER")
VOL_REGIME_FILTER_ENABLED = (
    _vr_env.strip().lower() not in ("false", "0", "no", "off")
    if _vr_env is not None and _vr_env.strip() != ""
    else bool(_CFG.get("vol_regime_filter_enabled", False))
)
VOL_REGIME_RISK_MULT = float(_CFG.get("vol_regime_risk_mult", 0.5))
VOL_WIN = 20          # SPY realized-vol lookback (trading days)
VOL_MED_WIN = 126     # trailing window for the vol threshold

# Short side (regime-gated). See live/config.py for the validation basis.
SHORT_ENABLED = bool(_CFG["short_enabled"])
SHORT_SYMBOLS = set(_CFG["short_symbols"])
REGIME_REF_SYMBOL = str(_CFG["regime_ref_symbol"])
REGIME_SMA_WINDOW = int(_CFG["regime_sma_window"])
REGIME_CONFIRM_DAYS = int(_CFG["regime_confirm_days"])

LATE_START_CUTOFF_MINUTES = int(os.environ.get("ORB_LATE_START_CUTOFF") or 10)  # halt NEW entries if started > N min after OR close. News bot starts later (after the scan) -> raises it.
POLL_SECONDS = 10

# Notional sizing mode (news-edge / catalyst names). When ORB_NOTIONAL_PER_TRADE is set,
# size by a FIXED DOLLAR amount per position (whole shares) instead of by risk-per-share,
# and SKIP the max_risk_per_share rejection. Rationale (2026-06-12): news catalysts land on
# high-priced, volatile names whose OR-low stops give huge risk/share ($18-87 on 2026-06-12's
# AMD/STX/SNDK). Risk-based sizing ($50 / risk_per_share) then yields 0-2 shares or rejects
# outright, so the catalyst bot only ever traded the cheapest, lowest-conviction name. Notional
# sizing makes every catalyst name tradeable with controlled dollar exposure. The baseline ORB
# never sets this -> risk-based sizing, behaviour unchanged. ORB_NOTIONAL_CAP optionally caps a
# single position (defaults to max_position_dollars / max_position_pct).
_notional_env = os.environ.get("ORB_NOTIONAL_PER_TRADE")
NOTIONAL_PER_TRADE = (float(_notional_env) if _notional_env and _notional_env.strip() else None)
_notcap_env = os.environ.get("ORB_NOTIONAL_CAP")
NOTIONAL_CAP = (float(_notcap_env) if _notcap_env and _notcap_env.strip() else None)
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
    side: str = "long"  # "long" or "short" — set when a breakout is taken
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
    # Populated when a bracket leg fills or EOD-flatten closes the position
    exited: bool = False
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # "target", "stop", or "EOD"
    realized_pnl: Optional[float] = None
    # Slippage tracking (Tier-1 #2): theoretical entry = last close at breakout;
    # actual entry = parent order's filled_avg_price. Logged once on first
    # observation in poll_exits.
    entry_estimate: Optional[float] = None
    entry_slippage_logged: bool = False
    # Trailing-stop exit (TRAILING_EXIT_ENABLED): the protective trailing stop is
    # attached on first-observed entry fill (Alpaca can't trail a bracket leg).
    trail_amount: Optional[float] = None   # $/share to trail below the peak (= 1R)
    trail_stop_id: Optional[str] = None    # id of the placed trailing-stop order


def open_risk_now(run: "RunState") -> float:
    """Total INITIAL $ risk across open (non-exited) positions:
    sum of |entry - stop| x shares. Initial risk is a safe upper bound — the
    trailing exit only ever ratchets a position's remaining risk DOWN."""
    total = 0.0
    for s in run.states.values():
        if (s.entered and s.entry_order_id and not s.exited and not s.reject_reason
                and s.entry_price is not None and s.stop_price is not None and s.shares):
            total += abs(s.entry_price - s.stop_price) * s.shares
    return total


@dataclass
class RunState:
    states: dict[str, SymbolState] = field(default_factory=dict)
    halted: bool = False
    halt_reason: Optional[str] = None  # "late-start", "loss-cap", or None when not halted
    starting_equity: float = 0.0
    or_lock_notified: bool = False
    shorts_enabled: bool = False       # set once at session start from the regime gate
    regime_note: str = ""              # human-readable explanation of the gate decision
    risk_mult: float = 1.0             # vol-regime risk dial (1.0 normal, <1 in high-vol)
    # Trend filter (multi-timeframe): symbols eligible for LONG entries today.
    # Computed once at session start from prior daily closes. None = filter
    # disabled or unavailable (fail-open, every name eligible).
    trend_eligible: Optional[set] = None


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
    # Paper vs LIVE is env-gated with a SAFE default: PAPER unless ALPACA_PAPER_TRADE
    # is explicitly set falsy. This REUSES the same var the Alpaca MCP server reads,
    # so one toggle controls both — no confusing near-duplicate. Going to real money
    # is then a deliberate .env change (ALPACA_PAPER_TRADE=false + live keys), never
    # an accidental code edit. Live keys won't authenticate against the paper
    # endpoint (and vice-versa), so a mismatched flag/keys fails safe (errors).
    paper = os.environ.get("ALPACA_PAPER_TRADE", "true").strip().lower() not in ("false", "0", "no", "off")
    if paper:
        log.info("Alpaca: PAPER trading (set ALPACA_PAPER_TRADE=false + live keys for real money).")
    else:
        log.warning("*** Alpaca: LIVE / REAL-MONEY TRADING (ALPACA_PAPER_TRADE=false). REAL FUNDS AT RISK. ***")
    return (
        TradingClient(api_key, secret, paper=paper),
        StockHistoricalDataClient(api_key, secret),
    )


# ---------- logging ----------
def setup_logging(dry_run: bool) -> None:
    today = datetime.now(ET).strftime("%Y-%m-%d")
    tag = os.environ.get("ORB_LOG_TAG", "")   # e.g. "news_" so the 2nd runner logs separately
    log_path = LOG_DIR / f"orb_{tag}{today}{'_dryrun' if dry_run else ''}.log"
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
def compute_trend_eligibility(dc: StockHistoricalDataClient,
                              watchlist: list[str]) -> Optional[set]:
    """Symbols passing the trend filter as of the PRIOR trading day's close:
    (close > 200d SMA) AND (20d return > SPY's 20d return).

    Validated by backtest/compare_trend_filter.py — nearly doubles avg_R, lifts
    win rate to 50.4%, cuts max DD ~79%, OOS-robust both halves.

    Fails open on any error: returns None and the caller treats every name as
    eligible (baseline behaviour). Never raises. Runs once at session start.
    """
    syms = sorted(set(watchlist) | {"SPY"})
    end = datetime.now(ET)
    start = end - timedelta(days=int(TREND_SMA_DAYS * 1.6))
    try:
        req = StockBarsRequest(
            symbol_or_symbols=syms, timeframe=TimeFrame.Day,
            start=start.astimezone(UTC), end=end.astimezone(UTC),
            feed=DataFeed.IEX, adjustment=Adjustment.ALL,
        )
        df = dc.get_stock_bars(req).df
    except Exception as e:
        log.warning(f"Trend filter: daily fetch failed ({e}); FAIL-OPEN (baseline).")
        try:
            notify(f"Trend filter daily fetch failed: {e}\nFalling back to baseline "
                   f"(all names eligible).", title="ORB trend filter degraded",
                   tags=["warning"])
        except Exception:
            pass
        return None
    if df.empty:
        log.warning("Trend filter: no daily bars returned; FAIL-OPEN (baseline).")
        return None
    closes = df["close"].unstack(level=0)
    closes.index = (pd.to_datetime(closes.index).tz_convert(ET)
                    .normalize().tz_localize(None))
    closes = closes.sort_index()
    today_naive = pd.Timestamp(datetime.now(ET).date())
    closes_prior = closes.loc[closes.index < today_naive]
    if len(closes_prior) < TREND_SMA_DAYS:
        log.warning(f"Trend filter: only {len(closes_prior)} prior days available "
                    f"(need {TREND_SMA_DAYS}); FAIL-OPEN (baseline).")
        return None
    prior_close = closes_prior.iloc[-1]
    sma = closes_prior.rolling(TREND_SMA_DAYS).mean().iloc[-1]
    ret = closes_prior.pct_change(TREND_RET_DAYS).iloc[-1]
    spy_ret = ret.get("SPY") if "SPY" in ret.index else None
    if spy_ret is None or pd.isna(spy_ret):
        log.warning("Trend filter: SPY 20d return unavailable; FAIL-OPEN.")
        return None
    eligible = set()
    for sym in watchlist:
        if sym not in closes_prior.columns:
            continue
        pc, s, r = prior_close.get(sym), sma.get(sym), ret.get(sym)
        if any(pd.isna(x) for x in (pc, s, r)):
            continue
        if pc > s and r > spy_ret:
            eligible.add(sym)
    log.info(f"Trend filter: {len(eligible)}/{len(watchlist)} names eligible today "
             f"(close > {TREND_SMA_DAYS}d SMA AND {TREND_RET_DAYS}d return > "
             f"SPY's {spy_ret:+.2%}).")
    if eligible:
        log.info(f"  eligible: {', '.join(sorted(eligible))}")
    filtered = [s for s in watchlist if s not in eligible
                and s in closes_prior.columns and not pd.isna(prior_close.get(s))]
    if filtered:
        preview = ', '.join(sorted(filtered)[:25])
        more = f" ... (+{len(filtered) - 25} more)" if len(filtered) > 25 else ""
        log.info(f"  filtered: {preview}{more}")
    return eligible


def compute_vol_regime(dc: StockHistoricalDataClient, today: date) -> Optional[bool]:
    """True if SPY's 20d realized vol (as of the PRIOR close) is above its
    trailing-126d median — a 'high-vol regime' where we dial risk down. None on
    any error (caller fails safe to full risk). Lookahead-free: only closes
    strictly before `today`. Mirrors backtest/compare_volpause.py.
    """
    try:
        end = combine_et(today, RTH_OPEN)        # pre-open today
        start = end - timedelta(days=400)        # ~270 trading days; need 146+
        req = StockBarsRequest(
            symbol_or_symbols=["SPY"], timeframe=TimeFrame.Day,
            start=start.astimezone(UTC), end=end.astimezone(UTC),
            feed=DataFeed.IEX, adjustment=Adjustment.ALL,
        )
        df = dc.get_stock_bars(req).df
        if df.empty:
            return None
        closes = (df.xs("SPY", level=0)["close"] if isinstance(df.index, pd.MultiIndex)
                  else df["close"]).astype(float).sort_index()
        # belt-and-braces: drop any bar dated today or later (no lookahead)
        keep = [(ts.tz_convert(ET).date() if getattr(ts, "tzinfo", None) else ts.date()) < today
                for ts in closes.index]
        closes = closes[keep]
        vol = closes.pct_change().rolling(VOL_WIN).std()
        med = vol.rolling(VOL_MED_WIN, min_periods=40).median()
        v, m = vol.iloc[-1], med.iloc[-1]
        if pd.isna(v) or pd.isna(m):
            log.info("Vol regime: insufficient history; full risk.")
            return None
        return bool(v > m)
    except Exception as e:
        log.warning(f"Vol regime check failed ({e}); FAIL-SAFE to full risk.")
        return None


def wait_for_network(tc: TradingClient, max_attempts: int = 5,
                     delay_sec: float = 30.0) -> bool:
    """Block until Alpaca is reachable; handle 'laptop just woke up' wifi lag.

    On 2026-05-29 the scheduled 06:15 fire crashed with DNS resolution failure
    because the laptop's network hadn't joined yet (user travelling). Retries
    the lightest API call (get_clock) before letting preflight try and crash.

    Returns True if reachable within the budget, False otherwise.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            tc.get_clock()
            if attempt > 1:
                log.info(f"Network reachable on attempt {attempt}/{max_attempts}.")
            return True
        except Exception as e:
            if attempt < max_attempts:
                log.warning(f"Network unreachable on attempt {attempt}/{max_attempts} "
                            f"({type(e).__name__}); retrying in {delay_sec:.0f}s")
                time_mod.sleep(delay_sec)
            else:
                log.error(f"Network unreachable after {max_attempts} attempts; aborting. "
                          f"Last error: {e}")
                return False
    return False


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

    # Use a 7-day lookback rather than fetch_today_bars: smoke test runs
    # pre-market (06:05 PDT) and Alpaca rejects when end < start.
    try:
        end_et = datetime.now(ET)
        start_et = end_et - timedelta(days=7)
        req = StockBarsRequest(
            symbol_or_symbols=["SPY"],
            timeframe=TimeFrame.Minute,
            start=start_et.astimezone(UTC),
            end=end_et.astimezone(UTC),
            feed=DataFeed.IEX,
        )
        bars = dc.get_stock_bars(req).df
        log.info(f"PASS  data feed reachable (SPY bars in last 7 days: {len(bars)})")
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
    """Find any orb-* orders submitted today and rehydrate state from them.

    Pulls entry fill price, share count, stop/target levels, and exit status
    (if a bracket leg has filled) so the UI shows realistic data on restart
    instead of just "ENTERED" with no numbers.
    """
    today_start = datetime.combine(today, time(0, 0, tzinfo=ET))
    # alpaca-py supports `nested=True` to embed bracket legs; fall back if not.
    try:
        req = GetOrdersRequest(
            status=QueryOrderStatus.ALL,
            after=today_start.astimezone(UTC),
            limit=500,
            nested=True,
        )
    except TypeError:
        req = GetOrdersRequest(
            status=QueryOrderStatus.ALL,
            after=today_start.astimezone(UTC),
            limit=500,
        )
    try:
        orders = tc.get_orders(filter=req)
    except TypeError:
        orders = tc.get_orders(req)
    coid_prefix = f"orb-{today.strftime('%Y%m%d')}-"
    resumed = 0
    for o in orders:
        if not getattr(o, "client_order_id", None):
            continue
        if not o.client_order_id.startswith(coid_prefix):
            continue
        sym = o.symbol
        if sym not in run.states:
            continue
        st = run.states[sym]
        st.entered = True
        st.entry_order_id = str(o.id)
        # Recover direction from the parent order side (SELL = short).
        st.side = "short" if _status_str(getattr(o, "side", "")) == "SELL" else "long"
        # Entry fill details (so the UI shows real entry prices, not blanks)
        entry_fill = getattr(o, "filled_avg_price", None)
        if entry_fill is not None:
            try:
                st.entry_price = float(entry_fill)
            except Exception:
                pass
        filled_qty = getattr(o, "filled_qty", None)
        if filled_qty is not None:
            try:
                st.shares = int(float(filled_qty))
            except Exception:
                pass
        # Bracket legs: capture stop/target levels and detect already-filled exits
        for leg in (getattr(o, "legs", None) or []):
            limit_px = getattr(leg, "limit_price", None)
            stop_px = getattr(leg, "stop_price", None)
            is_target = limit_px is not None
            try:
                if is_target and st.target_price is None and limit_px is not None:
                    st.target_price = float(limit_px)
                if (not is_target) and stop_px is not None and st.stop_price is None:
                    st.stop_price = float(stop_px)
            except Exception:
                pass
            leg_status = _status_str(getattr(leg, "status", ""))
            if leg_status == "FILLED" and not st.exited:
                exit_fill = getattr(leg, "filled_avg_price", None)
                if exit_fill is not None and st.entry_price is not None:
                    try:
                        epx = float(exit_fill)
                        exit_qty_raw = getattr(leg, "filled_qty", None)
                        qty = int(float(exit_qty_raw)) if exit_qty_raw else (st.shares or 0)
                        st.exited = True
                        st.exit_price = epx
                        st.exit_reason = "target" if is_target else "stop"
                        st.realized_pnl = ((st.entry_price - epx) if st.side == "short"
                                           else (epx - st.entry_price)) * qty
                        st.exit_notified = True  # poll_exits already missed this fill
                    except Exception:
                        pass
        resumed += 1
    if resumed:
        resumed_syms = [s for s, sx in run.states.items()
                        if sx.entered and sx.entry_order_id]
        exited_syms = [s for s, sx in run.states.items() if sx.exited]
        log.info(f"Resumed {resumed} entry state(s) from existing orders today "
                 f"(of which exited: {len(exited_syms)}).")
        try:
            parts = [f"Recovered {resumed} in-flight position(s): {', '.join(resumed_syms)}."]
            if exited_syms:
                parts.append(f"Already exited: {', '.join(exited_syms)}.")
            parts.append("Existing brackets continue server-side; "
                         "EOD-flat at 15:55 ET unchanged.")
            notify(
                "\n".join(parts),
                title="ORB recovered after restart",
                priority=4,
                tags=["arrows_counterclockwise"],
            )
        except Exception as e:
            log.warning(f"Recovery notification failed: {e}")


def detect_untracked_positions(tc: TradingClient, run: RunState,
                               watchlist: list[str]) -> None:
    """Flag any open watchlist position that sync_existing_orders_today didn't
    rehydrate as 'entered'. These are positions held across sessions (or by
    manual action) that the script doesn't own.

    Effect: mark the symbol entered with reject_reason='untracked' so the
    breakout loop skips it (prevents double-entry). Push a priority-5 ntfy
    so the user can close manually if undesired.

    Call AFTER sync_existing_orders_today and BEFORE prebuild_or_if_late.
    Origin: 2026-05-27 morning — yesterday's OPG sell on the stranded QQQ
    position expired without filling, leaving QQQ naked while the script
    was unaware (different coid prefix).
    """
    try:
        positions = tc.get_all_positions()
    except Exception as e:
        log.warning(f"detect_untracked_positions: get_all_positions failed: {e}")
        return

    untracked = []
    for p in positions:
        if p.symbol not in watchlist:
            continue
        st = run.states.get(p.symbol)
        if st is None or st.entered:
            continue  # already rehydrated from today's orders
        try:
            qty = int(float(p.qty))
            entry = float(p.avg_entry_price)
            current = float(getattr(p, "current_price", 0) or 0)
            unrealized = float(getattr(p, "unrealized_pl", 0) or 0)
        except Exception:
            qty, entry, current, unrealized = 0, 0.0, 0.0, 0.0
        st.entered = True
        st.entry_price = entry
        st.shares = qty
        st.reject_reason = "untracked"
        untracked.append((p.symbol, qty, entry, current, unrealized))
        log.warning(
            f"UNTRACKED position {p.symbol}: qty={qty} avg_entry=${entry:.2f} "
            f"current=${current:.2f} unrealized=${unrealized:+.2f}. "
            f"Marked entered to block double-entry; close manually if undesired."
        )

    if untracked:
        try:
            lines = ["Open positions the script does NOT own:", ""]
            for sym, qty, entry, current, unrl in untracked:
                lines.append(
                    f"  {sym}: {qty} sh @ ${entry:.2f}  now ${current:.2f}  "
                    f"unrealized ${unrl:+,.0f}"
                )
            lines.append("")
            lines.append("Script will NOT enter new orders in these symbols today. "
                         "Close manually via Alpaca if undesired.")
            notify(
                "\n".join(lines),
                title=f"ORB untracked positions ({datetime.now(ET).date().isoformat()})",
                priority=5,
                tags=["warning"],
            )
        except Exception as e:
            log.warning(f"Untracked positions notification failed: {e}")


def prebuild_or_if_late(dc: StockHistoricalDataClient, run: RunState,
                       watchlist: list[str], today: date,
                       or_end_dt: datetime) -> None:
    """If we start after the OR window closes, fetch today's bars and lock OR
    levels from them now — instead of waiting for the first main-loop iteration.

    No-op if now < or_end_dt. Safe to call even when bars haven't materialized
    (e.g., the market hasn't opened or it's a closed day).
    """
    if datetime.now(ET) < or_end_dt:
        return
    try:
        bars = fetch_today_bars(dc, watchlist, today)
    except Exception as e:
        log.warning(f"prebuild_or: bar fetch failed: {e}")
        return
    if bars.empty:
        log.info("prebuild_or: no bars for today; skipping (closed day or pre-open).")
        return
    symbols_in_data = set(bars.index.get_level_values(0).unique())
    locked = 0
    for sym in watchlist:
        if sym not in symbols_in_data:
            continue
        state = run.states[sym]
        if state.or_locked:
            continue
        sym_bars = bars.xs(sym, level=0)
        or_bars = sym_bars[sym_bars.index < or_end_dt]
        if or_bars.empty:
            continue
        state.or_high = float(or_bars["high"].max())
        state.or_low = float(or_bars["low"].min())
        state.or_locked = True
        locked += 1
        log.info(f"{sym} OR pre-locked from history: "
                 f"high=${state.or_high:.2f} low=${state.or_low:.2f}")
    if locked:
        log.info(f"prebuild_or: locked {locked}/{len(watchlist)} symbols from history.")


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
    _t0 = time_mod.perf_counter()
    bars = dc.get_stock_bars(req).df
    _dt = time_mod.perf_counter() - _t0
    # Capacity watch for the 122-name watchlist (2026-06-10 expansion): the poll
    # cycle is 10s, so a slow full-session fetch late in the day is the constraint.
    # Warn only when it matters — if this fires repeatedly, trim the expansion names.
    if _dt > 5.0:
        log.warning(f"fetch_today_bars slow: {_dt:.1f}s for {len(symbols)} symbols "
                    f"(>5s; 10s poll cycle — trim watchlist if persistent)")
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


# ---------- regime gate (short side) ----------
def compute_short_regime(dc: StockHistoricalDataClient, today: date) -> tuple[bool, str]:
    """Decide, pre-open, whether shorts are enabled today.

    Bearish regime = SPY's daily close below its REGIME_SMA_WINDOW-day SMA for
    REGIME_CONFIRM_DAYS consecutive sessions, using only sessions strictly
    BEFORE today (no lookahead). Returns (shorts_enabled, human_note). Any
    failure fails safe to shorts-OFF.
    """
    if not SHORT_ENABLED:
        return False, "shorts OFF (SHORT_ENABLED=False)"
    try:
        end_et = datetime.combine(today, RTH_OPEN, tzinfo=ET)  # pre-open today
        start_et = end_et - timedelta(days=REGIME_SMA_WINDOW * 3 + 30)
        req = StockBarsRequest(
            symbol_or_symbols=[REGIME_REF_SYMBOL],
            timeframe=TimeFrame.Day,
            start=start_et.astimezone(UTC),
            end=end_et.astimezone(UTC),
            feed=DataFeed.IEX,
        )
        df = dc.get_stock_bars(req).df
        if df.empty:
            return False, f"shorts OFF (no {REGIME_REF_SYMBOL} daily bars)"
        closes = df.xs(REGIME_REF_SYMBOL, level=0)["close"] if isinstance(df.index, pd.MultiIndex) else df["close"]
        closes = closes.astype(float)
        # Drop any bar dated today or later (belt-and-braces against lookahead).
        keep = [ts.tz_convert(ET).date() < today if getattr(ts, "tzinfo", None) else ts.date() < today
                for ts in closes.index]
        closes = closes[keep]
        if len(closes) < REGIME_SMA_WINDOW + REGIME_CONFIRM_DAYS:
            return False, f"shorts OFF (only {len(closes)} daily bars, need {REGIME_SMA_WINDOW + REGIME_CONFIRM_DAYS})"
        sma = closes.rolling(REGIME_SMA_WINDOW).mean()
        below = (closes < sma)
        n_below = int(below.tail(REGIME_CONFIRM_DAYS).sum())
        bearish = n_below == REGIME_CONFIRM_DAYS
        note = (f"shorts {'ON' if bearish else 'OFF'}: {REGIME_REF_SYMBOL} "
                f"{closes.iloc[-1]:.2f} vs SMA{REGIME_SMA_WINDOW} {sma.iloc[-1]:.2f}; "
                f"{n_below}/{REGIME_CONFIRM_DAYS} latest closes below SMA "
                f"(as of {closes.index[-1].date() if hasattr(closes.index[-1], 'date') else closes.index[-1]})")
        return bearish, note
    except Exception as e:
        return False, f"shorts OFF (regime check failed: {e})"


# ---------- sizing / guardrails ----------
def size_position(entry: float, stop: float, equity: float,
                  risk_mult: float = 1.0) -> tuple[int, str]:
    """Returns (shares, reason). shares==0 means reject; reason explains.

    Direction-agnostic: risk per share is |entry - stop| (stop is below entry
    for longs, above for shorts). risk_mult scales risk_per_trade for the
    vol-regime dial (1.0 normal, 0.5 in high-vol).
    """
    risk_per_share = abs(entry - stop)
    if risk_per_share < MIN_RISK_PER_SHARE:
        return 0, f"risk_per_share ${risk_per_share:.4f} < min ${MIN_RISK_PER_SHARE}"

    # --- notional sizing mode (news-edge): fixed $ per name, ignores the risk/share cap ---
    if NOTIONAL_PER_TRADE is not None:
        cap_dollars = equity * PARAMS.max_position_pct
        if PARAMS.max_position_dollars is not None:
            cap_dollars = min(cap_dollars, PARAMS.max_position_dollars)
        if NOTIONAL_CAP is not None:
            cap_dollars = min(cap_dollars, NOTIONAL_CAP)
        budget = min(NOTIONAL_PER_TRADE * risk_mult, cap_dollars)
        shares = math.floor(budget / entry)
        if shares <= 0:
            return 0, (f"notional sizing produced 0 shares "
                       f"(budget ${budget:.0f}, price ${entry:.2f})")
        return shares, "ok"

    # --- risk-based sizing (baseline ORB) ---
    if risk_per_share > MAX_RISK_PER_SHARE:
        return 0, f"risk_per_share ${risk_per_share:.2f} > max ${MAX_RISK_PER_SHARE}"
    shares_by_risk = math.floor(PARAMS.risk_per_trade * risk_mult / risk_per_share)
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
                   stop: float, target: float, side: OrderSide,
                   today: date, dry_run: bool) -> Optional[str]:
    """Submit a bracket market order. For a long (BUY) the take-profit sits above
    and stop below entry; for a short (SELL) Alpaca expects the opposite, which
    is exactly how the caller computes target (< entry) and stop (> entry)."""
    side_name = "BUY" if side == OrderSide.BUY else "SELL"
    coid = f"orb-{today.strftime('%Y%m%d')}-{sym}-entry"
    # TODO (cost-side improvement, deferred 2026-06-04): entries are MARKET orders
    # and pay the full spread. Measured incremental entry slippage ~0.020R (median
    # 0.014R), with a fat tail on high-priced / tight-stop names. A marketable-LIMIT
    # entry (limit a few cents above the breakout) would cap the worst fills and
    # could lift the thin net edge (~+0.039R after costs). TRADE-OFF: a limit can
    # MISS the trade if price runs past it before filling — and the runaway
    # breakouts are exactly the trend-day winners the trailing exit lives on, so a
    # too-tight limit systematically skips the best trades. Hard to backtest (needs
    # a fill-PROBABILITY model, not just a fill price; 1-min bars are coarse for it).
    # Plan: implement behind a config flag like trailing_exit_enabled, paper-first,
    # AFTER accumulating real fill/slippage data from live trailing sessions. See
    # backtest/compare_exits_slippage.py + the 2026-06-04 slippage measurement.
    #
    # Trailing-exit (long only): a trailing stop can't be a bracket leg, so the
    # entry is a plain market order and the protective trailing stop is attached
    # on fill in poll_exits. Shorts (if ever enabled) keep the bracket.
    trailing = TRAILING_EXIT_ENABLED and side == OrderSide.BUY
    if trailing:
        req = MarketOrderRequest(
            symbol=sym, qty=qty, side=side,
            time_in_force=TimeInForce.DAY, client_order_id=coid,
        )
    else:
        req = MarketOrderRequest(
            symbol=sym,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=round(target, 2)),
            stop_loss=StopLossRequest(stop_price=round(stop, 2)),
            client_order_id=coid,
        )
    if dry_run:
        if trailing:
            log.info(f"[DRY-RUN] WOULD SUBMIT trailing-entry {side_name} {qty} {sym} "
                     f"(market entry; trail ${abs(entry_est - stop):.2f}/sh attached on fill) coid={coid}")
        else:
            log.info(f"[DRY-RUN] WOULD SUBMIT bracket {side_name} {qty} {sym} "
                     f"target=${target:.2f} stop=${stop:.2f} coid={coid}")
        return f"dryrun-{coid}"
    try:
        o = tc.submit_order(req)
        log.info(f"SUBMITTED {sym} {side_name}: id={o.id} status={o.status} coid={coid}")
        try:
            exit_desc = (f"trail ${abs(entry_est - stop):.2f}/sh (init stop ${stop:.2f})"
                         if trailing else f"stop ${stop:.2f}  target ${target:.2f}")
            notify(
                f"{side_name} {qty} {sym} @ ~${entry_est:.2f}  "
                f"{exit_desc}  "
                f"risk ${abs(entry_est - stop) * qty:.0f}",
                title=f"ORB entry: {sym} ({side_name})",
                tags=["chart_with_upwards_trend"] if side == OrderSide.BUY
                     else ["chart_with_downwards_trend"],
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
    """For each entered symbol, check the parent + bracket legs:

    - On first observed entry fill: log slippage (actual - theoretical) once.
    - On first observed leg fill: log exit slippage AND push the ntfy.

    Pure read-only: never modifies orders/positions, only updates flags.
    Swallows all errors — notifications and logs are decoration.
    """
    for sym, st in run.states.items():
        if not (st.entered and st.entry_order_id) or st.reject_reason:
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
        is_short = st.side == "short"
        entry_action = "SELL" if is_short else "BUY"   # how the position was OPENED
        # ---- Entry slippage (once per symbol per session) ----
        if not st.entry_slippage_logged and st.entry_estimate is not None:
            try:
                actual = float(entry_fill)
                slip = actual - st.entry_estimate
                qty = int(getattr(parent, "filled_qty", None) or st.shares or 0)
                # BUY entry: positive slip = paid more (bad). SELL (short) entry:
                # positive slip = sold higher than expected (good).
                bps = (slip / st.entry_estimate * 10000) if st.entry_estimate else 0.0
                log.info(
                    f"SLIPPAGE entry {sym} {entry_action}: theoretical ${st.entry_estimate:.4f} "
                    f"actual ${actual:.4f} -> {slip:+.4f}/sh ({bps:+.2f} bps, "
                    f"cost ${slip * qty:+.2f} on {qty} sh)"
                )
            except Exception as e:
                log.debug(f"poll_exits: entry slippage log failed for {sym}: {e}")
            st.entry_slippage_logged = True
        # ---- Trailing exit: attach the protective trailing stop once the entry
        # has filled (Alpaca can't trail a bracket leg, so it's a standalone order
        # placed here). Retries each loop until placed; the unprotected window is
        # at most one poll and the initial trail level ≈ OR_low (the fixed stop). ----
        if TRAILING_EXIT_ENABLED and st.trail_amount and not st.trail_stop_id and not st.exited:
            try:
                fqty = int(getattr(parent, "filled_qty", None) or st.shares or 0)
                if fqty > 0:
                    treq = TrailingStopOrderRequest(
                        symbol=sym, qty=fqty,
                        side=OrderSide.BUY if is_short else OrderSide.SELL,
                        time_in_force=TimeInForce.DAY,
                        trail_price=round(st.trail_amount, 2),
                    )
                    to = tc.submit_order(treq)
                    st.trail_stop_id = str(to.id)
                    log.info(f"TRAILING stop attached {sym}: trail ${st.trail_amount:.2f}/sh "
                             f"id={to.id} (initial stop ~${st.stop_price:.2f})")
            except Exception as e:
                log.error(f"{sym} trailing-stop placement FAILED: {e}")
                # Safety net: never leave the position naked. Fall back to a plain
                # protective stop at OR_low (the old fixed-stop behavior). Tracked
                # via trail_stop_id so exit detection below handles it identically.
                try:
                    fqty = int(getattr(parent, "filled_qty", None) or st.shares or 0)
                    if fqty > 0 and st.stop_price is not None:
                        so = tc.submit_order(StopOrderRequest(
                            symbol=sym, qty=fqty,
                            side=OrderSide.BUY if is_short else OrderSide.SELL,
                            time_in_force=TimeInForce.DAY,
                            stop_price=round(st.stop_price, 2),
                        ))
                        st.trail_stop_id = str(so.id)
                        log.warning(f"{sym} FELL BACK to fixed stop ${st.stop_price:.2f} id={so.id}")
                except Exception as e2:
                    log.error(f"{sym} fixed-stop fallback ALSO failed (retry next loop): {e2}")
        # ---- Exit slippage + notify (once per symbol per session) ----
        if st.exit_notified:
            continue
        # Trailing-exit path: the exit is the trailing-stop fill, not a bracket leg.
        if TRAILING_EXIT_ENABLED and st.trail_stop_id:
            try:
                to = tc.get_order_by_id(st.trail_stop_id)
                if _status_str(getattr(to, "status", "")) == "FILLED":
                    exit_fill = float(to.filled_avg_price)
                    entry_px = float(entry_fill)
                    qty = int(getattr(to, "filled_qty", None) or st.shares or 0)
                    pnl = (entry_px - exit_fill) * qty if is_short else (exit_fill - entry_px) * qty
                    notify(
                        f"{sym} trail-stop @ ${exit_fill:.2f}  "
                        f"(entry ${entry_px:.2f}, qty {qty})  PnL ${pnl:+,.0f}",
                        title=f"ORB exit: {sym} trail",
                        tags=["octagonal_sign"],
                    )
                    st.exit_notified = True
            except Exception as e:
                log.warning(f"poll_exits: trailing exit check failed for {sym}: {e}")
            continue  # don't fall through to bracket-leg detection
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
                # Direction-aware P&L: long profits when exit > entry; short
                # profits when exit (buy-to-cover) < entry.
                pnl = (entry_px - exit_fill) * qty if is_short else (exit_fill - entry_px) * qty
                # Exit closes the position: a long exits via SELL, a short via BUY-to-cover.
                exit_action = "BUY" if is_short else "SELL"
                theoretical_exit = (st.target_price if is_target else st.stop_price)
                if theoretical_exit is not None:
                    slip = exit_fill - theoretical_exit
                    bps = (slip / theoretical_exit * 10000) if theoretical_exit else 0.0
                    log.info(
                        f"SLIPPAGE exit {sym} {reason} {exit_action}: "
                        f"theoretical ${theoretical_exit:.4f} "
                        f"actual ${exit_fill:.4f} -> {slip:+.4f}/sh "
                        f"({bps:+.2f} bps, {slip * qty:+.2f} on {qty} sh)"
                    )
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


# Closing a position requires the qty held by cancelled bracket legs to be
# released first. Empirically that release can lag 15-20s+ after the cancel, so
# the close budget must comfortably exceed it (the old 4x2s=8s budget was the
# 2026-05-27 naked-AAPL bug: cancels released at +17s, every close had already
# failed and the loop gave up silently).
CLOSE_MAX_ATTEMPTS = 12          # ~ up to ~36s of close retries
CLOSE_RETRY_DELAY_SEC = 3.0
CANCEL_CONFIRM_TIMEOUT_SEC = 25.0   # poll until our open orders actually clear
CANCEL_CONFIRM_POLL_SEC = 2.0
# Poll until close MARKET ORDERS fill before alerting. On 2026-05-28 the close
# orders queued behind leg-cancellations and didn't fill for ~29s, so the window
# must be generous. EOD flatten runs at 15:55 ET (300s before the 16:00 close),
# so 60s is safe headroom and still leaves time to react to a genuine failure.
FLATTEN_VERIFY_TIMEOUT_SEC = 60.0
FLATTEN_VERIFY_POLL_SEC = 3.0


def _our_open_orders(tc: TradingClient, watchlist: list[str]) -> list:
    """Open orders whose symbol is in our watchlist. Empty list on API error."""
    try:
        try:
            oo = tc.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500))
        except TypeError:
            oo = tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500))
        return [o for o in oo if o.symbol in watchlist]
    except Exception as e:
        log.warning(f"Could not list open orders: {e}")
        return []


def _our_positions(tc: TradingClient, watchlist: list[str]) -> list:
    """Open positions whose symbol is in our watchlist. Empty list on API error."""
    try:
        return [p for p in tc.get_all_positions() if p.symbol in watchlist]
    except Exception as e:
        log.warning(f"get_all_positions failed: {e}")
        return []


def _close_position_with_retry(tc: TradingClient, sym: str, watchlist: list[str]) -> bool:
    """close_position with a generous retry budget for the cancel-then-close race.

    While bracket legs are being cancelled Alpaca reports `insufficient qty
    available` until the held qty is released. On each race we also re-cancel
    any lingering orders for this symbol (in case a leg cancel didn't take),
    then wait and retry. Returns True only once Alpaca accepts the close.
    """
    for attempt in range(1, CLOSE_MAX_ATTEMPTS + 1):
        try:
            tc.close_position(sym)
            if attempt > 1:
                log.info(f"Close {sym} succeeded on attempt {attempt}")
            return True
        except APIError as e:
            msg = str(e).lower()
            is_race = ("insufficient qty" in msg or '"available":"0"' in msg
                       or "held for orders" in msg)
            if is_race and attempt < CLOSE_MAX_ATTEMPTS:
                # Re-cancel anything still holding this symbol's qty.
                for o in _our_open_orders(tc, [sym]):
                    try:
                        tc.cancel_order_by_id(o.id)
                    except Exception:
                        pass
                log.info(f"Close {sym} attempt {attempt}: qty still held by pending "
                         f"cancels, retrying in {CLOSE_RETRY_DELAY_SEC}s")
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
    """Cancel open orders for our symbols, close any positions, then VERIFY flat.

    Order of operations matters: (1) cancel bracket legs, (2) poll until those
    cancels actually clear so the held qty is released, (3) market-close each
    position with a generous retry budget, (4) re-check positions and fire a
    high-priority alert if anything is left open (a naked position with no stop
    is the worst-case outcome — the user must know to intervene).
    """
    # 1. Cancel all open orders for our symbols.
    for o in _our_open_orders(tc, watchlist):
        log.info(f"EOD cancel: order {o.id} ({o.symbol} {o.side} qty={o.qty})")
        if not dry_run:
            try:
                tc.cancel_order_by_id(o.id)
            except APIError as e:
                log.warning(f"Cancel {o.id} failed: {e}")

    # 2. Poll until our open orders clear (qty released), up to a timeout.
    if not dry_run:
        deadline = time_mod.monotonic() + CANCEL_CONFIRM_TIMEOUT_SEC
        while time_mod.monotonic() < deadline:
            remaining = _our_open_orders(tc, watchlist)
            if not remaining:
                break
            log.info(f"Waiting for {len(remaining)} order cancel(s) to clear...")
            time_mod.sleep(CANCEL_CONFIRM_POLL_SEC)
        else:
            log.warning("Cancellations did not fully clear within "
                        f"{CANCEL_CONFIRM_TIMEOUT_SEC}s; closing anyway (with retries).")

    # 3. Close positions (notify once each), with generous retry.
    for p in _our_positions(tc, watchlist):
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
            _close_position_with_retry(tc, p.symbol, watchlist)

    # 4. VERIFY flat. close_position only SUBMITS a market sell; it doesn't wait
    #    for the fill, so positions still show as held for a few seconds. Poll
    #    until they clear before alerting — otherwise every busy EOD false-alarms
    #    (observed 2026-05-28: ADBE/CRM/MSFT flagged "INCOMPLETE" but had filled
    #    seconds later). Only a position still open AFTER the poll window is a
    #    genuine naked-position emergency.
    if not dry_run:
        leftover = _our_positions(tc, watchlist)
        if leftover:
            deadline = time_mod.monotonic() + FLATTEN_VERIFY_TIMEOUT_SEC
            while leftover and time_mod.monotonic() < deadline:
                time_mod.sleep(FLATTEN_VERIFY_POLL_SEC)
                leftover = _our_positions(tc, watchlist)
                if leftover:
                    log.info(f"Waiting for {len(leftover)} close order(s) to fill...")
        if leftover:
            desc = ", ".join(f"{p.symbol} {p.side} {p.qty}" for p in leftover)
            log.error(f"EOD FLATTEN INCOMPLETE — still holding: {desc}. "
                      f"These positions have NO stop attached (legs were cancelled).")
            try:
                notify(
                    f"EOD flatten INCOMPLETE. Still holding: {desc}.\n"
                    f"Protective brackets were cancelled — these positions are "
                    f"UNHEDGED. Close them manually in Alpaca ASAP.",
                    title="ORB EOD FLATTEN FAILED",
                    priority=5,
                    tags=["rotating_light"],
                )
            except Exception as e:
                log.warning(f"Naked-position alert notification failed: {e}")
        else:
            log.info("EOD flatten verified: no open positions remain in watchlist.")


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
        "halt_reason": run.halt_reason,
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
        if st.exited and st.exit_price is not None:
            pnl_str = (f" PnL ${st.realized_pnl:+,.0f}"
                       if st.realized_pnl is not None else "")
            status = f"EXITED {st.side} {st.exit_reason} @ ${st.exit_price:.2f}{pnl_str}"
        elif st.entered and st.entry_order_id and not st.reject_reason:
            status = f"ENTERED {st.side} (bracket live on Alpaca)"
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
            "side": st.side,
            "entered": st.entered, "exited": st.exited, "status": status,
            "entry_price": st.entry_price, "stop_price": st.stop_price,
            "target_price": st.target_price, "shares": st.shares,
            "exit_price": st.exit_price, "exit_reason": st.exit_reason,
            "realized_pnl": st.realized_pnl,
        }
    return snap


def _emit_heartbeat(run: "RunState", watchlist: list[str], today: date,
                    phase: str, valid_for_s: float, dry_run: bool) -> None:
    """Write one liveness beat. Cheap (no API call) — counts from run state only.

    valid_for_s: how long this beat stays fresh. The status server treats
    now > (this beat's ts + valid_for_s) as "no fresh beat". Pass the poll
    interval (+grace) in the loop, or the seconds-until-open during the wait.
    """
    open_positions = sum(
        1 for s in run.states.values()
        if s.entered and s.entry_order_id and not s.reject_reason and not s.exited
    )
    heartbeat.write(
        time_mod.time() + valid_for_s,
        phase=phase,
        halted=run.halted,
        halt_reason=run.halt_reason or "",
        session_date=today.isoformat(),
        open_positions=open_positions,
        dry_run=dry_run,
    )


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
    # Catch cross-day or manual positions that don't have today's orb-* coid.
    # Blocks double-entry on already-held symbols (e.g., yesterday's QQQ that
    # rode overnight because the OPG sell expired).
    detect_untracked_positions(tc, run, watchlist)

    # Short-side regime gate: decided once, pre-open, from prior SPY closes.
    run.shorts_enabled, run.regime_note = compute_short_regime(dc, today)
    log.info(f"Regime gate: {run.regime_note}")
    if run.shorts_enabled:
        log.info(f"Short-eligible symbols today: {sorted(SHORT_SYMBOLS & set(watchlist))}")

    # Trend filter (multi-timeframe momentum confirmation): decide which names
    # are eligible for LONG entries today from prior daily closes. None = filter
    # disabled or fetch failed -> treat all as eligible (baseline behaviour).
    if TREND_FILTER_ENABLED:
        run.trend_eligible = compute_trend_eligibility(dc, watchlist)
    else:
        log.info("Trend filter: DISABLED in config.")
        run.trend_eligible = None

    # Vol-regime risk dial: in a high-vol regime, scale risk down for the whole
    # session (reactive — sits out the turbulent aftermath; see compare_volpause.py).
    run.risk_mult = 1.0
    vol_line = ""
    if VOL_REGIME_FILTER_ENABLED:
        high_vol = compute_vol_regime(dc, today)
        if high_vol:
            run.risk_mult = VOL_REGIME_RISK_MULT
            eff = PARAMS.risk_per_trade * VOL_REGIME_RISK_MULT
            log.warning(f"Vol regime: HIGH-VOL -> risk dialed to {VOL_REGIME_RISK_MULT:.0%} "
                        f"(${eff:.0f}/trade this session).")
            vol_line = f"\nVol regime: HIGH — risk dialed to {VOL_REGIME_RISK_MULT:.0%} (${eff:.0f}/trade)."
        else:
            state_txt = "unknown (fail-safe)" if high_vol is None else "calm"
            log.info(f"Vol regime: {state_txt} -> full risk.")

    open_dt = combine_et(today, RTH_OPEN)
    or_end_dt = open_dt + timedelta(minutes=PARAMS.or_minutes)
    eod_dt = combine_et(today, EOD_FLAT_TIME)
    close_dt = combine_et(today, RTH_CLOSE)

    # If we're starting after the OR window has already closed (restart or
    # late fire), lock OR levels from today's historical bars now so the UI
    # shows them immediately.
    prebuild_or_if_late(dc, run, watchlist, today, or_end_dt)

    log.info(f"Today: open={open_dt.strftime('%H:%M %Z')}  "
             f"OR ends={or_end_dt.strftime('%H:%M')}  "
             f"EOD flat={eod_dt.strftime('%H:%M')}  "
             f"close={close_dt.strftime('%H:%M')}")

    # Phone notification: session is up. Sent once per run, before waiting for open.
    shorts_line = (f"Shorts ON ({', '.join(sorted(SHORT_SYMBOLS & set(watchlist)))})"
                   if run.shorts_enabled else "Shorts OFF (regime not bearish)")
    notify(
        f"Account ${run.starting_equity:,.0f}. Waiting for 09:30 ET market open. "
        f"Watchlist: {', '.join(watchlist)}.\n{shorts_line}.{vol_line}",
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

    # First beat before the pre-open wait blocks. Keep it fresh until just past
    # the open so the status server shows "starting up" rather than "DOWN"
    # during the 09:25->09:30 gap when nothing else is running.
    _emit_heartbeat(run, watchlist, today,
                    phase="pre-market (waiting for 9:30 ET open)",
                    valid_for_s=max(0.0, (open_dt - datetime.now(ET)).total_seconds()) + 60,
                    dry_run=dry_run)

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
        run.halt_reason = f"late-start (+{late_by_min}m)"
        if ui is not None:
            ui.set_state("warning")
        try:
            notify(
                f"Started {late_by_min} min after OR window closed "
                f"(threshold {LATE_START_CUTOFF_MINUTES}m). "
                f"NEW entries halted for today; existing positions still EOD-flatten at 15:55 ET.",
                title="ORB late-start halt",
                priority=4,
                tags=["warning"],
            )
        except Exception as e:
            log.warning(f"Late-start notification failed: {e}")

    while True:
        now = datetime.now(ET)
        # Liveness beat — written every iteration before any work that could
        # raise/continue, so a stuck or crashed loop stops refreshing it and
        # the status server flips to DOWN. Valid for one poll interval + grace.
        _emit_heartbeat(run, watchlist, today,
                        phase=_phase_for(now, open_dt, or_end_dt, eod_dt),
                        valid_for_s=POLL_SECONDS + 45, dry_run=dry_run)
        if now >= eod_dt:
            log.info("EOD flat time reached. Flattening.")
            flatten_all(tc, watchlist, dry_run, run=run)
            _send_eod_notification(tc, run, watchlist, today)
            return
        if now >= close_dt:
            log.info("Past market close. Exiting.")
            _send_eod_notification(tc, run, watchlist, today)
            return

        # Time-of-day entry cutoff (halts NEW entries only; existing trades ride)
        if (not run.halted and PARAMS.no_entry_after_time is not None
                and now.time() >= PARAMS.no_entry_after_time):
            cutoff_str = PARAMS.no_entry_after_time.strftime("%H:%M")
            log.info(f"Past no-entry cutoff {cutoff_str} ET; halting new entries.")
            run.halted = True
            run.halt_reason = f"cutoff ({cutoff_str} ET)"
            if ui is not None:
                ui.set_state("warning")
            try:
                notify(
                    f"Past {cutoff_str} ET no-entry cutoff. NEW entries halted; "
                    f"existing positions ride their brackets to EOD-flat at 15:55 ET.",
                    title="ORB cutoff halt",
                    priority=3,
                    tags=["alarm_clock"],
                )
            except Exception as e:
                log.warning(f"Cutoff notification failed: {e}")

        # Daily-loss circuit breaker (halts NEW entries only)
        if not run.halted:
            pnl = realized_pnl_today(tc, today)
            if pnl <= -DAILY_LOSS_CAP:
                log.warning(f"Daily loss cap hit: realized PnL ${pnl:+,.2f}. Halting new entries.")
                run.halted = True
                run.halt_reason = f"loss-cap (${pnl:+,.0f})"
                if ui is not None:
                    ui.set_state("halted")
                try:
                    notify(
                        f"Daily loss cap hit: realized PnL ${pnl:+,.2f} (cap ${DAILY_LOSS_CAP:,.0f}).\n"
                        f"NEW entries halted; existing positions ride their brackets.",
                        title="ORB loss-cap halt",
                        priority=5,
                        tags=["octagonal_sign"],
                    )
                except Exception as e:
                    log.warning(f"Loss-cap notification failed: {e}")

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
            state.last_close = last_close

            # Direction of breakout. Longs (close > OR_high) are always eligible.
            # Shorts (close < OR_low) only for short-eligible names when today's
            # regime gate is bearish (run.shorts_enabled). Stop/target mirror the
            # backtest: long stop=OR_low/target above; short stop=OR_high/target below.
            short_ok = run.shorts_enabled and sym in SHORT_SYMBOLS
            entry_estimate = last_close
            if last_close > state.or_high:
                side, side_name = OrderSide.BUY, "long"
                stop = state.or_low
                target = entry_estimate + PARAMS.target_r * (entry_estimate - stop)
                ref_label, ref_val, arrow = "OR_high", state.or_high, ">"
            elif short_ok and last_close < state.or_low:
                side, side_name = OrderSide.SELL, "short"
                stop = state.or_high
                target = entry_estimate - PARAMS.target_r * (stop - entry_estimate)
                ref_label, ref_val, arrow = "OR_low", state.or_low, "<"
            else:
                continue

            # Tight-OR filter: skip breakouts whose opening range is too WIDE relative
            # to price (> TIGHT_OR_MAX_PCT). Wide-OR = far stop, weak follow-through,
            # unreliable edge; the validated edge lives in tight ranges + the trailing
            # exit (biggest single win, 2026-06-09). Direction-agnostic (OR width is).
            if TIGHT_OR_FILTER_ENABLED and state.or_high is not None and state.or_low is not None:
                or_width_pct = ((state.or_high - state.or_low) / entry_estimate * 100.0
                                if entry_estimate else 0.0)
                if or_width_pct > TIGHT_OR_MAX_PCT:
                    log.info(f"{sym} {side_name} breakout SKIPPED: OR width {or_width_pct:.2f}% "
                             f"> {TIGHT_OR_MAX_PCT:.2f}% (tight-OR filter).")
                    state.entered = True
                    state.side = side_name
                    state.reject_reason = "OR too wide"
                    continue

            # Trend filter (longs only): skip names that didn't pass the daily
            # SMA + relative-strength check at session start. run.trend_eligible
            # is None when the filter is off or the daily fetch failed
            # (fail-open -> baseline behaviour).
            if (side_name == "long" and TREND_FILTER_ENABLED
                    and run.trend_eligible is not None
                    and sym not in run.trend_eligible):
                log.info(f"{sym} long breakout SKIPPED: not in trend-eligible set today.")
                state.entered = True
                state.side = side_name
                state.reject_reason = "trend filter"
                continue

            # Concurrency cap: with a broad watchlist, many names break out near
            # the open. Don't exceed MAX_CONCURRENT_POSITIONS open at once. When
            # full, this breakout is skipped for the day (first-come, matching
            # the greedy fill in backtest/universe_portfolio.py).
            if MAX_CONCURRENT_POSITIONS is not None:
                open_now = sum(
                    1 for s in run.states.values()
                    if s.entered and s.entry_order_id and not s.exited and not s.reject_reason
                )
                if open_now >= MAX_CONCURRENT_POSITIONS:
                    log.info(f"{sym} {side_name} breakout SKIPPED: at concurrency cap "
                             f"({open_now}/{MAX_CONCURRENT_POSITIONS} open).")
                    state.entered = True
                    state.side = side_name
                    state.reject_reason = "cap full"
                    continue

            equity = float(tc.get_account().equity)
            qty, reason = size_position(entry_estimate, stop, equity, run.risk_mult)

            log.info(f"BREAKOUT {sym} {side_name.upper()}: close=${last_close:.2f} {arrow} "
                     f"{ref_label}=${ref_val:.2f}  -> entry~=${entry_estimate:.2f} "
                     f"stop=${stop:.2f} target=${target:.2f}  qty={qty} (sizing: {reason})")

            if qty == 0:
                log.warning(f"{sym} REJECTED: {reason}")
                state.entered = True  # don't keep retrying a rejected setup
                state.side = side_name
                state.reject_reason = reason
                try:
                    notify(
                        f"{sym} {side_name} breakout REJECTED: {reason}\n"
                        f"entry~${entry_estimate:.2f}  stop ${stop:.2f}  target ${target:.2f}",
                        title=f"ORB rejected: {sym}",
                        tags=["no_entry"],
                    )
                except Exception as e:
                    log.warning(f"Reject notification failed: {e}")
                continue

            # Open-risk cap: never let the book's total initial risk exceed the
            # rail, no matter what sizing said. At the auto default (risk x
            # concurrency cap) this only trips on bugs — which is the point.
            if MAX_OPEN_RISK is not None:
                cand_risk = abs(entry_estimate - stop) * qty
                cur_risk = open_risk_now(run)
                if cur_risk + cand_risk > MAX_OPEN_RISK + 1e-6:
                    log.warning(f"{sym} {side_name} breakout BLOCKED by open-risk cap: "
                                f"${cur_risk:,.0f} open + ${cand_risk:,.0f} new > "
                                f"${MAX_OPEN_RISK:,.0f} rail.")
                    state.entered = True
                    state.side = side_name
                    state.reject_reason = "open-risk cap"
                    try:
                        notify(
                            f"{sym} {side_name} entry BLOCKED by the open-risk cap:\n"
                            f"${cur_risk:,.0f} already at risk + ${cand_risk:,.0f} new "
                            f"> ${MAX_OPEN_RISK:,.0f}.\n"
                            f"At the auto rail this should only happen on a sizing "
                            f"bug or config mistake — check the log.",
                            title=f"ORB open-risk cap: {sym}",
                            priority=4, tags=["rotating_light"],
                        )
                    except Exception as e:
                        log.warning(f"Open-risk-cap notification failed: {e}")
                    continue

            oid = submit_bracket(tc, sym, qty, entry_estimate, stop, target, side, today, dry_run)
            state.entered = True
            state.side = side_name
            state.entry_order_id = oid
            state.entry_price = entry_estimate
            state.entry_estimate = entry_estimate  # frozen for slippage comparison
            state.stop_price = stop
            if TRAILING_EXIT_ENABLED and side == OrderSide.BUY:
                # No fixed target; trailing stop (= initial risk/share) attached on fill.
                state.trail_amount = round(abs(entry_estimate - stop), 2)
                state.target_price = None
            else:
                state.target_price = target
            state.shares = qty

        # One-time OR-locked summary push (after the OR window has closed and
        # at least one symbol has locked).
        if (not run.or_lock_notified
                and now >= or_end_dt
                and any(s.or_locked for s in run.states.values())):
            run.or_lock_notified = True
            try:
                lines = []
                for sym in watchlist:
                    s = run.states[sym]
                    if s.or_locked and s.or_high is not None and s.or_low is not None:
                        rng = s.or_high - s.or_low
                        lines.append(f"{sym}: ${s.or_low:.2f} – ${s.or_high:.2f}  (range ${rng:.2f})")
                    else:
                        lines.append(f"{sym}: no OR data")
                halt_note = "  (NEW entries halted)" if run.halted else ""
                dir_note = ("long breakouts (OR-high) + short breakdowns (OR-low) on "
                            f"{', '.join(sorted(SHORT_SYMBOLS & set(watchlist)))}"
                            if run.shorts_enabled else "long breakouts above OR-high")
                notify(
                    f"Opening range locked. Watching for {dir_note}"
                    f"{halt_note}.\n\n" + "\n".join(lines),
                    title=f"ORB OR locked ({today.isoformat()})",
                    tags=["lock"],
                )
            except Exception as e:
                log.warning(f"OR-lock notification failed: {e}")

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
             f"loss_cap=${DAILY_LOSS_CAP} "
             f"open_risk_cap={'$' + format(MAX_OPEN_RISK, ',.0f') if MAX_OPEN_RISK else 'OFF'}")
    if SHORT_ENABLED:
        log.info(f"Shorts: regime-gated (SPY < SMA{REGIME_SMA_WINDOW} x{REGIME_CONFIRM_DAYS} days), "
                 f"symbols={sorted(SHORT_SYMBOLS)}, flips=0")
    else:
        log.info("Shorts: DISABLED (SHORT_ENABLED=False)")
    log.info(f"Tight-OR filter: {'ON (max OR ' + format(TIGHT_OR_MAX_PCT, '.2f') + '% of price)' if TIGHT_OR_FILTER_ENABLED else 'OFF'}"
             f"  |  Trend filter: {'ON' if TREND_FILTER_ENABLED else 'OFF'}")

    load_env()
    try:
        tc, dc = build_clients()
    except RuntimeError as e:
        log.error(str(e))
        return 1

    today = datetime.now(ET).date()
    if args.preflight_only:
        return smoke_test(tc, dc, today)

    # Tolerate a brief 'laptop just woke / wifi still joining' startup lag at
    # the 06:15 fire (2026-05-29 incident). 5 attempts x 30s = up to 2.5 min.
    if not wait_for_network(tc):
        return 1

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
                f"Check logs at {LOG_DIR}",
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
