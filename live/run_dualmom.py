"""Monthly dual-momentum rebalance runner (Alpaca paper).

The "bedrock" sleeve from the strategy research (backtest/run_trend.py +
trend_robustness.py): blended-lookback (3/6/12-month) momentum, hold the top-3
risk-ETFs that beat cash, equal-weight, else cash. ~0.9 Sharpe / -13% maxDD /
crisis-alpha in the 2007-2026 backtest. Monthly cadence -> trivial to automate.

Behaviour:
  - Self-gating: only rebalances on the FIRST trading day of the month (or with
    --force). Safe to fire every weekday morning via Task Scheduler; no-ops
    otherwise. So it never double-rebalances and never misses the month start.
  - Signal from Alpaca DAILY bars (adjustment=all), last completed month-end.
  - Rebalances ONLY its own universe symbols (never touches ORB positions or the
    BTC orphan). Sells/reductions first, brief pause, then buys (notional).
  - --dry-run prints the plan and places nothing.

  # Windows
  .venv\\Scripts\\python.exe live\\run_dualmom.py --dry-run --force   # test now
  .venv\\Scripts\\python.exe live\\run_dualmom.py                      # live, self-gated
  # Linux
  ./.venv/bin/python live/run_dualmom.py --dry-run --force
  ./.venv/bin/python live/run_dualmom.py

ACCOUNT NOTE: do NOT run this in the same Alpaca account as the ORB bot if their
symbol universes overlap (SPY/QQQ): ORB's 15:55 ET EOD-flatten liquidates any
position in its watchlist, which would wipe this sleeve's holdings daily. Point
it at a SEPARATE paper account via DUALMOM_ALPACA_API_KEY / _SECRET_KEY, or use a
non-overlapping universe. See the deployment discussion.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time as time_mod
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    GetCalendarRequest, MarketOrderRequest, ClosePositionRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from live.notify import notify  # noqa: E402

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

# Universe (matches the validated backtest). SHY = cash/safe harbour.
RISK = ["SPY", "QQQ", "EFA", "EEM", "VNQ", "GLD", "DBC", "TLT", "AGG"]
CASH = "SHY"
LOOKBACKS_M = (3, 6, 12)     # blended-lookback months
TOPK = 3
MIN_TRADE = 50.0             # skip rebalance deltas under $50 (noise)
SELL_SETTLE_SEC = 3.0        # pause after sells so buying power frees up
DUALMOM_CAPITAL = float(os.environ.get("DUALMOM_CAPITAL", "50000"))

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
log = logging.getLogger("dualmom")


def setup_logging(dry_run: bool) -> None:
    today = datetime.now(ET).strftime("%Y-%m-%d")
    path = LOG_DIR / f"dualmom_{today}{'_dryrun' if dry_run else ''}.log"
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S")
    log.setLevel(logging.INFO)
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(path, encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.info(f"Log: {path}")


def load_env() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def build_clients():
    # Prefer a dedicated dual-momentum account if its keys are set, else fall
    # back to the main Alpaca keys (with a loud warning about the ORB conflict).
    key = os.environ.get("DUALMOM_ALPACA_API_KEY") or os.environ.get("ALPACA_API_KEY")
    sec = os.environ.get("DUALMOM_ALPACA_SECRET_KEY") or os.environ.get("ALPACA_SECRET_KEY")
    dedicated = bool(os.environ.get("DUALMOM_ALPACA_API_KEY"))
    if not key or not sec:
        raise RuntimeError("Alpaca keys missing from .env")
    if not dedicated:
        log.warning("Using MAIN Alpaca account (no DUALMOM_ALPACA_API_KEY set). "
                    "If the ORB bot runs here too, its EOD-flatten will liquidate "
                    "SPY/QQQ holdings — use a separate paper account for this sleeve.")
    tc = TradingClient(key, sec, paper=True)
    dc = StockHistoricalDataClient(key, sec)
    return tc, dc, dedicated


def is_first_trading_day(tc: TradingClient, today) -> bool:
    first = today.replace(day=1)
    cal = tc.get_calendar(GetCalendarRequest(start=first, end=today))
    return len(cal) > 0 and cal[0].date == today


def fetch_monthly_closes(dc: StockHistoricalDataClient, symbols):
    """Daily adjusted closes -> completed month-end closes (wide frame)."""
    start = (datetime.now(ET) - timedelta(days=520)).astimezone(UTC)
    end = datetime.now(ET).astimezone(UTC)
    req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame.Day,
                           start=start, end=end, feed=DataFeed.IEX,
                           adjustment=Adjustment.ALL)  # total-return basis (divs+splits)
    df = dc.get_stock_bars(req).df
    if df.empty:
        raise RuntimeError("No daily bars returned")
    close = df["close"].unstack(level=0)  # index=ts, columns=symbol
    close.index = pd.to_datetime(close.index).tz_convert(ET).normalize()
    monthly = close.resample("ME").last().dropna(how="all")
    now = datetime.now(ET)
    if monthly.index[-1].month == now.month and monthly.index[-1].year == now.year:
        monthly = monthly.iloc[:-1]  # drop the incomplete current month
    return monthly


def target_weights(monthly: pd.DataFrame) -> dict:
    """Blended (3/6/12mo) momentum; top-K risk assets that beat cash, equal
    weight; unfilled slots -> cash."""
    if len(monthly) <= max(LOOKBACKS_M):
        raise RuntimeError("Not enough monthly history for momentum")
    moms = [monthly.iloc[-1] / monthly.iloc[-1 - n] - 1 for n in LOOKBACKS_M]
    blend = sum(moms) / len(moms)
    cash_mom = blend.get(CASH, 0.0)
    scores = {s: blend[s] for s in RISK if s in blend.index and pd.notna(blend[s])}
    eligible = {s: v for s, v in scores.items() if v > cash_mom}
    top = sorted(eligible, key=eligible.get, reverse=True)[:TOPK]
    w = {s: 0.0 for s in RISK + [CASH]}
    for s in top:
        w[s] = 1.0 / TOPK
    w[CASH] = 1.0 - len(top) / TOPK
    return w, blend


def rebalance(tc, weights, capital, dry_run):
    universe = set(RISK) | {CASH}
    positions = {p.symbol: float(p.market_value) for p in tc.get_all_positions()
                 if p.symbol in universe}
    plan = []
    # 1) reductions / exits first
    for sym in sorted(universe):
        cur = positions.get(sym, 0.0)
        tgt = weights.get(sym, 0.0) * capital
        if cur - tgt > MIN_TRADE:
            if tgt <= MIN_TRADE:
                plan.append((sym, "EXIT", cur, 0.0))
            else:
                pct = round((cur - tgt) / cur * 100, 2)
                plan.append((sym, f"REDUCE {pct:.1f}%", cur, tgt))
    # 2) buys
    for sym in sorted(universe):
        cur = positions.get(sym, 0.0)
        tgt = weights.get(sym, 0.0) * capital
        if tgt - cur > MIN_TRADE:
            plan.append((sym, "BUY", cur, tgt))

    log.info(f"Rebalance plan (capital ${capital:,.0f}, {len(plan)} actions):")
    for sym, action, cur, tgt in plan:
        log.info(f"  {sym:<5} {action:<12} cur ${cur:>9,.0f} -> tgt ${tgt:>9,.0f}")
    if not plan:
        log.info("  (already at target — nothing to do)")
    if dry_run:
        log.info("DRY-RUN: no orders placed.")
        return plan

    # Execute: sells/reductions first, settle, then buys.
    did_sell = False
    for sym, action, cur, tgt in plan:
        if action.startswith("EXIT") or action.startswith("REDUCE"):
            try:
                if action == "EXIT":
                    tc.close_position(sym)
                else:
                    pct = (cur - tgt) / cur * 100
                    tc.close_position(sym, ClosePositionRequest(percentage=str(round(pct, 2))))
                did_sell = True
                log.info(f"  submitted {action} {sym}")
            except Exception as e:
                log.warning(f"  {action} {sym} failed: {e}")
    if did_sell:
        time_mod.sleep(SELL_SETTLE_SEC)
    for sym, action, cur, tgt in plan:
        if action == "BUY":
            notional = round(tgt - cur, 2)
            try:
                tc.submit_order(MarketOrderRequest(
                    symbol=sym, notional=notional, side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY))
                log.info(f"  submitted BUY {sym} ${notional:,.0f}")
            except Exception as e:
                log.warning(f"  BUY {sym} failed: {e}")
    return plan


def main() -> int:
    ap = argparse.ArgumentParser(description="Monthly dual-momentum rebalance")
    ap.add_argument("--dry-run", action="store_true", help="Print plan, place nothing.")
    ap.add_argument("--force", action="store_true", help="Bypass first-trading-day gate.")
    ap.add_argument("--preflight-only", action="store_true", help="Show target weights + exit.")
    args = ap.parse_args()

    setup_logging(args.dry_run)
    load_env()
    try:
        tc, dc, dedicated = build_clients()
    except RuntimeError as e:
        log.error(str(e))
        return 1

    today = datetime.now(ET).date()
    monthly = fetch_monthly_closes(dc, RISK + [CASH])
    weights, blend = target_weights(monthly)
    asof = monthly.index[-1].date()
    held = [f"{s} {w:.0%}" for s, w in weights.items() if w > 0]
    log.info(f"Signal as-of completed month-end {asof}; blended-momentum ranks:")
    for s in sorted(blend.index, key=lambda x: blend[x], reverse=True):
        mark = " <-- HOLD" if weights.get(s, 0) > 0 else ""
        log.info(f"  {s:<5} {blend[s]:+7.2%}{mark}")
    log.info(f"Target: {', '.join(held)}")

    if args.preflight_only:
        return 0

    if not args.force and not is_first_trading_day(tc, today):
        log.info(f"{today} is not the first trading day of the month — no rebalance. "
                 f"(use --force to override)")
        return 0

    # Safety: never place live dual-momentum orders in the shared ORB account.
    # Live trading requires a dedicated DUALMOM_ALPACA_API_KEY (separate account).
    if not args.dry_run and not dedicated:
        log.error("REFUSING live orders: no dedicated DUALMOM_ALPACA_API_KEY set, so "
                  "this would trade in the shared ORB account (whose EOD-flatten would "
                  "then liquidate the holdings). Add the dedicated key to .env, or use "
                  "--dry-run.")
        return 2

    acct = tc.get_account()
    capital = min(float(acct.equity), DUALMOM_CAPITAL)
    log.info(f"Account equity ${float(acct.equity):,.0f}; sleeve capital ${capital:,.0f} "
             f"({'dedicated acct' if dedicated else 'SHARED acct — see warning'})")
    plan = rebalance(tc, weights, capital, args.dry_run)

    if not args.dry_run:
        try:
            lines = [f"Dual-momentum rebalance ({asof}):", f"Target: {', '.join(held)}"]
            if plan:
                lines += [f"{s} {a}" for s, a, _, _ in plan]
            else:
                lines.append("No trades (already at target).")
            notify("\n".join(lines), title="Dual-momentum rebalanced",
                   tags=["arrows_counterclockwise"])
        except Exception as e:
            log.warning(f"ntfy failed: {e}")
    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
