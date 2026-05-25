"""Opening Range Breakout — pure strategy logic, no I/O.

Same module powers backtest and live execution.

Rules (long-only MVP):
  - Opening range: first 15 minutes of RTH (9:30-9:44:59 ET).
  - Entry: market buy on the NEXT bar's open after a 1-min bar CLOSES above OR high.
  - Stop: OR low.
  - Target: entry + target_r * (entry - stop).  Default target_r = 2.0.
  - Time stop: flat at/after eod_flat (default 15:55 ET) on that bar's close.
  - One shot per symbol per session. No re-entries.
  - Optional: when price has moved +move_stop_to_be_at_r * initial_risk in our favor,
    lift the stop to entry (lag one bar — checked at end of bar, applies to next).

Sizing:
  - shares = floor(risk_per_trade / (entry - stop))
  - Capped so that shares * entry <= min(equity * max_position_pct, max_position_dollars).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import time
from typing import Optional

import pandas as pd  # noqa: F401  (re-exported for callers)

RTH_OPEN = time(9, 30)
EOD_FLAT_TIME = time(15, 55)


@dataclass(frozen=True)
class Params:
    or_minutes: int = 15
    target_r: float = 2.0
    risk_per_trade: float = 100.0
    max_position_pct: float = 0.25
    max_position_dollars: Optional[float] = None  # absolute cap; if set, combines with pct cap
    move_stop_to_be_at_r: Optional[float] = None  # e.g. 1.0 = lift stop to entry once +1R reached
    eod_flat: time = EOD_FLAT_TIME


@dataclass(frozen=True)
class Trade:
    symbol: str
    date: pd.Timestamp
    or_high: float
    or_low: float
    entry_time: pd.Timestamp
    entry_price: float
    stop_price: float
    target_price: float
    shares: int
    risk_dollars: float
    exit_time: pd.Timestamp
    exit_price: float
    exit_reason: str
    pnl_dollars: float
    pnl_r: float


def size_shares(entry: float, stop: float, equity: float, p: Params) -> int:
    risk_per_share = entry - stop
    if risk_per_share <= 0:
        return 0
    shares_by_risk = math.floor(p.risk_per_trade / risk_per_share)
    cap_dollars = equity * p.max_position_pct
    if p.max_position_dollars is not None:
        cap_dollars = min(cap_dollars, p.max_position_dollars)
    shares_by_cap = math.floor(cap_dollars / entry)
    return max(0, min(shares_by_risk, shares_by_cap))


def simulate_day(
    bars: pd.DataFrame,
    symbol: str,
    equity: float,
    p: Params = Params(),
) -> Optional[Trade]:
    """Run ORB on one symbol's intraday bars for one session.

    bars: 1-min RTH bars with tz-aware ET DatetimeIndex; columns open/high/low/close/volume.
    """
    if bars.empty:
        return None

    tz = bars.index.tz
    session_date = bars.index[0].normalize().tz_localize(None)
    session_open = pd.Timestamp.combine(session_date.date(), RTH_OPEN).tz_localize(tz)
    or_end = session_open + pd.Timedelta(minutes=p.or_minutes)
    eod_cutoff = pd.Timestamp.combine(session_date.date(), p.eod_flat).tz_localize(tz)

    or_bars = bars[bars.index < or_end]
    if or_bars.empty:
        return None
    or_high = float(or_bars["high"].max())
    or_low = float(or_bars["low"].min())

    post_or = bars[bars.index >= or_end]
    if post_or.empty:
        return None

    breakout_idx = None
    for ts, row in post_or.iterrows():
        if row["close"] > or_high:
            breakout_idx = ts
            break
    if breakout_idx is None:
        return None

    after_breakout = post_or[post_or.index > breakout_idx]
    if after_breakout.empty:
        return None

    entry_ts = after_breakout.index[0]
    entry_price = float(after_breakout.iloc[0]["open"])
    stop_price = or_low
    target_price = entry_price + p.target_r * (entry_price - stop_price)

    shares = size_shares(entry_price, stop_price, equity, p)
    if shares == 0:
        return None
    risk_dollars = shares * (entry_price - stop_price)

    exit_ts = None
    exit_price = None
    exit_reason = None
    current_stop = stop_price
    be_triggered = False

    for ts, row in after_breakout.iterrows():
        if ts >= eod_cutoff:
            exit_ts, exit_price, exit_reason = ts, float(row["close"]), "eod"
            break

        op = float(row["open"])
        high = float(row["high"])
        low = float(row["low"])

        # Gap-through fills (apply only after the entry bar).
        if ts != entry_ts:
            if op <= current_stop:
                exit_ts, exit_price, exit_reason = ts, op, "stop"
                break
            if op >= target_price:
                exit_ts, exit_price, exit_reason = ts, op, "target"
                break

        # Same-bar both-touched: conservatively assume stop hit first.
        if low <= current_stop:
            exit_ts, exit_price, exit_reason = ts, current_stop, "stop"
            break
        if high >= target_price:
            exit_ts, exit_price, exit_reason = ts, target_price, "target"
            break

        # End-of-bar: lift stop to break-even if this bar's high crossed the BE trigger.
        if p.move_stop_to_be_at_r is not None and not be_triggered:
            be_trigger_price = entry_price + p.move_stop_to_be_at_r * (entry_price - stop_price)
            if high >= be_trigger_price:
                current_stop = entry_price
                be_triggered = True

    if exit_ts is None:
        last_ts = after_breakout.index[-1]
        exit_ts = last_ts
        exit_price = float(after_breakout.iloc[-1]["close"])
        exit_reason = "eod"

    pnl_dollars = (exit_price - entry_price) * shares
    pnl_r = (exit_price - entry_price) / (entry_price - stop_price)

    return Trade(
        symbol=symbol,
        date=session_date,
        or_high=or_high,
        or_low=or_low,
        entry_time=entry_ts,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        shares=shares,
        risk_dollars=risk_dollars,
        exit_time=exit_ts,
        exit_price=exit_price,
        exit_reason=exit_reason,
        pnl_dollars=pnl_dollars,
        pnl_r=pnl_r,
    )
