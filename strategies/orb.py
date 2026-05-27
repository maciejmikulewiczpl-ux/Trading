"""Opening Range Breakout — pure strategy logic, no I/O.

Same module powers backtest and live execution.

Rules:
  - Opening range: first 15 minutes of RTH (9:30-9:44:59 ET).
  - Long entry: market buy on the NEXT bar's open after a 1-min bar CLOSES above OR high.
    Stop: OR low (minus stop_offset buffer). Target: entry + target_r * (entry - stop).
  - Short entry: market sell on the NEXT bar's open after a 1-min bar CLOSES below OR low.
    Stop: OR high (plus stop_offset buffer). Target: entry - target_r * (stop - entry).
  - Direction toggles via enable_long / enable_short. First breakout in either direction wins.
  - Time stop: flat at/after eod_flat (default 15:55 ET) on that bar's close.
  - Flips: with max_flips > 0, a stop-out may be followed by an opposite-direction entry
    later the same session (see Params.max_flips). One shot per session when max_flips == 0.
  - Optional: when price has moved +move_stop_to_be_at_r * initial_risk in our favor,
    lift the stop to entry (lag one bar — checked at end of bar, applies to next).

Sizing:
  - shares = floor(risk_per_trade / abs(entry - stop))
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
    # No new entries after this time-of-day (existing trades still ride to eod_flat).
    # ORB breakouts after the first ~90 min of the session have weaker follow-through.
    no_entry_after_time: Optional[time] = None
    # Stop buffer beyond the OR boundary as a fraction of the OR range. 0.0 = no
    # buffer (stop is exactly OR_low for longs / OR_high for shorts, current
    # default). 0.10 = stop is 10% of the OR range beyond the boundary —
    # reduces "death by liquidity sweep" at the exact round number every retail
    # trader is watching.
    stop_offset_pct: float = 0.0
    # Direction toggles. A long enters on a close above OR_high; a short enters
    # on a close below OR_low. With both enabled, the first breakout in either
    # direction is taken. Shorts are OPT-IN: the default is long-only so every
    # existing backtest/compare_*.py keeps its current numbers unchanged.
    enable_long: bool = True
    enable_short: bool = False
    # Number of opposite-direction re-entries allowed after a stop-out within
    # the same session. 0 = one shot per session (legacy behavior); 1 = if the
    # first trade stops out, the opposite breakout may trigger once more. Flips
    # are never taken after a target or EOD exit.
    max_flips: int = 0


@dataclass(frozen=True)
class Trade:
    symbol: str
    date: pd.Timestamp
    side: str  # "long" or "short"
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
    risk_per_share = abs(entry - stop)
    if risk_per_share <= 0:
        return 0
    shares_by_risk = math.floor(p.risk_per_trade / risk_per_share)
    cap_dollars = equity * p.max_position_pct
    if p.max_position_dollars is not None:
        cap_dollars = min(cap_dollars, p.max_position_dollars)
    shares_by_cap = math.floor(cap_dollars / entry)
    return max(0, min(shares_by_risk, shares_by_cap))


def _simulate_one_trade(
    post_or: pd.DataFrame,
    symbol: str,
    equity: float,
    side: str,
    or_high: float,
    or_low: float,
    eod_cutoff: pd.Timestamp,
    session_date: pd.Timestamp,
    p: Params,
    start_after_ts: Optional[pd.Timestamp] = None,
    entry_cutoff_ts: Optional[pd.Timestamp] = None,
) -> Optional[Trade]:
    """Simulate one directional ORB trade.

    `post_or` is the FULL post-opening-range bar set (through eod_flat). The
    breakout search is limited to bars at/before `entry_cutoff_ts` (the
    no_entry_after_time gate), but once entered the trade is tracked over the
    full day so exits ride to eod_flat — matching live, where the cutoff blocks
    only new entries. Scans for the first breakout in `side` ("long" = close >
    or_high, "short" = close < or_low), optionally starting strictly after
    `start_after_ts` (used for flips). Returns the Trade or None. All long/short
    comparisons are mirrored; P&L is signed by direction.
    """
    long = side == "long"

    scan = post_or if start_after_ts is None else post_or[post_or.index > start_after_ts]
    if scan.empty:
        return None

    # Breakout (entry) search is bounded by the entry cutoff; exits are not.
    entry_scan = scan if entry_cutoff_ts is None else scan[scan.index <= entry_cutoff_ts]
    breakout_idx = None
    for ts, row in entry_scan.iterrows():
        c = row["close"]
        if (long and c > or_high) or (not long and c < or_low):
            breakout_idx = ts
            break
    if breakout_idx is None:
        return None

    after_breakout = scan[scan.index > breakout_idx]
    if after_breakout.empty:
        return None

    entry_ts = after_breakout.index[0]
    entry_price = float(after_breakout.iloc[0]["open"])
    or_range = or_high - or_low
    if long:
        stop_price = or_low - p.stop_offset_pct * or_range
        target_price = entry_price + p.target_r * (entry_price - stop_price)
    else:
        stop_price = or_high + p.stop_offset_pct * or_range
        target_price = entry_price - p.target_r * (stop_price - entry_price)

    shares = size_shares(entry_price, stop_price, equity, p)
    if shares == 0:
        return None
    risk_per_share = abs(entry_price - stop_price)
    risk_dollars = shares * risk_per_share

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

        # Gap-through fills (apply only after the entry bar). For a long, a gap
        # below the stop / above the target fills at the open; for a short the
        # inequalities flip (stop is above, target below).
        if ts != entry_ts:
            if long:
                if op <= current_stop:
                    exit_ts, exit_price, exit_reason = ts, op, "stop"
                    break
                if op >= target_price:
                    exit_ts, exit_price, exit_reason = ts, op, "target"
                    break
            else:
                if op >= current_stop:
                    exit_ts, exit_price, exit_reason = ts, op, "stop"
                    break
                if op <= target_price:
                    exit_ts, exit_price, exit_reason = ts, op, "target"
                    break

        # Same-bar both-touched: conservatively assume stop hit first.
        if long:
            if low <= current_stop:
                exit_ts, exit_price, exit_reason = ts, current_stop, "stop"
                break
            if high >= target_price:
                exit_ts, exit_price, exit_reason = ts, target_price, "target"
                break
        else:
            if high >= current_stop:
                exit_ts, exit_price, exit_reason = ts, current_stop, "stop"
                break
            if low <= target_price:
                exit_ts, exit_price, exit_reason = ts, target_price, "target"
                break

        # End-of-bar: lift stop to break-even once price has moved
        # +move_stop_to_be_at_r * initial_risk in our favor.
        if p.move_stop_to_be_at_r is not None and not be_triggered:
            be_move = p.move_stop_to_be_at_r * risk_per_share
            if long:
                if high >= entry_price + be_move:
                    current_stop = entry_price
                    be_triggered = True
            else:
                if low <= entry_price - be_move:
                    current_stop = entry_price
                    be_triggered = True

    if exit_ts is None:
        last_ts = after_breakout.index[-1]
        exit_ts = last_ts
        exit_price = float(after_breakout.iloc[-1]["close"])
        exit_reason = "eod"

    if long:
        pnl_dollars = (exit_price - entry_price) * shares
        pnl_r = (exit_price - entry_price) / (entry_price - stop_price)
    else:
        pnl_dollars = (entry_price - exit_price) * shares
        pnl_r = (entry_price - exit_price) / (stop_price - entry_price)

    return Trade(
        symbol=symbol,
        date=session_date,
        side=side,
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


def simulate_session(
    bars: pd.DataFrame,
    symbol: str,
    equity: float,
    p: Params = Params(),
) -> list[Trade]:
    """Run ORB on one symbol's intraday bars for one session; returns 0..N trades.

    bars: 1-min RTH bars with tz-aware ET DatetimeIndex; columns open/high/low/close/volume.

    The first trade is the earliest breakout across the enabled directions. If
    it stops out and flips remain (p.max_flips) and the opposite direction is
    enabled, the opposite breakout may re-enter later in the session — repeated
    until a non-stop exit, flips run out, or no further breakout occurs.
    """
    if bars.empty:
        return []

    tz = bars.index.tz
    session_date = bars.index[0].normalize().tz_localize(None)
    session_open = pd.Timestamp.combine(session_date.date(), RTH_OPEN).tz_localize(tz)
    or_end = session_open + pd.Timedelta(minutes=p.or_minutes)
    eod_cutoff = pd.Timestamp.combine(session_date.date(), p.eod_flat).tz_localize(tz)

    or_bars = bars[bars.index < or_end]
    if or_bars.empty:
        return []
    or_high = float(or_bars["high"].max())
    or_low = float(or_bars["low"].min())

    post_or = bars[bars.index >= or_end]
    if post_or.empty:
        return []

    # Time-of-day entry cutoff blocks only NEW entries; trades entered before it
    # still ride to eod_flat (matches live, where the cutoff gates entries and
    # the bracket rides to the 15:55 flatten). The exit scan is NOT truncated.
    entry_cutoff_ts = None
    if p.no_entry_after_time is not None:
        entry_cutoff_ts = pd.Timestamp.combine(
            session_date.date(), p.no_entry_after_time
        ).tz_localize(tz)

    enabled = [s for s, on in (("long", p.enable_long), ("short", p.enable_short)) if on]
    if not enabled:
        return []

    # First trade: earliest entry across enabled directions.
    candidates = [
        t for t in (
            _simulate_one_trade(post_or, symbol, equity, side, or_high, or_low,
                                eod_cutoff, session_date, p,
                                entry_cutoff_ts=entry_cutoff_ts)
            for side in enabled
        ) if t is not None
    ]
    if not candidates:
        return []
    trades = [min(candidates, key=lambda t: t.entry_time)]

    # Flips: an opposite-direction re-entry is allowed only after a stop-out.
    # The flip entry still respects the entry cutoff; its exit rides to eod_flat.
    flips_left = p.max_flips
    while flips_left > 0 and trades[-1].exit_reason == "stop":
        opp = "short" if trades[-1].side == "long" else "long"
        if opp not in enabled:
            break
        nxt = _simulate_one_trade(post_or, symbol, equity, opp, or_high, or_low,
                                  eod_cutoff, session_date, p,
                                  start_after_ts=trades[-1].exit_time,
                                  entry_cutoff_ts=entry_cutoff_ts)
        if nxt is None:
            break
        trades.append(nxt)
        flips_left -= 1

    return trades


def simulate_day(
    bars: pd.DataFrame,
    symbol: str,
    equity: float,
    p: Params = Params(),
) -> Optional[Trade]:
    """Backward-compatible wrapper: the first trade of the session, or None.

    With the default long-only, max_flips=0 Params this is identical to the
    legacy single-trade behavior. Callers that want flips should use
    simulate_session and collect the full list.
    """
    trades = simulate_session(bars, symbol, equity, p)
    return trades[0] if trades else None
