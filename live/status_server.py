"""Always-on status web page for the ORB paper-trading system.

Runs as its OWN long-lived service (separate from the ORB runner, which is
timer-fired and exits after each session). Serves one self-refreshing page that
answers the two questions the trader actually has:

  1. "What is the correct state of my trades?"  -> queried live from Alpaca,
     the authoritative source. Works whether or not the runner is alive.
  2. "Is the script actually running right now?" -> read from the heartbeat
     file the runner writes every loop, interpreted against the market clock so
     that "no heartbeat" outside session hours reads as IDLE (fine), but a
     missing/stale heartbeat *while the market is open* reads as DOWN (alarm).

Read-only. Places/cancels nothing. Binds to 127.0.0.1 by default — reach it
from your laptop over an SSH tunnel, so account figures never touch the public
internet:

    ssh -i <key> -L 8787:localhost:8787 ubuntu@<vm-ip>
    # then open http://localhost:8787 in your browser

Run on the VM:
    .venv/bin/python live/status_server.py            # 127.0.0.1:8787
    STATUS_PORT=9000 .venv/bin/python live/status_server.py
    STATUS_BIND=0.0.0.0 .venv/bin/python ...          # public (needs a firewall)
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time as _time
from datetime import datetime, time as dtime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from alpaca.data.enums import DataFeed  # noqa: E402
from alpaca.data.historical import StockHistoricalDataClient  # noqa: E402
from alpaca.data.requests import StockBarsRequest  # noqa: E402
from alpaca.data.timeframe import TimeFrame  # noqa: E402
from alpaca.trading.client import TradingClient  # noqa: E402
from alpaca.trading.enums import QueryOrderStatus  # noqa: E402
from alpaca.trading.requests import GetOrdersRequest, GetPortfolioHistoryRequest  # noqa: E402

from live import heartbeat  # noqa: E402
from live.paper_orb import (  # noqa: E402
    ET, UTC, EOD_FLAT_TIME, PARAMS, RTH_OPEN,
    build_clients, fetch_today_bars, load_env,
)

BIND = os.environ.get("STATUS_BIND", "127.0.0.1")
PORT = int(os.environ.get("STATUS_PORT", "8787"))
CACHE_TTL = 4.0  # seconds; one Alpaca gather shared across rapid page polls

# Hardening for public (0.0.0.0) exposure: this server shares a 1 GB VM with the
# live trading runners, so a connection flood must not be able to starve them.
# Cap concurrent in-flight requests (excess get a fast 503) and time out slow/
# half-open sockets so a thread can't be held hostage (slowloris). Read-only
# server, so dropping requests under load is harmless.
MAX_CONN = int(os.environ.get("STATUS_MAX_CONN", "24"))
_conn_sema = threading.BoundedSemaphore(MAX_CONN)

_cache: dict = {"ts": 0.0, "data": None}


def _f(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------
# Liveness verdict: heartbeat + market clock -> one of alive/warning/down/idle/done
# --------------------------------------------------------------------------
def _liveness(hb: dict | None, clock, now_et: datetime) -> dict:
    now = _time.time()
    # The runner's session runs market-open .. EOD-flatten (15:55 ET). After
    # 15:55 it exits on purpose, so a missing beat then is expected, not a fault.
    market_open = bool(clock and getattr(clock, "is_open", False))
    should_be_live = market_open and now_et.time() < EOD_FLAT_TIME

    if hb is None:
        if should_be_live:
            return _verdict("down", "SCRIPT DOWN",
                            "Market is open but no heartbeat file exists — the runner has not started this session.")
        return _verdict("idle", "Idle",
                        "Runner not running. Expected outside the trading session.", hb)

    age = now - _f(hb.get("ts"))
    fresh = now <= _f(hb.get("expected_next_by"))
    halted = bool(hb.get("halted"))
    phase = hb.get("phase", "running")
    dry = " (dry-run)" if hb.get("dry_run") else ""

    if fresh:
        if halted:
            return _verdict("warning", "RUNNING — entries halted",
                            f"{phase}{dry}. New entries halted: {hb.get('halt_reason') or 'see log'}. "
                            f"Existing positions still ride their brackets.", hb)
        return _verdict("alive", "RUNNING",
                        f"{phase}{dry}. Last beat {age:.0f}s ago.", hb)

    # Stale beat.
    if should_be_live:
        return _verdict("down", "SCRIPT DOWN",
                        f"Heartbeat is {age:.0f}s stale while the market is open — the runner appears to have "
                        f"crashed or hung (last phase: {phase}).", hb)
    if hb.get("session_date") == now_et.date().isoformat() and now_et.time() >= EOD_FLAT_TIME:
        return _verdict("done", "Session complete",
                        f"Today's session finished cleanly. Last phase: {phase}.", hb)
    return _verdict("idle", "Idle",
                    f"Runner not running (last beat {_fmt_ago(age)} ago). Expected outside the session.", hb)


def _verdict(state: str, headline: str, detail: str, hb: dict | None = None) -> dict:
    return {"state": state, "headline": headline, "detail": detail, "heartbeat": hb}


def _fmt_ago(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 90:
        return f"{seconds}s"
    if seconds < 5400:
        return f"{seconds // 60}m"
    return f"{seconds // 3600}h"


# --------------------------------------------------------------------------
# Opening-range levels (the "ORB fork") for currently-held symbols.
# Computed here from Alpaca minute bars rather than the runner's in-memory
# state, so the page stays self-contained and previewable. Mirrors the runner:
# OR high/low = max-high / min-low of bars in [09:30, 09:30 + or_minutes).
# --------------------------------------------------------------------------
def _or_levels(dc, symbols: list[str], now_et: datetime) -> dict:
    out: dict = {}
    if not symbols or dc is None:
        return out
    # Nothing to compute before the regular session opens — the minute-bar
    # window would start after "now". OR is an equities concept; this simply
    # yields no levels (—) for any symbol with no RTH bars yet (incl. crypto).
    if now_et.time() < RTH_OPEN:
        return out
    today = now_et.date()
    or_end = (datetime.combine(today, RTH_OPEN, tzinfo=ET)
              + timedelta(minutes=PARAMS.or_minutes))
    bars = fetch_today_bars(dc, list(symbols), today)
    if bars is None or bars.empty:
        return out
    syms_in = set(bars.index.get_level_values(0).unique())
    for sym in symbols:
        if sym not in syms_in:
            continue
        sb = bars.xs(sym, level=0)
        ob = sb[sb.index < or_end]
        if ob.empty:
            continue
        out[sym] = {"or_high": float(ob["high"].max()),
                    "or_low": float(ob["low"].min())}
    return out


# Volatility-regime ("calm day") definition — mirrors backtest/compare_regime_filter.py.
# A day is "calm" if SPY's 20d realized vol was BELOW its trailing-126d median as
# of the PRIOR close. ORB historically does better on calm days (validated 2026-06-04).
VOL_WIN = 20
VOL_MED_WIN = 126


def _spy_daily(dc, start_utc, end_utc) -> dict:
    """Map ET date -> {"ret": close-to-close %, "calm": bool|None}.

    ret = SPY % change (market benchmark). calm = vol-regime flag (above), None
    when there isn't enough history. Lookahead-free: the flag for a day uses only
    data through the prior session.
    """
    out: dict = {}
    if dc is None:
        return out
    req = StockBarsRequest(
        symbol_or_symbols=["SPY"], timeframe=TimeFrame.Day,
        start=start_utc, end=end_utc, feed=DataFeed.IEX,
    )
    df = dc.get_stock_bars(req).df
    if df is None or df.empty:
        return out
    closes = (df.xs("SPY", level=0)["close"]
              if hasattr(df.index, "levels") else df["close"]).astype(float).sort_index()
    ret = closes.pct_change()
    vol = ret.rolling(VOL_WIN).std()
    vol_med = vol.rolling(VOL_MED_WIN, min_periods=40).median()
    calm = (vol < vol_med).shift(1)  # decide a session from the prior close
    for ts in closes.index:
        d = ts.tz_convert(ET).date().isoformat() if getattr(ts, "tzinfo", None) else ts.date().isoformat()
        r, cf = ret.get(ts), calm.get(ts)
        out[d] = {
            "ret": (float(r) * 100.0) if pd.notna(r) else None,
            "calm": (bool(cf) if pd.notna(cf) else None),
        }
    return out


def _next_session_date(tc, after_iso: str) -> str | None:
    """ET date of the first trading session strictly after `after_iso`.

    Used to label the live (in-progress / just-closed) P/L row: Alpaca rolls
    `last_equity` and appends the prior day to portfolio history only at the
    next market open, so account.day_pnl belongs to the session *after* the
    last history point — today once it opens, or the just-finished session
    while the market is shut. Authoritative via the market calendar; falls
    back to the next weekday (holidays aside) if the calendar call fails.
    """
    from datetime import date as _date
    try:
        after = _date.fromisoformat(after_iso)
    except Exception:
        return None
    try:
        from alpaca.trading.requests import GetCalendarRequest
        cal = tc.get_calendar(GetCalendarRequest(
            start=after + timedelta(days=1), end=after + timedelta(days=7)))
        for d in sorted(getattr(c, "date") for c in cal):
            if d > after:
                return d.isoformat()
    except Exception:
        pass
    d = after + timedelta(days=1)
    while d.weekday() >= 5:  # skip Sat/Sun
        d += timedelta(days=1)
    return d.isoformat()


# --------------------------------------------------------------------------
# Trade state: authoritative, straight from Alpaca
# --------------------------------------------------------------------------
def _gather(tc, dc=None, since_date=None) -> dict:
    # `since_date` (YYYY-MM-DD ET str) clips P/L history to an account's inception — used
    # for the Hype account, which is the RETIRED dual-mom paper account: rows before the
    # first Hype trade belong to dual-mom and must not be attributed to Hype.
    # (Named `since_date`, NOT `since` — a local `since` datetime is reused below.)
    now_et = datetime.now(ET)
    out: dict = {
        "generated": now_et.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "errors": [],
        "market": None,
        "account": None,
        "positions": [],
        "open_orders": [],
        "orb_fills": [],
        "closed_today": [],
        "daily_pnl": [],
        "week_pnl": None,
        "month_pnl": None,
        "invested": 0.0,
    }

    clock = None
    try:
        clock = tc.get_clock()
        if clock.is_open:
            out["market"] = {"is_open": True,
                             "label": f"OPEN — closes {clock.next_close.astimezone(ET):%H:%M %Z}"}
        else:
            out["market"] = {"is_open": False,
                             "label": f"CLOSED — opens {clock.next_open.astimezone(ET):%a %b %d %H:%M %Z}"}
    except Exception as e:
        out["errors"].append(f"clock: {e}")

    try:
        a = tc.get_account()
        equity, last_eq = _f(a.equity), _f(a.last_equity)
        day_pnl = equity - last_eq
        out["account"] = {
            "number": a.account_number,
            "status": str(a.status).rsplit(".", 1)[-1],
            "equity": equity,
            "cash": _f(a.cash),
            "buying_power": _f(a.buying_power),
            "day_pnl": day_pnl,
            "day_pnl_pct": (day_pnl / last_eq * 100) if last_eq else 0.0,
        }
    except Exception as e:
        out["errors"].append(f"account: {e}")

    open_symbols: set[str] = set()
    try:
        for p in tc.get_all_positions():
            open_symbols.add(p.symbol)
            out["positions"].append({
                "symbol": p.symbol,
                "side": str(p.side).rsplit(".", 1)[-1].lower(),
                "qty": _f(p.qty),
                "avg_entry": _f(p.avg_entry_price),
                "current": _f(p.current_price),
                "cost_basis": _f(p.cost_basis),        # $ invested in this name
                "market_value": _f(p.market_value),
                "unrealized_pl": _f(p.unrealized_pl),
                "unrealized_plpc": _f(p.unrealized_plpc) * 100,
            })
        out["invested"] = sum(p["cost_basis"] for p in out["positions"])
    except Exception as e:
        out["errors"].append(f"positions: {e}")

    # Working orders. Bracket exits are quirky on Alpaca: once the entry fills,
    # the take-profit LIMIT leg is status "new" but the protective STOP leg goes
    # to status "held" — and a status=OPEN query returns NEITHER the held stop
    # nor nests it under the (now-filled) parent. So we pull recent orders with
    # nested=True and keep every still-working leg; that surfaces the stop legs
    # and their prices for both the table and the per-position risk calc below.
    WORKING_STATUSES = {"new", "accepted", "partially_filled", "held",
                        "pending_new", "accepted_for_bidding", "pending_replace"}
    try:
        since = (now_et - timedelta(days=7)).astimezone(UTC)
        req = GetOrdersRequest(status=QueryOrderStatus.ALL, after=since,
                               limit=500, nested=True)
        try:
            orders = tc.get_orders(filter=req)
        except TypeError:
            orders = tc.get_orders(req)
        seen: set = set()
        for parent in orders:
            for o in (parent, *(getattr(parent, "legs", None) or [])):
                oid = getattr(o, "id", None)
                if oid in seen:
                    continue
                if str(o.status).rsplit(".", 1)[-1].lower() not in WORKING_STATUSES:
                    continue
                seen.add(oid)
                otype = str(getattr(o, "order_type", "") or o.type).rsplit(".", 1)[-1]
                out["open_orders"].append({
                    "symbol": o.symbol,
                    "side": str(o.side).rsplit(".", 1)[-1].lower(),
                    "type": otype,
                    "qty": _f(o.qty),
                    "limit": _f(getattr(o, "limit_price", None)) or None,
                    "stop": _f(getattr(o, "stop_price", None)) or None,
                    "status": str(o.status).rsplit(".", 1)[-1],
                })
        out["open_orders"].sort(key=lambda r: (r["symbol"], r["type"]))
    except Exception as e:
        out["errors"].append(f"open orders: {e}")

    # ---- annotate open positions: OR "fork" (high/low) + actual $ risk ----
    # Risk = distance to the live protective stop × shares held. The stop price
    # comes from the open stop-loss bracket leg (already fetched above), so this
    # is the real risk on the book, not a re-derived estimate.
    try:
        stop_by_sym: dict[str, float] = {}
        for o in out["open_orders"]:
            if o.get("stop"):
                stop_by_sym.setdefault(o["symbol"], o["stop"])
        or_map = _or_levels(dc, [p["symbol"] for p in out["positions"]], now_et)
        for p in out["positions"]:
            lv = or_map.get(p["symbol"], {})
            p["or_high"] = lv.get("or_high")
            p["or_low"] = lv.get("or_low")
            stop = stop_by_sym.get(p["symbol"])
            p["stop"] = stop
            p["risk"] = (abs(p["avg_entry"] - stop) * p["qty"]) if stop else None
    except Exception as e:
        out["errors"].append(f"position OR/risk: {e}")

    try:
        today = now_et.date()
        start = datetime.combine(today, dtime(0, 0, tzinfo=ET)).astimezone(UTC)
        req = GetOrdersRequest(status=QueryOrderStatus.ALL, after=start, limit=200)
        try:
            todays = tc.get_orders(filter=req)
        except TypeError:
            todays = tc.get_orders(req)
        prefix = f"orb-{today:%Y%m%d}-"
        # Per-symbol fill aggregation for round-trip realized P&L.
        agg: dict[str, dict] = {}
        for o in todays:
            coid = getattr(o, "client_order_id", "") or ""
            fap = getattr(o, "filled_avg_price", None)
            fqty = _f(o.filled_qty)
            if fap is None or fqty <= 0:
                continue
            side = str(o.side).rsplit(".", 1)[-1].lower()
            price = _f(fap)
            if coid.startswith(prefix):
                out["orb_fills"].append({
                    "symbol": o.symbol, "side": side, "qty": fqty,
                    "price": price, "coid": coid,
                })
            d = agg.setdefault(o.symbol, {"buy_q": 0.0, "buy_n": 0.0,
                                          "sell_q": 0.0, "sell_n": 0.0, "first": None})
            if side == "buy":
                d["buy_q"] += fqty
                d["buy_n"] += fqty * price
            else:
                d["sell_q"] += fqty
                d["sell_n"] += fqty * price
            ft = getattr(o, "filled_at", None)
            if ft is not None and (d["first"] is None or ft < d["first"][0]):
                d["first"] = (ft, side)

        # A symbol is a "closed today" round-trip if it has both buys and sells
        # today and is now FLAT (not in open positions). Realized = sell proceeds
        # - buy cost (works for long and short alike when net flat).
        for sym, d in agg.items():
            if sym in open_symbols or d["buy_q"] <= 0 or d["sell_q"] <= 0:
                continue
            realized = d["sell_n"] - d["buy_n"]
            opened = d["first"][1] if d["first"] else ("buy" if d["buy_q"] >= d["sell_q"] else "sell")
            out["closed_today"].append({
                "symbol": sym,
                "side": "long" if opened == "buy" else "short",
                "qty": round(min(d["buy_q"], d["sell_q"]), 4),
                "entry_avg": d["buy_n"] / d["buy_q"],
                "exit_avg": d["sell_n"] / d["sell_q"],
                "realized": realized,
            })
        out["closed_today"].sort(key=lambda r: r["realized"])
    except Exception as e:
        out["errors"].append(f"today's orders: {e}")

    # ---- $ invested per day: gross ENTRY (buy) notional filled each ET day.
    # The strategy is long-only and intraday, so buy notional = capital deployed
    # that day. Lets each day's P/L be shown as a return on what was actually at
    # risk, not as a near-zero fraction of the mostly-cash account equity.
    invested_by_day: dict[str, float] = {}
    deployed_eod: dict[str, float] = {}   # date -> concurrent cost basis of open positions at EOD
    try:
        win_start = datetime.combine(today - timedelta(days=370), dtime(0, 0, tzinfo=ET)).astimezone(UTC)
        req = GetOrdersRequest(status=QueryOrderStatus.ALL, after=win_start, limit=500)
        try:
            hist = tc.get_orders(filter=req)
        except TypeError:
            hist = tc.get_orders(req)
        all_fills = []
        for o in hist:
            fap = getattr(o, "filled_avg_price", None)
            fqty = _f(o.filled_qty)
            ft = getattr(o, "filled_at", None)
            if fap is None or fqty <= 0 or ft is None:
                continue
            side = str(o.side).rsplit(".", 1)[-1].lower()
            d = ft.astimezone(ET).date().isoformat()
            all_fills.append((ft, d, o.symbol, side, fqty, _f(fap)))
            if side == "buy":
                invested_by_day[d] = invested_by_day.get(d, 0.0) + fqty * _f(fap)
        # concurrent deployed capital (cost basis of open positions) per day, from ALL fills
        all_fills.sort(key=lambda x: x[0])
        runpos: dict = {}   # sym -> [qty, cost]
        for (_ft, d, sym, side, fq, px) in all_fills:
            st = runpos.setdefault(sym, [0.0, 0.0])
            if side == "buy":
                st[0] += fq; st[1] += fq * px
            elif st[0] > 0:                                  # sell: avg-cost removal
                st[1] -= min(fq, st[0]) / st[0] * st[1]
                st[0] = max(0.0, st[0] - fq)
            deployed_eod[d] = sum(c for _q, c in runpos.values())
    except Exception as e:
        out["errors"].append(f"orders history: {e}")

    # ---- SPY daily % (benchmark) + calm-day vol-regime flag for the same window ----
    spy_info: dict = {}
    try:
        spy_start = datetime.combine(today - timedelta(days=372), dtime(0, 0, tzinfo=ET)).astimezone(UTC)
        spy_info = _spy_daily(dc, spy_start, now_et.astimezone(UTC))
    except Exception as e:
        out["errors"].append(f"SPY benchmark: {e}")

    # ---- previous days' P&L (account-level, from portfolio history) ----
    try:
        ph = tc.get_portfolio_history(
            GetPortfolioHistoryRequest(period="1A", timeframe="1D"))
        ts, pl, eq = ph.timestamp, ph.profit_loss, ph.equity
        rows = []
        for i in range(len(ts)):
            d = datetime.fromtimestamp(ts[i], UTC).astimezone(ET).date().isoformat()
            pnl = _f(pl[i]) if i < len(pl) else 0.0
            inv = invested_by_day.get(d, 0.0)
            rows.append({
                "date": d,
                "pnl": pnl,
                "invested": inv,
                # P/L as a % of capital deployed that day (None if nothing traded).
                "pnl_pct_inv": (pnl / inv * 100) if inv > 0 else None,
                "spy_pct": (spy_info.get(d) or {}).get("ret"),
                "calm": (spy_info.get(d) or {}).get("calm"),
                "equity": _f(eq[i]) if i < len(eq) else 0.0,
            })
        # Drop leading pre-funding days (portfolio history pads them with eq=0).
        # Show every real day since funding, newest first — no cap.
        rows = [r for r in rows if r["equity"] > 0]

        # Portfolio history lags by a session: account.last_equity (and the
        # newest 1D point) only roll forward at the next market open. So the
        # live day_pnl belongs to the session AFTER the last history point —
        # synthesize that row from the account snapshot so the current/just-
        # closed day is always present and counted in the roll-ups below.
        today_iso = today.isoformat()
        acct = out.get("account") or {}
        eq_now = acct.get("equity", 0.0)
        last_hist = rows[-1]["date"] if rows else None
        live_date = _next_session_date(tc, last_hist) if last_hist else today_iso
        if (eq_now > 0 and live_date and live_date <= today_iso
                and not any(r["date"] == live_date for r in rows)):
            pnl_now = acct.get("day_pnl", 0.0)
            inv_now = invested_by_day.get(live_date, 0.0)
            rows.append({
                "date": live_date,
                "pnl": pnl_now,
                "invested": inv_now,
                "pnl_pct_inv": (pnl_now / inv_now * 100) if inv_now > 0 else None,
                "spy_pct": (spy_info.get(live_date) or {}).get("ret"),
                "calm": (spy_info.get(live_date) or {}).get("calm"),
                "equity": eq_now,
            })

        # Clip to inception (Hype account): drop dual-mom-era rows BEFORE the rollups
        # and daily list are built, so every figure reflects only this bot's history.
        if since_date:
            rows = [r for r in rows if r["date"] >= since_date]

        # Return on CONCURRENT capital deployed (the strategy-quality denominator, not the
        # idle account balance): carry the EOD cost basis forward across no-fill days.
        dep_series, _last = [], 0.0
        for r in sorted(rows, key=lambda x: x["date"]):
            _last = deployed_eod.get(r["date"], _last)
            dep_series.append(_last)
        _active = [x for x in dep_series if x > 0]
        out["deployed_avg"] = (sum(_active) / len(_active)) if _active else 0.0
        out["deployed_peak"] = max(dep_series) if dep_series else 0.0
        out["window_pnl"] = sum(r["pnl"] for r in rows)

        # Trailing-window roll-ups for the stat cards (calendar windows incl. today).
        def _window(days: int) -> dict:
            cutoff = (today - timedelta(days=days - 1)).isoformat()
            sel = [r for r in rows if r["date"] >= cutoff]
            pnl_sum = sum(r["pnl"] for r in sel)
            inv_sum = sum(r["invested"] for r in sel)
            return {"pnl": pnl_sum,
                    "pnl_pct_inv": (pnl_sum / inv_sum * 100) if inv_sum > 0 else None}
        out["week_pnl"] = _window(7)
        out["month_pnl"] = _window(30)

        out["daily_pnl"] = list(reversed(rows))
    except Exception as e:
        out["errors"].append(f"portfolio history: {e}")

    out["liveness"] = _liveness(heartbeat.read(), clock, now_et)
    return out


NEWS_PICKS_DIR = ROOT / "experiments" / "news_edge" / "picks"

# Read-only clients for the 2nd paper account (the news-edge bot). Lazily built once
# from .env.news; (None, None) if the file/keys aren't present (e.g. not on the VM yet),
# in which case the tab just shows picks without the bot's account.
_NEWS = {"built": False, "tc": None, "dc": None}
_news_cache = {"ts": 0.0, "data": None}


def _news_clients():
    if not _NEWS["built"]:
        _NEWS["built"] = True
        f = ROOT / ".env.news"
        if f.exists():
            vals = {}
            for line in f.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    vals[k.strip()] = v.strip().strip('"').strip("'")
            key, sec = vals.get("ALPACA_API_KEY"), vals.get("ALPACA_SECRET_KEY")
            if key and sec:
                try:
                    _NEWS["tc"] = TradingClient(key, sec, paper=True)
                    _NEWS["dc"] = StockHistoricalDataClient(key, sec)
                except Exception:
                    _NEWS["tc"] = _NEWS["dc"] = None
    return _NEWS["tc"], _NEWS["dc"]


def _news_status() -> dict | None:
    """Same rich trade-state snapshot as the main account, but for the news bot's
    2nd account — so the tab can show its money next to the picks. Separate 4s cache."""
    tc, dc = _news_clients()
    if tc is None:
        return None
    now = _time.time()
    if _news_cache["data"] is None or now - _news_cache["ts"] > CACHE_TTL:
        try:
            _news_cache["data"] = _gather(tc, dc)
        except Exception as e:
            _news_cache["data"] = {"errors": [f"news account: {e}"]}
        _news_cache["ts"] = now
    return _news_cache["data"]


_news_live_cache: dict = {"ts": 0.0, "data": None}


def _news_live(symbols: list[str]) -> dict:
    """Live intraday behavior for today's news picks — bought or not. Per symbol:
    last, gap%, day%, day hi/lo, 15-min OR levels (what the news bot trades), and
    the 9:45 reference price the after-close scorer measures from. 30s cache —
    the OR-level fetch pulls today's minute bars, no need to do that every poll."""
    dc = Handler.dc or _news_clients()[1]
    if dc is None or not symbols:
        return {}
    now = _time.time()
    if _news_live_cache["data"] is not None and now - _news_live_cache["ts"] < 30.0:
        return _news_live_cache["data"]
    now_et = datetime.now(ET)
    snaps = {}
    try:
        from alpaca.data.requests import StockSnapshotRequest
        snaps = dc.get_stock_snapshot(
            StockSnapshotRequest(symbol_or_symbols=list(symbols), feed=DataFeed.IEX)) or {}
    except Exception:
        pass
    # 9:45 ET reference = open of the 09:45 minute bar (the scorer's entry point)
    px945: dict = {}
    try:
        ref_t = datetime.combine(now_et.date(), dtime(9, 45), tzinfo=ET)
        if now_et >= ref_t:
            req = StockBarsRequest(symbol_or_symbols=list(symbols), timeframe=TimeFrame.Minute,
                                   start=ref_t.astimezone(UTC),
                                   end=(ref_t + timedelta(minutes=1)).astimezone(UTC),
                                   feed=DataFeed.IEX)
            bars = dc.get_stock_bars(req).df
            if bars is not None and not bars.empty:
                for sym in set(bars.index.get_level_values(0)):
                    px945[sym] = float(bars.xs(sym, level=0)["open"].iloc[0])
    except Exception:
        pass
    try:
        or_map = _or_levels(dc, list(symbols), now_et)
    except Exception:
        or_map = {}
    out: dict = {}
    for sym in symbols:
        sn = snaps.get(sym)
        if sn is None:
            continue
        try:
            prev_c = float(sn.previous_daily_bar.close) if sn.previous_daily_bar else None
            db = sn.daily_bar
            last = (float(sn.latest_trade.price) if sn.latest_trade
                    else (float(db.close) if db else None))
            o = float(db.open) if db else None
            p945 = px945.get(sym)
            out[sym] = {
                "last": last,
                "gap_pct": ((o / prev_c - 1) * 100) if (o and prev_c) else None,
                "day_pct": ((last / prev_c - 1) * 100) if (last and prev_c) else None,
                "hi": float(db.high) if db else None,
                "lo": float(db.low) if db else None,
                "px945": p945,
                "since945_pct": ((last / p945 - 1) * 100) if (last and p945) else None,
                "or_high": (or_map.get(sym) or {}).get("or_high"),
                "or_low": (or_map.get(sym) or {}).get("or_low"),
            }
        except Exception:
            continue
    _news_live_cache["data"] = out
    _news_live_cache["ts"] = now
    return out


def _newsedge() -> dict:
    """Summarize the news-edge forward-test picks for the web tab (read-only).

    Reads experiments/news_edge/picks/*.json (written by newsedge.py). No Alpaca
    calls — outcomes are already baked into the files by the `outcomes` command,
    so this just tallies them: per-day picks + the running (+)-vs-(-) separation.
    """
    out = {"generated": datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S %Z"), "days": [], "overall": {}}
    if not NEWS_PICKS_DIR.exists():
        return out
    avg = lambda xs: (sum(xs) / len(xs)) if xs else None
    all_pos: list[float] = []
    all_neg: list[float] = []
    days = []
    for f in sorted(NEWS_PICKS_DIR.glob("*.json")):
        try:
            rec = json.load(open(f))
        except Exception:
            continue
        picks = rec.get("picks", [])
        scored = [p for p in picks if p.get("ret_945_close") is not None]
        pos = [p["ret_945_close"] for p in scored if p.get("signal", 0) > 0]
        neg = [p["ret_945_close"] for p in scored if p.get("signal", 0) < 0]
        all_pos += pos
        all_neg += neg
        days.append({
            "date": rec.get("date", f.stem),
            "logged_at": rec.get("logged_at"),
            "n": len(picks),
            "n_scored": len(scored),
            "avg_pos": avg(pos), "avg_neg": avg(neg),
            "sep": (avg(pos) - avg(neg)) if pos and neg else None,
            "picks": picks,
        })
    days.sort(key=lambda d: d["date"], reverse=True)
    out["days"] = days
    out["overall"] = {
        "n_days": len(days),
        "n_scored": len(all_pos) + len(all_neg),
        "avg_pos": avg(all_pos), "avg_neg": avg(all_neg),
        "win_pos": (sum(1 for r in all_pos if r > 0) / len(all_pos) * 100) if all_pos else None,
        "sep": (avg(all_pos) - avg(all_neg)) if all_pos and all_neg else None,
    }
    bot = _news_status()
    if bot is not None:
        out["bot"] = bot
    # live behavior of TODAY's picks (bought or not) for the live table
    today_iso = datetime.now(ET).date().isoformat()
    tday = next((d for d in days if d["date"] == today_iso), None)
    if tday:
        try:
            out["live"] = _news_live([p["symbol"] for p in tday["picks"]])
            out["live_date"] = today_iso
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------------
# Lottery experiment tab — mirrors the news-edge tab (read-only). Reads
# experiments/lottery/picks/*.json (written by board.py / scored by outcomes.py)
# and the lottery bot's repurposed dual-mom paper account via .env.lottery.
# ---------------------------------------------------------------------------
LOTTERY_PICKS_DIR = ROOT / "experiments" / "lottery" / "picks"
# Hype bot inception: its first real trades (NBIS/TRV/SPCL) filled 2026-06-15. The
# account is the retired dual-mom paper account (dual-mom liquidated 2026-06-12), so
# P/L history before this date is dual-mom's and is clipped out of the Hype views.
HYPE_INCEPTION = "2026-06-15"

_LOTTERY = {"built": False, "tc": None, "dc": None}
_lottery_cache = {"ts": 0.0, "data": None}
_lottery_live_cache: dict = {"ts": 0.0, "data": None}


def _lottery_clients():
    if not _LOTTERY["built"]:
        _LOTTERY["built"] = True
        f = ROOT / ".env.lottery"
        if f.exists():
            vals = {}
            for line in f.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    vals[k.strip()] = v.strip().strip('"').strip("'")
            key, sec = vals.get("ALPACA_API_KEY"), vals.get("ALPACA_SECRET_KEY")
            if key and sec:
                try:
                    _LOTTERY["tc"] = TradingClient(key, sec, paper=True)
                    _LOTTERY["dc"] = StockHistoricalDataClient(key, sec)
                except Exception:
                    _LOTTERY["tc"] = _LOTTERY["dc"] = None
    return _LOTTERY["tc"], _LOTTERY["dc"]


def _lottery_closed_today(tc) -> list[dict]:
    """Round-trips CLOSED today on the Hype account — including MULTI-DAY holds whose entry
    was on a prior day (the generic _gather closed_today only catches same-day round-trips,
    which the multi-day Hype bot rarely has). A symbol counts if it had a SELL filled today
    and is now flat; realized = matched (exit_avg - entry_avg) * qty, entry from the buy
    fills over a trailing window. Approximate if a name was round-tripped twice in the window
    (rare — the bot de-dups names)."""
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    today = datetime.now(ET).date()
    start = datetime.combine(today - timedelta(days=10), dtime(0, 0, tzinfo=ET)).astimezone(UTC)
    try:
        req = GetOrdersRequest(status=QueryOrderStatus.CLOSED, after=start, limit=500)
        try:
            orders = tc.get_orders(filter=req)
        except TypeError:
            orders = tc.get_orders(req)
        held = {p.symbol for p in tc.get_all_positions()}
    except Exception:
        return []
    agg: dict[str, dict] = {}
    for o in orders:
        fap = getattr(o, "filled_avg_price", None)
        fq = _f(o.filled_qty)
        ft = getattr(o, "filled_at", None)
        if fap is None or fq <= 0 or ft is None:
            continue
        side = str(o.side).rsplit(".", 1)[-1].lower()
        a = agg.setdefault(o.symbol, {"bq": 0.0, "bn": 0.0, "sq": 0.0, "sn": 0.0,
                                      "tsq": 0.0, "tsn": 0.0})
        if side == "buy":
            a["bq"] += fq; a["bn"] += fq * _f(fap)
        elif side == "sell":
            a["sq"] += fq; a["sn"] += fq * _f(fap)
            if ft.astimezone(ET).date() == today:
                a["tsq"] += fq; a["tsn"] += fq * _f(fap)   # today's sells only
    out = []
    for sym, a in agg.items():
        # any name with a SELL filled TODAY against a known entry — full OR partial.
        # (A trailing stop can sell part of a position and leave a remnant held; the old
        # "must be flat" gate hid those exits + their realized P/L entirely.)
        if a["tsq"] <= 0 or a["bq"] <= 0:
            continue
        entry_avg = a["bn"] / a["bq"]
        exit_avg = a["tsn"] / a["tsq"]
        qty = a["tsq"]                                   # shares sold today
        partial = (sym in held) or (a["sq"] + 1e-9 < a["bq"])
        out.append({"symbol": sym, "qty": qty, "entry_avg": entry_avg,
                    "exit_avg": exit_avg, "realized": (exit_avg - entry_avg) * qty,
                    "partial": partial})
    out.sort(key=lambda r: r["realized"])
    return out


def _lottery_status() -> dict | None:
    """Trade-state snapshot for the Hype bot's account (the RETIRED dual-mom account).
    P/L history is clipped to HYPE_INCEPTION so dual-mom's pre-06-15 days/equity swings
    are not shown as Hype's (its first trades — NBIS/TRV/SPCL — landed 2026-06-15)."""
    tc, dc = _lottery_clients()
    if tc is None:
        return None
    now = _time.time()
    if _lottery_cache["data"] is None or now - _lottery_cache["ts"] > CACHE_TTL:
        try:
            data = _gather(tc, dc, since_date=HYPE_INCEPTION)
            data["closed_today"] = _lottery_closed_today(tc)  # multi-day-aware override
            _lottery_cache["data"] = data
        except Exception as e:
            _lottery_cache["data"] = {"errors": [f"lottery account: {e}"]}
        _lottery_cache["ts"] = now
    return _lottery_cache["data"]


def _lottery_live(symbols: list[str]) -> dict:
    """Live intraday behavior for today's lottery picks — bought or not. Same per-symbol
    fields as the news tab (last, gap%, day%, 9:45 ref, since-9:45). 30s cache."""
    dc = Handler.dc or _lottery_clients()[1]
    if dc is None or not symbols:
        return {}
    now = _time.time()
    if _lottery_live_cache["data"] is not None and now - _lottery_live_cache["ts"] < 30.0:
        return _lottery_live_cache["data"]
    now_et = datetime.now(ET)
    snaps = {}
    try:
        from alpaca.data.requests import StockSnapshotRequest
        snaps = dc.get_stock_snapshot(
            StockSnapshotRequest(symbol_or_symbols=list(symbols), feed=DataFeed.IEX)) or {}
    except Exception:
        pass
    px945: dict = {}
    try:
        ref_t = datetime.combine(now_et.date(), dtime(9, 45), tzinfo=ET)
        if now_et >= ref_t:
            req = StockBarsRequest(symbol_or_symbols=list(symbols), timeframe=TimeFrame.Minute,
                                   start=ref_t.astimezone(UTC),
                                   end=(ref_t + timedelta(minutes=1)).astimezone(UTC),
                                   feed=DataFeed.IEX)
            bars = dc.get_stock_bars(req).df
            if bars is not None and not bars.empty:
                for sym in set(bars.index.get_level_values(0)):
                    px945[sym] = float(bars.xs(sym, level=0)["open"].iloc[0])
    except Exception:
        pass
    out: dict = {}
    for sym in symbols:
        sn = snaps.get(sym)
        if sn is None:
            continue
        try:
            prev_c = float(sn.previous_daily_bar.close) if sn.previous_daily_bar else None
            db = sn.daily_bar
            last = (float(sn.latest_trade.price) if sn.latest_trade
                    else (float(db.close) if db else None))
            o = float(db.open) if db else None
            p945 = px945.get(sym)
            out[sym] = {
                "last": last,
                "gap_pct": ((o / prev_c - 1) * 100) if (o and prev_c) else None,
                "day_pct": ((last / prev_c - 1) * 100) if (last and prev_c) else None,
                "px945": p945,
                "since945_pct": ((last / p945 - 1) * 100) if (last and p945) else None,
            }
        except Exception:
            continue
    _lottery_live_cache["data"] = out
    _lottery_live_cache["ts"] = now
    return out


def _source_daily(days_limit: int = 14) -> dict:
    """Per-source daily performance grid (descriptive) for the Hype + Summary tabs.
    For each SCORED day, the average 9:45->close return of the names each source flagged
    (top_k_of), plus 'combined3' (the Hype bot's traded top-3) and 'random' (luck baseline).
    Returns {dates (newest-first), sources (benchmarks then by cum avg), grid:{src:{date:
    {n,avg,hit}}}, cum:{src:{n,avg,hit}}}. Shared so both tabs read one source of truth."""
    out = {"dates": [], "sources": [], "grid": {}, "cum": {}}
    if not LOTTERY_PICKS_DIR.exists():
        return out
    recs = []
    for f in sorted(LOTTERY_PICKS_DIR.glob("*.json")):
        try:
            r = json.load(open(f))
            recs.append((r.get("date", f.stem), r.get("picks", [])))
        except Exception:
            continue
    recs.sort(key=lambda x: x[0], reverse=True)   # newest first
    WIN1 = 5.0
    grid: dict = {}
    pool: dict = {}
    dates_list: list = []
    for date, ps in recs:
        if not any(p.get("ret_945_close") is not None for p in ps):
            continue       # skip unscored days (e.g. today before the 1:10pm scorer)
        dates_list.append(date)
        groups: dict = {}
        ranked = sorted([p for p in ps if p.get("combined_score") is not None],
                        key=lambda x: -x["combined_score"])
        groups["combined3"] = ranked[:3]
        groups["random"] = [p for p in ps if p.get("basket") == "random"]
        for p in ps:
            for sig in p.get("top_k_of", []):
                groups.setdefault(sig, []).append(p)
        for src, gp in groups.items():
            vals = [p["ret_945_close"] for p in gp if p.get("ret_945_close") is not None]
            if not vals:
                continue
            grid.setdefault(src, {})[date] = {
                "n": len(vals), "avg": round(sum(vals) / len(vals), 2),
                "hit": round(sum(1 for x in vals if x >= WIN1) / len(vals), 3)}
            pool.setdefault(src, []).extend(vals)
    cum = {src: {"n": len(v), "avg": round(sum(v) / len(v), 2),
                 "hit": round(sum(1 for x in v if x >= WIN1) / len(v), 3)}
           for src, v in pool.items()}
    # classify: online SOURCES (a place publishing data) vs computed FEATURES (price/volume).
    kind = {"wsb": "src", "stocktwits": "src", "pennystocks": "src", "shortsqueeze": "src",
            "gtrends": "src", "finra_shortvol": "src", "halts": "src",
            "ignition": "feat", "pm_rvol": "feat", "squeeze": "feat", "uoa": "feat",
            "gappers": "feat", "combined3": "bench", "filtered3": "bench", "random": "bench"}
    _bench_names = ("combined3", "filtered3", "random")
    bench = [s for s in _bench_names if s in cum]
    rest = [s for s in cum if s not in _bench_names]
    srcs = sorted([s for s in rest if kind.get(s) == "src"], key=lambda s: -cum[s]["avg"])
    feats = sorted([s for s in rest if kind.get(s) == "feat"], key=lambda s: -cum[s]["avg"])
    other = sorted([s for s in rest if s not in kind], key=lambda s: -cum[s]["avg"])
    return {"dates": dates_list[:days_limit], "sources": bench + srcs + feats + other,
            "kinds": {s: kind.get(s, "src") for s in cum}, "grid": grid, "cum": cum}


def _driver_text(pick: dict) -> str:
    """Plain-English 'why was this name picked' from its signals + top_k_of flags."""
    sg = pick.get("signals", {}) or {}
    tk = pick.get("top_k_of", []) or []
    parts = []
    if "ignition" in tk and sg.get("ignition") is not None:
        parts.append(f"price momentum (ignition {int(sg['ignition'])}/4)")
    if "wsb" in tk:
        s = sg.get("wsb_surge")
        parts.append(f"Reddit WSB surge{f' {s:.0f}x' if s else ''}")
    if "stocktwits" in tk:
        parts.append("StockTwits trending")
    if "pennystocks" in tk:
        parts.append("r/pennystocks buzz")
    if "shortsqueeze" in tk:
        parts.append("r/Shortsqueeze buzz")
    if "squeeze" in tk:
        sq = sg.get("squeeze")
        parts.append(f"short-squeeze setup{f' ({sq:.1f})' if sq else ''}")
    if "pm_rvol" in tk:
        rv = sg.get("pm_rvol")
        parts.append(f"high premarket volume{f' {rv:.1f}x' if rv else ''}")
    if "gtrends" in tk:
        parts.append("Google-Trends spike")
    if "finra_shortvol" in tk:
        parts.append("heavy short volume")
    if "halts" in tk:
        parts.append("trading halt")
    gp = sg.get("gap_pct")
    if gp is not None and abs(gp) >= 2 and not any("gap" in p for p in parts):
        parts.append(f"premarket gap {gp:+.0f}%")
    if not parts:
        # no single hype signal flagged it — it made the cut on the blended composite alone
        cs = pick.get("combined_score")
        parts.append("broad composite strength, no single hype signal stood out"
                     if cs is not None else "—")
        if pick.get("basket") == "random":
            parts.append("from the random-control basket")
    return ", ".join(parts)


_asset_name_cache: dict = {}


def _asset_names(tc, symbols: list[str]) -> dict:
    """{symbol: company name} via Alpaca asset metadata, cached (names don't change)."""
    out = {}
    for s in symbols:
        if s in _asset_name_cache:
            out[s] = _asset_name_cache[s]
            continue
        try:
            nm = getattr(tc.get_asset(s), "name", None)
            if nm:                       # trim boilerplate suffixes for readability
                for suf in (" Common Stock", " Class A Common Stock", ", Inc.", " Inc."):
                    nm = nm.replace(suf, "")
                nm = nm.strip().rstrip(",")
        except Exception:
            nm = None
        _asset_name_cache[s] = nm
        out[s] = nm
    return out


def _lottery_drivers(symbols: list[str]) -> dict:
    """{symbol: {date, why, score}} — the most recent board pick record per held symbol,
    translated into a plain-English driver string."""
    want = set(symbols)
    out: dict = {}
    if not LOTTERY_PICKS_DIR.exists():
        return out
    for f in sorted(LOTTERY_PICKS_DIR.glob("*.json"), reverse=True):   # newest first
        if not want:
            break
        try:
            rec = json.load(open(f))
        except Exception:
            continue
        for p in rec.get("picks", []):
            s = p.get("symbol")
            if s in want:
                out[s] = {"date": rec.get("date"), "why": _driver_text(p),
                          "score": p.get("combined_score")}
                want.discard(s)
    return out


def _lottery() -> dict:
    """Summarize the lottery forward-test for its web tab (read-only).

    Per-day picks + a signal scoreboard: for each signal/basket, the W1/W2/W3
    hit-rate, the lift vs the RANDOM-basket luck baseline, and a binomial p.
    Reuses the pre-registered stat helpers from experiments/lottery/analyze.py.
    """
    out = {"generated": datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S %Z"),
           "days": [], "scoreboard": {}}
    if not LOTTERY_PICKS_DIR.exists():
        return out
    days = []
    for f in sorted(LOTTERY_PICKS_DIR.glob("*.json")):
        try:
            rec = json.load(open(f))
        except Exception:
            continue
        picks = rec.get("picks", [])
        baskets: dict = {}
        for p in picks:
            b = p.get("basket", "?")
            baskets[b] = baskets.get(b, 0) + 1
        n_scored = sum(1 for p in picks if p.get("ret_945_close") is not None)
        days.append({
            "date": rec.get("date", f.stem),
            "logged_at": rec.get("logged_at"),
            "n": len(picks),
            "n_scored": n_scored,
            "baskets": baskets,
            "picks": picks,
        })
    days.sort(key=lambda d: d["date"], reverse=True)
    out["days"] = days

    # --- signal scoreboard (reuse the pre-registered verdict math) ---
    try:
        from experiments.lottery.analyze import hit_stats, _binom_p, WINDEFS, SUCCESS_LIFT, SUCCESS_NDAYS
        recs = [{"picks": d["picks"]} for d in days]
        by_basket: dict = {}
        by_signal: dict = {}
        combined_top3: list = []
        for rec in recs:
            ranked = sorted([p for p in rec["picks"] if p.get("combined_score") is not None],
                            key=lambda x: -x["combined_score"])
            combined_top3.extend(ranked[:3])
            for p in rec["picks"]:
                by_basket.setdefault(p.get("basket", "?"), []).append(p)
                for sig in p.get("top_k_of", []):
                    by_signal.setdefault(sig, []).append(p)
        random_picks = by_basket.get("random", [])
        base = {}
        for key, field, thr in WINDEFS:
            w, s = hit_stats(random_picks, field, thr)
            base[key] = (w / s) if s else None

        def rows_for(name, kind, picks):
            r = {"name": name, "kind": kind, "wins": {}}
            for key, field, thr in WINDEFS:
                w, s = hit_stats(picks, field, thr)
                br = base.get(key)
                rate = (w / s) if s else None
                lift = (rate / br) if (rate is not None and br) else None
                p = _binom_p(w, s, br) if (s and br) else None
                r["wins"][key] = {"w": w, "n": s, "rate": rate, "lift": lift, "p": p}
            return r

        scoreboard = {
            "n_days": len(days),
            "success_lift": SUCCESS_LIFT, "success_ndays": SUCCESS_NDAYS,
            "base": base,
            "signals": [rows_for(s, "signal", by_signal[s]) for s in sorted(by_signal)],
            "baskets": [rows_for(b, "basket", by_basket[b])
                        for b in ["wsb", "stocktwits", "gappers", "control", "random"]
                        if by_basket.get(b)],
            "combined_top3": rows_for("combined top-3/day", "combined", combined_top3),
        }
        out["scoreboard"] = scoreboard
    except Exception as e:
        out["scoreboard"] = {"error": f"scoreboard unavailable: {e}"}

    # --- daily source-performance grid (descriptive; shared helper, also used by Summary) ---
    try:
        out["source_daily"] = _source_daily(days_limit=12)
    except Exception as e:
        out["source_daily"] = {"error": f"source_daily unavailable: {e}"}

    bot = _lottery_status()
    if bot is not None:
        out["bot"] = bot
        try:
            if bot.get("positions"):
                syms = [p["symbol"] for p in bot["positions"]]
                drv = _lottery_drivers(syms)
                try:
                    tc2, _dc2 = _lottery_clients()
                    names = _asset_names(tc2, syms) if tc2 else {}
                    for s in drv:
                        drv[s]["name"] = names.get(s)
                except Exception:
                    pass
                out["drivers"] = drv
        except Exception:
            pass
    today_iso = datetime.now(ET).date().isoformat()
    tday = next((d for d in days if d["date"] == today_iso), None)
    if tday:
        try:
            out["live"] = _lottery_live([p["symbol"] for p in tday["picks"]])
            out["live_date"] = today_iso
        except Exception:
            pass
    return out


# Market regime gauge (scripts/market_regime.py snapshot) for the Market tab.
# Daily-bar data, so a long cache: 15 min when healthy, 60 s retry after an error.
# The snapshot fetches ~125 names of daily closes — cheap, but no reason to refetch
# on every page poll. A background thread (started in serve()) refreshes it every
# 15 min even when nobody has the page open, so the verdict-change phone push
# below fires regardless of page views.
_regime_cache: dict = {"ts": 0.0, "data": None, "ttl": 0.0}
REGIME_STATE_FILE = ROOT / "logs" / "regime_state.json"


def _regime_notify(data: dict) -> None:
    """Push to the phone (ntfy, same topic as the trading bot) when the regime
    verdict or the dip read CHANGES. State persists across restarts; the very
    first run only seeds the state file (no push for 'no change')."""
    cur = {"verdict": data.get("verdict"),
           "dip": (data.get("dip") or {}).get("verdict")}
    prev = None
    try:
        if REGIME_STATE_FILE.exists():
            prev = json.load(open(REGIME_STATE_FILE))
    except Exception:
        prev = None
    if prev is not None and {k: prev.get(k) for k in cur} == cur:
        return
    if prev is not None:
        try:
            from live.notify import notify
            tone = data.get("tone")
            tags = {"good": ["chart_with_upwards_trend"], "caution": ["warning"],
                    "bad": ["rotating_light"], "neutral": ["scales"]}.get(tone)
            notify(f"{cur['verdict']}\n"
                   f"dip read: {cur['dip']}\n"
                   f"structure {data.get('struct_score', 0):+d} ({data.get('struct_state')}) · "
                   f"momentum {data.get('mom_score', 0):+d} ({data.get('mom_state')})"
                   + (f"\nwas: {prev.get('verdict')}" if prev.get("verdict") else ""),
                   title="Market regime changed",
                   priority=4 if tone == "bad" else 3, tags=tags)
        except Exception:
            pass  # decoration only — the page must serve regardless
    try:
        REGIME_STATE_FILE.parent.mkdir(exist_ok=True)
        json.dump(cur, open(REGIME_STATE_FILE, "w"))
    except Exception:
        pass


def _regime() -> dict:
    now = _time.time()
    if _regime_cache["data"] is None or now - _regime_cache["ts"] > _regime_cache["ttl"]:
        try:
            from scripts.market_regime import snapshot
            data = snapshot()
        except Exception as e:
            data = {"error": f"regime snapshot failed: {e}"}
        _regime_cache["data"] = data
        _regime_cache["ts"] = now
        _regime_cache["ttl"] = 60.0 if data.get("error") else 900.0
        if not data.get("error"):
            _regime_notify(data)
    return _regime_cache["data"]


def _regime_loop() -> None:
    """Daemon thread: keep the regime fresh so verdict-change pushes don't
    depend on someone polling the page."""
    while True:
        try:
            _regime()
        except Exception:
            pass
        _time.sleep(300)  # cheap check; _regime() itself enforces the 15-min TTL


def _status(tc, dc=None) -> dict:
    now = _time.time()
    if _cache["data"] is None or now - _cache["ts"] > CACHE_TTL:
        _cache["data"] = _gather(tc, dc)
        _cache["ts"] = now
    return _cache["data"]


def _summary(tc, dc=None) -> dict:
    """Compact day-by-day P/L for all three bots (ORB baseline, news-edge, lottery)
    for the Summary tab. Reuses the cached per-account gathers — no extra Alpaca calls.
    Each entry carries that account's daily_pnl rows (same shape the P/L breakdown uses)."""
    out = {"generated": datetime.now(ET).isoformat(timespec="seconds"), "bots": []}

    def _add(key: str, name: str, getter):
        try:
            data = getter()
        except Exception as e:
            out["bots"].append({"key": key, "name": name, "daily_pnl": [], "error": str(e)})
            return
        if not data:
            out["bots"].append({"key": key, "name": name, "daily_pnl": [], "absent": True})
            return
        out["bots"].append({
            "key": key, "name": name,
            "daily_pnl": data.get("daily_pnl", []),
            "week_pnl": data.get("week_pnl"),
            "month_pnl": data.get("month_pnl"),
            "deployed_avg": data.get("deployed_avg"),
            "deployed_peak": data.get("deployed_peak"),
            "window_pnl": data.get("window_pnl"),
        })

    _add("orb", "ORB baseline", lambda: _status(tc, dc))
    _add("news", "News-Edge", _news_status)
    _add("lottery", "Lottery", _lottery_status)
    # source-pick daily performance (measured-only signals) for the bots-vs-sources grid
    try:
        out["source_daily"] = _source_daily()
    except Exception:
        out["source_daily"] = {}
    return out


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------
PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ORB status</title>
<style>
  :root{ --bg:#0e1116; --card:#171c24; --line:#262e3a; --txt:#d6dde7; --dim:#7c8694;
         --green:#22a35a; --red:#d63b40; --orange:#e0871f; --grey:#5a6675; }
  *{box-sizing:border-box} body{margin:0;background:#0e1116;color:#d6dde7;
    font:14px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
  .wrap{max-width:880px;margin:0 auto;padding:18px}
  .banner{border-radius:12px;padding:18px 20px;margin-bottom:16px;border:1px solid var(--line)}
  .banner .dot{display:inline-block;width:14px;height:14px;border-radius:50%;margin-right:10px;vertical-align:middle}
  .banner h1{margin:0;font-size:20px;display:inline;vertical-align:middle}
  .banner p{margin:8px 0 0;color:#c8d2de}
  .s-alive{background:#10231a;border-color:#1d4d2e}  .s-alive .dot{background:#22a35a;box-shadow:0 0 10px #22a35a}
  .s-warning{background:#241c0e}  .s-warning .dot{background:#e0871f;box-shadow:0 0 10px #e0871f}
  .s-down{background:#27110f}  .s-down .dot{background:#d63b40;box-shadow:0 0 12px #d63b40}
  .s-idle,.s-done{background:#171c24}  .s-idle .dot,.s-done .dot{background:#5a6675}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin-bottom:16px}
  .stat{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
  .stat .k{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.05em}
  .stat .v{font-size:18px;margin-top:4px}
  .pos{color:#3fbf72} .neg{color:#e0594f}
  .card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin-bottom:14px}
  .card h2{margin:0 0 10px;font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--dim)}
  table{width:100%;border-collapse:collapse;font-size:13px}
  th{text-align:right;color:var(--dim);font-weight:500;padding:4px 8px;border-bottom:1px solid var(--line)}
  th:first-child,td:first-child{text-align:left}
  td{text-align:right;padding:5px 8px;border-bottom:1px solid #1c232d}
  tr:last-child td{border-bottom:none}
  .empty{color:var(--dim);font-style:italic}
  .foot{color:var(--dim);font-size:12px;margin-top:18px;display:flex;justify-content:space-between}
  .err{color:#e0871f;font-size:12px;margin-top:8px}
  .tabs{display:flex;gap:6px;margin-bottom:10px}
  .tab{background:#0e1116;color:var(--dim);border:1px solid var(--line);border-radius:7px;
       padding:5px 12px;font:inherit;font-size:12px;cursor:pointer}
  .tab:hover{color:var(--txt)}
  .tab.active{background:#1f2630;color:var(--txt);border-color:#3a4350}
  .tab.toggle.active{background:#10231a;color:#3fbf72;border-color:#1d4d2e}
  .hint{color:var(--dim);font-size:11px;margin-top:8px}
  #tip{position:fixed;z-index:60;max-width:300px;background:#0b0e13;border:1px solid #3a4350;
       color:#d6dde7;font-size:12px;line-height:1.5;padding:8px 10px;border-radius:8px;
       box-shadow:0 8px 24px rgba(0,0,0,.55);pointer-events:none;opacity:0;transition:opacity .08s}
  #tip.show{opacity:1}
  th[data-tip],.k[data-tip]{cursor:help;text-decoration:underline dotted #586273;text-underline-offset:3px}
  details.explain summary{cursor:pointer;color:var(--txt);font-size:13px;list-style:none}
  details.explain summary::before{content:"▸ ";color:var(--dim)}
  details.explain[open] summary::before{content:"▾ "}
  details.explain .ex{margin:10px 0 4px;color:#b6c0cc;font-size:12.5px;line-height:1.55}
  details.explain .ex b{color:var(--txt)}
  details.explain .ex .pts{color:var(--dim);font-size:11px}
  details.explain h3{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--dim);margin:14px 0 2px}
  .cap{color:#8a95a3;font-size:11.5px;line-height:1.5;margin:4px 0 12px;border-left:2px solid #262e3a;padding-left:8px}
  tr.dim td{opacity:.4}
  .hivol{color:var(--orange);font-size:10px;margin-left:6px;white-space:nowrap}
</style></head>
<body><div class="wrap" id="root">loading…</div>
<script>
const money = v => (v<0?"-$":"$") + Math.abs(v).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2});
const sign = v => (v>=0?"+":"") + money(v);
const cls = v => v>=0 ? "pos":"neg";
function row(cells, trCls){ return `<tr${trCls?` class="${trCls}"`:""}>`+cells.map((c,i)=>`<td${c.cls?` class="${c.cls}"`:""}>${c.v}</td>`).join("")+"</tr>"; }
// ---- hover tooltips: one glossary, matched to a header/label's text on hover ----
const GLOSSARY={
 "sym":"Ticker symbol of the stock.",
 "side":"Direction: long = bought expecting it to rise (ORB only trades long).",
 "qty":"Number of shares.",
 "avg":"Average price you paid to enter the position.",
 "last":"Most recent traded price.",
 "value":"Current market value of the position (shares × last price).",
 "invested":"Dollars actually put into the position (cost basis) — the capital at work, not the whole account.",
 "avg invested":"Average daily capital deployed over the period (the same money is redeployed each day, so it isn't summed).",
 "or low":"Low of the first 15 minutes (the 'opening range'). A long breakout's protective stop sits here.",
 "or high":"High of the first 15 minutes. A long entry triggers when price breaks above this.",
 "risk $":"Dollars at risk right now = distance from entry to the protective stop × shares held.",
 "unreal p/l":"Unrealized profit/loss — paper gain/loss on positions still open (not yet sold).",
 "realized p/l":"Profit/loss locked in by closing the position today.",
 "realized":"Profit/loss locked in by closing the position.",
 "entry":"Average price the position was opened at.",
 "exit":"Average price the position was closed at.",
 "target":"Take-profit price (legacy bracket leg; the live bot now mostly uses a trailing stop instead).",
 "stop":"Protective stop price — the position auto-sells here to cap the loss.",
 "status":"Order state at the broker (e.g. new, held, filled).",
 "fill":"Price at which the entry order actually executed.",
 "id":"Broker order ID for this entry (for cross-referencing logs).",
 "date":"Trading day.",
 "week":"Week (Monday-dated) the rows are grouped into.",
 "month":"Calendar month the rows are grouped into.",
 "p/l":"Profit or loss in dollars.",
 "%":"Return on the capital actually invested that day (P/L ÷ invested) — NOT on total account equity.",
 "s&p 500 %":"How the S&P 500 (SPY) moved over the same period — the market benchmark to beat.",
 "equity":"Total account value (cash + positions) at the end of that period.",
 "signal":"The morning scan's call: ▲ +1 frontrunner (bullish catalyst), ▼ −1 avoid (bearish), 0 neutral, ctrl = mechanical control.",
 "held":"✓ = this name is currently held in the news bot's account.",
 "conf":"The analyst's confidence in the call, 0–100%.",
 "gap%":"How far the stock opened today versus yesterday's close.",
 "day%":"Total move so far today versus yesterday's close.",
 "vs or":"Whether price is above the opening-range high (a breakout — what the bot trades), inside it, or below the low.",
 "9:45 px":"The stock's price at 9:45 ET — the reference point the after-close scorer measures from.",
 "since 9:45":"Move since 9:45 ET. This IS the 'buy the pick outright, no ORB' result, updating live.",
 "9:45→close":"The pick's actual move from 9:45 ET to the close — the outcome the experiment scores.",
 "hit":"✓ if the pick's direction was right (a +1 that rose, or a −1 that fell), ✗ if not.",
 "reason":"The catalyst/rationale the scan recorded for the call.",
 "ma50":"50-day average closing price (~10 weeks) — the medium-term trend line.",
 "ma120":"120-day average closing price (~6 months).",
 "ma200":"200-day average closing price (~10 months) — the major bull/bear dividing line.",
 "200d slope":"Direction of the 200-day average over the last month — rising = long-term uptrend intact.",
 "rsi":"RSI-14 (0–100): under 35 washed out, 45–55 neutral, over 70 overbought. A momentum speedometer.",
 "macd hist":"Momentum gauge: above 0 = rising, below 0 = falling. The slope (growing/shrinking) matters most.",
 "off 52w hi":"How far below the 52-week high — small = near highs (healthy), under −15% = serious damage.",
 "20d vol":"Annualized size of daily swings over 20 days. When elevated, the live bot halves its risk automatically.",
 "pts":"Points this reading contributes to the structure or momentum score (+ bullish, − bearish).",
 "component":"The individual indicator being scored.",
 "reading":"Its current value and what it implies.",
 "event":"A level/condition worth watching — if reached it would flip part of the read.",
 "trigger":"The value at which this event fires.",
 "now":"The current value, for comparison with the trigger.",
 "effect":"What changes in the verdict if the trigger is hit.",
 // stat-card labels
 "day p/l":"Profit/loss so far today (vs yesterday's close).",
 "week p/l":"Profit/loss over the last 7 days.",
 "month p/l":"Profit/loss over the last 30 days.",
 "cash":"Uninvested cash in the account.",
 "buying power":"Maximum position value you can hold — equity × intraday margin (up to 4×).",
 "days logged":"Number of days the news scan has recorded picks.",
 "scored picks":"Picks that have a measured 9:45→close outcome.",
 "verdict":"The overall market read, combining the structure and momentum scores.",
};
function attachTips(){
  document.querySelectorAll("#root th, #root .stat .k").forEach(el=>{
    if(el.hasAttribute("data-tip")) return;
    const g=GLOSSARY[el.textContent.trim().toLowerCase()];
    if(g) el.setAttribute("data-tip", g);
  });
}
(function initTips(){
  const tip=document.createElement("div"); tip.id="tip"; document.body.appendChild(tip);
  function show(el){
    const t=el.getAttribute("data-tip"); if(!t) return;
    tip.textContent=t; tip.classList.add("show");
    const r=el.getBoundingClientRect(), w=tip.offsetWidth, h=tip.offsetHeight;
    let x=r.left+r.width/2-w/2; x=Math.max(8,Math.min(x,innerWidth-w-8));
    let y=r.top-h-8; if(y<8) y=r.bottom+8;
    tip.style.left=x+"px"; tip.style.top=y+"px";
  }
  document.addEventListener("mouseover",e=>{ const el=e.target.closest("[data-tip]"); if(el) show(el); });
  document.addEventListener("mouseout",e=>{ if(e.target.closest("[data-tip]")) tip.classList.remove("show"); });
  const root=document.getElementById("root");
  if(root) new MutationObserver(attachTips).observe(root,{childList:true,subtree:true});
  attachTips();
})();
function pctTxt(v){ return (v>=0?"+":"")+v.toFixed(2)+"%"; }
function winStat(label, w){
  if(!w) return `<div class="stat"><div class="k">${label}</div><div class="v">—</div></div>`;
  const sub = (w.pnl_pct_inv===null||w.pnl_pct_inv===undefined) ? "" : `<br><small>${pctTxt(w.pnl_pct_inv)} on inv.</small>`;
  return `<div class="stat"><div class="k">${label}</div><div class="v ${cls(w.pnl)}">${sign(w.pnl)}${sub}</div></div>`;
}
// Monday (ISO) of the week containing an ISO date string.
function weekStart(iso){
  const dt=new Date(iso+"T00:00:00Z");
  dt.setUTCDate(dt.getUTCDate()-((dt.getUTCDay()+6)%7));
  return dt.toISOString().slice(0,10);
}
// Roll daily rows up into period buckets. keyFn maps a date->bucket key;
// rows are newest-first, buckets returned newest-first. P/L sums; "invested"
// is the AVERAGE daily capital deployed over the bucket's trading days (the
// same money is redeployed each day, so summing it would overstate capital);
// % = period P/L / that average (return on typical daily capital); S&P 500 %
// compounds the daily returns; equity = the most recent day's equity.
function groupRows(rows, keyFn, labelFn){
  const map=new Map();
  for(const r of rows.slice().reverse()){            // oldest-first for compounding
    const k=keyFn(r.date);
    let g=map.get(k);
    if(!g){ g={label:labelFn(r.date),pnl:0,investedSum:0,days:0,spyMul:1,hasSpy:false,equity:r.equity}; map.set(k,g); }
    g.pnl+=r.pnl; g.investedSum+=r.invested; g.days++; g.equity=r.equity;
    if(r.spy_pct!=null){ g.spyMul*=(1+r.spy_pct/100); g.hasSpy=true; }
  }
  return Array.from(map.values()).reverse().map(g=>{
    const avgInv = g.days>0 ? g.investedSum/g.days : 0;
    return {
      label:g.label, pnl:g.pnl, invested:avgInv, equity:g.equity,
      pnl_pct_inv: avgInv>0 ? g.pnl/avgInv*100 : null,
      spy_pct: g.hasSpy ? (g.spyMul-1)*100 : null,
    };
  });
}
let lastData=null;
let topView="trading";
let lastNews=null;
// Per-tab P/L-breakdown view state (each tab keeps its own Day/Week/Month + calm toggle).
let plState={trading:{view:"day",calm:false},news:{view:"day",calm:false},lottery:{view:"day",calm:false}};
function rerenderPL(tab){
  if(tab==="news"){ if(lastNews) renderNews(lastNews); }
  else if(tab==="lottery"){ if(lastLottery) renderLottery(lastLottery); }
  else if(lastData) render(lastData);
}
function setPLView(tab,v){ plState[tab].view=v; rerenderPL(tab); }
function setCalm(tab){ plState[tab].calm=!plState[tab].calm; rerenderPL(tab); }
// Shared "P/L breakdown" card — identical table for the ORB, News-Edge and Lottery tabs.
// dp = that account's daily_pnl rows (from _gather); tab keys into plState for view/calm.
function plBreakdownCard(dp, todayStr, tab){
  dp=dp||[];
  const st=plState[tab], plView=st.view, calmOnly=st.calm;
  const calmKnown=dp.some(r=>r.calm!=null);
  const dpView = calmOnly ? dp.filter(r=>r.calm===true) : dp;
  let h=`<div class="card"><h2>P/L breakdown</h2>`;
  h+=`<div class="tabs">`+["day","week","month"].map(v=>
       `<button class="tab${plView===v?" active":""}" onclick="setPLView('${tab}','${v}')">${v[0].toUpperCase()+v.slice(1)}</button>`).join("")
     +(calmKnown?`<button class="tab toggle${calmOnly?" active":""}" onclick="setCalm('${tab}')" title="Count only low-volatility (calm) days — where ORB does better. Does not change live trading.">Calm days only</button>`:``)+`</div>`;
  if(calmKnown){
    const calmRows=dp.filter(r=>r.calm===true);
    const tAll=dp.reduce((s,r)=>s+r.pnl,0), tCalm=calmRows.reduce((s,r)=>s+r.pnl,0);
    h+=`<div class="hint">vol filter: <b>${calmRows.length}/${dp.length}</b> days calm · P/L calm-only <span class="${cls(tCalm)}">${sign(tCalm)}</span> vs all-days <span class="${cls(tAll)}">${sign(tAll)}</span> &nbsp;(high-vol days marked <span class="hivol">hi-vol</span>)</div>`;
  }
  if(dpView.length){
    let rowsV, lblHead, fmtLbl, invHead="invested";
    if(plView==="week"){
      rowsV=groupRows(dpView, weekStart, dt=>"wk of "+weekStart(dt)); lblHead="week"; invHead="avg invested"; fmtLbl=r=>r.label;
    } else if(plView==="month"){
      rowsV=groupRows(dpView, dt=>dt.slice(0,7), dt=>dt.slice(0,7)); lblHead="month"; invHead="avg invested"; fmtLbl=r=>r.label;
    } else {
      rowsV=dpView; lblHead="date";
      fmtLbl=r=>(r.date===todayStr?`${r.date} <small>(today)</small>`:r.date)+(r.calm===false?` <span class="hivol">hi-vol</span>`:``);
    }
    h+=`<table><tr><th>${lblHead}</th><th>${invHead}</th><th>P/L</th><th>%</th><th>S&P 500 %</th><th>equity</th></tr>`;
    rowsV.forEach((r)=>{
      const pct = (r.pnl_pct_inv===null||r.pnl_pct_inv===undefined)
        ? {v:"—"}
        : {v:`${r.pnl_pct_inv>=0?"+":""}${r.pnl_pct_inv.toFixed(1)}%`,cls:cls(r.pnl_pct_inv)};
      const spy = (r.spy_pct===null||r.spy_pct===undefined)
        ? {v:"—"}
        : {v:`${r.spy_pct>=0?"+":""}${r.spy_pct.toFixed(2)}%`,cls:cls(r.spy_pct)};
      // high-vol days keep their "hi-vol" label but are NOT faded (full contrast).
      h+=row([{v:fmtLbl(r)},{v:r.invested>0?money(r.invested):"—"},{v:sign(r.pnl),cls:cls(r.pnl)},pct,spy,{v:money(r.equity)}]); });
    h+=`</table>`;
    h+=`<div class="hint">% = P/L vs capital invested (avg daily capital for Week/Month) · S&P 500 % = SPY close-to-close (compounded) · calm = SPY 20d vol below its trailing median</div>`;
  } else h+=`<div class="empty">${calmOnly?"no calm days in range":"no history"}</div>`;
  h+=`</div>`;
  return h;
}
function topNav(active){
  return `<div class="tabs" style="margin-bottom:14px">`+[["trading","Trading"],["news","News-Edge"],["lottery","Hype"],["summary","Summary"],["regime","Market"]].map(
    ([k,l])=>`<button class="tab${active===k?" active":""}" onclick="setTopView('${k}')">${l}</button>`).join("")+`</div>`;
}
let lastRegime=null, lastLottery=null, lastSummary=null;
function showLoading(tab){
  const r=document.getElementById("root");
  if(r) r.innerHTML=topNav(tab)+`<div class="card"><div class="empty">loading…</div></div>`;
}
// Rebuild #root WITHOUT losing scroll position. The 3s auto-refresh replaces the DOM,
// which otherwise snaps the page (and any horizontally-scrolled wide table) back to the
// top-left. Preserve the page scroll + every inner ".scrollx" container's scrollLeft.
function setRoot(html){
  const root=document.getElementById("root");
  if(!root){ return; }
  const sx=window.pageXOffset, sy=window.pageYOffset;
  const inner=[].map.call(root.querySelectorAll(".scrollx"), e=>e.scrollLeft);
  root.innerHTML=html;
  const now=root.querySelectorAll(".scrollx");
  for(let i=0;i<inner.length && i<now.length;i++){ now[i].scrollLeft=inner[i]; }
  window.scrollTo(sx, sy);
}
// One-time, on-load warm of the bot tabs so even the FIRST click on each is instant
// (stores the payload, does NOT render — we may be on another tab). Sequential so only
// one request is in flight at a time: gentle on the 1 GB VM, and /api/summary then reuses
// the freshly-warmed news/lottery server caches.
let _prefetched=false;
async function prefetchTabs(){
  if(_prefetched) return; _prefetched=true;
  const grab=async(url,set)=>{ try{ const r=await fetch(url,{cache:"no-store"}); set(await r.json()); }catch(e){} };
  await grab("/api/newsedge", d=>{lastNews=d;});
  await grab("/api/lottery", d=>{lastLottery=d;});
  await grab("/api/summary", d=>{lastSummary=d;});
}
function setTopView(v){
  topView=v;
  // Paint the tab IMMEDIATELY from the last-known payload (or a loading shell on first
  // visit) so switching feels instant; the fetch then refreshes it in place a moment
  // later. Avoids staring at the previous tab while the server round-trip completes.
  if(v==="news"){ lastNews?renderNews(lastNews):showLoading("news"); fetchNews(); }
  else if(v==="lottery"){ lastLottery?renderLottery(lastLottery):showLoading("lottery"); fetchLottery(); }
  else if(v==="summary"){ lastSummary?renderSummary(lastSummary):showLoading("summary"); fetchSummary(); }
  else if(v==="regime"){ lastRegime?renderRegime(lastRegime):showLoading("regime"); fetchRegime(); }
  else { lastData?render(lastData):showLoading("trading"); }
}
let lastRegimeTxt=null;
let explainOpen=false;
function toggleExplain(v){
  explainOpen=v;
  const ex=document.querySelector("details.explain");
  if(ex) ex.open=v;
}
async function fetchRegime(){
  try{
    const r=await fetch("/api/regime",{cache:"no-store"});
    const txt=await r.text();
    // identical payload + tab already rendered -> leave the DOM alone, so an
    // open explainer (or text selection) survives the 3s poll
    if(txt===lastRegimeTxt && document.getElementById("regime-marker")) return;
    lastRegimeTxt=txt;
    lastRegime=JSON.parse(txt);
    if(topView==="regime") renderRegime(lastRegime);   // don't clobber a tab switched to mid-flight
  }
  catch(e){ if(topView==="regime") document.getElementById("root").innerHTML=topNav("regime")+`<div class="card empty">market regime data unavailable</div>`; }
}
// Tiny SVG line chart: series=[{vals,color,w}], guides=[{y,label}]. Drawn in the
// browser from the numbers the API ships — zero chart libs, zero VM load.
function spark(series, opts){
  opts=opts||{};
  const w=760, hh=opts.h||64;
  const all=series.flatMap(s=>s.vals).filter(v=>v!=null);
  if(!all.length) return "";
  let mn=Math.min(...all), mx=Math.max(...all);
  for(const g of (opts.guides||[])){ mn=Math.min(mn,g.y); mx=Math.max(mx,g.y); }
  const rg=(mx-mn)||1, n=series[0].vals.length;
  const X=i=>(i/(n-1))*w, Y=v=>hh-3-((v-mn)/rg)*(hh-8);
  let out="";
  for(const g of (opts.guides||[]))
    out+=`<line x1="0" x2="${w}" y1="${Y(g.y).toFixed(1)}" y2="${Y(g.y).toFixed(1)}" stroke="#39424f" stroke-dasharray="3,5"/>`
        +(g.label?`<text x="${w-4}" y="${(Y(g.y)-3).toFixed(1)}" text-anchor="end" font-size="9" fill="#5a6675">${g.label}</text>`:"");
  for(const s of series){
    const pts=s.vals.map((v,i)=>v==null?null:X(i).toFixed(1)+","+Y(v).toFixed(1)).filter(Boolean).join(" ");
    out+=`<polyline points="${pts}" fill="none" stroke="${s.color}" stroke-width="${s.w||1.4}"/>`;
  }
  return `<svg viewBox="0 0 ${w} ${hh}" width="100%" height="${hh}" preserveAspectRatio="none" style="display:block">${out}</svg>`;
}
function chartLabel(parts){
  return `<div class="hint" style="margin:10px 0 2px">`+parts.map(([t,c])=>`<span style="color:${c}">● ${t}</span>`).join(" &nbsp; ")+`</div>`;
}
// Plain-language basis of the analysis — static text, one entry per vote.
const REGIME_EXPLAIN = `
<div class="card"><details class="explain"><summary><b>How to read this page</b> — every parameter explained in plain language</summary>
<div class="ex">No single number decides anything here. Each indicator below casts a small <b>vote</b> (its points are shown next to it in the tables above), and the votes add up into two separate scores. Think of it like a doctor reading vitals: one bad reading is a note, several together are a diagnosis.</div>

<h3>Why two scores?</h3>
<div class="ex"><b>STRUCTURE</b> (±7) answers "what kind of market has this been for months?" — it moves slowly and ignores any single bad week. <b>MOMENTUM</b> (±6) answers "which way is it moving right now?" — it reacts within days. They often disagree, and the disagreement IS the information: a falling week inside a healthy year is a pullback; a rising week inside a broken year is a bounce. The headline verdict is just the combination of the two.</div>

<h3>Structure votes (the slow axis)</h3>
<div class="ex"><b>Price vs 200-day moving average</b> <span class="pts">(±2 — the heavyweight)</span><br>
The 200d MA is simply the average closing price of the last ~10 months — the most-watched line in all of markets. Price above it = the long-term trend is up; below it = broken. It gets double weight because history is unambiguous: most of the stock market's gains happen above this line, and most disasters happen below it. Our live bot's trend filter uses the same line.</div>
<div class="ex"><b>200d MA slope</b> <span class="pts">(±1)</span><br>
Whether that 10-month average itself is rising or falling. Price can pop above a still-falling average in a bear-market rally — the slope catches that trick.</div>
<div class="ex"><b>50d vs 200d MA ("golden/death cross")</b> <span class="pts">(±1)</span><br>
The 10-week average above the 10-month average = "golden cross" (the recent past is stronger than the distant past — healthy). Below = "death cross." Slow but famously hard to fake.</div>
<div class="ex"><b>Breadth above 200d MA</b> <span class="pts">(±2 — the other heavyweight)</span><br>
The % of our own 122 tradable names that sit above their own 200d line. This is the honesty check on the index: SPY can be held up by five mega-caps while the average stock quietly breaks down. Above 55% = broad health; below 45% = the average stock is already in a downtrend regardless of what the index says.</div>
<div class="ex"><b>Distance from the 52-week high</b> <span class="pts">(±1)</span><br>
Within 5% of the 1-year high = near the highs (healthy). More than 15% below = serious damage. (Trader vocabulary: 5–10% down = "pullback/correction," 20%+ = "bear market.")</div>

<h3>Momentum votes (the fast axis)</h3>
<div class="ex"><b>5-day return</b> <span class="pts">(±1)</span><br>
Literally just the last week, with a ±1% dead zone so ordinary wiggle doesn't vote. The bluntest, fastest input.</div>
<div class="ex"><b>Price vs 50-day MA</b> <span class="pts">(±1)</span><br>
The ~10-week average — the medium-term line trend traders defend. Holding above it during a slide means the dip is still orderly; losing it is the first real crack.</div>
<div class="ex"><b>MACD (12-26-9)</b> <span class="pts">(±1)</span><br>
Compares a fast (12-day) average of price against a slow (26-day) one. When the fast falls below the slow, recent buying has weakened — momentum turned down. The "histogram" number is how hard it's leaning; its direction day to day shows the move accelerating or fading.</div>
<div class="ex"><b>RSI-14</b> <span class="pts">(±1)</span><br>
A 0–100 speedometer of the last 14 days' up-moves vs down-moves. Above 55 = buyers in charge; below 45 = sellers in charge; below 35 = "washed out" — historically the zone where selling exhausts itself and dips become buyable (which is why 35 is the dip trigger below).</div>
<div class="ex"><b>Volatility regime</b> <span class="pts">(±1)</span><br>
The exact same measure the live bot uses to cut its risk in half: are SPY's daily swings bigger than their 6-month norm? Calm markets drift up; stressed ones whipsaw. When this is ELEVATED, the bot is already trading half-size automatically.</div>
<div class="ex"><b>Breadth above 50d MA</b> <span class="pts">(±1)</span><br>
Short-term participation: what % of our names are above their own 10-week line. Falling breadth here shows a sell-off spreading; recovering breadth is usually the first sign of repair.</div>

<h3>From scores to the verdict</h3>
<div class="ex">Structure ≥ +3 → <b>UP</b>, ≤ −3 → <b>BROKEN</b>, between → MIXED. Momentum ≥ +2 → <b>RISING</b>, ≤ −2 → <b>FALLING</b>, between → FLAT. The pair picks one of nine labels (e.g. UP + FALLING = "uptrend under pressure"). The banner color follows the label, not just the math.</div>

<h3>The dip read</h3>
<div class="ex">One rule with strong historical support: a pullback is only worth buying when the <b>long-term structure is intact</b> (price above a rising 200d MA) <b>and</b> price has actually stretched (RSI under 35, or below the 50d MA while 3–12% off the high). Same chart below a broken 200d MA = a falling knife, not a dip. "DIP FORMING" means the first condition holds and the second is approaching — wait for the trigger, don't front-run it.</div>

<h3>The turn checklist</h3>
<div class="ex">Only matters when structure is BROKEN. Bottoms don't announce themselves, but they tend to share fingerprints: an RSI washout that recovers, MACD turning back up, price reclaiming the 50d line, volatility compressing, breadth repairing. 4+ of 6 lit up = a turn forming; 2 or fewer = the downtrend is intact.</div>

<h3>The honest caveat</h3>
<div class="ex">All of this <b>describes</b> the tape; none of it <b>predicts</b> it — in our own backtests, indicator-based entry signals keep failing to add value over the validated system. Use this page for context and sizing courage, not as a trade signal. The live bot doesn't read it; its own validated regime logic (vol-dial + trend filter) is unchanged.</div>
<button class="tab" style="margin-top:10px" onclick="toggleExplain(false)">▴ Close explainer</button>
</details></div>`;
function renderRegime(rd){
  let h=topNav("regime");
  if(rd.error){
    document.getElementById("root").innerHTML=h+`<div class="card empty">${rd.error}</div>`;
    return;
  }
  const bcls = {good:"s-alive",caution:"s-warning",bad:"s-down",neutral:"s-idle"}[rd.tone]||"s-idle";
  const ss=rd.struct_score, ms=rd.mom_score;
  h+=`<div class="banner ${bcls}"><span class="dot"></span><h1>${rd.verdict}</h1><p>structure <b>${ss>=0?"+":""}${ss}</b>/±7 (${rd.struct_state}) · momentum <b>${ms>=0?"+":""}${ms}</b>/±6 (${rd.mom_state}) · daily closes through ${rd.asof} · refreshes ~15 min</p></div>`;
  // risk-off radar — cross-asset "market-feel" safety read (descriptive, not predictive)
  const ro=rd.risk_off||{};
  if(ro.read){
    const readHtml = ro.read==="NORMAL" ? `<span class="pos">NORMAL</span>`
      : ro.read==="WATCH" ? `<span style="color:var(--orange)">WATCH</span>`
      : `<span class="neg">ELEVATED</span>`;
    const leanCls = l => (l==="risk-off"||l==="stress") ? "neg" : l==="risk-on" ? "pos" : "";
    h+=`<div class="card"><h2>Risk-off radar — ${readHtml} <small style="color:var(--dim)">(${ro.n_disc||0}/2 key signals risk-off)</small></h2>`;
    h+=`<table><tr><th style="text-align:left">signal</th><th style="text-align:left">reading</th><th>lean</th></tr>`;
    for(const s of (ro.signals||[])){
      const nm = s.key ? `<b>${s.name}</b>` : `<span style="color:#9aa6b4">${s.name}</span>`;
      h+=`<tr><td style="text-align:left">${nm}</td><td style="text-align:left;color:#9aa6b4">${s.value}</td><td class="${leanCls(s.lean)}">${s.lean}</td></tr>`;
    }
    h+=`</table><div class="hint">${ro.note||""}</div></div>`;
  }
  // dip read — the question the tab exists to answer
  const d=rd.dip||{};
  h+=`<div class="card"><h2>Buy-the-dip read</h2>
    <div class="grid">
      <div class="stat"><div class="k">Long-term structure</div><div class="v ${d.structure?"pos":"neg"}">${d.structure?"INTACT":"BROKEN"}</div></div>
      <div class="stat"><div class="k">Short-term stretch</div><div class="v">${d.oversold?"OVERSOLD":"not oversold"}</div></div>
      <div class="stat" style="grid-column:span 2"><div class="k">Verdict</div><div class="v">${d.verdict||"—"}</div></div>
    </div>
    <div class="hint">${d.note||""}</div></div>`;
  h+=`<span id="regime-marker" style="display:none"></span>`;
  h+=REGIME_EXPLAIN.replace('<details class="explain"', explainOpen?'<details class="explain" open':'<details class="explain"');
  // index table
  const idx=rd.indexes||{};
  h+=`<div class="card"><h2>Indexes (daily)</h2><table><tr><th>sym</th><th>last</th><th>MA50</th><th>MA120</th><th>MA200</th><th>200d slope</th><th>RSI</th><th>MACD hist</th><th>off 52w hi</th><th>20d vol</th></tr>`;
  for(const [sym,a] of Object.entries(idx)){
    h+=row([{v:sym},{v:a.px.toFixed(0)},{v:a.ma["50"].toFixed(0)},{v:a.ma["120"].toFixed(0)},{v:a.ma["200"].toFixed(0)},
            {v:a.ma200_rising?"rising":"falling",cls:a.ma200_rising?"pos":"neg"},
            {v:a.rsi.toFixed(0)},{v:(a.macd_hist>=0?"+":"")+a.macd_hist.toFixed(2),cls:cls(a.macd_hist)},
            {v:pctTxt(a.dd52),cls:cls(a.dd52)},{v:a.vol20_ann.toFixed(0)+"%"+(a.vol_hot?" ⚠":"")}]);
  }
  h+=`</table></div>`;
  // breadth
  const br=rd.breadth||{};
  h+=`<div class="card"><h2>Breadth — our ${br.n||0}-name watchlist</h2><div class="grid">
    <div class="stat"><div class="k">Above 200d MA</div><div class="v ${br.pct200>55?"pos":br.pct200<45?"neg":""}">${(br.pct200||0).toFixed(0)}%</div></div>
    <div class="stat"><div class="k">Above 50d MA</div><div class="v ${br.pct50>55?"pos":br.pct50<45?"neg":""}">${(br.pct50||0).toFixed(0)}%</div></div>
  </div><div class="hint">breadth leads the index at turns — indexes can be held up by a few mega-caps while the average stock is already in a downtrend</div></div>`;
  // votes — two groups: slow structure axis, fast momentum axis
  for(const [grp,lim,score,state] of [["structure",7,ss,rd.struct_state],["momentum",6,ms,rd.mom_state]]){
    const sCls = score>=(grp==="structure"?3:2)?"pos":score<=-(grp==="structure"?3:2)?"neg":"";
    h+=`<div class="card"><h2>${grp} — <span class="${sCls}">${score>=0?"+":""}${score}</span> of ±${lim} (${state})</h2><table><tr><th>pts</th><th style="text-align:left">component</th><th style="text-align:left">reading</th></tr>`;
    for(const v of (rd.votes||[])){
      if(v.grp!==grp) continue;
      const p=v.pts>0?`+${v.pts}`:`${v.pts}`;
      h+=`<tr><td class="${v.pts>0?"pos":v.pts<0?"neg":""}">${p}</td><td style="text-align:left">${v.name}</td><td style="text-align:left;color:#9aa6b4">${v.detail}</td></tr>`;
    }
    h+=`</table></div>`;
  }
  // charts — last 120 sessions, drawn client-side as SVG
  const H=rd.history;
  if(H && H.spy_close){
    const last=a=>{ for(let i=a.length-1;i>=0;i--) if(a[i]!=null) return a[i]; return null; };
    h+=`<div class="card"><h2>Charts — last ${H.dates.length} sessions (${H.dates[0]} → ${H.dates[H.dates.length-1]})</h2>`;
    h+=chartLabel([[`SPY ${last(H.spy_close)}`,"#d6dde7"],[`MA50 ${Math.round(last(H.spy_ma50))}`,"#e0871f"],[`MA200 ${Math.round(last(H.spy_ma200))}`,"#5a8bd6"]]);
    h+=spark([{vals:H.spy_close,color:"#d6dde7",w:1.7},{vals:H.spy_ma50,color:"#e0871f"},{vals:H.spy_ma200,color:"#5a8bd6"}],{h:120});
    h+=`<div class="cap"><b>Price vs its averages</b> — SPY (white) with its 50-day (orange) and 200-day (blue) average closing prices. The market's skeleton: price above both rising averages = healthy uptrend. In a pullback, watch the orange 50d line — holding above it = an orderly dip; losing it = the correction deepening. The blue 200d line is the bull/bear divide itself.</div>`;
    h+=chartLabel([[`RSI-14: ${Math.round(last(H.rsi))}`,"#3fbf72"]]);
    h+=spark([{vals:H.rsi,color:"#3fbf72"}],{h:56,guides:[{y:35,label:"35 washout"},{y:55,label:"55"}]});
    h+=`<div class="cap"><b>RSI-14 (0–100)</b> — a speedometer comparing the size of up-days vs down-days over the last 14 sessions. Ranges: <b>70+</b> overbought (rallies often pause) · <b>55–70</b> buyers in charge · <b>45–55</b> neutral · <b>35–45</b> sellers in charge · <b>under 35</b> washed out — selling tends to exhaust itself there, which is why 35 is the dip trigger. Mid-range readings say little on their own.</div>`;
    h+=chartLabel([[`MACD histogram: ${last(H.macd_hist)>=0?"+":""}${last(H.macd_hist)}`,"#c77dff"]]);
    h+=spark([{vals:H.macd_hist,color:"#c77dff"}],{h:56,guides:[{y:0,label:"0"}]});
    h+=`<div class="cap"><b>MACD histogram</b> — in plain terms: is recent buying stronger or weaker than its own recent norm? Above the 0-line = momentum up, below = momentum down — but the <b>slope</b> matters most: bars growing away from zero = the move accelerating; shrinking back toward zero = the move fading (often the earliest sign a sell-off or rally is ending).</div>`;
    h+=chartLabel([[`breadth >200d: ${Math.round(last(H.breadth200))}%`,"#3fbf72"],[`>50d: ${Math.round(last(H.breadth50))}%`,"#e0871f"]]);
    h+=spark([{vals:H.breadth200,color:"#3fbf72"},{vals:H.breadth50,color:"#e0871f"}],{h:56,guides:[{y:45,label:"45%"},{y:55,label:"55%"}]});
    h+=`<div class="cap"><b>Breadth</b> — the % of our own 122 tradable names above their own 200-day (green) and 50-day (orange) averages. It answers "is the average stock OK, or is the index being carried by a few giants?" Above 55% = broad participation; below 45% = the average stock is already breaking down. Green moves slowly (long-term health); orange reacts fast — it's where a sell-off spreads first and heals first.</div>`;
    h+=chartLabel([[`SPY 20d realized vol: ${Math.round(last(H.vol20))}% ann`,"#e0594f"]]);
    h+=spark([{vals:H.vol20,color:"#e0594f"}],{h:56});
    h+=`<div class="cap"><b>20-day realized volatility</b> — how big SPY's daily swings have <i>actually</i> been over the last 20 sessions, scaled to a yearly number (14% ≈ daily wiggle consistent with ±14%/yr). Calm bull markets typically run ~8–12%; when this rises above its 6-month norm, the live bot automatically cuts position size in half (the "vol-dial"). Spikes mark stress; vol falling back after a spike is one of the turn markers.</div>`;
    h+=`</div>`;
  }
  // what to watch — the concrete trigger levels that would flip the read
  if(rd.levels && rd.levels.length){
    h+=`<div class="card"><h2>What would change this read — levels to watch (not predictions)</h2><table><tr><th style="text-align:left">event</th><th>trigger</th><th>now</th><th style="text-align:left">effect</th></tr>`;
    for(const lv of rd.levels)
      h+=`<tr><td style="text-align:left">${lv.name}</td><td>${lv.trigger}</td><td>${lv.now}</td><td style="text-align:left;color:#9aa6b4">${lv.effect}</td></tr>`;
    h+=`</table><div class="hint">a phone push (same ntfy topic as the bot) fires automatically whenever the verdict or dip read changes</div></div>`;
  }
  // turn checklist
  h+=`<div class="card"><h2>Downtrend-ending checklist — ${rd.n_turn_on}/6 markers on${rd.n_turn_on>=4?" · turn forming":rd.n_turn_on<=2?" · no confirmed turn":""}</h2><table>`;
  for(const c of (rd.turn||[])){
    h+=`<tr><td style="text-align:left">${c.on?'<span class="pos">✓</span>':'<span style="color:#5a6675">✗</span>'}</td><td style="text-align:left;color:${c.on?'#d6dde7':'#7c8694'}">${c.name}</td></tr>`;
  }
  h+=`</table><div class="hint">matters most when structure is BROKEN — these are the classic bottoming markers to wait for before buying weakness in a downtrend</div></div>`;
  h+=`<div class="hint">Descriptive read for the human — indicators describe the tape, they don't predict it. NOT a bot input: the live bot's regime logic (vol-dial + trend filter) is unchanged.</div>`;
  h+=`<div class="foot"><span>regime · ${rd.generated||""}</span><span></span></div>`;
  setRoot(h);
  const ex=document.querySelector("details.explain");
  if(ex) ex.addEventListener("toggle",()=>{ explainOpen=ex.open; });
}
async function fetchNews(){
  try{ const r=await fetch("/api/newsedge",{cache:"no-store"}); lastNews=await r.json(); if(topView==="news") renderNews(lastNews); }
  catch(e){ if(topView==="news") document.getElementById("root").innerHTML=topNav("news")+`<div class="card empty">news-edge data unavailable</div>`; }
}
function renderNews(nd){
  const o=nd.overall||{};
  const sepC=v=>(v>=0?"pos":"neg");
  let h=topNav("news");
  h+=`<div class="banner s-idle"><h1>News-Edge — catalyst-selected ORB</h1><p>A separate paper bot trades the morning's <b>positive</b> news picks (same ORB engine, no trend filter) on its own account — run alongside the baseline for a money-to-money comparison. Plus the daily signal measurement below. Never touches the baseline bot.</p></div>`;
  // --- the bot's live account (money — the whole point of the comparison) ---
  if(nd.bot){
    const b=nd.bot, a=b.account;
    h+=`<div class="card"><h2>News bot — live paper account${a&&a.number?` &nbsp;·&nbsp; #${a.number}`:''}</h2>`;
    if(a){
      h+=`<div class="grid">
        <div class="stat"><div class="k">Equity</div><div class="v">${money(a.equity)}</div></div>
        <div class="stat"><div class="k">Day P/L</div><div class="v ${cls(a.day_pnl)}">${sign(a.day_pnl)}<br><small>${a.day_pnl_pct>=0?"+":""}${a.day_pnl_pct.toFixed(2)}%</small></div></div>
        ${winStat("Week P/L", b.week_pnl)}
        ${winStat("Month P/L", b.month_pnl)}
        <div class="stat"><div class="k">Cash</div><div class="v">${money(a.cash)}</div></div>
        <div class="stat"><div class="k">Invested</div><div class="v">${money(b.invested||0)}</div></div>
      </div>`;
    }
    const bp=b.positions||[];
    h+=`<div style="color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.06em;margin:6px 0 6px">Open positions (${bp.length})</div>`;
    if(bp.length){
      h+=`<table><tr><th>sym</th><th>side</th><th>qty</th><th>avg</th><th>last</th><th>invested</th><th>unreal P/L</th></tr>`;
      for(const p of bp) h+=row([{v:p.symbol},{v:p.side},{v:p.qty.toFixed(0)},{v:money(p.avg_entry)},{v:money(p.current)},{v:money(p.cost_basis)},{v:`${sign(p.unrealized_pl)} <small>(${p.unrealized_plpc>=0?"+":""}${p.unrealized_plpc.toFixed(1)}%)</small>`,cls:cls(p.unrealized_pl)}]);
      h+=`</table>`;
    } else h+=`<div class="empty">flat</div>`;
    const bct=b.closed_today||[];
    const bpnl=bct.reduce((s,c)=>s+c.realized,0);
    h+=`<div style="color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.06em;margin:12px 0 6px">Closed today (${bct.length})${bct.length?` &nbsp;·&nbsp; realized <span class="${cls(bpnl)}">${sign(bpnl)}</span> &nbsp;·&nbsp; <span>${money(bpnl/bct.length)}/trade</span>`:""}</div>`;
    if(bct.length){
      h+=`<table><tr><th>sym</th><th>side</th><th>qty</th><th>entry</th><th>exit</th><th>realized</th></tr>`;
      for(const c of bct) h+=row([{v:c.symbol},{v:c.side},{v:c.qty.toFixed(0)},{v:money(c.entry_avg)},{v:money(c.exit_avg)},{v:sign(c.realized),cls:cls(c.realized)}]);
      h+=`</table>`;
    } else h+=`<div class="empty">no round-trips closed today</div>`;
    if(b.errors&&b.errors.length) h+=`<div class="err">⚠ ${b.errors.join(" · ")}</div>`;
    h+=`</div>`;
  } else {
    h+=`<div class="card"><div class="empty">News bot account not visible here yet (needs .env.news where the status server runs). Picks measurement below still works.</div></div>`;
  }
  // --- P/L breakdown (same table as the ORB tab, for the news bot's account) ---
  if(nd.bot && nd.bot.daily_pnl) h+=plBreakdownCard(nd.bot.daily_pnl, (nd.generated||"").slice(0,10), "news");
  // --- live behavior of today's picks (bought or not) ---
  if(nd.live && Object.keys(nd.live).length){
    const held=new Set(((nd.bot||{}).positions||[]).map(p=>p.symbol));
    const today=(nd.days||[]).find(d=>d.date===nd.live_date)||{picks:[]};
    const sigOrd=p=>p.control?2:(p.signal>0?0:p.signal<0?3:1);
    h+=`<div class="card"><h2>Today's picks — live behavior (${nd.live_date})</h2>`;
    h+=`<table><tr><th>sym</th><th>signal</th><th>held</th><th>last</th><th>gap%</th><th>day%</th><th>OR low</th><th>OR high</th><th>vs OR</th><th>9:45 px</th><th>since 9:45</th></tr>`;
    for(const p of today.picks.slice().sort((a,b)=>sigOrd(a)-sigOrd(b))){
      const L=nd.live[p.symbol]; if(!L) continue;
      const sig=p.control?`<span style="color:#7c8694">ctrl</span>`
        :p.signal>0?`<span class="pos">▲ +1</span>`
        :p.signal<0?`<span class="neg">▼ −1</span>`:`<span style="color:#7c8694">0</span>`;
      const pc=v=>v==null?{v:"—"}:{v:pctTxt(v),cls:cls(v)};
      let vsOR="—";
      if(L.last!=null&&L.or_high!=null)
        vsOR=L.last>L.or_high?`<span class="pos">above OR-high</span>`
          :(L.or_low!=null&&L.last<L.or_low)?`<span class="neg">below OR-low</span>`:`inside`;
      h+=row([{v:p.symbol},{v:sig},{v:held.has(p.symbol)?`<span class="pos">✓</span>`:""},
              {v:L.last!=null?money(L.last):"—"},pc(L.gap_pct),pc(L.day_pct),
              {v:L.or_low!=null?money(L.or_low):"—"},{v:L.or_high!=null?money(L.or_high):"—"},
              {v:vsOR},{v:L.px945!=null?money(L.px945):"—"},pc(L.since945_pct)]);
    }
    h+=`</table><div class="hint">every pick tracked live whether or not the bot bought it · "since 9:45" is the exact reference the after-close scorer uses (the no-ORB buy-and-hold arm) · OR low/high = the 15-min opening range the news bot trades · "held" = currently in the news bot's book</div></div>`;
  }
  // --- the daily signal measurement (does my read predict the day?) ---
  h+=`<div class="card"><h2>Signal measurement — do (+) picks outrun (−)?</h2>`;
  h+=`<div class="grid">
    <div class="stat"><div class="k">Days logged</div><div class="v">${o.n_days||0}</div></div>
    <div class="stat"><div class="k">Scored picks</div><div class="v">${o.n_scored||0}</div></div>
    <div class="stat"><div class="k">(+) avg move</div><div class="v ${o.avg_pos==null?'':sepC(o.avg_pos)}">${o.avg_pos==null?'—':pctTxt(o.avg_pos)}</div></div>
    <div class="stat"><div class="k">(−) avg move</div><div class="v ${o.avg_neg==null?'':sepC(o.avg_neg)}">${o.avg_neg==null?'—':pctTxt(o.avg_neg)}</div></div>
    <div class="stat"><div class="k">(+)−(−) edge</div><div class="v ${o.sep==null?'':sepC(o.sep)}">${o.sep==null?'—':pctTxt(o.sep)}</div></div>
  </div></div>`;
  const days=nd.days||[];
  if(!days.length){ h+=`<div class="card"><div class="empty">No picks logged yet — the first scan happens live near the open (9:30–9:45 ET).</div></div>`; }
  for(const day of days){
    const sb = day.sep==null ? "" : ` &nbsp;·&nbsp; edge <span class="${sepC(day.sep)}">${pctTxt(day.sep)}</span>`;
    h+=`<div class="card"><h2>${day.date} &nbsp;·&nbsp; ${day.n} picks${sb}</h2>`;
    h+=`<table><tr><th>sym</th><th>signal</th><th>conf</th><th>9:45→close</th><th>hit</th><th>reason</th></tr>`;
    for(const p of (day.picks||[])){
      const sig = p.signal>0?`<span class="pos">▲ long</span>`:p.signal<0?`<span class="neg">▼ avoid</span>`:`<span style="color:#7c8694">– neutral</span>`;
      const ret=p.ret_945_close;
      const retCell = ret==null?"—":`<span class="${sepC(ret)}">${pctTxt(ret)}</span>`;
      let hit="";
      if(ret!=null && p.signal!==0){ const ok=(p.signal>0&&ret>0)||(p.signal<0&&ret<0); hit=ok?`<span class="pos">✓</span>`:`<span class="neg">✗</span>`; }
      const reason=(p.reason||"").replace(/&/g,"&amp;").replace(/</g,"&lt;");
      h+=`<tr><td>${p.symbol}</td><td style="text-align:right">${sig}</td><td>${Math.round((p.confidence||0)*100)}%</td><td>${retCell}</td><td>${hit}</td><td style="text-align:left;color:#9aa6b4">${reason}</td></tr>`;
    }
    h+=`</table></div>`;
  }
  h+=`<div class="hint">edge = avg move of (+) picks minus avg move of (−) picks. Positive and growing with the sample = a real signal worth automating. One good week is noise.</div>`;
  h+=`<div class="foot"><span>news-edge · ${nd.generated||''}</span><span></span></div>`;
  setRoot(h);
}
async function fetchLottery(){
  try{ const r=await fetch("/api/lottery",{cache:"no-store"}); lastLottery=await r.json(); if(topView==="lottery") renderLottery(lastLottery); }
  catch(e){ if(topView==="lottery") document.getElementById("root").innerHTML=topNav("lottery")+`<div class="card empty">lottery data unavailable</div>`; }
}
function renderLottery(ld){
  const sb=ld.scoreboard||{};
  const liftC=v=>(v==null?"":(v>=2?"pos":(v<1?"neg":"")));
  const pct1=v=>v==null?"—":(v*100).toFixed(0)+"%";
  let h=topNav("lottery");
  h+=`<div class="banner s-idle"><h1>Hype — can ANY hype metric pick the day's winners?</h1><p>A pre-registered forward test: every morning, mechanically log the top names per hype signal (WSB surge, StockTwits, premarket RVOL, squeeze, options flow, ignition) plus a <b>random control basket</b>. A signal "works" only if it hits real winners ≥2× more often than the random picks. Bold paper bot buys the top-3 combined-score picks. Verdict at 30 trading days.</p></div>`;
  // --- bot account ---
  if(ld.bot){
    const b=ld.bot, a=b.account;
    h+=`<div class="card"><h2>Hype bot — live paper account${a&&a.number?` &nbsp;·&nbsp; #${a.number}`:''} <small style="color:var(--dim)">(repurposed dual-mom)</small></h2>`;
    if(a){
      h+=`<div class="grid">
        <div class="stat"><div class="k">Equity</div><div class="v">${money(a.equity)}</div></div>
        <div class="stat"><div class="k">Day P/L</div><div class="v ${cls(a.day_pnl)}">${sign(a.day_pnl)}<br><small>${a.day_pnl_pct>=0?"+":""}${a.day_pnl_pct.toFixed(2)}%</small></div></div>
        ${winStat("Week P/L", b.week_pnl)}
        ${winStat("Month P/L", b.month_pnl)}
        <div class="stat"><div class="k">Cash</div><div class="v">${money(a.cash)}</div></div>
        <div class="stat"><div class="k">Invested</div><div class="v">${money(b.invested||0)}</div></div>
      </div>`;
    }
    const bp=b.positions||[];
    h+=`<div style="color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.06em;margin:6px 0 6px">Open positions (${bp.length})</div>`;
    if(bp.length){
      h+=`<table><tr><th>sym</th><th>qty</th><th>avg</th><th>last</th><th>invested</th><th>unreal P/L</th></tr>`;
      for(const p of bp) h+=row([{v:p.symbol},{v:p.qty.toFixed(0)},{v:money(p.avg_entry)},{v:money(p.current)},{v:money(p.cost_basis)},{v:`${sign(p.unrealized_pl)} <small>(${p.unrealized_plpc>=0?"+":""}${p.unrealized_plpc.toFixed(1)}%)</small>`,cls:cls(p.unrealized_pl)}]);
      h+=`</table>`;
      // why each open position was picked (plain-English signal drivers)
      const drv=ld.drivers||{};
      if(Object.keys(drv).length){
        h+=`<div style="color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.06em;margin:10px 0 4px">Why these were picked</div>`;
        for(const p of bp){ const d=drv[p.symbol]; if(!d) continue;
          h+=`<div class="hint" style="margin:3px 0"><b>${p.symbol}</b>`
            +`${d.name?` <span style="color:var(--dim)">(${d.name})</span>`:''} — ${d.why}`
            +`${d.score!=null?` <span style="color:var(--dim)">· score ${d.score.toFixed(2)}, picked ${d.date||''}</span>`:''}</div>`;
        }
      }
    } else h+=`<div class="empty">flat</div>`;
    const bct=b.closed_today||[];
    const bpnl=bct.reduce((s,c)=>s+c.realized,0);
    h+=`<div style="color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.06em;margin:12px 0 6px">Closed today (${bct.length})${bct.length?` &nbsp;·&nbsp; realized <span class="${cls(bpnl)}">${sign(bpnl)}</span> &nbsp;·&nbsp; <span>${money(bpnl/bct.length)}/trade</span>`:""}</div>`;
    if(bct.length){
      h+=`<table><tr><th>sym</th><th>qty</th><th>entry</th><th>exit</th><th>realized</th></tr>`;
      for(const c of bct) h+=row([{v:c.symbol+(c.partial?` <small style="color:var(--dim)">partial</small>`:"")},{v:c.qty.toFixed(0)},{v:money(c.entry_avg)},{v:money(c.exit_avg)},{v:sign(c.realized),cls:cls(c.realized)}]);
      h+=`</table><div class="hint">includes multi-day holds closed today + PARTIAL trailing-stop exits (tagged "partial" — some shares may still be held); realized = (exit − entry) × shares sold today</div>`;
    } else h+=`<div class="empty">no positions closed today</div>`;
    if(b.errors&&b.errors.length) h+=`<div class="err">⚠ ${b.errors.join(" · ")}</div>`;
    h+=`</div>`;
  } else {
    h+=`<div class="card"><div class="empty">Hype bot account not visible here yet (needs .env.lottery where the status server runs). First live run Mon 09:44 ET. Signal measurement below still works.</div></div>`;
  }
  // --- P/L breakdown (same table as the ORB tab, for the lottery bot's account) ---
  if(ld.bot && ld.bot.daily_pnl) h+=plBreakdownCard(ld.bot.daily_pnl, (ld.generated||"").slice(0,10), "lottery");
  // --- signal scoreboard (the verdict engine) ---
  h+=`<div class="card"><h2>Signal scoreboard — does any hype metric beat luck?</h2>`;
  if(sb.error){ h+=`<div class="err">${sb.error}</div>`; }
  else if(!sb.n_days){ h+=`<div class="empty">No picks logged yet — the first board logs near 6:24am PT.</div>`; }
  else {
    const base=sb.base||{};
    h+=`<div class="hint" style="margin-bottom:8px">Random-basket base rate (the luck line) — W1 9:45→close ≥+5%: <b>${base.W1==null?'n/a':(base.W1*100).toFixed(0)+'%'}</b> · W2 next-day ≥+10%: <b>${base.W2==null?'n/a':(base.W2*100).toFixed(0)+'%'}</b> · W3 3-day ≥+20%: <b>${base.W3==null?'n/a':(base.W3*100).toFixed(0)+'%'}</b>. A signal "works" iff W1 lift ≥${sb.success_lift||2}× with p<0.05 over ≥${sb.success_ndays||30} days (${sb.n_days} so far).</div>`;
    const mkRow=(r)=>{
      const w1=r.wins.W1||{}, w2=r.wins.W2||{}, w3=r.wins.W3||{};
      const liftCell=w=>w.lift==null?{v:"—"}:{v:`${w.lift.toFixed(2)}×`,cls:liftC(w.lift)};
      const rateCell=w=>w.n?`${pct1(w.rate)} <small style="color:var(--dim)">(${w.w}/${w.n})</small>`:"—";
      let verdict="";
      if(w1.lift!=null && w1.p!=null){
        if(w1.lift>=(sb.success_lift||2) && w1.p<0.05) verdict = sb.n_days>=(sb.success_ndays||30)?`<span class="pos">PASSES</span>`:`<span class="pos">on track</span>`;
      }
      const nm = r.kind==="combined"?`<b>${r.name}</b>`:(r.kind==="basket"?`<span style="color:#9aa6b4">${r.name}</span>`:r.name);
      return `<tr><td style="text-align:left">${nm}</td><td>${rateCell(w1)}</td>`+
             row_cell(liftCell(w1))+`<td>${w1.p==null?"—":"p="+w1.p.toFixed(3)}</td>`+
             `<td>${rateCell(w2)}</td><td>${rateCell(w3)}</td><td>${verdict}</td></tr>`;
    };
    h+=`<div style="color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.06em;margin:10px 0 4px">By signal (names that were top-K for that metric)</div>`;
    h+=`<table><tr><th>signal</th><th>W1 hit</th><th>W1 lift</th><th>W1 p</th><th>W2 hit</th><th>W3 hit</th><th></th></tr>`;
    for(const r of (sb.signals||[])) h+=mkRow(r);
    h+=`</table>`;
    h+=`<div style="color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.06em;margin:14px 0 4px">By basket &amp; the combined top-3 bot picks</div>`;
    h+=`<table><tr><th>basket</th><th>W1 hit</th><th>W1 lift</th><th>W1 p</th><th>W2 hit</th><th>W3 hit</th><th></th></tr>`;
    for(const r of (sb.baskets||[])) h+=mkRow(r);
    if(sb.combined_top3) h+=mkRow(sb.combined_top3);
    h+=`</table>`;
  }
  h+=`<div class="hint">W1 = 9:45→close ≥+5% · W2 = next-day ≥+10% · W3 = 3-day ≥+20%. Lift = signal hit-rate ÷ random base rate. ≥2× and growing with the sample = a real winner-picker. One good week is noise.</div></div>`;
  // --- daily source performance grid (descriptive day-by-day comparison) ---
  const sd=ld.source_daily||{};
  if(sd.error){ h+=`<div class="card"><h2>Daily source performance</h2><div class="err">${sd.error}</div></div>`; }
  else if(sd.dates && sd.dates.length){
    h+=`<div class="card"><h2>Daily source performance — avg 9:45→close % per source</h2>`;
    h+=`<div class="hint" style="margin-bottom:8px">Each cell = the average same-day return (9:45→close) of the names that source flagged that day. <b>combined3</b> = the bot's traded top-3 · <b>random</b> = the luck baseline. Scan a row left-to-right for a source's daily trend; compare each against random. Newest day first · hover a cell for pick-count &amp; W1 hit-rate.</div>`;
    const dcell=o=>o==null?`<td style="color:var(--dim)">·</td>`:`<td class="${cls(o.avg)}" title="${o.n} picks · ${(o.hit*100).toFixed(0)}% W1 hit">${o.avg>=0?"+":""}${o.avg.toFixed(1)}</td>`;
    h+=`<table><tr><th style="text-align:left">source</th><th>cum avg</th>`;
    for(const dt of sd.dates) h+=`<th>${dt.slice(5)}</th>`;
    h+=`</tr>`;
    for(const src of (sd.sources||[])){
      const c=(sd.cum||{})[src]||{};
      const nm=srcLabel(src,sd.kinds);
      h+=`<tr><td style="text-align:left">${nm}</td>`+
         `<td class="${cls(c.avg)}" title="${c.n||0} picks total">${c.avg==null?"—":((c.avg>=0?"+":"")+c.avg.toFixed(1))}</td>`;
      for(const dt of sd.dates) h+=dcell((sd.grid[src]||{})[dt]);
      h+=`</tr>`;
    }
    h+=`</table><div class="hint">descriptive daily color only (green up / red down) — the statistical verdict is the lift table above, which needs ≥30 days. A source beating random here day after day is what eventually shows up there.</div></div>`;
  }
  // --- today's picks live behavior ---
  if(ld.live && Object.keys(ld.live).length){
    const held=new Set(((ld.bot||{}).positions||[]).map(p=>p.symbol));
    const today=(ld.days||[]).find(d=>d.date===ld.live_date)||{picks:[]};
    const ord=p=>({hype:0,wsb:0,stocktwits:0,gappers:1,control:2,random:3}[p.basket]??1);
    h+=`<div class="card"><h2>Today's board — live behavior (${ld.live_date})</h2>`;
    h+=`<table><tr><th>sym</th><th>basket</th><th>score</th><th>held</th><th>last</th><th>gap%</th><th>day%</th><th>9:45 px</th><th>since 9:45</th></tr>`;
    for(const p of today.picks.slice().sort((a,b)=>ord(a)-ord(b)||(b.combined_score||0)-(a.combined_score||0))){
      const L=ld.live[p.symbol]; if(!L) continue;
      const pc=v=>v==null?{v:"—"}:{v:pctTxt(v),cls:cls(v)};
      h+=row([{v:p.symbol},{v:basketBadge(p.basket)},{v:p.combined_score==null?"—":p.combined_score.toFixed(2)},
              {v:held.has(p.symbol)?`<span class="pos">✓</span>`:""},
              {v:L.last!=null?money(L.last):"—"},pc(L.gap_pct),pc(L.day_pct),
              {v:L.px945!=null?money(L.px945):"—"},pc(L.since945_pct)]);
    }
    h+=`</table><div class="hint">every candidate tracked live whether or not the bot bought it · "since 9:45" is the exact reference the after-close scorer measures · "held" = currently in the hype bot's book</div></div>`;
  }
  // --- per-day board breakdown ---
  const days=ld.days||[];
  if(!days.length){ h+=`<div class="card"><div class="empty">No boards logged yet — the first board logs near 6:24am PT.</div></div>`; }
  for(const day of days){
    const bk=day.baskets||{};
    const bkTxt=Object.keys(bk).map(k=>`${k} ${bk[k]}`).join(" · ");
    h+=`<div class="card"><h2>${day.date} &nbsp;·&nbsp; ${day.n} picks &nbsp;<small style="color:var(--dim)">${bkTxt}</small></h2>`;
    h+=`<table><tr><th>sym</th><th>basket</th><th>signals</th><th>score</th><th>9:45→close</th><th>1d</th><th>3d</th><th>win</th></tr>`;
    for(const p of (day.picks||[]).slice().sort((a,b)=>(b.combined_score||0)-(a.combined_score||0))){
      const sg=p.signals||{};
      const tags=(p.top_k_of||[]).join(",")||"—";
      const r1=p.ret_945_close, r2=p.ret_1d, r3=p.ret_3d;
      const rc=v=>v==null?"—":`<span class="${cls(v)}">${pctTxt(v)}</span>`;
      let win="";
      if(r1!=null||r2!=null||r3!=null){
        const isW=(r1!=null&&r1>=5)||(r2!=null&&r2>=10)||(r3!=null&&r3>=20);
        win=isW?`<span class="pos">★</span>`:"";
      }
      h+=`<tr><td>${p.symbol}</td><td style="text-align:right">${basketBadge(p.basket)}</td>`+
         `<td style="text-align:left;color:#9aa6b4;font-size:11px">${tags}</td>`+
         `<td>${p.combined_score==null?"—":p.combined_score.toFixed(2)}</td>`+
         `<td>${rc(r1)}</td><td>${rc(r2)}</td><td>${rc(r3)}</td><td>${win}</td></tr>`;
    }
    h+=`</table></div>`;
  }
  h+=`<div class="foot"><span>hype · ${ld.generated||''}</span><span></span></div>`;
  setRoot(h);
}
function row_cell(c){ return `<td${c.cls?` class="${c.cls}"`:''}>${c.v}</td>`; }
function basketBadge(b){
  const m={wsb:["#d97706","WSB"],stocktwits:["#2563eb","ST"],gappers:["#7c8694","gap"],
           control:["#7c8694","ctrl"],random:["#7c8694","rand"]};
  const x=m[b]||["#7c8694",b||"?"];
  return `<span style="color:${x[0]};font-size:11px;font-weight:600">${x[1]}</span>`;
}
function srcLabel(src,kinds){
  if(src==="combined3"||src==="random") return `<b>${src}</b>`;
  if(src==="filtered3") return `<b>filtered3</b> <small style="color:var(--dim)">vol-filter</small>`;
  const k=(kinds||{})[src]||"src";
  const tag = k==="feat"?`<span title="computed price/volume feature">📈</span> `
                        :`<span title="online source">📡</span> `;
  return `${tag}${src}`;
}
async function fetchSummary(){
  try{ const r=await fetch("/api/summary",{cache:"no-store"}); lastSummary=await r.json(); if(topView==="summary") renderSummary(lastSummary); }
  catch(e){ if(topView==="summary") document.getElementById("root").innerHTML=topNav("summary")+`<div class="card empty">summary data unavailable</div>`; }
}
// Summary tab: all three bots' daily P/L side by side (invested · P/L · % per bot) vs the S&P 500.
function renderSummary(sd){
  let h=topNav("summary");
  h+=`<div class="banner s-idle"><h1>Summary — all three bots, day by day</h1><p>Daily P/L, capital invested and return for each paper bot side by side, against the S&P 500. News-Edge and Hype both size ~$2,000/name; the ORB baseline sizes by risk. Same money-to-money lens for every strategy.</p></div>`;
  const keys=["orb","news","lottery"], names={orb:"ORB baseline",news:"News-Edge",lottery:"Hype"};
  const bots=sd.bots||[];
  const byKey={}; bots.forEach(b=>byKey[b.key]=b);
  // merge all three accounts' daily rows by ET date
  const map={};
  for(const b of bots) for(const r of (b.daily_pnl||[])){ (map[r.date]=map[r.date]||{})[b.key]=r; }
  const dates=Object.keys(map).sort().reverse();
  const todayStr=(sd.generated||"").slice(0,10);
  // note any bot whose account isn't visible where the server runs
  const missing=keys.filter(k=>byKey[k] && (byKey[k].absent||byKey[k].error));
  if(missing.length) h+=`<div class="card"><div class="hint">Not visible here yet: ${missing.map(k=>names[k]).join(", ")} (account keys absent where the status server runs). Their columns show — until then.</div></div>`;
  if(!dates.length){ h+=`<div class="card"><div class="empty">No P/L history yet for any bot.</div></div>`; h+=`<div class="foot"><span>summary · ${sd.generated||''}</span><span></span></div>`; setRoot(h); return; }

  const plCell=v=>v==null?{v:"—"}:{v:sign(v),cls:cls(v)};
  const pctCell=(v,dp)=>(v==null||v===undefined)?{v:"—"}:{v:`${v>=0?"+":""}${v.toFixed(dp)}%`,cls:cls(v)};
  const invCell=v=>(v&&v>0)?{v:money(v)}:{v:"—"};
  const spyOf=d=>{ for(const k of keys){ const r=(map[d]||{})[k]; if(r&&r.spy_pct!=null) return r.spy_pct; } return null; };

  h+=`<div class="card"><h2>Daily P/L — ORB · News-Edge · Hype</h2><div class="scrollx" style="overflow-x:auto">`;
  h+=`<table><tr><th rowspan="2">date</th>`
    +keys.map(k=>`<th colspan="3" style="text-align:center;color:var(--txt)">${names[k]}</th>`).join("")
    +`<th rowspan="2">S&P 500 %</th></tr>`;
  h+=`<tr>`+keys.map(()=>`<th>inv</th><th>P/L</th><th>%</th>`).join("")+`</tr>`;

  // totals across the shown window (per bot: Σ invested, Σ P/L; S&P compounded)
  const tot={}; keys.forEach(k=>tot[k]={inv:0,pnl:0,any:false});
  let spyMul=1, spyAny=false;
  for(const d of dates){
    for(const k of keys){ const r=(map[d]||{})[k]; if(r){ tot[k].inv+=r.invested||0; tot[k].pnl+=r.pnl||0; tot[k].any=true; } }
    const s=spyOf(d); if(s!=null){ spyMul*=(1+s/100); spyAny=true; }
  }
  let trow=[{v:`<b>Σ window</b>`}];
  keys.forEach(k=>{ trow.push(invCell(tot[k].any?tot[k].inv:null)); trow.push(plCell(tot[k].any?tot[k].pnl:null)); trow.push({v:""}); });
  trow.push(pctCell(spyAny?(spyMul-1)*100:null,2));
  h+=row(trow);

  for(const d of dates){
    const cells=[{v:(d===todayStr?`${d} <small>(today)</small>`:d)}];
    for(const k of keys){
      const r=(map[d]||{})[k];
      cells.push(invCell(r?r.invested:null));
      cells.push(plCell(r?r.pnl:null));
      cells.push(pctCell(r?r.pnl_pct_inv:null,1));
    }
    cells.push(pctCell(spyOf(d),2));
    h+=row(cells);
  }
  h+=`</table></div>`;
  h+=`<div class="hint">inv = capital deployed that day (gross buy notional) · % = that bot's P/L ÷ its invested · S&P 500 % = SPY close-to-close · "Σ window" = totals over all days shown (S&P compounded). Each bot trades its own paper account.</div></div>`;

  // --- return on CAPITAL ACTUALLY DEPLOYED (the strategy-quality denominator) ---
  h+=`<div class="card"><h2>Return on capital deployed (window shown)</h2>`;
  h+=`<div class="hint" style="margin-bottom:8px">P/L vs the capital actually <b>at work</b> — the avg / peak concurrent cost basis of open positions — <b>not</b> the idle account balance. This is the strategy-quality number (e.g. a small book turning over efficiently).</div>`;
  h+=`<table><tr><th style="text-align:left">bot</th><th>avg deployed</th><th>peak deployed</th><th>P/L</th><th>% on avg deployed</th></tr>`;
  for(const k of keys){
    const b=byKey[k]; if(!b||b.absent||b.error) continue;
    const dep=b.deployed_avg, pk=b.deployed_peak, pnl=b.window_pnl;
    const roc=(dep&&dep>0&&pnl!=null)?(pnl/dep*100):null;
    h+=row([{v:names[k]},{v:dep?money(dep):"—"},{v:pk?money(pk):"—"},
            {v:pnl!=null?sign(pnl):"—",cls:cls(pnl)},
            {v:roc!=null?`${roc>=0?"+":""}${roc.toFixed(1)}%`:"—",cls:cls(roc)}]);
  }
  h+=`</table><div class="hint">avg/peak deployed = mean/max concurrent position cost basis over the days shown · % on avg deployed = total window P/L ÷ avg deployed. Holds multi-day (Hype/News) keep capital at work; the ORB baseline flattens daily so its deployed reading is choppier. Small sample — read the magnitude with caution.</div></div>`;

  // --- bots vs source picks — one daily-% lens ---
  const sdg=sd.source_daily||{};
  if(sdg.sources && sdg.sources.length){
    h+=`<div class="card"><h2>Daily % — bots vs source picks</h2>`;
    h+=`<div class="hint" style="margin-bottom:8px"><b>Bot</b> rows = that bot's account return (P/L ÷ invested). <b>Source</b> rows = the average 9:45→close return of the names that source flagged that day (measured only — not traded). combined3 = the Hype bot's traded top-3 · random = luck baseline. Compare any source against the bots, day by day · hover a source cell for pick-count &amp; hit-rate.</div>`;
    h+=`<div class="scrollx" style="overflow-x:auto"><table><tr><th style="text-align:left">row</th>`;
    for(const d of dates) h+=`<th>${d.slice(5)}</th>`;
    h+=`</tr>`;
    const num=(v)=>v==null?`<td style="color:var(--dim)">·</td>`:`<td class="${cls(v)}">${v>=0?"+":""}${v.toFixed(1)}</td>`;
    for(const k of keys){
      h+=`<tr><td style="text-align:left"><b>${names[k]}</b> <small style="color:var(--dim)">bot</small></td>`;
      for(const d of dates){ const r=(map[d]||{})[k]; h+=num(r?r.pnl_pct_inv:null); }
      h+=`</tr>`;
    }
    h+=`<tr><td colspan="${dates.length+1}" style="text-align:left;color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.06em;padding-top:10px">source picks — avg 9:45→close %</td></tr>`;
    for(const src of sdg.sources){
      const nm=srcLabel(src,sdg.kinds);
      h+=`<tr><td style="text-align:left">${nm}</td>`;
      for(const d of dates){ const o=(sdg.grid[src]||{})[d];
        h+= o==null?`<td style="color:var(--dim)">·</td>`:`<td class="${cls(o.avg)}" title="${o.n} picks · ${(o.hit*100).toFixed(0)}% W1 hit">${o.avg>=0?"+":""}${o.avg.toFixed(1)}</td>`; }
      h+=`</tr>`;
    }
    h+=`</table></div>`;
    h+=`<div class="hint">bot % and source % aren't the same unit — a bot row is whole-account return (sizing, cash drag, exits); a source row is the raw average move of its flagged names. Use it for direction/feel, not a precise head-to-head. Green up / red down · "·" = no scored picks that day.</div></div>`;
  }
  h+=`<div class="foot"><span>summary · ${sd.generated||''}</span><span></span></div>`;
  setRoot(h);
}
function render(d){
  lastData=d;
  const L=d.liveness;
  let h=topNav("trading")+`<div class="banner s-${L.state}"><span class="dot"></span><h1>${L.headline}</h1><p>${L.detail}</p></div>`;
  if(d.market){ h+=`<div class="card"><h2>Market</h2>${d.market.label}</div>`; }
  const a=d.account;
  if(a){
    h+=`<div class="grid">
      <div class="stat"><div class="k">Equity</div><div class="v">${money(a.equity)}</div></div>
      <div class="stat"><div class="k">Day P/L</div><div class="v ${cls(a.day_pnl)}">${sign(a.day_pnl)}<br><small>${a.day_pnl_pct>=0?"+":""}${a.day_pnl_pct.toFixed(2)}%</small></div></div>
      ${winStat("Week P/L", d.week_pnl)}
      ${winStat("Month P/L", d.month_pnl)}
      <div class="stat"><div class="k">Cash</div><div class="v">${money(a.cash)}</div></div>
      <div class="stat"><div class="k">Invested</div><div class="v">${money(d.invested||0)}</div></div>
      <div class="stat"><div class="k">Buying power</div><div class="v">${money(a.buying_power)}</div></div>
    </div>`;
  }
  // positions — "invested" = cost basis ($ put into the name); "value" = current mkt value
  h+=`<div class="card"><h2>Open positions (${d.positions.length})</h2>`;
  if(d.positions.length){
    h+=`<table><tr><th>sym</th><th>side</th><th>qty</th><th>avg</th><th>last</th><th>OR low</th><th>OR high</th><th>risk $</th><th>invested</th><th>value</th><th>unreal P/L</th></tr>`;
    for(const p of d.positions) h+=row([{v:p.symbol},{v:p.side},{v:p.qty.toFixed(2)},{v:money(p.avg_entry)},{v:money(p.current)},{v:(p.or_low!=null?money(p.or_low):"—")},{v:(p.or_high!=null?money(p.or_high):"—")},{v:(p.risk!=null?money(p.risk):"—")},{v:money(p.cost_basis)},{v:money(p.market_value)},{v:`${sign(p.unrealized_pl)} <small>(${p.unrealized_plpc>=0?"+":""}${p.unrealized_plpc.toFixed(1)}%)</small>`,cls:cls(p.unrealized_pl)}]);
    h+=`</table>`;
  } else h+=`<div class="empty">flat</div>`;
  h+=`</div>`;
  // closed round-trips today (realized P/L)
  const ct=d.closed_today||[];
  const ctPnl=ct.reduce((s,c)=>s+c.realized,0);
  h+=`<div class="card"><h2>Closed today (${ct.length})${ct.length?` &nbsp;·&nbsp; realized <span class="${cls(ctPnl)}">${sign(ctPnl)}</span>`:""}</h2>`;
  if(ct.length){
    h+=`<table><tr><th>sym</th><th>side</th><th>qty</th><th>entry</th><th>exit</th><th>realized P/L</th></tr>`;
    for(const c of ct) h+=row([{v:c.symbol},{v:c.side},{v:c.qty.toFixed(0)},{v:money(c.entry_avg)},{v:money(c.exit_avg)},{v:sign(c.realized),cls:cls(c.realized)}]);
    h+=`</table>`;
  } else h+=`<div class="empty">no round-trips closed today</div>`;
  h+=`</div>`;
  // open orders — merge a bracket's two legs (take-profit + stop-loss) into one
  // row per position so target and stop sit side by side, not on separate rows.
  const oo=d.open_orders||[];
  const og=(()=>{ const m=new Map();
    for(const o of oo){ const k=o.symbol+"|"+o.side+"|"+o.qty;
      let g=m.get(k); if(!g){ g={symbol:o.symbol,side:o.side,qty:o.qty,limit:null,stop:null,st:new Set()}; m.set(k,g); }
      if(o.limit!=null) g.limit=o.limit; if(o.stop!=null) g.stop=o.stop; g.st.add(o.status); }
    return Array.from(m.values()); })();
  h+=`<div class="card"><h2>Open orders (${og.length})</h2>`;
  if(og.length){
    h+=`<table><tr><th>sym</th><th>side</th><th>qty</th><th>target</th><th>stop</th><th>status</th></tr>`;
    for(const o of og) h+=row([{v:o.symbol},{v:o.side},{v:o.qty.toFixed(0)},{v:o.limit?money(o.limit):"—"},{v:o.stop?money(o.stop):"—"},{v:Array.from(o.st).join("/").toLowerCase()}]);
    h+=`</table>`;
  } else h+=`<div class="empty">none</div>`;
  h+=`</div>`;
  // today's ORB fills
  h+=`<div class="card"><h2>Today's ORB entries (${d.orb_fills.length})</h2>`;
  if(d.orb_fills.length){
    h+=`<table><tr><th>sym</th><th>side</th><th>qty</th><th>fill</th><th>id</th></tr>`;
    for(const o of d.orb_fills) h+=row([{v:o.symbol},{v:o.side},{v:o.qty.toFixed(0)},{v:money(o.price)},{v:`<span style="color:#7c8694">${o.coid}</span>`}]);
    h+=`</table>`;
  } else h+=`<div class="empty">no ORB entries filled today</div>`;
  h+=`</div>`;
  // P/L breakdown — account-level, newest first (shared with the News-Edge & Lottery tabs).
  h+=plBreakdownCard(d.daily_pnl||[], (d.generated||"").slice(0,10), "trading");
  if(d.errors && d.errors.length) h+=`<div class="err">⚠ ${d.errors.join(" · ")}</div>`;
  h+=`<div class="foot"><span>snapshot ${d.generated}</span><span id="tick"></span></div>`;
  setRoot(h);
}
let fails=0;
async function tick(){
  try{ const r=await fetch("/api/status",{cache:"no-store"}); const d=await r.json(); lastData=d; fails=0;
       if(topView==="trading") render(d);
       const t=document.getElementById("tick"); if(t) t.textContent="live ●"; }
  catch(e){ fails++; const t=document.getElementById("tick");
       if(t) t.textContent=`page offline (${fails}) — is the SSH tunnel up?`; }
  if(topView==="news") fetchNews();
  if(topView==="lottery") fetchLottery();
  if(topView==="summary") fetchSummary();
  if(topView==="regime") fetchRegime();
}
tick(); setInterval(tick, 3000);
setTimeout(prefetchTabs, 2000);  // warm bot tabs ~2s after load (after the first paint)
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    tc = None  # set in serve()
    dc = None  # data client, set in serve()
    timeout = 15  # drop slow/half-open sockets so a thread can't be held hostage

    def log_message(self, *a):  # silence per-request stderr spam
        pass

    def handle_one_request(self):
        # Shed load instead of spawning unbounded work: if we're already at the
        # concurrency cap, answer 503 immediately and move on. acquire() is
        # non-blocking so a flood can't queue up and pin the VM.
        if not _conn_sema.acquire(blocking=False):
            try:
                self.send_response(503)
                self.send_header("Content-Length", "0")
                self.send_header("Retry-After", "2")
                self.end_headers()
            except Exception:
                pass
            self.close_connection = True
            return
        try:
            super().handle_one_request()
        finally:
            _conn_sema.release()

    def _send(self, code, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/status"):
            try:
                body = json.dumps(_status(self.tc, self.dc)).encode("utf-8")
                self._send(200, body, "application/json")
            except Exception as e:
                self._send(500, json.dumps({"error": str(e)}).encode(), "application/json")
        elif self.path.startswith("/api/newsedge"):
            try:
                body = json.dumps(_newsedge()).encode("utf-8")
                self._send(200, body, "application/json")
            except Exception as e:
                self._send(500, json.dumps({"error": str(e)}).encode(), "application/json")
        elif self.path.startswith("/api/lottery"):
            try:
                body = json.dumps(_lottery()).encode("utf-8")
                self._send(200, body, "application/json")
            except Exception as e:
                self._send(500, json.dumps({"error": str(e)}).encode(), "application/json")
        elif self.path.startswith("/api/summary"):
            try:
                body = json.dumps(_summary(self.tc, self.dc)).encode("utf-8")
                self._send(200, body, "application/json")
            except Exception as e:
                self._send(500, json.dumps({"error": str(e)}).encode(), "application/json")
        elif self.path.startswith("/api/regime"):
            try:
                body = json.dumps(_regime()).encode("utf-8")
                self._send(200, body, "application/json")
            except Exception as e:
                self._send(500, json.dumps({"error": str(e)}).encode(), "application/json")
        elif self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/healthz":
            self._send(200, b"ok", "text/plain")
        else:
            self._send(404, b"not found", "text/plain")


def serve() -> int:
    load_env()
    try:
        tc, dc = build_clients()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    Handler.tc = tc
    Handler.dc = dc
    threading.Thread(target=_regime_loop, daemon=True,
                     name="regime-refresh").start()
    httpd = ThreadingHTTPServer((BIND, PORT), Handler)
    print(f"ORB status server on http://{BIND}:{PORT}  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    return 0


if __name__ == "__main__":
    sys.exit(serve())
