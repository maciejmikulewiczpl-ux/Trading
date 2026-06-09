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
from alpaca.data.requests import StockBarsRequest  # noqa: E402
from alpaca.data.timeframe import TimeFrame  # noqa: E402
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
def _gather(tc, dc=None) -> dict:
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
    try:
        win_start = datetime.combine(today - timedelta(days=370), dtime(0, 0, tzinfo=ET)).astimezone(UTC)
        req = GetOrdersRequest(status=QueryOrderStatus.ALL, after=win_start, limit=500)
        try:
            hist = tc.get_orders(filter=req)
        except TypeError:
            hist = tc.get_orders(req)
        for o in hist:
            fap = getattr(o, "filled_avg_price", None)
            fqty = _f(o.filled_qty)
            ft = getattr(o, "filled_at", None)
            if fap is None or fqty <= 0 or ft is None:
                continue
            if str(o.side).rsplit(".", 1)[-1].lower() != "buy":
                continue
            d = ft.astimezone(ET).date().isoformat()
            invested_by_day[d] = invested_by_day.get(d, 0.0) + fqty * _f(fap)
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
    return out


def _status(tc, dc=None) -> dict:
    now = _time.time()
    if _cache["data"] is None or now - _cache["ts"] > CACHE_TTL:
        _cache["data"] = _gather(tc, dc)
        _cache["ts"] = now
    return _cache["data"]


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
  tr.dim td{opacity:.4}
  .hivol{color:var(--orange);font-size:10px;margin-left:6px;white-space:nowrap}
</style></head>
<body><div class="wrap" id="root">loading…</div>
<script>
const money = v => (v<0?"-$":"$") + Math.abs(v).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2});
const sign = v => (v>=0?"+":"") + money(v);
const cls = v => v>=0 ? "pos":"neg";
function row(cells, trCls){ return `<tr${trCls?` class="${trCls}"`:""}>`+cells.map((c,i)=>`<td${c.cls?` class="${c.cls}"`:""}>${c.v}</td>`).join("")+"</tr>"; }
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
let plView="day";
let calmOnly=false;
let lastData=null;
let topView="trading";
let lastNews=null;
function setPLView(v){ plView=v; if(lastData) render(lastData); }
function setCalm(){ calmOnly=!calmOnly; if(lastData) render(lastData); }
function topNav(active){
  return `<div class="tabs" style="margin-bottom:14px">`+[["trading","Trading"],["news","News-Edge"]].map(
    ([k,l])=>`<button class="tab${active===k?" active":""}" onclick="setTopView('${k}')">${l}</button>`).join("")+`</div>`;
}
function setTopView(v){
  topView=v;
  if(v==="news") fetchNews();
  else if(lastData) render(lastData);
}
async function fetchNews(){
  try{ const r=await fetch("/api/newsedge",{cache:"no-store"}); lastNews=await r.json(); renderNews(lastNews); }
  catch(e){ document.getElementById("root").innerHTML=topNav("news")+`<div class="card empty">news-edge data unavailable</div>`; }
}
function renderNews(nd){
  const o=nd.overall||{};
  const sepC=v=>(v>=0?"pos":"neg");
  let h=topNav("news");
  h+=`<div class="banner s-idle"><h1>News-Edge — forward test</h1><p>Manual prototype: does my morning read of the news predict the day's move? Logs only — never trades, never touches the ORB bot.</p></div>`;
  h+=`<div class="grid">
    <div class="stat"><div class="k">Days logged</div><div class="v">${o.n_days||0}</div></div>
    <div class="stat"><div class="k">Scored picks</div><div class="v">${o.n_scored||0}</div></div>
    <div class="stat"><div class="k">(+) avg move</div><div class="v ${o.avg_pos==null?'':sepC(o.avg_pos)}">${o.avg_pos==null?'—':pctTxt(o.avg_pos)}</div></div>
    <div class="stat"><div class="k">(−) avg move</div><div class="v ${o.avg_neg==null?'':sepC(o.avg_neg)}">${o.avg_neg==null?'—':pctTxt(o.avg_neg)}</div></div>
    <div class="stat"><div class="k">(+)−(−) edge</div><div class="v ${o.sep==null?'':sepC(o.sep)}">${o.sep==null?'—':pctTxt(o.sep)}</div></div>
  </div>`;
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
  document.getElementById("root").innerHTML=h;
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
  // P/L breakdown — account-level, newest first. Day rows come from the server;
  // Week/Month are aggregated client-side from those same rows via the tabs.
  const dp=d.daily_pnl||[];
  const todayStr=(d.generated||"").slice(0,10);  // "YYYY-MM-DD" of this snapshot (ET)
  // Calm-day vol filter: "calm" days are low-volatility sessions (SPY 20d vol below
  // its trailing median) where ORB historically does better. The toggle shows the
  // counterfactual — your P/L counting ONLY calm days — without changing live trading.
  const calmKnown=dp.some(r=>r.calm!=null);
  const dpView = calmOnly ? dp.filter(r=>r.calm===true) : dp;
  h+=`<div class="card"><h2>P/L breakdown</h2>`;
  h+=`<div class="tabs">`+["day","week","month"].map(v=>
       `<button class="tab${plView===v?" active":""}" onclick="setPLView('${v}')">${v[0].toUpperCase()+v.slice(1)}</button>`).join("")
     +(calmKnown?`<button class="tab toggle${calmOnly?" active":""}" onclick="setCalm()" title="Count only low-volatility (calm) days — where ORB does better. Does not change live trading.">Calm days only</button>`:``)+`</div>`;
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
      const trCls=(plView==="day" && !calmOnly && r.calm===false)?"dim":"";
      h+=row([{v:fmtLbl(r)},{v:r.invested>0?money(r.invested):"—"},{v:sign(r.pnl),cls:cls(r.pnl)},pct,spy,{v:money(r.equity)}], trCls); });
    h+=`</table>`;
    h+=`<div class="hint">% = P/L vs capital invested (avg daily capital for Week/Month) · S&P 500 % = SPY close-to-close (compounded) · calm = SPY 20d vol below its trailing median</div>`;
  } else h+=`<div class="empty">${calmOnly?"no calm days in range":"no history"}</div>`;
  h+=`</div>`;
  if(d.errors && d.errors.length) h+=`<div class="err">⚠ ${d.errors.join(" · ")}</div>`;
  h+=`<div class="foot"><span>snapshot ${d.generated}</span><span id="tick"></span></div>`;
  document.getElementById("root").innerHTML=h;
}
let fails=0;
async function tick(){
  try{ const r=await fetch("/api/status",{cache:"no-store"}); const d=await r.json(); lastData=d; fails=0;
       if(topView==="trading") render(d);
       const t=document.getElementById("tick"); if(t) t.textContent="live ●"; }
  catch(e){ fails++; const t=document.getElementById("tick");
       if(t) t.textContent=`page offline (${fails}) — is the SSH tunnel up?`; }
  if(topView==="news") fetchNews();
}
tick(); setInterval(tick, 3000);
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
    httpd = ThreadingHTTPServer((BIND, PORT), Handler)
    print(f"ORB status server on http://{BIND}:{PORT}  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    return 0


if __name__ == "__main__":
    sys.exit(serve())
