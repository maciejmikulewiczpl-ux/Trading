"""Per-trade outcome ledger for the Hype bot — the raw material for deep-dive analysis
(trail-width tuning, which signals actually pick winners, exit-timing).

Reconstructs every trade from the account's FILL history (robust + idempotent — handles
partial trailing-stop exits and re-entries, which an in-bot exit-logger would miss), then
joins each trade's ENTRY CONTEXT from the immutable picks files and enriches with bar-
derived outcome metrics:

  - realized P/L + % and hold length, with the EXIT REASON (trailing_stop vs time_stop)
  - combined_score + which signals flagged it (top_k_of) + key signal values at entry
    -> lets us later ask "do ignition / wsb / rvol picks make more money?"
  - MFE / MAE (max favourable / adverse excursion over the hold, from daily highs/lows)
    -> "how much did the 10% trail leave on the table? how close did we come to stopping?"
  - post-exit 5-day drift -> "did we exit too early?" (the SLS question)

Rebuilds the whole ledger each run -> logs/lottery_trade_ledger.csv. Loads .env.lottery
(the Hype account). Daily bars only (cheap); a precise trail-width simulation can pull
minute bars in a dedicated study later.

Run:  .venv/Scripts/python.exe experiments/lottery/trade_ledger.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
PICKS_DIR = ROOT / "experiments" / "lottery" / "picks"
OUT = ROOT / "logs" / "lottery_trade_ledger.csv"
INCEPTION = datetime(2026, 6, 15, tzinfo=UTC)
SIG_COLS = ["ignition", "wsb_surge", "pm_rvol", "gap_pct", "squeeze", "uoa_z",
            "gtrends_spike", "finra_short_ratio"]


def _load_env() -> None:
    f = ROOT / ".env.lottery"
    if not f.exists():
        print("FATAL: .env.lottery not found.", file=sys.stderr); sys.exit(2)
    for line in f.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip().strip('"').strip("'")


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _weekdays(d0, d1) -> int:
    n, d = 0, d0
    while d < d1:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n += 1
    return n


def _fills(tc):
    """All filled buy/sell legs since inception, chronological: (sym, ts, side, qty, px, type)."""
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    out = []
    req = GetOrdersRequest(status=QueryOrderStatus.CLOSED, after=INCEPTION, limit=500)
    try:
        orders = tc.get_orders(filter=req)
    except TypeError:
        orders = tc.get_orders(req)
    for o in orders:
        fq, fap, ft = _f(o.filled_qty), getattr(o, "filled_avg_price", None), getattr(o, "filled_at", None)
        if fq <= 0 or fap is None or ft is None:
            continue
        side = str(o.side).rsplit(".", 1)[-1].lower()
        otype = str(o.type).rsplit(".", 1)[-1].lower()
        out.append((o.symbol, ft, side, fq, _f(fap), otype))
    out.sort(key=lambda x: x[1])
    return out


def _entry_context() -> dict:
    """{(symbol, date): {combined_score, top_k_of, basket, signals}} from the picks files."""
    ctx = {}
    for fp in sorted(PICKS_DIR.glob("*.json")):
        try:
            rec = json.load(open(fp))
        except Exception:
            continue
        for p in rec.get("picks", []):
            ctx[(p["symbol"], rec.get("date", fp.stem))] = {
                "combined_score": p.get("combined_score"),
                "top_k_of": ",".join(p.get("top_k_of", []) or []),
                "basket": p.get("basket"),
                "signals": p.get("signals", {}) or {}}
    return ctx


def _build_trades(fills) -> list[dict]:
    """Walk fills per symbol; a trade = a buy lot held until flat (handles partials/re-entry)."""
    by_sym: dict[str, list] = {}
    for f in fills:
        by_sym.setdefault(f[0], []).append(f)
    trades = []
    for sym, legs in by_sym.items():
        open_t = None
        for (_, ts, side, qty, px, otype) in legs:
            if side == "buy":
                if open_t is None:
                    open_t = {"symbol": sym, "entry_ts": ts, "bq": 0.0, "bn": 0.0,
                              "sq": 0.0, "sn": 0.0, "exit_ts": None, "reasons": set()}
                open_t["bq"] += qty; open_t["bn"] += qty * px
            elif side == "sell" and open_t is not None:
                open_t["sq"] += qty; open_t["sn"] += qty * px
                open_t["exit_ts"] = ts
                open_t["reasons"].add("trail" if "trail" in otype else "time_stop")
                if open_t["sq"] >= open_t["bq"] - 1e-6:      # flat -> trade closed
                    trades.append(open_t); open_t = None
        if open_t is not None:                                # still open (full or partial)
            trades.append(open_t)
    return trades


def _daily_bars(dc, symbols):
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    import pandas as pd
    out = {}
    syms = sorted(set(symbols))
    for i in range(0, len(syms), 50):
        grp = syms[i:i + 50]
        try:
            df = dc.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=grp, timeframe=TimeFrame.Day, start=INCEPTION,
                end=datetime.now(UTC), feed=DataFeed.IEX)).df
        except Exception:
            continue
        for s in grp:
            try:
                sb = df.xs(s, level=0).copy()
                sb.index = [t.tz_convert(ET).date() for t in sb.index]
                out[s] = sb
            except KeyError:
                pass
    return out


def main() -> int:
    _load_env()
    from alpaca.trading.client import TradingClient
    from alpaca.data.historical import StockHistoricalDataClient
    tc = TradingClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], paper=True)
    dc = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])

    fills = _fills(tc)
    trades = _build_trades(fills)
    ctx = _entry_context()
    try:
        live_px = {p.symbol: _f(p.current_price) for p in tc.get_all_positions()}
    except Exception:
        live_px = {}
    bars = _daily_bars(dc, [t["symbol"] for t in trades])
    today = datetime.now(ET).date()

    rows = []
    for t in trades:
        sym = t["symbol"]
        entry_date = t["entry_ts"].astimezone(ET).date()
        entry_avg = t["bn"] / t["bq"] if t["bq"] else 0.0
        sold = t["sq"]
        open_qty = t["bq"] - sold
        is_open = open_qty > 1e-6
        exit_date = t["exit_ts"].astimezone(ET).date() if t["exit_ts"] else None
        exit_avg = t["sn"] / t["sq"] if t["sq"] else None
        # realized on the sold portion; if still open, mark the remainder at current price
        realized = (exit_avg - entry_avg) * sold if exit_avg is not None else 0.0
        unreal = (live_px.get(sym, entry_avg) - entry_avg) * open_qty if is_open else 0.0
        status = ("open" if t["sq"] == 0 else "partial-open") if is_open else "closed"
        reason = "+".join(sorted(t["reasons"])) if t["reasons"] else ""
        hold = _weekdays(entry_date, exit_date or today)

        # bar-derived outcome metrics over the hold window
        mfe = mae = post5 = None
        sb = bars.get(sym)
        if sb is not None and entry_avg > 0:
            end = exit_date or today
            win = sb[(sb.index >= entry_date) & (sb.index <= end)]
            if len(win):
                mfe = round((float(win["high"].max()) / entry_avg - 1) * 100, 2)
                mae = round((float(win["low"].min()) / entry_avg - 1) * 100, 2)
            if exit_date and exit_avg:
                after = sb[sb.index > exit_date]
                if len(after):
                    ref = after.iloc[min(4, len(after) - 1)]["close"]
                    post5 = round((float(ref) / exit_avg - 1) * 100, 2)

        c = ctx.get((sym, entry_date.isoformat()), {})
        sig = c.get("signals", {})
        row = {"symbol": sym, "entry_date": entry_date.isoformat(), "entry_avg": round(entry_avg, 4),
               "qty": round(t["bq"], 0), "status": status,
               "exit_date": exit_date.isoformat() if exit_date else "",
               "exit_avg": round(exit_avg, 4) if exit_avg is not None else "",
               "exit_reason": reason, "sold_qty": round(sold, 0), "open_qty": round(open_qty, 0),
               "realized": round(realized, 2), "unrealized": round(unreal, 2),
               "realized_pct": round((exit_avg / entry_avg - 1) * 100, 2) if exit_avg else "",
               "hold_days": hold, "combined_score": c.get("combined_score"),
               "top_k_of": c.get("top_k_of", ""), "basket": c.get("basket", ""),
               "mfe_pct": mfe, "mae_pct": mae, "post_exit_5d_pct": post5}
        for s in SIG_COLS:
            row[f"sig_{s}"] = sig.get(s)
        rows.append(row)

    rows.sort(key=lambda r: (r["entry_date"], r["symbol"]))
    OUT.parent.mkdir(exist_ok=True)
    import csv
    cols = (["symbol", "entry_date", "entry_avg", "qty", "status", "exit_date", "exit_avg",
             "exit_reason", "sold_qty", "open_qty", "realized", "unrealized", "realized_pct",
             "hold_days", "combined_score", "top_k_of", "basket", "mfe_pct", "mae_pct",
             "post_exit_5d_pct"] + [f"sig_{s}" for s in SIG_COLS])
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    closed = [r for r in rows if r["status"] == "closed"]
    wins = [r for r in closed if r["realized"] > 0]
    tot_real = sum(r["realized"] for r in rows)
    print(f"ledger: {len(rows)} trades ({len(closed)} closed, {len(rows)-len(closed)} open) -> {OUT.name}")
    if closed:
        wr = len(wins) / len(closed) * 100
        print(f"  closed: {len(wins)}/{len(closed)} winners ({wr:.0f}%), realized ${tot_real:,.0f}")
        mfes = [r["mfe_pct"] for r in closed if r["mfe_pct"] is not None]
        reals = [r["realized_pct"] for r in closed if r["realized_pct"] != ""]
        left = [r["mfe_pct"] - r["realized_pct"] for r in closed
                if r["mfe_pct"] is not None and r["realized_pct"] != ""]
        if mfes and reals and left:
            print(f"  avg MFE {sum(mfes)/len(mfes):+.1f}% vs avg realized {sum(reals)/len(reals):+.1f}% "
                  f"-> ~{sum(left)/len(left):+.1f}% avg left on the table (trail-width signal)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
