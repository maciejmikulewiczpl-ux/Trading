"""Lottery paper bot (Track C) — the BOLD day-1 forward-test trader.

At ~09:45 ET it reads today's lottery board (experiments/lottery/picks/<ET-date>.json),
takes the TOP-3 by combined_score (the hype basket), and for each:
  - buys ~$2,000 of WHOLE shares at market (qty = floor($2000/price); names pricier than
    $2000 are skipped -- NOT fractional/notional, because Alpaca rejects trailing stops
    on fractional quantities (found in the 2026-06-12 end-to-end trace; a notional buy
    would have left positions with NO stop at all). $2,000/name MATCHES the news-edge
    bot's ORB_NOTIONAL_PER_TRADE=2000 (also whole-share floored, paper_orb.py) so the two
    bots' dollar PnL is directly comparable. Sizing is COLOR only -- the lottery verdict
    is Track B's size-independent hit-rate, so this budget change doesn't affect it,
  - attaches a 10% NATIVE Alpaca trailing stop on fill, GTC (holds up to T+3 -- a DAY trail
    would expire at the close and leave the rest unprotected),
  - records the entry so a later run can time-stop close at T+3 if still open (cancelling the
    GTC trail first, else Alpaca rejects the close for insufficient available qty).

EXIT = 10% trailing stop + T+3 max hold. A same-day 15:55 exit was TESTED 2026-06-30 and REVERTED
2026-07-01: horizon_curve.py showed the bot's PnL is TAIL-driven and the fat tail lives in the +1
to +2 day move -- same-day exit captured only ~1/6th of the total (top-3 SUM +7.8% vs +49.7% at
+2d), i.e. it capped the moonshots that drive the realized PnL. The ~-0.64%/night overnight gap
bleed is real but the multi-day tail more than pays for it. `--eod-close` REMAINS as a MANUAL
emergency "close everything now" command; it is NOT scheduled. (Sweet spot looks ~T+2; the trailing
stop exits most winners before T+3 anyway -- a T+2 refinement is a separate, still-to-validate tweak.)

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

# Leveraged / inverse / single-stock-leveraged ETFs: derivative VEHICLES, not "hype lottery"
# names. They surface on the board via gappers (SOXL topped it 2026-06-23) but are the worst
# overnight-gap offenders and don't fit the thesis, so the bot SKIPS them at entry (the board
# still scores them for measurement continuity). Curated core set; extend as needed. (An
# asset-name keyword check -- "Bull/Bear/Ultra/3X/2X" -- is a future, more-complete approach.)
LEVERAGED_INVERSE_ETFS = frozenset({
    # broad-index leveraged/inverse
    "TQQQ", "SQQQ", "UPRO", "SPXU", "SPXL", "SPXS", "SSO", "SDS", "QLD", "QID",
    "UDOW", "SDOW", "DDM", "DXD", "TNA", "TZA", "UWM", "TWM", "MIDU", "URTY", "SRTY",
    # volatility
    "UVXY", "SVXY", "VIXY", "UVIX", "SVIX", "VXX",
    # sector / thematic leveraged
    "SOXL", "SOXS", "LABU", "LABD", "TECL", "TECS", "FAS", "FAZ", "NUGT", "DUST",
    "JNUG", "JDST", "GUSH", "DRIP", "ERX", "ERY", "CURE", "DPST", "WEBL", "WEBS",
    "HIBL", "HIBS", "RETL", "NAIL", "DFEN", "DUSL", "YINN", "YANG", "TMF", "TMV",
    "BOIL", "KOLD", "UCO", "SCO", "AGQ", "ZSL", "UGL", "GLL", "DRN", "DRV",
    # single-stock leveraged/inverse (the fast-growing crop)
    "TSLL", "TSLQ", "TSLT", "TSLS", "NVDL", "NVDU", "NVDD", "NVDX", "CONL", "COND",
    "MSTU", "MSTX", "MSTZ", "AAPU", "AAPD", "GGLL", "GGLS", "MSFU", "MSFD",
    "AMZU", "AMZD", "METU", "METD", "AMDL", "AMDS", "PLTU", "AMUU", "AVL",
})

NOTIONAL = 2000.0          # $ per pick (matches news-edge ORB_NOTIONAL_PER_TRADE for a direct PnL comparison)
TRAIL_PCT = 10.0           # native trailing stop % (INTRADAY protection only now)
MAX_SPREAD_PCT = 3.0       # skip names whose bid/ask spread is wider than this (illiquid / heavy slippage)
LIQUID_DOLLAR_VOL = 5_000_000.0  # prior-session CONSOLIDATED (SIP) $-volume above which a $2000 clip has
                           # ~zero market impact -> BYPASS the spread guard. The free feed's single-venue
                           # IEX NBBO is noisy and FALSE-REJECTS liquid names (2026-07-02: RDDT showed a
                           # 4.7% IEX spread, OUST 14%, yet both trade $150M+/day -- the bot bought nothing
                           # that day). SIP prior-day daily bars ARE readable on free tier (recent SIP
                           # quotes are not), so SIP $-vol is the reliable market-impact measure. Names
                           # with NO SIP daily bar (true microcaps, e.g. RGC) fall through to the spread guard.
TOP_N = 3                  # top-3 by combined_score
TIME_STOP_DAYS = 3         # close at T+3 trading-ish days if still open (the 10% trailing stop
                           # exits most winners before then). Same-day exit tested + REVERTED
                           # 2026-07-01 (capped the +1-2d tail that drives PnL); --eod-close is a manual tool.
STATE_FILE = ROOT / "logs" / "lottery_positions.json"
EXEC_LOG = ROOT / "logs" / "lottery_execution.csv"   # intended-quote vs actual-fill (slippage/capacity)

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


def _latest_prices(symbols: list[str]) -> dict[str, float]:
    """Latest trade price per symbol (IEX) -- for whole-share sizing."""
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestTradeRequest
        from alpaca.data.enums import DataFeed
        dc = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"],
                                       os.environ["ALPACA_SECRET_KEY"])
        res = dc.get_stock_latest_trade(StockLatestTradeRequest(
            symbol_or_symbols=symbols, feed=DataFeed.IEX))
        return {s: float(t.price) for s, t in res.items() if t and t.price}
    except Exception as e:
        print(f"latest-price fetch failed: {e}")
        return {}


def _latest_quotes(symbols: list[str]) -> dict[str, tuple]:
    """Latest NBBO (bid, ask) per symbol at trade time -- spread/slippage context."""
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestQuoteRequest
        from alpaca.data.enums import DataFeed
        dc = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"],
                                       os.environ["ALPACA_SECRET_KEY"])
        res = dc.get_stock_latest_quote(StockLatestQuoteRequest(
            symbol_or_symbols=symbols, feed=DataFeed.IEX))
        return {s: (float(q.bid_price), float(q.ask_price)) for s, q in res.items()
                if q and q.bid_price and q.ask_price}
    except Exception as e:
        print(f"quote fetch failed: {e}")
        return {}


def _prev_dollar_volumes(symbols: list[str]) -> dict[str, float]:
    """Prior-session CONSOLIDATED (SIP) dollar volume per symbol -- the real market-impact
    proxy for the liquidity bypass. Free tier can't query recent SIP *quotes* but CAN read
    prior-day SIP daily bars (>15 min old), so this is the reliable liquidity signal the
    noisy single-venue IEX NBBO can't give. Names with no SIP daily bar (true microcaps)
    are simply absent -> the caller keeps applying the spread guard to them."""
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from alpaca.data.enums import DataFeed
        dc = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"],
                                       os.environ["ALPACA_SECRET_KEY"])
        bars = dc.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=symbols, timeframe=TimeFrame.Day, limit=2, feed=DataFeed.SIP))
        data = getattr(bars, "data", {}) or {}
        out: dict[str, float] = {}
        for s in symbols:
            sb = data.get(s) or []
            if sb:
                b = sb[-1]   # last COMPLETED session (SIP's 15-min delay excludes today's partial bar)
                vwap = float(getattr(b, "vwap", 0) or getattr(b, "close", 0) or 0)
                if vwap > 0:
                    out[s] = float(b.volume) * vwap
        return out
    except Exception as e:
        print(f"dollar-volume fetch failed: {e}")
        return {}


def _log_execution(row: dict) -> None:
    """Append one trade-time execution record (intended vs fill) to lottery_execution.csv."""
    import csv
    cols = ["date", "submit_ts", "symbol", "qty", "intended_px", "bid", "ask", "mid",
            "spread_bps", "fill_avg", "slip_vs_intended_bps", "slip_vs_mid_bps", "order_id"]
    EXEC_LOG.parent.mkdir(exist_ok=True)
    new = not EXEC_LOG.exists()
    try:
        with open(EXEC_LOG, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            if new:
                w.writeheader()
            w.writerow({k: row.get(k) for k in cols})
    except Exception as e:
        print(f"  exec-log write failed: {e}")


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


def run_time_stops(tc, dry_run: bool, close_all: bool = False) -> int:
    """Close tracked positions. close_all=True (the ~15:55 ET --eod-close run) force-closes
    EVERY tracked position the same day it was opened; close_all=False (the morning run) is
    the T+TIME_STOP_DAYS BACKSTOP for anything a missed EOD run left open overnight."""
    from alpaca.trading.requests import GetOrdersRequest
    state = _load_state()
    if not state:
        print("close: no tracked positions.")
        return 0
    try:
        positions = {p.symbol: p for p in tc.get_all_positions()}
    except Exception as e:
        print(f"close: get_all_positions failed: {e}")
        positions = {}
    closed = 0
    for sym in list(state.keys()):
        info = state[sym]
        age = _trading_days_since(info["entry_date"])
        if sym not in positions:
            # position gone (trailing stop hit or manually closed) -> drop tracking,
            # and cancel any leftover GTC trail so it can't fire with no position
            oid = info.get("trail_order_id")
            if oid and not dry_run:
                try:
                    tc.cancel_order_by_id(oid)
                except Exception:
                    pass   # usually already filled/cancelled -- that's how the position closed
            print(f"close: {sym} no longer held (age {age}) -> untracked.")
            del state[sym]
            continue
        if close_all or age >= TIME_STOP_DAYS:
            reason = "EOD same-day" if close_all else f"T+{age} backstop (>= {TIME_STOP_DAYS})"
            if dry_run:
                print(f"[DRY-RUN] WOULD close {sym} ({reason})")
            else:
                try:
                    # cancel the GTC trailing stop FIRST -- with it open, the shares are
                    # reserved and Alpaca rejects the close (insufficient qty available)
                    oid = info.get("trail_order_id")
                    if oid:
                        try:
                            tc.cancel_order_by_id(oid)
                            print(f"close: cancelled trail order {oid} for {sym}.")
                        except Exception as ce:
                            print(f"close: trail cancel {sym} ({oid}) failed/already done: {ce}")
                    tc.close_position(sym)
                    print(f"CLOSED {sym} ({reason}, age {age}).")
                    del state[sym]
                    closed += 1
                except Exception as e:
                    print(f"close {sym} FAILED: {e}")
        else:
            print(f"close: {sym} age {age} < {TIME_STOP_DAYS}, hold (backstop not reached).")
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

    pick_syms = [p["symbol"] for p in picks]
    prices = _latest_prices(pick_syms)
    quotes = _latest_quotes(pick_syms)   # bid/ask at trade time for slippage/capacity analysis
    dvols = _prev_dollar_volumes(pick_syms)   # prior-day SIP $-vol: liquidity bypass for IEX-noise spreads
    placed = 0
    for p in picks:
        sym = p["symbol"]
        if sym in held or sym in state:
            print(f"  {sym}: already held/tracked, skip.")
            continue
        if sym in LEVERAGED_INVERSE_ETFS:
            print(f"  {sym}: leveraged/inverse ETF -- skip (derivative vehicle, not a hype name).")
            continue
        px = prices.get(sym)
        if px is None or px <= 0:
            print(f"  {sym}: no price available -- skip.")
            continue
        qty = int(NOTIONAL // px)   # WHOLE shares: trailing stops reject fractional qty
        if qty <= 0:
            print(f"  {sym}: price ${px:.2f} > ${NOTIONAL:.0f} budget -- skip "
                  f"(whole-share constraint).")
            continue
        # LIQUIDITY guard (market-impact): skip names with an absurdly wide quote spread =
        # untradeable / heavy slippage. The free feed's IEX NBBO is single-venue and noisy, so
        # a wide reading alone false-rejects genuinely liquid names -- prior-day SIP dollar
        # volume (LIQUID_DOLLAR_VOL) overrides the spread for names that clearly can absorb the clip.
        bid, ask = quotes.get(sym, (None, None))
        if bid and ask and ask > 0:
            spread_pct = (ask - bid) / ((ask + bid) / 2.0) * 100
            if spread_pct > MAX_SPREAD_PCT:
                dv = dvols.get(sym, 0.0)
                if dv >= LIQUID_DOLLAR_VOL:
                    # liquid name: a $2000 clip is a rounding error vs $-vol, so the wide IEX
                    # quote is single-venue noise, not real illiquidity -> keep it.
                    print(f"  {sym}: IEX spread {spread_pct:.1f}% but prior SIP $-vol "
                          f"${dv/1e6:.0f}M >= ${LIQUID_DOLLAR_VOL/1e6:.0f}M -- IEX noise, KEEP.")
                else:
                    print(f"  {sym}: spread {spread_pct:.1f}% > {MAX_SPREAD_PCT:.0f}% and $-vol "
                          f"${dv/1e6:.1f}M < ${LIQUID_DOLLAR_VOL/1e6:.0f}M -- illiquid, "
                          f"skip (market-impact guard).")
                    continue
        if dry_run:
            print(f"  [DRY-RUN] WOULD BUY {sym} {qty} sh @ ~${px:.2f} (~${qty*px:.0f}) "
                  f"+ {TRAIL_PCT:.0f}% GTC trailing stop on fill")
            continue
        try:
            order = tc.submit_order(MarketOrderRequest(
                symbol=sym, qty=qty, side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY))
            print(f"  BUY submitted {sym} {qty} sh @ ~${px:.2f} (order {order.id}).")
            bid, ask = quotes.get(sym, (None, None))
            state[sym] = {"entry_date": date_str, "qty": qty, "ref_price": px,
                          "combined_score": p["combined_score"], "trail_attached": False,
                          "buy_order_id": str(order.id),
                          "exec": {"submit_ts": datetime.now(ET).isoformat(timespec="seconds"),
                                   "intended_px": px, "bid": bid, "ask": ask, "qty": qty},
                          "exec_logged": False}
            placed += 1
        except Exception as e:
            print(f"  {sym} BUY FAILED: {e}")
    if not dry_run:
        _save_state(state)
    return placed


def _filled_qty_when_done(tc, order_id: str, max_wait_s: int = 12) -> tuple:
    """Poll a buy order until it's terminally filled, returning (full filled_qty, fill_avg).
    A market order on a thin name can fill progressively over several seconds; reading the
    position too early (the old fixed 3s sleep) left late-settling shares with NO stop —
    that's how SLS ended up with 178 bought but only 170 trailed (2026-06-29). Poll the
    order itself so the trail always covers the complete fill."""
    import time as _t
    best, avg = 0, None
    for _ in range(max_wait_s):
        try:
            o = tc.get_order_by_id(order_id)
            fq = int(float(getattr(o, "filled_qty", 0) or 0))
            if fq >= best:
                best = fq
                fap = getattr(o, "filled_avg_price", None)
                avg = float(fap) if fap else avg
            st = str(getattr(o, "status", "")).rsplit(".", 1)[-1].lower()
            if st in ("filled", "canceled", "cancelled", "expired", "rejected"):
                return best, avg
        except Exception:
            pass
        _t.sleep(1)
    return best, avg


def attach_trailing_stops(tc, dry_run: bool) -> int:
    """For any tracked position with no trailing stop yet, attach a 10% native trailing
    stop covering the FULL filled quantity. Waits for the buy order to finish filling
    (not a fixed sleep) so no shares are left unprotected; tops up if the held position
    still exceeds what we trailed. Idempotent (skips ones already attached)."""
    from alpaca.trading.requests import TrailingStopOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    if dry_run:
        return 0
    state = _load_state()
    attached = 0
    for sym, info in state.items():
        if info.get("trail_attached"):
            continue
        # full filled qty + avg price from the buy order (handles slow/partial market fills)
        filled, fill_avg = (_filled_qty_when_done(tc, info["buy_order_id"])
                            if info.get("buy_order_id") else (0, None))
        # log execution quality (intended quote vs actual fill) once per trade
        ex = info.get("exec")
        if ex and not info.get("exec_logged") and fill_avg:
            bid, ask, intended = ex.get("bid"), ex.get("ask"), ex.get("intended_px")
            mid = ((bid + ask) / 2) if (bid and ask) else None
            _log_execution({
                "date": info.get("entry_date"), "submit_ts": ex.get("submit_ts"),
                "symbol": sym, "qty": filled, "intended_px": intended, "bid": bid, "ask": ask,
                "mid": round(mid, 4) if mid else None,
                "spread_bps": round((ask - bid) / mid * 10000, 1) if mid else None,
                "fill_avg": round(fill_avg, 4),
                "slip_vs_intended_bps": round((fill_avg / intended - 1) * 10000, 1) if intended else None,
                "slip_vs_mid_bps": round((fill_avg / mid - 1) * 10000, 1) if mid else None,
                "order_id": info.get("buy_order_id")})
            info["exec_logged"] = True
        try:
            pos = {p.symbol: p for p in tc.get_all_positions()}.get(sym)
            held_qty = int(float(pos.qty)) if pos else 0
        except Exception:
            held_qty = 0
        qty = max(filled, held_qty)        # cover everything that actually landed
        qty = min(qty, held_qty) if held_qty else qty   # never exceed what's held
        if qty <= 0:
            print(f"  {sym}: no filled shares to protect yet -- skip.")
            continue
        try:
            to = tc.submit_order(TrailingStopOrderRequest(
                symbol=sym, qty=qty, side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC, trail_percent=TRAIL_PCT))
            info["trail_attached"] = True
            info["trail_order_id"] = str(to.id)
            info["trail_qty"] = qty
            print(f"  trailing stop attached {sym}: {TRAIL_PCT:.0f}% on {qty} sh (order {to.id}).")
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
    eod_close = "--eod-close" in args   # the ~15:55 ET run: force-close everything SAME DAY

    tc = _trading_client()
    try:
        acct = tc.get_account()
        print(f"lottery bot account: equity ${float(acct.equity):,.2f}  "
              f"cash ${float(acct.cash):,.2f}  (paper)")
    except Exception as e:
        print(f"account fetch failed: {e}")

    _emit_heartbeat("start")

    # SAME-DAY EOD close (~15:55 ET): force-close every tracked position, then exit.
    if eod_close:
        n = run_time_stops(tc, dry_run, close_all=True)
        print(f"eod-close: {n} position(s) closed.")
        _emit_heartbeat("done")
        return 0

    # 1. backstop close first (T+1 safety for anything a missed EOD run left open overnight)
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
