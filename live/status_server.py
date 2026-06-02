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
from datetime import datetime, time as dtime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alpaca.trading.enums import QueryOrderStatus  # noqa: E402
from alpaca.trading.requests import GetOrdersRequest  # noqa: E402

from live import heartbeat  # noqa: E402
from live.paper_orb import ET, UTC, EOD_FLAT_TIME, build_clients, load_env  # noqa: E402

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
# Trade state: authoritative, straight from Alpaca
# --------------------------------------------------------------------------
def _gather(tc) -> dict:
    now_et = datetime.now(ET)
    out: dict = {
        "generated": now_et.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "errors": [],
        "market": None,
        "account": None,
        "positions": [],
        "open_orders": [],
        "orb_fills": [],
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

    try:
        for p in tc.get_all_positions():
            out["positions"].append({
                "symbol": p.symbol,
                "side": str(p.side).rsplit(".", 1)[-1].lower(),
                "qty": _f(p.qty),
                "avg_entry": _f(p.avg_entry_price),
                "current": _f(p.current_price),
                "market_value": _f(p.market_value),
                "unrealized_pl": _f(p.unrealized_pl),
            })
    except Exception as e:
        out["errors"].append(f"positions: {e}")

    try:
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=100)
        try:
            orders = tc.get_orders(filter=req)
        except TypeError:
            orders = tc.get_orders(req)
        for o in orders:
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
    except Exception as e:
        out["errors"].append(f"open orders: {e}")

    try:
        today = now_et.date()
        start = datetime.combine(today, dtime(0, 0, tzinfo=ET)).astimezone(UTC)
        req = GetOrdersRequest(status=QueryOrderStatus.ALL, after=start, limit=200)
        try:
            todays = tc.get_orders(filter=req)
        except TypeError:
            todays = tc.get_orders(req)
        prefix = f"orb-{today:%Y%m%d}-"
        for o in todays:
            coid = getattr(o, "client_order_id", "") or ""
            if coid.startswith(prefix) and getattr(o, "filled_avg_price", None) is not None:
                out["orb_fills"].append({
                    "symbol": o.symbol,
                    "side": str(o.side).rsplit(".", 1)[-1].lower(),
                    "qty": _f(o.filled_qty),
                    "price": _f(o.filled_avg_price),
                    "coid": coid,
                })
    except Exception as e:
        out["errors"].append(f"today's orders: {e}")

    out["liveness"] = _liveness(heartbeat.read(), clock, now_et)
    return out


def _status(tc) -> dict:
    now = _time.time()
    if _cache["data"] is None or now - _cache["ts"] > CACHE_TTL:
        _cache["data"] = _gather(tc)
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
</style></head>
<body><div class="wrap" id="root">loading…</div>
<script>
const money = v => (v<0?"-$":"$") + Math.abs(v).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2});
const sign = v => (v>=0?"+":"") + money(v);
const cls = v => v>=0 ? "pos":"neg";
function row(cells){ return "<tr>"+cells.map((c,i)=>`<td${c.cls?` class="${c.cls}"`:""}>${c.v}</td>`).join("")+"</tr>"; }
function render(d){
  const L=d.liveness;
  let h=`<div class="banner s-${L.state}"><span class="dot"></span><h1>${L.headline}</h1><p>${L.detail}</p></div>`;
  if(d.market){ h+=`<div class="card"><h2>Market</h2>${d.market.label}</div>`; }
  const a=d.account;
  if(a){
    h+=`<div class="grid">
      <div class="stat"><div class="k">Equity</div><div class="v">${money(a.equity)}</div></div>
      <div class="stat"><div class="k">Day P/L</div><div class="v ${cls(a.day_pnl)}">${sign(a.day_pnl)}<br><small>${a.day_pnl_pct>=0?"+":""}${a.day_pnl_pct.toFixed(2)}%</small></div></div>
      <div class="stat"><div class="k">Cash</div><div class="v">${money(a.cash)}</div></div>
      <div class="stat"><div class="k">Buying power</div><div class="v">${money(a.buying_power)}</div></div>
    </div>`;
  }
  // positions
  h+=`<div class="card"><h2>Open positions (${d.positions.length})</h2>`;
  if(d.positions.length){
    h+=`<table><tr><th>sym</th><th>side</th><th>qty</th><th>avg</th><th>last</th><th>value</th><th>unreal P/L</th></tr>`;
    for(const p of d.positions) h+=row([{v:p.symbol},{v:p.side},{v:p.qty.toFixed(2)},{v:money(p.avg_entry)},{v:money(p.current)},{v:money(p.market_value)},{v:sign(p.unrealized_pl),cls:cls(p.unrealized_pl)}]);
    h+=`</table>`;
  } else h+=`<div class="empty">flat</div>`;
  h+=`</div>`;
  // open orders
  h+=`<div class="card"><h2>Open orders (${d.open_orders.length})</h2>`;
  if(d.open_orders.length){
    h+=`<table><tr><th>sym</th><th>side</th><th>type</th><th>qty</th><th>limit</th><th>stop</th><th>status</th></tr>`;
    for(const o of d.open_orders) h+=row([{v:o.symbol},{v:o.side},{v:o.type},{v:o.qty.toFixed(0)},{v:o.limit?money(o.limit):"—"},{v:o.stop?money(o.stop):"—"},{v:o.status}]);
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
  if(d.errors && d.errors.length) h+=`<div class="err">⚠ ${d.errors.join(" · ")}</div>`;
  h+=`<div class="foot"><span>snapshot ${d.generated}</span><span id="tick"></span></div>`;
  document.getElementById("root").innerHTML=h;
}
let fails=0;
async function tick(){
  try{ const r=await fetch("/api/status",{cache:"no-store"}); render(await r.json()); fails=0;
       document.getElementById("tick").textContent="live ●"; }
  catch(e){ fails++; const t=document.getElementById("tick");
       if(t) t.textContent=`page offline (${fails}) — is the SSH tunnel up?`; }
}
tick(); setInterval(tick, 3000);
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    tc = None  # set in serve()
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
                body = json.dumps(_status(self.tc)).encode("utf-8")
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
        tc, _ = build_clients()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    Handler.tc = tc
    httpd = ThreadingHTTPServer((BIND, PORT), Handler)
    print(f"ORB status server on http://{BIND}:{PORT}  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    return 0


if __name__ == "__main__":
    sys.exit(serve())
