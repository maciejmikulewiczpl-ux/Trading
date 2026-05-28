"""One-shot account status snapshot — works any time, trader running or not.

Unlike the tray UI (which is a thread inside the live runner and dies when the
session ends), this queries Alpaca directly and prints a current snapshot:
market clock, account equity/cash/day-PnL, open positions, open orders, and
today's ORB order activity. Read-only — places/cancels nothing.

Run:
    .venv/Scripts/python.exe live/status.py
"""
from __future__ import annotations

import sys
from datetime import datetime, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alpaca.trading.enums import QueryOrderStatus  # noqa: E402
from alpaca.trading.requests import GetOrdersRequest  # noqa: E402

from live.paper_orb import ET, UTC, build_clients, load_env  # noqa: E402


def _f(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def main() -> int:
    load_env()
    try:
        tc, _ = build_clients()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    now = datetime.now(ET)
    print(f"ORB status - {now:%Y-%m-%d %H:%M:%S %Z}")

    # ---- market clock ----
    try:
        clock = tc.get_clock()
        if clock.is_open:
            nxt = clock.next_close.astimezone(ET)
            print(f"Market: OPEN  (closes {nxt:%H:%M %Z})")
        else:
            nxt = clock.next_open.astimezone(ET)
            print(f"Market: CLOSED  (opens {nxt:%Y-%m-%d %H:%M %Z})")
    except Exception as e:
        print(f"Market: (clock unavailable: {e})")
    print("=" * 64)

    # ---- account ----
    try:
        a = tc.get_account()
        equity = _f(a.equity)
        last_eq = _f(a.last_equity)
        day_pnl = equity - last_eq
        print(f"Account {a.account_number}   status={a.status}")
        print(f"  Equity      ${equity:>13,.2f}")
        print(f"  Cash        ${_f(a.cash):>13,.2f}")
        print(f"  Day PnL     ${day_pnl:>+13,.2f}   ({day_pnl / last_eq * 100:+.2f}% vs prior close)"
              if last_eq else f"  Day PnL     ${day_pnl:>+13,.2f}")
        print(f"  Buying pwr  ${_f(a.buying_power):>13,.2f}")
    except Exception as e:
        print(f"  (account unavailable: {e})")
    print()

    # ---- open positions ----
    try:
        positions = tc.get_all_positions()
    except Exception as e:
        positions = []
        print(f"(positions unavailable: {e})")
    print(f"Open positions ({len(positions)}):")
    if positions:
        print(f"  {'sym':<8}{'side':<6}{'qty':>10}{'avg_entry':>12}"
              f"{'current':>12}{'mkt_value':>13}{'unreal PnL':>13}")
        for p in positions:
            print(f"  {p.symbol:<8}{str(p.side).rsplit('.', 1)[-1].lower():<6}"
                  f"{_f(p.qty):>10.4f}{_f(p.avg_entry_price):>12,.2f}"
                  f"{_f(p.current_price):>12,.2f}{_f(p.market_value):>13,.2f}"
                  f"{_f(p.unrealized_pl):>+13,.2f}")
    else:
        print("  (flat)")
    print()

    # ---- open orders ----
    try:
        open_orders = tc.get_orders(filter=GetOrdersRequest(
            status=QueryOrderStatus.OPEN, limit=100))
    except TypeError:
        open_orders = tc.get_orders(GetOrdersRequest(
            status=QueryOrderStatus.OPEN, limit=100))
    except Exception as e:
        open_orders = []
        print(f"(open orders unavailable: {e})")
    print(f"Open orders ({len(open_orders)}):")
    if open_orders:
        print(f"  {'sym':<8}{'side':<6}{'type':<12}{'qty':>8}"
              f"{'limit':>10}{'stop':>10}  status")
        for o in open_orders:
            otype = str(getattr(o, "order_type", "") or o.type).rsplit(".", 1)[-1]
            lp = getattr(o, "limit_price", None)
            sp = getattr(o, "stop_price", None)
            print(f"  {o.symbol:<8}{str(o.side).rsplit('.', 1)[-1].lower():<6}"
                  f"{otype:<12}{_f(o.qty):>8.0f}"
                  f"{('$' + format(_f(lp), ',.2f')) if lp else '-':>10}"
                  f"{('$' + format(_f(sp), ',.2f')) if sp else '-':>10}"
                  f"  {str(o.status).rsplit('.', 1)[-1]}")
    else:
        print("  (none)")
    print()

    # ---- today's ORB order activity ----
    today = now.date()
    today_start = datetime.combine(today, time(0, 0, tzinfo=ET))
    try:
        req = GetOrdersRequest(status=QueryOrderStatus.ALL,
                               after=today_start.astimezone(UTC), limit=200)
        try:
            todays = tc.get_orders(filter=req)
        except TypeError:
            todays = tc.get_orders(req)
    except Exception as e:
        todays = []
        print(f"(today's orders unavailable: {e})")
    coid_prefix = f"orb-{today:%Y%m%d}-"
    orb_fills = [o for o in todays
                 if getattr(o, "client_order_id", "")
                 and o.client_order_id.startswith(coid_prefix)
                 and getattr(o, "filled_avg_price", None) is not None]
    print(f"Today's ORB entries filled ({len(orb_fills)}):")
    if orb_fills:
        for o in orb_fills:
            print(f"  {o.symbol:<6} {str(o.side).rsplit('.', 1)[-1].lower():<5} "
                  f"{_f(o.filled_qty):>4.0f} @ ${_f(o.filled_avg_price):,.2f}  "
                  f"{o.client_order_id}")
    else:
        print("  (no ORB entries filled today)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
