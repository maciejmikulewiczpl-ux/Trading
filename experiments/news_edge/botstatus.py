"""Quick live status of the news-edge bot's 2nd paper account (positions, day P/L, fills)."""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for line in (ROOT / ".env.news").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip().strip('"').strip("'")

from alpaca.trading.client import TradingClient  # noqa: E402
from alpaca.trading.enums import QueryOrderStatus  # noqa: E402
from alpaca.trading.requests import GetOrdersRequest  # noqa: E402

tc = TradingClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], paper=True)
a = tc.get_account()
eq, le = float(a.equity), float(a.last_equity)
dp = eq - le
print(f"NEWS BOT  acct {a.account_number} | equity ${eq:,.2f} | day P/L ${dp:+,.2f} ({(dp/le*100) if le else 0:+.2f}%)")

pos = tc.get_all_positions()
print(f"open positions: {len(pos)}")
for p in pos:
    side = str(p.side).split(".")[-1]
    print(f"  {p.symbol:5} {side:5} qty {float(p.qty):>4.0f}  avg ${float(p.avg_entry_price):7.2f}  "
          f"cur ${float(p.current_price):7.2f}  uPL ${float(p.unrealized_pl):+8.2f} ({float(p.unrealized_plpc)*100:+.2f}%)")

since = datetime.now(timezone.utc) - timedelta(hours=12)
try:
    orders = tc.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.ALL, after=since, limit=100))
except TypeError:
    orders = tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, after=since, limit=100))
fills = [o for o in orders if getattr(o, "filled_at", None) and float(o.filled_qty or 0) > 0]
print(f"fills today: {len(fills)}")
for o in sorted(fills, key=lambda o: o.filled_at):
    side = str(o.side).split(".")[-1]
    print(f"  {o.symbol:5} {side:4} {float(o.filled_qty):>4.0f} @ ${float(o.filled_avg_price):7.2f}  "
          f"({o.filled_at.astimezone().strftime('%H:%M')})")
