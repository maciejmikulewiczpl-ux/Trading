"""TSMOM sleeve REBALANCE bot -- Alpaca paper, monthly. The diversifier that runs ALONGSIDE the MES
intraday-momentum bot (the two are negatively correlated -> combined Sharpe ~1.8, [[futures_diversification_wip]]).

Runs on the VM (reliable -- no laptop/TWS dependency, unlike the futures bot). Once a month it rebalances
the account to tsmom.signal.target_weights (long-only cross-asset trend, inverse-vol). Between rebalances
it just holds. Its own Alpaca paper account (.env.tsmom) so positions don't mix with the ORB/hype bots.

    .venv/Scripts/python.exe tsmom/rebalance.py            # DRY-RUN (no orders)
    DRY_RUN=0 .venv/Scripts/python.exe tsmom/rebalance.py  # arm (paper account only)
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tsmom.weights import BASKET, weights_from_daily  # noqa: E402

ET = ZoneInfo("America/New_York")
STATUS = ROOT / "tsmom" / "status.json"
MIN_TRADE_FRAC = 0.005   # skip rebalancing a name whose drift is < 0.5% of equity (avoid churn)


def _load_env() -> None:
    f = ROOT / ".env.tsmom"
    if not f.exists():
        print("FATAL: .env.tsmom not found. Create an Alpaca PAPER account (free) and add its keys "
              "(ALPACA_API_KEY / ALPACA_SECRET_KEY). See .env.tsmom.example.", file=sys.stderr)
        raise SystemExit(1)
    for line in f.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.split("#", 1)[0].strip())


def _clients():
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.trading.client import TradingClient
    k, s = os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"]
    return TradingClient(k, s, paper=True), StockHistoricalDataClient(k, s)


def _prices(dc, syms):
    from alpaca.data.requests import StockLatestTradeRequest
    if not syms:
        return {}
    r = dc.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=list(syms)))
    return {s: float(r[s].price) for s in r}


def _daily_closes(dc):
    """Daily close DataFrame (cols=ETFs) from Alpaca IEX -- the momentum signal input (~15 months)."""
    from datetime import datetime, timedelta, timezone
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    df = dc.get_stock_bars(StockBarsRequest(
        symbol_or_symbols=list(BASKET), timeframe=TimeFrame.Day,
        start=datetime.now(timezone.utc) - timedelta(days=470),
        adjustment=Adjustment.ALL,          # dividend+split adjusted -> matches the research; bonds count
        feed=DataFeed.IEX)).df
    return df["close"].unstack(level=0)   # (date x symbol)


def _log(msg):
    print(f"[{datetime.now(ET):%Y-%m-%d %H:%M:%S}] {msg}")


def main() -> int:
    _load_env()
    dry = os.environ.get("DRY_RUN", "1") != "0"
    tc, dc = _clients()
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    acct = tc.get_account()
    capital = float(os.environ.get("TSMOM_CAPITAL", "250000"))   # fixed slice -> safe to SHARE an account
    tw = weights_from_daily(_daily_closes(dc))               # {ETF: weight} from Alpaca daily bars
    allpos = {p.symbol: p for p in tc.get_all_positions()}
    pos = {s: p for s, p in allpos.items() if s in BASKET}   # ONLY our ETFs -- never touch other bots' names
    syms = sorted(BASKET)                                    # manage only the basket
    px = _prices(dc, syms)
    _log(f"TSMOM rebalance | DRY_RUN={dry} | capital ${capital:,.0f} | {len(tw)} targets, "
         f"{len(pos)} basket holdings (account holds {len(allpos)} total)")

    orders, plan = [], []
    for s in syms:
        if s not in px or px[s] <= 0:
            continue
        tgt_sh = int((tw.get(s, 0.0) * capital) / px[s])     # size off the fixed slice, not shared equity
        cur_sh = int(float(pos[s].qty)) if s in pos else 0
        delta = tgt_sh - cur_sh
        if delta == 0 or abs(delta) * px[s] < MIN_TRADE_FRAC * capital:
            continue
        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        plan.append({"symbol": s, "side": side.value, "qty": abs(delta), "px": round(px[s], 2)})
        if not dry:
            try:
                tc.submit_order(MarketOrderRequest(symbol=s, qty=abs(delta), side=side,
                                                   time_in_force=TimeInForce.DAY))
                orders.append(s)
            except Exception as e:
                _log(f"  order {s} failed: {str(e)[:60]}")
    for p in plan:
        _log(f"  {'WOULD ' if dry else ''}{p['side']} {p['qty']} {p['symbol']} @ ~${p['px']}")
    if not plan:
        _log("  already at target -- no rebalance needed.")

    # status for the dashboard -- the SLEEVE's own P&L (its ETF holdings), not the shared account total
    positions = [{"symbol": p.symbol, "qty": float(p.qty), "value": float(p.market_value),
                  "unreal": float(p.unrealized_pl), "unreal_pc": float(p.unrealized_plpc) * 100}
                 for p in pos.values()]
    sleeve_value = sum(p["value"] for p in positions)
    STATUS.write_text(json.dumps({
        "updated": datetime.now(ET).isoformat(timespec="seconds"), "dry_run": dry,
        "capital": capital, "sleeve_value": round(sleeve_value),
        "sleeve_unreal": round(sum(p["unreal"] for p in positions)),
        "invested_pct": round(sleeve_value / capital * 100) if capital else 0,
        "shared_account": len(allpos) > len(pos),
        "targets": {t: round(w * 100, 1) for t, w in sorted(tw.items(), key=lambda x: -x[1])},
        "positions": sorted(positions, key=lambda x: -x["value"]),
        "planned": plan, "executed": orders,
    }, indent=2, default=str))
    _log(f"status written ({'dry-run' if dry else str(len(orders))+' orders sent'}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
