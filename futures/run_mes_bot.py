"""MES paper bot -- the LIVE intraday-momentum strategy (the one validated candidate).

Strategy (from futures/backtest_momentum.py, the config that won the fair 5-min test): each :00/:30 ET,
compare price to a dynamic gap-adjusted noise band (open x (1 +/- 14d avg move-by-time)); above upper
-> long, below lower -> short; RIDE TO THE CLOSE (NO trailing stop -- stops hurt), flat at 15:55.

Because there is no intra-bar stop, each pass just REPLAYS today's completed :00/:30 decisions to get
the current target position (-1/0/+1) x MES_QTY, then reconciles the IBKR position to it with a market
order. Idempotent: a missed pass self-corrects next time; the broker position IS the state (no state
file). Fired every ~5 min during RTH by mes-bot.timer. DRY_RUN=1 logs the intended action, no orders.

    .venv-openbb/Scripts/python.exe futures/run_mes_bot.py            # DRY-RUN (needs TWS for data)
    DRY_RUN=0 .venv-openbb/Scripts/python.exe futures/run_mes_bot.py  # arm (PAPER account only)
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from futures import broker_ibkr as brk  # noqa: E402
from futures.backtest_momentum import EOD, RTH_OPEN, build_sigma  # same setup as the backtest  # noqa: E402

ET = ZoneInfo("America/New_York")


def _load_dotenv(name: str = ".env.futures") -> None:
    f = Path(__file__).resolve().parents[1] / name
    if not f.exists():
        return
    for line in f.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.split("#", 1)[0].strip())


_load_dotenv()
QTY = int(os.environ.get("MES_QTY", "1"))
DRY_RUN = os.environ.get("DRY_RUN", "1") != "0"


def _log(msg: str) -> None:
    print(f"[{datetime.now(ET):%Y-%m-%d %H:%M:%S}] {msg}")


def compute_target(df, today, now) -> tuple[int, str]:
    """Replay today's completed :00/:30 decisions vs the gap-adjusted noise band -> target pos -1/0/+1.
    df = intraday OHLCV (ET) with >=14 prior days of history + today so far."""
    rth = df[(df.index.time >= RTH_OPEN) & (df.index.time <= EOD)]
    day_df = {d: rth[rth.index.date == d] for d in sorted({t.date() for t in rth.index})}
    sigma = build_sigma(day_df)                      # sigma[(d,key)] uses only PRIOR days -> no leak
    today_rth = day_df.get(today)
    if today_rth is None or len(today_rth) == 0:
        return 0, "no RTH bars yet today"
    op = today_rth["open"].iloc[0]
    prior = [d for d in sorted(day_df) if d < today]
    pc = day_df[prior[-1]]["close"].iloc[-1] if prior else None
    gap = (op / pc - 1.0) if pc else 0.0
    dec = today_rth[((today_rth.index.minute == 0) | (today_rth.index.minute == 30))
                    & (today_rth.index.time > RTH_OPEN) & (today_rth.index <= now)]
    pos, since = 0, None
    for ts, row in dec.iterrows():
        s = sigma.get((today, f"{ts.hour:02d}:{ts.minute:02d}"))
        if s is None:
            continue
        up, lo = op * (1 + s), op * (1 - s)
        if gap < 0:
            up += -gap * op
        elif gap > 0:
            lo -= gap * op
        px = row["close"]
        new = 1 if px > up else (-1 if px < lo else pos)
        if new != pos:                                # position established/flipped at this bar
            pos, since = new, f"{'LONG' if new > 0 else 'SHORT'} from {ts:%H:%M} @{px:.2f}"
    info = (f"holding {since}, gap{gap*100:+.2f}%" if pos else
            f"flat (band unbroken so far), gap{gap*100:+.2f}%")
    return pos, info


STATUS_FILE = Path(__file__).resolve().parent / "status.json"
WIN_OPEN, WIN_CLOSE = time(9, 25), time(16, 5)   # ET market-hours guard for the every-5-min scheduler


TRADES_FILE = STATUS_FILE.parent / "trades.json"


def _write_status(**kw) -> None:
    """Dump a small status.json for the dashboard Futures tab (best-effort)."""
    try:
        STATUS_FILE.write_text(json.dumps({"updated": datetime.now(ET).isoformat(timespec="seconds"),
                                           "dry_run": DRY_RUN, "qty": QTY, **kw}, indent=2, default=str))
    except Exception:
        pass


def _record_trades(ib) -> None:
    """Merge today's closing fills (realized P&L) into futures/trades.json for the dashboard history.
    Dedup by execId; ib.fills() is session-scoped so this accumulates the persistent record."""
    try:
        existing = json.loads(TRADES_FILE.read_text()).get("trades", []) if TRADES_FILE.exists() else []
        seen = {t.get("execId") for t in existing}
        new = [t for t in brk.closed_trades(ib) if t.get("execId") not in seen]
        if new:
            TRADES_FILE.write_text(json.dumps({"trades": existing + new}, indent=2, default=str))
            _log(f"recorded {len(new)} closed trade(s): "
                 + ", ".join(f"{t['side']} {t['qty']} P/L {t['pnl']:+.0f}" for t in new))
    except Exception as e:
        _log(f"trade-record skipped ({str(e)[:50]}).")


def main() -> int:
    now = datetime.now(ET)
    # ET market-hours guard: off-hours/weekend passes exit instantly (no TWS connect), so the
    # every-5-min Surface scheduler is cheap and timezone-agnostic (the bot decides in ET).
    if now.weekday() >= 5 or not (WIN_OPEN <= now.time() <= WIN_CLOSE):
        return 0
    _log(f"MES momentum pass | DRY_RUN={DRY_RUN} | qty={QTY}")
    try:
        ib = brk.connect()
    except Exception as e:
        _log(f"TWS not reachable ({str(e)[:50]}) -- skip pass."); return 0
    try:
        df = brk.fetch_intraday(ib, duration="40 D", bar_size="5 mins")  # ~28 trading days > 14d sigma
        if df is None:
            _log("no history fetched -- abort pass."); return 0
        target, info = compute_target(df, now.date(), now)
        if now.time() >= EOD:                        # ride to close: flat by 15:55
            target, info = 0, "EOD -> flat  [" + info + "]"
        target *= QTY
        cur = brk.position(ib)
        _log(f"target={target:+d}  current={cur:+d}  | {info}")
        order = None
        if DRY_RUN:
            _log("DRY_RUN -> no order placed.")
        else:
            order = brk.reconcile_to(ib, target)
            _log(f"ORDER {order}" if order else "already at target -- hold.")
        _write_status(position=(target if not DRY_RUN else cur), target=target, signal=info,
                      last_order=order, account=brk.account_snapshot(ib))
        if not DRY_RUN:
            _record_trades(ib)
    finally:
        ib.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
