"""MES paper bot -- the live counterpart of futures/backtest_orb.py (Phase 3 skeleton).

Runs the tight-OR-day ORB on MES against the IBKR PAPER account:
  1. At the RTH open, build the 09:30-09:45 ET opening range from live 5-min bars.
  2. Apply the tight-OR-day gate (OR range vs the trailing OR% distribution -> only trade tight days).
  3. On the first breakout close beyond the OR, enter market + attach a native trailing stop
     (TRAIL_R * OR_range) via broker_ibkr.market_with_trailing.
  4. Flat by 15:55 ET (same-day; no overnight). Log entry/exit + slippage-vs-arrival to the ledger.

State in futures/state.json (one trade/day/side). Idempotent restarts. This is a SKELETON: the
decision logic mirrors the backtest; it is guarded so it will not place orders until (a) IBKR paper
Gateway is live and (b) DRY_RUN is turned off. Nothing here runs without the account.

    .venv-openbb/Scripts/python.exe futures/run_mes_bot.py           # DRY-RUN (no orders)
    DRY_RUN=0 .venv-openbb/Scripts/python.exe futures/run_mes_bot.py # arm (paper account only)
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
from futures.backtest_orb import OR_END, RTH_OPEN, TRAIL_R  # reuse the SAME setup constants  # noqa: E402
from futures.data import POINT_VALUE  # noqa: E402

ET = ZoneInfo("America/New_York")
STATE = Path(__file__).resolve().parent / "state.json"
EOD = time(15, 55)


def _load_dotenv(name: str = ".env.futures") -> None:
    """Minimal KEY=VALUE loader (matches how the other bots load their own env). No-op if absent."""
    f = Path(__file__).resolve().parents[1] / name
    if not f.exists():
        return
    for line in f.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            v = v.split("#", 1)[0].strip()          # drop inline comments
            os.environ.setdefault(k.strip(), v)


_load_dotenv()
OR_TIGHT_PCTL = float(os.environ.get("OR_TIGHT_PCTL", "0.5"))
QTY = int(os.environ.get("MES_QTY", "1"))
DRY_RUN = os.environ.get("DRY_RUN", "1") != "0"


def _load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {}


def _save_state(s: dict) -> None:
    STATE.write_text(json.dumps(s, indent=2, default=str))


def _log(msg: str) -> None:
    print(f"[{datetime.now(ET):%Y-%m-%d %H:%M:%S}] {msg}")


def decide_and_trade(ib) -> None:
    """One evaluation pass. Pulls today's 5-min bars, computes the OR + tight gate, and if a breakout
    has printed, enters (unless already traded today / DRY_RUN). Designed to be called on a schedule
    between ~09:45 and 15:55 ET."""
    today = datetime.now(ET).date()
    state = _load_state()
    if state.get("date") == str(today) and state.get("entered"):
        _log("already traded today; nothing to do."); return

    # recent intraday incl. today, from IBKR (fallback to the free yfinance loader for dry-run dev)
    try:
        df = brk.fetch_intraday(ib, duration="10 D", bar_size="5 mins") if ib else None
    except Exception as e:
        _log(f"history fetch failed ({str(e)[:60]}); aborting pass."); return
    if df is None:
        from futures.data import load_mes_intraday
        df = load_mes_intraday("5m", "10d")

    day = df[df.index.date == today]
    rth = day[(day.index.time >= RTH_OPEN) & (day.index.time <= EOD)]
    ob = rth[rth.index.time < OR_END]
    if len(ob) < 2:
        _log("opening range not complete yet; wait."); return
    or_hi, or_lo = ob["high"].max(), ob["low"].min()
    or_range = or_hi - or_lo
    price0 = rth["close"].iloc[0]

    # tight-OR-day gate vs trailing OR% distribution (prior days in the pulled window)
    import numpy as np
    prior = sorted({ts.date() for ts in df.index if ts.date() < today})
    hist = []
    for d in prior:
        dd = df[(df.index.date == d)]
        dob = dd[(dd.index.time >= RTH_OPEN) & (dd.index.time < OR_END)]
        drth = dd[(dd.index.time >= RTH_OPEN) & (dd.index.time <= EOD)]
        if len(dob) and len(drth) and drth["close"].iloc[0] > 0:
            hist.append((dob["high"].max() - dob["low"].min()) / drth["close"].iloc[0])
    if len(hist) >= 10:
        if (or_range / price0) > float(np.quantile(hist, OR_TIGHT_PCTL)):
            _log(f"OR not tight (OR%={or_range/price0:.4f}); stand down today."); return
    else:
        _log(f"insufficient OR history ({len(hist)}); stand down."); return

    # breakout check on the latest closed bar
    post = rth[rth.index.time >= OR_END]
    if post.empty:
        _log("no post-OR bars yet."); return
    last = post["close"].iloc[-1]
    side = "BUY" if last > or_hi else ("SELL" if last < or_lo else None)
    if side is None:
        _log(f"no breakout yet (last {last:.2f} within OR [{or_lo:.2f},{or_hi:.2f}])."); return

    trail_pts = TRAIL_R * or_range
    _log(f"SIGNAL {side} MESx{QTY} | OR[{or_lo:.2f},{or_hi:.2f}] range {or_range:.2f} "
         f"| trail {trail_pts:.2f}pts (~${trail_pts*POINT_VALUE:.0f})")
    if DRY_RUN or ib is None:
        _log("DRY_RUN -> no order placed."); return
    parent, stop = brk.market_with_trailing(ib, side, QTY, trail_pts)
    state.update({"date": str(today), "entered": True, "side": side, "or_hi": or_hi, "or_lo": or_lo,
                  "trail_pts": trail_pts, "entry_order": parent.order.orderId})
    _save_state(state)
    _log(f"ENTERED {side} (order {parent.order.orderId}); trailing stop attached.")


def main() -> int:
    now = datetime.now(ET)
    _log(f"MES bot pass | DRY_RUN={DRY_RUN} | qty={QTY} | tightPctl={OR_TIGHT_PCTL}")
    if now.time() >= EOD:
        # end-of-day: flatten anything still open (same-day strategy)
        if not DRY_RUN:
            ib = brk.connect()
            try:
                if brk.position(ib) != 0:
                    brk.flatten(ib); _log("EOD flatten sent.")
            finally:
                ib.disconnect()
        else:
            _log("EOD (DRY_RUN): would flatten any open position.")
        return 0
    ib = None
    if not DRY_RUN:
        ib = brk.connect()
    try:
        decide_and_trade(ib)
    finally:
        if ib is not None:
            ib.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
