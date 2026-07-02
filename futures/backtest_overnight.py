"""Extended-hours / OVERNIGHT study for MES -- is there a tradeable edge outside the 9:30-16:00 session?

The documented "overnight drift": historically ~all of the S&P's return accrued close->open, intraday
was ~flat. BUT it's decaying (NY Fed 2026 "disappearing overnight drift"; the ES 2-3am window that
drove it went flat; NightShares ETFs closed). This tests it on OUR 2yr MES 5-min data (RTH-anchored):

  intraday-only  : buy the 09:30 open, sell the 15:55 close (flat overnight) -- what our momentum bot does
  overnight-only : buy the 15:55 close, sell the next 09:30 open (flat all day) -- the overnight-drift play
  buy & hold     : hold continuously (both sessions, no daily round-trip cost)

Reports each as a daily-rebalanced strategy net of the round-turn cost, an OOS H1/H2 decay split, and a
"points earned by ET hour" profile. MEASUREMENT ONLY.

    .venv-openbb/Scripts/python.exe futures/backtest_overnight.py
"""
from __future__ import annotations

import sys
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from futures.data import COST_RT_USD, POINT_VALUE, load_mes_intraday_cache  # noqa: E402

RTH_OPEN, RTH_CLOSE = time(9, 30), time(15, 55)


def _stat(pnl: pd.Series, label: str) -> None:
    r = pnl.dropna()
    if len(r) < 20:
        print(f"  {label:<34} n<20"); return
    eq = r.cumsum()
    dd = (eq - eq.cummax()).min()
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() else float("nan")
    print(f"  {label:<34} net${r.sum():+7.0f}  perday${r.mean():+5.1f}  win{100*(r>0).mean():3.0f}%  "
          f"Sharpe~{sharpe:+.2f}  maxDD${dd:+7.0f}")


def main() -> int:
    df = load_mes_intraday_cache()
    rth = df[(df.index.time >= RTH_OPEN) & (df.index.time <= RTH_CLOSE)]
    days = sorted({t.date() for t in rth.index})
    opens = {d: rth[rth.index.date == d]["open"].iloc[0] for d in days if len(rth[rth.index.date == d])}
    closes = {d: rth[rth.index.date == d]["close"].iloc[-1] for d in days if len(rth[rth.index.date == d])}
    rows, prev = [], None
    for d in days:
        if d not in opens:
            continue
        intr = closes[d] - opens[d]
        overn = (opens[d] - closes[prev]) if (prev in closes) else np.nan
        rows.append({"date": pd.Timestamp(d), "intraday": intr, "overnight": overn})
        prev = d
    t = pd.DataFrame(rows).dropna().set_index("date")

    print(f"=== MES overnight vs intraday: {len(t)} days {t.index.min().date()} -> {t.index.max().date()} "
          f"(1 MES, net ${COST_RT_USD:.0f}/RT) ===\n")
    intr_pnl = t["intraday"] * POINT_VALUE - COST_RT_USD
    over_pnl = t["overnight"] * POINT_VALUE - COST_RT_USD
    bh_pnl = (t["intraday"] + t["overnight"]) * POINT_VALUE   # continuous hold, no daily round-trip
    _stat(intr_pnl, "intraday-only (open->close)")
    _stat(over_pnl, "overnight-only (close->open)")
    _stat(bh_pnl, "buy & hold (both, no daily cost)")

    # decay check: is the overnight edge weaker in the recent half?
    mid = t.index[len(t) // 2]
    print(f"\n  overnight-only OOS split at {mid.date()}:")
    _stat(over_pnl[t.index < mid], "  H1 overnight")
    _stat(over_pnl[t.index >= mid], "  H2 overnight (recent)")

    # points earned by ET hour (intra-bar close-open, so no cross-bar/gap contamination)
    move = (df["close"] - df["open"]) * POINT_VALUE
    by_hour = move.groupby(df.index.hour).sum()
    print("\n  $ earned by ET hour (1 MES, whole sample; shows where the drift lives):")
    for h in sorted(by_hour.index):
        bar = "#" * int(abs(by_hour[h]) / max(1, by_hour.abs().max()) * 30)
        tag = "RTH" if 9 <= h <= 15 else "o/n"
        print(f"    {h:02d}:00 [{tag}] {by_hour[h]:+7.0f}  {bar}")

    print("\nRead: if overnight-only is net-positive with Sharpe > intraday AND H2 (recent) still holds,")
    print("there's a live overnight edge worth a strategy. If H2 has decayed to ~flat/negative, the")
    print("literature's 'disappearing drift' holds on MES too -> not worth trading. Net of costs; 2yr only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
