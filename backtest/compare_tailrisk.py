"""Tail / correlation risk (Fable #5): what does the WORST day look like, and does the
daily-loss-cap actually protect?

16 concurrent tight-OR longs are NOT 16 independent bets — they're one leveraged beta
bet expressed many ways. The $500 daily-loss-cap only halts NEW entries; positions
already open keep riding. This dig quantifies the left tail on the live (HAND) tight-OR
trailing config, cap-aware real $, 1.0x slip:

  1. Daily PnL distribution + worst days (with # positions / # stopped that day).
  2. PEAK simultaneous OPEN RISK ($) across concurrent positions vs the ~$800 nominal
     budget — does the real exposure cluster, and how big can a single bad moment get?
  3. CORRELATION proxy: on the worst days, what fraction of that day's positions lost?
  4. DAILY-LOSS-CAP efficacy: event-sim the cap (realized PnL accrues as trades CLOSE;
     once <= -$500, block new entries; open ones ride). Capped vs uncapped worst days —
     does it actually cut the tail, or do the already-open positions blow through it?

Run:
    .venv/Scripts/python.exe backtest/compare_tailrisk.py
"""
from __future__ import annotations

import math
import pickle
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.universe_portfolio import portfolio  # noqa: E402
from backtest.compare_selection import _tday  # noqa: E402
from backtest.compare_norefill_trend import trend_eligibility, apply_filter  # noqa: E402
from backtest.compare_volpause import prior_vol_flags, RISK, CAP  # noqa: E402
from backtest.compare_or_range_realcost import or_pct, risk_ps  # noqa: E402
from backtest.universe_scan import UNIVERSE  # noqa: E402

import pandas as pd  # noqa: E402

WINDOW = 730
OR_THR = 0.5
TARGET_MEDIAN_R = 0.042
NOTIONAL_CAP = 10_000.0
LOSS_CAP = 500.0


def main() -> int:
    blob = pickle.load(open(ROOT / "backtest" / f".pit_trailtrades_{WINDOW}d.pkl", "rb"))
    daily = pickle.load(open(ROOT / "backtest" / f".pit_daily_{WINDOW}d.pkl", "rb"))
    closes_all = daily["close"]

    all_tr = [t for syms in blob.values() for t in syms if t.symbol in set(UNIVERSE)]
    days = sorted({_tday(t) for t in all_tr})
    present = sorted({t.symbol for t in all_tr})
    closes = closes_all[[c for c in sorted(set(present) | {"SPY"}) if c in closes_all.columns]]
    elig = trend_eligibility(closes, present, days)
    prior = prior_vol_flags(closes, days)
    mult = {d: (0.5 if prior.get(d) else 1.0) for d in days}

    hand = [t for t in apply_filter(all_tr, elig) if or_pct(t) <= OR_THR]
    taken = portfolio(hand, CAP)
    cents = TARGET_MEDIAN_R * statistics.median(risk_ps(t) for t in hand) / 2.0

    # per-trade cap-aware shares, $ risk, $ pnl
    rec = []
    for t in taken:
        rps = risk_ps(t)
        sh = min(math.floor(RISK * mult.get(_tday(t), 1.0) / rps), math.floor(NOTIONAL_CAP / t.entry_price))
        if sh <= 0:
            continue
        rec.append({"day": _tday(t), "entry": t.entry_time, "exit": t.exit_time,
                    "risk": sh * rps, "pnl": (t.exit_price - t.entry_price) * sh - 2 * cents * sh,
                    "stopped": t.exit_price <= t.stop_price + 1e-9})

    df = pd.DataFrame(rec)
    daily_pnl = df.groupby("day")["pnl"].sum().sort_values()
    n_pos = df.groupby("day").size()
    n_stop = df.groupby("day")["stopped"].sum()

    print(f"\n{'='*72}\nTAIL / CORRELATION RISK — live HAND tight-OR trailing, cap-aware (1.0x slip)")
    print(f"{len(df)} trades over {df['day'].nunique()} active days")
    print(f"{'='*72}")
    s = daily_pnl
    print(f"\n  Daily PnL: mean +${s.mean():.0f}  std ${s.std():.0f}  "
          f"min ${s.min():+,.0f}  p5 ${s.quantile(.05):+,.0f}  p95 ${s.quantile(.95):+,.0f}  "
          f"%down {100*(s<0).mean():.0f}%")
    print(f"\n  WORST 6 days (date | PnL | #pos | #stopped | % of day's pos that lost):")
    for d in s.head(6).index:
        day_tr = df[df["day"] == d]
        losers = (day_tr["pnl"] < 0).mean()
        print(f"    {d}  ${s[d]:+,.0f}   {int(n_pos[d]):>2} pos   {int(n_stop[d]):>2} stop   {100*losers:.0f}% lost")

    # peak simultaneous open RISK per day (sweep-line over entry/exit), then the worst across all days
    peak_risk_by_day = {}
    for d, g in df.groupby("day"):
        evs = []
        for _, r in g.iterrows():
            evs.append((r["entry"], r["risk"]))
            evs.append((r["exit"], -r["risk"]))
        evs.sort(key=lambda x: x[0])
        cur = peak = 0.0
        for _, dr in evs:
            cur += dr
            peak = max(peak, cur)
        peak_risk_by_day[d] = peak
    pk = pd.Series(peak_risk_by_day)
    print(f"\n  Peak simultaneous OPEN RISK ($ at risk at once): median ${pk.median():,.0f}  "
          f"max ${pk.max():,.0f}  (nominal budget {CAP}x${RISK:.0f}=${CAP*RISK:,.0f})")
    print(f"  -> if EVERY position open at that peak stopped at once, worst-moment loss ~ ${pk.max():,.0f}")

    # daily-loss-cap event sim: realized accrues as trades CLOSE; once <= -cap, block new entries
    saved = 0.0
    trig_days = 0
    capped_min = None
    capped_series = {}
    for d, g in df.groupby("day"):
        evs = []
        for i, r in g.iterrows():
            evs.append((r["entry"], "in", i, r))
            evs.append((r["exit"], "out", i, r))
        evs.sort(key=lambda x: (x[0], 0 if x[1] == "out" else 1))  # process exits before entries at a tie
        realized = 0.0
        halted = False
        taken_ids, day_pnl = set(), 0.0
        triggered = False
        for _, kind, i, r in evs:
            if kind == "in":
                if not halted:
                    taken_ids.add(i)
            else:
                if i in taken_ids:
                    realized += r["pnl"]
                    if realized <= -LOSS_CAP and not halted:
                        halted = True
                        triggered = True
        day_pnl = sum(df.loc[i, "pnl"] for i in taken_ids)
        capped_series[d] = day_pnl
        if triggered:
            trig_days += 1
            saved += (daily_pnl[d] - day_pnl)  # uncapped - capped (negative if cap hurt by blocking winners)
    cs = pd.Series(capped_series)
    print(f"\n  DAILY-LOSS-CAP (${LOSS_CAP:.0f}) efficacy:")
    print(f"    triggers on {trig_days} of {len(s)} days; net $ effect vs uncapped: ${saved:+,.0f} "
          f"(positive = saved losses, negative = blocked winners)")
    print(f"    worst day  uncapped ${s.min():+,.0f}  ->  capped ${cs.min():+,.0f}")
    print(f"    total PnL  uncapped ${s.sum():+,.0f}  ->  capped ${cs.sum():+,.0f}")
    print("\nRead: if peak open-risk >> the loss-cap, a correlated gap-through can lose multiples of")
    print("$500 before the cap (which only blocks NEW entries) bites. If 'worst day capped' ~ 'uncapped',")
    print("the cap doesn't protect the tail -> the real guardrail is per-moment open-risk, not a realized cap.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
