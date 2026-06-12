# Lottery Scanner — can ANY metric predict the irrational daily winners?

**Pre-registration (immutable, written 2026-06-12 before any data was collected).**

All shipped engines are backtest-validated mechanical edges, but the market's biggest
daily winners are often attention-driven lottery tickets (squeezes, rumors, meme
momentum) that mechanical backtests on liquid universes never see. This experiment
tests every *other* measurable hype angle — falsifiably: pre-registered "winner"
definitions, six mechanical signals logged every morning, hit-rate lift vs a **random-
basket luck baseline**, verdict at 30 trading days. Paper money. Full small/micro-cap
universe; a day-1 paper bot runs on the repurposed dual-momentum paper account.

Reuses the proven news-edge pipeline (picks JSON -> outcomes scorer -> analyze ->
status tab -> ntfy). No LLM in the daily loop — plain Python, zero quota.

## Pre-registration (immutable)

- **Winners:** W1 = same-day `ret_945_close >= +5%`; W2 = `ret_1d >= +10%`;
  W3 = `ret_3d >= +20%`.
- **Signals (6):** WSB mention surge (apewisdom, exists), StockTwits trending rank
  (exists), premarket RVOL+gap (exists), short-squeeze score (short%float ×
  days-to-cover, yfinance weekly cache), unusual-options call-volume z-score (Alpaca
  chain snapshots, 20d trailing state), price/volume ignition score (green streak +
  vol accel + 52w-high proximity + prior-day winner).
- **Baselines:** seeded random basket (n=10 from liquid universe) + existing >3%
  gapper control.
- **Success bar:** a signal "works" iff W1 hit-rate >= 2× random base rate, p < 0.05
  (binomial + day-resampled bootstrap), n >= 30 days. `combined_score` = mean
  percentile rank across non-null signals (fixed, never tuned).

## Tracks

- **Track A (backtest):** `backtest/lottery_ignition.py` — 10y classification-lift
  study on `.swing_daily_cache.pkl` (no fetching): does the ignition score predict
  next-day cross-sectional top-decile gainers? Liquid-universe bound documented in
  its docstring. `backtest/lottery_uoa.py` (week 2, deferred) — call-volume z>=2 on
  day T vs return T+1 on a pilot of hype-prone names.
- **Track B (daily hype board, forward test, no LLM):** this package. `board.py`
  logs the six signals + baskets to `picks/<date>.json` (immutable) every morning;
  `outcomes.py` fills `ret_945_close` after the close and `ret_1d/ret_3d` on later
  runs; `analyze.py` computes per-signal hit-rate lift vs the random base rate and
  prints the pre-registered verdict.
- **Track C (bold paper bot, day 1):** repurposes the dual-momentum paper account
  (dual-mom edge retired 2026-06-12). `scripts/run_lottery_bot.py` buys the top-3
  `combined_score` hype picks at 09:45 ET ($500 notional each, 10% native trailing
  stop, T+3 time-stop). Bot PnL is *color*, not the verdict — the verdict is Track
  B's hit-rate stats.

## Schema (`picks/<date>.json`)

```
{
  "date": "YYYY-MM-DD",
  "logged_at": "...ET ISO...",
  "picks": [
    {
      "symbol": "GME",
      "basket": "wsb" | "stocktwits" | "gappers" | "random" | "control",
      "top_k_of": ["wsb", "ignition", ...],   # which signals top-K'd this name
      "signals": {                            # raw per-signal values (null if offline)
        "wsb_surge": 4.2, "wsb_rank": 1,
        "st_rank": 3,
        "pm_rvol": 5.1, "gap_pct": 8.3,
        "squeeze": null, "uoa_z": null,
        "ignition": 3
      },
      "combined_score": 0.71,                 # mean percentile rank across non-null signals
      "ret_945_close": null, "ret_1d": null, "ret_3d": null
    }
  ]
}
```

## Signals come online over days 1–4

Signals 1–3 (WSB, StockTwits, premarket RVOL/gap) + 6 (ignition) are live day 1.
`squeeze_scores()` returns None until `.short_interest_cache.json` exists (filled by
the Sunday `update_short_interest.py` run). `uoa_snapshot()` returns None until
`.uoa_state.json` has accumulated ~20 sessions of trailing call-volume history. The
board logs fine with the available signals; `combined_score` averages only non-null
ones.

## Risks (documented)

- yfinance short interest is 2–4 weeks stale (a structural feature, not a morning
  signal). · apewisdom has no SLA + serves junk tickers (filtered via the Alpaca
  tradable-asset list). · StockTwits throttling (degrade gracefully, `_error`
  pattern). · IEX thin minute bars on small caps — the 1d/3d horizons are the robust
  fallback for those names. · UOA z needs ~20 sessions of state before it goes live.

## Verification milestones

- 30 trading days: `analyze.py` prints the pre-registered verdict per signal.
  Anything failing the 2×-lift bar is closed; anything passing graduates to a proper
  sizing discussion.
