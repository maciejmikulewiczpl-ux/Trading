# Engine #3 — Swing-Horizon Breakout: SPEC + TEST PLAN (pre-registered)

**Status: REJECTED 2026-06-12 after audit (verify_swing.py).** Initial run reported
ALL GATES PASS (Sharpe 3.03), but review found (A) the daily-PnL series counted each
trade's PnL ~5.65x (MTM added daily AND full realized PnL added again at exit) — the
corrected Sharpe is 0.57; and (B) the candidate universe was the top-500 by 2024-26
dollar volume, i.e. future winners — the survivorship-clean 2024-07+ window shows
$3.00/trade (vs $8.20 contaminated) and Sharpe 0.45. Corrected G1a FAIL (0.57 < 1.0),
G2a FAIL (2022 = -$629 < -$500). Cell closed per pre-registered discipline. Design locked before any data was
looked at. The executing session implements EXACTLY this; if results are bad, the
verdict is FAIL — no post-hoc tuning, no new variants invented mid-run. (Tuning
discipline = same as tight-OR / meanrev / options precedents.)

## 0. Motivation (one paragraph)

Portfolio currently spans intraday (ORB, validated+live) and monthly rotation
(dual-mom, live). The 2–15 day swing cell between them is empty. Time-series
momentum / breakout continuation at multi-day horizon is the best-documented
retail-accessible effect not yet tested here. Our own tight-OR finding (intraday
compression → expansion) gives a project-native hypothesis at daily scale:
**range compression precedes tradeable breakouts**. This engine must EARN its slot:
it holds overnight (gap risk ORB deliberately avoids), so gates are strict and
include crisis behavior + decorrelation.

## 1. Strategy definition (all variants long-only, stocks)

Signals computed on DAILY bars at close of day T; entries at OPEN of T+1.

- **V0 baseline (Donchian-style):** enter when close(T) > max(high) of the prior
  55 sessions (T-55..T-1). Initial stop = entry − 2.5 × ATR(14). Trailing
  Chandelier stop: max(close since entry) − 2.5 × ATR(14, updated daily); stop
  only ratchets up. Exit when stop is hit (see §3 fill rules) — no profit target
  (let winners run; mirrors validated trailing-exit finding).
- **V1 compression (primary hypothesis):** V0 + precondition at T:
  10-day range pct = (max(high,10d) − min(low,10d)) / close(T), and this value is
  ≤ its own 33rd percentile over the trailing 252 sessions (per symbol).
  This is the daily analog of tight-OR. Pre-registered expectation: V1 ≥ V0 on
  Sharpe with fewer trades.
- **V2 horizon check:** V0 with 20-day-high entry (same exits). Not a candidate
  to ship by itself unless it dominates; it exists to detect horizon sensitivity.

**Regime filter (all variants):** new entries only when SPY close > SPY 200-day
SMA at T. Open positions are NOT force-closed on regime flip (stops handle exits).

**Risk & portfolio (mirror live conventions):**
- Equity $100k notional context. Risk per trade $50 (= entry−stop distance × shares).
- Shares = floor(50 / (entry − initial_stop)); skip if entry−stop < $0.05.
- Max concurrent positions 12; max total open risk $600; per-position notional
  cap $10k. One position per symbol at a time. If more signals than slots on a
  day: take largest (dollar-volume) first — mechanical, no ranking cleverness.
- Re-entry after stop-out allowed only on a NEW signal (close > 55d high again).

## 2. Data

- **Source:** yfinance daily OHLCV (auto_adjust=True everywhere, signals AND fills
  — consistent adjusted series), `.venv-openbb` has yfinance. Alpaca NOT needed.
- **Window:** 2016-01-01 → run date (~10y). This is the point of daily bars:
  includes 2018 vol shock, 2020 crash, 2022 bear, 2024-26.
- **Universe — SURVIVORSHIP WARNING (hard-won lesson, see pit_survivorship_finding):**
  do NOT hand the 2026 watchlist to a 2016 backtest and call it a day.
  Required approach: reuse the PIT machinery (`backtest/pit_universe.py`,
  `pit_snapshot.py`, `pit_expand.py`) — mechanical high-realized-vol selection,
  refreshed yearly in the sim, exactly as pit_survivorship_finding validated
  (volatility is the selector that reproduces the curated edge without hindsight).
  Universe size ~100-150 names per year-slice, min dollar-volume floor as in
  pit_universe. Delisted names yfinance can't serve: log how many drop and report
  the count in the verdict (if >15% of PIT names are unfetchable, flag it —
  partial survivorship remains and the verdict must say so).
- SPY daily for regime + benchmark.
- Cache everything to `backtest/.swing_daily_cache.pkl` (one fetch, then iterate).

## 3. Simulation rules (no-fantasy-fills — the options G3 lesson)

- Entry: open(T+1). If open(T+1) already < initial stop → skip the entry (gap
  collapsed the setup).
- Stop checking, daily, for each open position:
  - if open ≤ stop → exit at OPEN (gap-through fills at the gap, never at stop)
  - elif low ≤ stop → exit at stop price
- Costs: 0.10% of notional round-trip (≈5 bps/side — conservative for our liquid
  names) charged per trade. No financing/borrow (long-only cash).
- No lookahead anywhere: ATR/percentile/SMA computed strictly from data ≤ T.

## 4. Outputs (per variant)

Per-trade list (entry/exit dates+prices, R multiple, hold days) + summary:
trades, win%, avg R, PnL$, Sharpe (daily PnL), max drawdown $, avg/median hold
days, exposure (avg open positions), PnL by year, PnL in named windows:
2020-02→2020-04, 2022 full year, 2025-04 (tariff spike). Benchmarks: SPY
buy-hold same window; report side-by-side.

## 5. GATES (pre-registered — ALL must pass to proceed to paper)

- **G0 validity:** ≥150 closed trades full-window for the candidate variant;
  unfetchable-PIT-name dropout ≤15%.
- **G1 economics:** net-of-cost full-window Sharpe ≥ 1.0 AND positive net PnL in
  BOTH window halves (2016-2020 / 2021-2026).
- **G2 crisis:** 2022 full-year net PnL ≥ −$500 (one bad month of the target run
  rate) AND 2020-02→04 net PnL ≥ −$1000. The regime filter + stops must contain
  bears, or an overnight engine is unshippable.
- **G3 decorrelation:** daily-PnL correlation vs the ORB backtest over the
  overlapping 730d ≤ 0.30. (Regenerate ORB daily PnL from the cached trades via
  the `three()`/portfolio machinery in `compare_volpause.py` /
  `compare_or_range_filter.py`.) If it just shadows ORB, it adds risk, not diversification.
- **G4 character:** median hold ≥ 3 days (else it's a worse ORB) and avg
  winner/loser ratio ≥ 1.8 (trend engines must have right-skew; a 50/50 grinder
  at this horizon is noise).
- **Robustness (after gates, only if V-candidate passes):** fixed grid sweep —
  entry {20, 55}-day high × stop {2.0, 2.5, 3.0}×ATR × compression {on, off}.
  PASS requires a plateau: ≥⅔ of neighboring cells within 30% of the chosen
  cell's Sharpe. A knife-edge = overfit = FAIL regardless of headline numbers.

**Ship path if ALL pass:** separate systemd paper runner on the VM (pattern:
`live/run_dualmom.py` + `scripts/launch_dualmom.sh`), risk budget capped at 25%
of total, ≥4 weeks live paper before any real-money discussion. Memory + MEMORY.md
updated per memory_index_hygiene either way (PASS or FAIL).

## 6. Implementation plan (file by file)

1. `backtest/fetch_swing_data.py` — PIT yearly universes + yfinance daily fetch
   2016→now + SPY → `.swing_daily_cache.pkl`. Run with
   `.venv-openbb\Scripts\python.exe`. Print dropout count.
2. `backtest/run_swing.py` — the event-driven daily simulator (§1+§3), variant
   selected by flag. Pure pandas/numpy, runs in either venv.
3. `backtest/compare_swing_variants.py` — V0/V1/V2 + gates table (§4+§5),
   halves, named windows, G3 correlation vs ORB.
4. `backtest/swing_robustness.py` — the fixed grid of §5, plateau verdict.
5. Verdict block printed by 3+4; executor writes memory file + MEMORY.md line.

## 7. Environment gotchas for the executing session (read before coding)

- Windows console is cp1252: NO unicode (≤ → ✓) in `print()` strings.
- Two venvs: yfinance/openbb in `.venv-openbb`, alpaca-py in `.venv`. Plain
  `python` is NOT on PATH — always use the venv exe explicitly.
- `load_env()` pattern lives in `backtest/run_orb.py` (only needed if touching
  Alpaca for the G3 ORB series).
- Minute-bar caches: timestamp `.asi8` is MICROSECONDS (pandas 2.x), not ns.
- Don't start heavy/long runs in the ~5h before 6:33am PDT (cron quota guard —
  see news_edge_experiment memory). Overnight pattern: `backtest/overnight.sh`
  + results to `OVERNIGHT_RESULTS*.md` if runs are long.
- Commit scripts + verdict after the run only if the user asks (repo convention:
  findings get committed, but ask first).
- If a gate result is MARGINAL (e.g., Sharpe 0.95, or one half barely negative):
  do not argue it across the line and do not iterate — record exact numbers and
  stop; the user will escalate the judgment call to a stronger model/session.
