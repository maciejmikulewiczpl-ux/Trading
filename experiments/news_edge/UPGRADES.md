# News-edge harness upgrades (Fable review, 2026-06-09) — implementation brief

> **STATUS: IMPLEMENTED 2026-06-10.** All 5 done + tested (EDGAR live-tested 104 filings;
> pm-rvol/pm-gappers smoke-tested; newsedge.py fields/gap-backfill/control/analyze validated
> end-to-end on the 2026-06-09 day file; scan prompt rewritten). Effective for the next
> morning scan. Verdict still needs ~25-30 trading days of the NEW data (control basket +
> gap/rvol fields) to be interpretable.

Five small changes from the expert review of the news-edge experiment. None disturb the
immutable-log protocol; old pick files stay valid (new fields are additive/optional).
Verdict still needs 25-30 trading days — these make that verdict *interpretable*.

**The core critique they address:** the current candidate net (StockTwits trending +
premarket movers) conditions on the MOVE, not the NEWS — it buys names after the crowd
piled in (day 1: INTC +10% premarket → −4.5% from 9:45; gap-and-fade). And without a
mechanical control, a positive verdict can't distinguish "Claude's judgment adds value"
from "gapper-list membership." Day 1 also had 3 chip names = ONE bet (clustering).

## 1. EDGAR real-time filings connector — the big one
`sources.py` gets an `edgar` subcommand. SEC EDGAR is free, official, real-time,
machine-readable, no key. Primary-source catalysts BEFORE the aggregators rewrite them
(day 1's best (−) pick, RDW −8.2%, was a $500M ATM offering = an SEC filing).
- Use the full-text search API: `https://efts.sec.gov/LATEST/search-index?q=...&dateRange=custom`
  (JSON; set a proper User-Agent per SEC policy, e.g. "news-edge research <email>"), and/or
  the Atom feed `https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=8-K&...&output=atom`.
- `sources.py edgar <hours_back>` → recent 8-K / S-1 / 424B5 / Form 4 filings mapped to
  tickers (CIK→ticker via the free `company_tickers.json`), printed as
  `{ticker: [{form, time, title}]}`. Filter to liquid names (reuse a price/volume floor).
- Scan prompt step 1 adds: "EDGAR filings (fresh 8-K/dilution/insider): `sources.py edgar 16`" —
  candidates from filings are the "fresh, not yet priced" category the movers net misses.

## 2. Three logged fields per pick (newsedge.py `log` + scan prompt)
Additive schema; `cmd_log` passes them through if present, omits if not:
- `gap_pct`        : premarket gap % vs prior close (analyst supplies; or compute in
                     `outcomes` retroactively from daily bars — cleaner, zero scan effort).
- `premarket_rvol` : today's premarket $-volume / 20-day avg premarket $-volume (Alpaca
                     minute bars 04:00-09:30 ET; add `sources.py pm-rvol T1,T2,...`).
                     "Story + abnormal participation" is the filter; story w/o volume is noise.
- `earnings_day`   : bool — name reports today/tonight (Alpha Vantage EARNINGS_CALENDAR CSV,
                     key already in .env; cache the daily CSV). Earnings gaps behave differently;
                     analyze() must be able to split on this.
Optional 4th: `theme` (short string, e.g. "chips-rebound") so clustered picks are countable
as one bet.

## 3. Mechanical control basket — the missing experiment arm
Each scan day, ALSO log every liquid premarket gapper >3% (mechanical, no judgment,
signal=0, `sources:["control"]`, flagged `control: true`) — either appended to the same
day file under `"controls": [...]` or signal-tagged; scorer scores them identically.
`analyze` then reports Claude's (+) picks vs the control basket: **Claude earns his keep
only if (+) beats the mechanical screen he draws from**, not just the (−) bucket.
Implementation: `sources.py pm-gappers 3.0` (Alpaca snapshot/bars premarket change), and
the scan prompt logs them in the same JSON.

## 4. Scan-prompt edits (news_scan_prompt.md)
- Add EDGAR + pm-rvol + pm-gappers steps (above).
- REQUIRE a named, dated catalyst for any signal != 0; "momentum, catalyst unconfirmed"
  picks (day-1 TXRH/WWD) are capped at confidence <= 0.3 or logged signal 0.
- Require `theme` per pick; discourage >2 picks on one theme (one bet, not three).
- StockTwits reframed: log it as participation/crowding gauge; the head-to-head decides
  if it's contrarian.

## 5. analyze() upgrades (newsedge.py)
- Split every source's separation by: confidence bucket (>=0.6 vs <0.6), earnings_day,
  gap bucket (|gap| <3% vs >=3%), and claude-vs-control (the #3 comparison).
- Optional: theme-collapsed view (average within a day-theme first, then across) so the
  n isn't inflated by clusters.
- Keep output compact — it goes in the 1:05pm ntfy push.

Prediction to test (write it down now, judge in 30 days): the edge, if any, lives in
SMALL-gap fresh-news names (EDGAR/late-breaking), not the big movers; big-gap (+) picks
will show gap-fade. If (+)-vs-control separation is ~0, the LLM read adds nothing over a
mechanical gapper screen and the experiment's answer is "automate the screen, drop the analyst."
