# Second-Wave / Intraday Entry-Timing — MEASURED experiment spec

**Status:** spec only (2026-07-01). Execute alongside #3 (horizon-curve logging) / #4.
**Guiding principle:** MEASURE, don't trade. This adds *logged, scored* artifacts; it does NOT
change the live bot and involves NO discretionary/LLM entries. A trading change follows only a
PASSED pre-registered test — same discipline as the rest of the lottery experiment.

**Question we're answering:** the reviews + our data say 09:45 may "arrive after the party." Is the
hype edge better captured by entering LATER in the day, and/or do NEW names that heat up after the
open have edge (a genuine "second wave")?

---

## TIER 1 — Entry-timing curve (cheap; do first, pairs with #3)

Test whether entering the SAME picks later beats 09:45, using intraday prices — no new signals.

- Script: `experiments/lottery/entry_timing_probe.py` (or fold into the outcome scorer).
- For every logged pick, fetch same-day intraday price at: 09:45 (have), 10:30, 12:00, 14:00, 15:55.
- Compute forward return from EACH timestamp to the same-day close (and to next open, to test the
  overnight bleed direction). Append as new immutable fields (`px_1030`, `ret_1030_close`, ...);
  only fill if empty (never overwrite — matches the picks-file immutability rule).
- `analyze.py`: report same-day expectancy **by entry time** = the entry-timing curve, for the
  bot's top-3 and per-signal baskets, with the bootstrap-CI + vol-matched machinery already there.
- **Data:** intraday minute bars for the picked names only (IEX) — this is the same pull as the #3
  horizon curve, so build them together.
- **Pre-registered verdict:** shift the live entry time ONLY if a later entry beats 09:45 on
  same-day expectancy with a day-resampled bootstrap-CI edge > 0, n_days >= 30. Else 09:45 stands.

## TIER 2 — True second-wave board (heavier; ONLY if Tier 1 shows later entries help)

Test whether names that heat up AFTER the open (not in the morning board) have edge.

- A midday board run (~12:30 ET) re-discovering candidates with INTRADAY-appropriate signals:
  - **intraday RVOL** = volume so far today vs typical-by-this-time (replaces premarket `pm_rvol`).
  - **intraday move** = % from today's open (replaces overnight `gap_pct`).
  - **attention velocity** = change in WSB/StockTwits mentions since the morning snapshot (the
    reviewers' key idea) — REQUIRES persisting the morning mention counts per ticker to diff at noon.
  - `ignition` / `realized_vol` carry over (daily-bar based, stable through the day).
- Log to a SEPARATE artifact `experiments/lottery/picks_pm/<date>.json` so the morning file stays
  immutable/clean. Reuse the picks schema + a midday `combined_score`.
- Score the midday top-3 vs (a) the random basket [vol-matched], AND (b) the SAME day's morning
  top-3 — forward to close. The midday arm must beat BOTH to justify a live second-wave entry.
- Scheduling: a measurement-only VM timer (~12:30 ET) running `board.py --wave=pm`. NO trading. If
  it proves out, THEN add a second entry run (mirror the `lottery-eod` timer pattern we just shipped).
- New plumbing: attention-velocity state (store morning mention counts), intraday-RVOL fetch.

---

## OFF-LIMITS (explicitly)
- No discretionary / LLM-in-the-loop entries or position amendments. The live bot's rules are
  unchanged. This is a MEASUREMENT arm; any trading change is a separate, pre-registered step.
- Don't overwrite the immutable morning picks file; second-wave data lives in new fields / new files.

## Effort / cost
- Tier 1: MEDIUM (intraday-bar pull + scorer + analyze reporting). Build with #3.
- Tier 2: HEAVY (intraday signals + attention-velocity state + a 2nd VM timer + separate scoring).
  A real build; gated on Tier 1 justifying it.
