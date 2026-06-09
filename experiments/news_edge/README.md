# News-edge forward test (experiment — NOT part of the live bot)

An un-backtestable idea, evaluated the only honest way: **forward-test it.** Near the open,
during the 9:30–9:45 ET opening-range window, the analyst (Claude, manually, on mornings we
run it) scans news/sentiment for the day's movers and most-active names and logs a short
ranked list of **frontrunners (+)** and **avoid (−)**. End of day, a script records what those
names actually did (9:45→close). Over weeks we ask one question: **do the (+) picks actually
outrun the (−) picks?** If yes, it's real; if not, we drop it — same bar as every other idea.

This lives in `experiments/` and writes only to `experiments/news_edge/picks/`. It does **not**
touch the validated ORB pipeline (`backtest/`, `live/`) or place any orders. It is a logging +
evaluation harness for a manual prototype, not an automated strategy.

## Protocol
1. **~09:30–09:45 ET (live):** analyst scans (Alpaca news + market-movers + web tone) and
   produces a picks JSON: per symbol `{symbol, signal: +1|0|-1, confidence: 0-1, reason, sources}`.
2. **Log it:** `python experiments/news_edge/newsedge.py log <picks.json>` →
   saves `picks/YYYY-MM-DD.json` (timestamped, immutable record — no look-ahead).
3. **After close (or next day):** `python experiments/news_edge/newsedge.py outcomes YYYY-MM-DD`
   → pulls each name's 09:45→15:55 return and appends it to that day's file.
4. **Weekly:** `python experiments/news_edge/newsedge.py analyze` → avg return by signal bucket,
   (+) vs (−) separation, hit rate, n. The verdict accumulates with the sample.

## Honest caveats
- Naive news-sentiment edges are often arbitraged away; this is a test, not an assumption.
- The signal is the analyst's discretionary read — noisy, and only as good as the sources + judgment.
- A meaningful verdict needs a real sample (dozens of days). Early results are noise.
- For the always-on VM this would need an *automated* sentiment source (not the analyst). The
  manual prototype only tells us whether the edge is worth automating.
