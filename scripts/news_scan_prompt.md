You are running HEADLESS (no human present) as the automated morning news-edge scan, ~9:33 ET, inside the ORB opening-range window. There is no chat to reply to — just DO the work, then print a short summary. Use your Bash tool for python/git commands; use forward-slash paths (they work in git-bash on this Windows machine). Do NOT launch any trading bot locally — the VM news-orb (09:40 ET) pulls the picks you push and trades them.

GOAL: produce ~10-20 LIQUID frontrunner/avoid stock picks for today AND a mechanical control basket, then push them.

WHY THE CONTROL BASKET + FIELDS (read once): the experiment must prove the ANALYST adds value over a dumb screen. So we also log every >3% premarket gapper mechanically (signal 0, control), and tag each pick with gap%, premarket RVOL, earnings-day, and a theme. The verdict compares YOUR (+) picks vs the control basket — not just (+) vs (-). Favor FRESH catalysts (filings/news) over names that already ran; the pre-registered bet is the edge lives in smaller-gap fresh-news names, while big premarket gappers tend to fade.

STEPS (in order):
1) CANDIDATE NET (cast wide, liquid only — skip sub-$5 / penny pumps):
   - SEC EDGAR fresh filings (primary-source catalysts BEFORE aggregators rewrite them — 8-K material events, 424B5/S-1 = dilution/offering): `.venv/Scripts/python.exe experiments/news_edge/sources.py edgar 16`
   - StockTwits trending (free, real-time crowd — a CROWDING gauge, not direction): `.venv/Scripts/python.exe experiments/news_edge/sources.py st-trending 30`
   - Reddit/WSB mention SURGE (same rule — crowding, not direction; a big overnight mention spike means "something happened, find the catalyst", never a signal by itself): `.venv/Scripts/python.exe experiments/news_edge/sources.py reddit 25`
   - Market tone + single-stock catalysts: WebSearch (include Yahoo Finance + CNBC) for today's premarket movers / earnings / upgrades / FDA / deals.
   - If mcp__alpaca__ tools are available, also use get_market_movers, get_most_active_stocks, and get_news for fresh Benzinga catalysts. If NOT, rely on EDGAR + StockTwits + WebSearch — fine.
2) MY CALL per candidate: signal +1 (frontrunner, clear bullish catalyst) / -1 (avoid, clear bearish) / 0 (unclear), each with a short reason + confidence 0-1 + sources + a `theme` (short tag, e.g. "ai-chips", "fda", "dilution"). RULES:
   - A nonzero signal REQUIRES a named, dated catalyst (deal/upgrade/guidance/FDA/filing/earnings). "Momentum / catalyst unconfirmed" names get signal 0 OR confidence <= 0.3 — never a confident call on price action alone.
   - Earnings-day names: allowed but set `earnings_day: true` (gappy; behave differently).
   - Don't pile >2 picks on one theme (3 chip names = ONE bet, not three) — pick the best 1-2 per theme.
   - Liquid only. Aim ~10-20 across DIFFERENT themes.
3) PER-PICK SIGNALS + CONTEXT (for ALL candidate tickers, comma-separated):
   - `.venv/Scripts/python.exe experiments/news_edge/sources.py st-sentiment <TICKERS>`  (StockTwits crowd — main scored source; contrarian/crowding read)
   - `.venv/Scripts/python.exe experiments/news_edge/sources.py av <TICKERS>`  (Alpha Vantage — often empty on fresh news; include ONLY if n>0)
   - `.venv/Scripts/python.exe experiments/news_edge/sources.py pm-rvol <TICKERS>`  (premarket relative volume — abnormal participation is what makes a story tradeable)
   Each pick: source_signals {"stocktwits": sig, "alphavantage": sig if n>0}, and set `premarket_rvol` to the pm-rvol value (story + volume > story alone). If a pick appears in the reddit list, set `reddit_rank` to its rank (crowding context for later analysis — do NOT put reddit in source_signals; it has no direction). Viral X/Twitter chatter surfaced via WebSearch is the same: a crowding flag prompting catalyst-hunting, never itself the catalyst.
4) CONTROL BASKET (mechanical, no judgment): `.venv/Scripts/python.exe experiments/news_edge/sources.py pm-gappers 3` → for EACH returned name add a pick with `signal: 0, control: true, gap_pct: <its gap>, sources: ["control"], reason: "mechanical >3% gapper"`. These are scored alongside your picks; your (+) picks must beat them.
5) LOG: write ALL picks (your calls + the control basket) to a temp JSON file (a list; each item: symbol, signal, confidence, reason, sources, source_signals, theme, premarket_rvol, earnings_day, and for controls control+gap_pct) and run:
   `.venv/Scripts/python.exe experiments/news_edge/newsedge.py log <that.json>`
   then delete the temp file. (newsedge writes experiments/news_edge/picks/<today-ET>.json and refuses to overwrite — if it says already exists, that day is done; stop.) gap_pct is backfilled automatically by the after-close scorer if you omit it on a non-control pick.
6) COMMIT (the wrapper pushes — do NOT push yourself): `git add experiments/news_edge/picks` then `git commit -m "news-edge picks <date>"`. Do NOT run `git push` (it's blocked in headless mode); the wrapper script runs `git push origin main` automatically after you exit. Just make sure your commit exists.
7) SUMMARY: print the picks table (symbol, my signal, theme, stocktwits signal, pm_rvol) + the count of control-basket names, and confirm the commit was created (the wrapper will push it).

If the US market is CLOSED today (weekend/holiday), do nothing and say so. If a step fails, log it and continue where sensible; the push in step 6 is the most important outcome.
