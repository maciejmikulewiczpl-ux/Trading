You are running HEADLESS (no human present) as the automated morning news-edge scan, ~9:33 ET, inside the ORB opening-range window. There is no chat to reply to — just DO the work, then print a short summary. Use your Bash tool for python/git commands; use forward-slash paths (they work in git-bash on this Windows machine). Do NOT launch any trading bot locally — the VM news-orb (09:40 ET) pulls the picks you push and trades them.

GOAL: produce ~10-20 LIQUID frontrunner/avoid stock picks for today and push them.

STEPS (in order):
1) CANDIDATE NET (cast wide, liquid only — skip sub-$5 / penny pumps):
   - StockTwits trending (free, real-time): `.venv/Scripts/python.exe experiments/news_edge/sources.py st-trending 30`
   - Market tone + single-stock catalysts: WebSearch (include Yahoo Finance + CNBC) for today's premarket movers / earnings / upgrades.
   - If mcp__alpaca__ tools are available, also use get_market_movers (stocks), get_most_active_stocks, and get_news for fresh Benzinga catalysts. If NOT available, rely on StockTwits + WebSearch — fine.
2) MY CALL per candidate: signal +1 (frontrunner, clear bullish catalyst) / -1 (avoid, clear bearish) / 0 (unclear), each with a short reason + confidence 0-1 + sources. Favor clear fresh catalysts (deals, upgrades, guidance, FDA); momentum names ok; earnings-day names cautious (gappy) but allowed with a note. Liquid only. Aim ~10-20.
3) PER-SOURCE SIGNALS (head-to-head): for ALL candidate tickers:
   `.venv/Scripts/python.exe experiments/news_edge/sources.py st-sentiment <COMMA_TICKERS>`  (StockTwits crowd — main scored source)
   `.venv/Scripts/python.exe experiments/news_edge/sources.py av <COMMA_TICKERS>`  (Alpha Vantage — often empty on fresh news; include a signal ONLY if n>0)
   Each pick: source_signals {"stocktwits": sig, "alphavantage": sig if n>0}.
4) LOG: write the picks to a temp JSON file (a list; each item: symbol, signal, confidence, reason, sources, source_signals) and run:
   `.venv/Scripts/python.exe experiments/news_edge/newsedge.py log <that.json>`
   then delete the temp file. (newsedge writes experiments/news_edge/picks/<today-ET>.json and refuses to overwrite — if it says already exists, that day is done; stop.)
5) COMMIT (the wrapper pushes — do NOT push yourself): `git add experiments/news_edge/picks` then `git commit -m "news-edge picks <date>"`. Do NOT run `git push` (it's blocked in headless mode); the wrapper script runs `git push origin main` automatically after you exit. Just make sure your commit exists.
6) SUMMARY: print the picks table (symbol, my signal, stocktwits signal) and confirm the commit was created (the wrapper will push it).

If the US market is CLOSED today (weekend/holiday), do nothing and say so. If a step fails, log it and continue where sensible; the push in step 5 is the most important outcome.
