You are running HEADLESS (no human present) as the automated morning news-edge scan, ~9:33 ET, inside the ORB opening-range window. There is no chat to reply to — just DO the work, then print a short summary at the end. Use PowerShell for python/git (not Bash) on this Windows machine. Do NOT launch any trading bot locally — the VM news-orb (09:40 ET) pulls the picks you push and trades them.

GOAL: produce ~10-20 LIQUID frontrunner/avoid stock picks for today and push them.

STEPS (in order):
1) CANDIDATE NET (cast wide, liquid only — skip sub-$5 / penny pumps):
   - StockTwits trending (free, real-time): `.venv\Scripts\python.exe experiments\news_edge\sources.py st-trending 30`
   - Market tone + single-stock catalysts: WebSearch (include Yahoo Finance + CNBC) for "stock market today premarket movers catalysts earnings upgrades" for today's date.
   - If the mcp__alpaca__ tools are available, also use get_market_movers (stocks), get_most_active_stocks, and get_news for fresh Benzinga catalysts. If they are NOT available, rely on StockTwits + WebSearch — that is fine.
2) MY CALL per candidate: signal +1 (frontrunner, clear bullish catalyst) / -1 (avoid, clear bearish) / 0 (unclear), each with a short reason + confidence 0-1 + sources. Favor clear fresh catalysts (deals, upgrades, guidance, FDA); momentum names ok. Earnings-day names: cautious (gappy) but allowed with a note. Keep only liquid names. Aim for ~10-20 total.
3) PER-SOURCE SIGNALS (the head-to-head): for ALL candidate tickers, run:
   `.venv\Scripts\python.exe experiments\news_edge\sources.py st-sentiment <COMMA_TICKERS>`  (StockTwits crowd — main scored source)
   `.venv\Scripts\python.exe experiments\news_edge\sources.py av <COMMA_TICKERS>`  (Alpha Vantage — often empty on fresh news; include a signal ONLY if n>0)
   Each pick gets source_signals {"stocktwits": sig, "alphavantage": sig if n>0}.
4) LOG: write the picks to a temp JSON file (a list; each item: symbol, signal, confidence, reason, sources, source_signals) and run:
   `.venv\Scripts\python.exe experiments\news_edge\newsedge.py log <that.json>`
   then delete the temp file. (newsedge writes experiments\news_edge\picks\<today-ET>.json; it refuses to overwrite — if it says already exists, that day is already done, stop.)
5) COMMIT + PUSH (critical — the VM pulls this): `git add experiments/news_edge/picks; git commit -m "news-edge picks <date>"; git push origin main`. Confirm the push succeeded (it must reach origin/main).
6) SUMMARY: print the picks table (symbol, my signal, stocktwits signal) and confirm the push reached origin.

If the US market is CLOSED today (weekend/holiday), do nothing and say so. If a step fails, log the error and continue where sensible; the push in step 5 is the most important outcome.
