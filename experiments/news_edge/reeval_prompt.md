You are running HEADLESS (no human present) as the automated INTRADAY NEWS RE-EVALUATION, ~12:30 ET. There is no chat to reply to — just DO the work, then print a short summary. Use your Bash tool for python/git commands; use forward-slash paths (git-bash on Windows). This is a SHADOW layer: you LOG a recommendation per held name, you do NOT trade and you do NOT launch any bot. Nothing you write here places or cancels an order.

GOAL: for each name the news-hold bot is CURRENTLY HOLDING, re-read today's fresh news and record whether — with what you now know — the position still deserves to be held, or the catalyst has changed enough that a smart trader would trim/exit. This is measured against the bot's mechanical exit (T+3 / 10% trail) to learn whether an LLM catalyst-aware exit adds value.

STEPS (in order):
1) GET THE HELD NAMES (the re-eval universe):
   `.venv/Scripts/python.exe experiments/news_edge/reeval.py positions`
   This prints the .env.news account's current holdings as JSON: [{symbol, qty, avg_px, mkt_px, upl_pct}, ...].
   If it prints `[]` (flat, no positions), there is nothing to re-evaluate — print "flat, nothing to re-eval" and STOP (do not create a log file).

2) FRESH NEWS PER HELD NAME (only the held symbols; do not scan the market):
   - `.venv/Scripts/python.exe experiments/news_edge/sources.py edgar 16` then filter to held names (fresh 8-K/424B5/S-1 filings on a HELD name = the highest-value trigger: offering priced = dilution = exit signal).
   - WebSearch each held symbol for TODAY's news (Yahoo Finance + CNBC): did the catalyst play out, reverse, or get superseded (offering priced, deal broke, guidance cut, downgrade, halt)?
   - If mcp__alpaca__ tools are available, use get_news for the held symbols (fresh Benzinga catalysts).

3) YOUR CALL per held name — action ∈ {hold, trim, exit}, with:
   - `catalyst_changed`: true only if there is a NAMED, DATED new development since the morning (offering priced, deal terminated, guidance cut, FDA CRL, downgrade, halt). Absent fresh news = catalyst_changed:false.
   - RULES:
     * NO fresh material news → `hold` (do NOT exit on price action alone — an LLM has no edge reading ticks, and the analysis shows bad-catalyst names often drop then BOUNCE the next day, so exiting on a red number is exactly the trap).
     * Confirmed BAD development (dilution priced, deal broke, guidance/FDA miss, downgrade) → `exit`, catalyst_changed:true.
     * Catalyst intact but extended / crowded / partial bad news → `trim`.
   - `confidence` 0-1, a short `reason` (name the development or say "no new catalyst"), and `sources`.
   - Copy `px_at_reeval` = the name's mkt_px from step 1, and `upl_pct_at_reeval` = its upl_pct (snapshot for later scoring).

4) LOG (a list; each item: symbol, action, catalyst_changed, confidence, reason, sources, px_at_reeval, upl_pct_at_reeval) to a temp JSON file, then:
   `.venv/Scripts/python.exe experiments/news_edge/reeval.py log <that.json>`
   then delete the temp file. (It writes experiments/news_edge/reeval/<today-ET>.json and refuses to overwrite — if it says already exists, today's re-eval is done; stop.)

5) COMMIT (the wrapper pushes — do NOT push yourself): `git add experiments/news_edge/reeval` then `git commit -m "news re-eval <date>"`. Do NOT run `git push`.

6) SUMMARY: print a table (symbol, action, catalyst_changed, reason) and confirm the commit exists.

If the market is CLOSED today, or the held list is empty, do nothing and say so. Remember: SHADOW only — recommendations are logged and scored later, never executed.
