# Headless INTRADAY news re-evaluation — launched by the Windows scheduled task
# "NewsReeval" at ~9:30am PT (12:30 ET) on weekdays. Runs Claude Code non-interactively to
# re-read fresh news on the CURRENTLY-HELD news-hold names and LOG a hold/trim/exit
# recommendation per name (SHADOW: it never trades — it writes experiments/news_edge/reeval/
# <date>.json for later scoring vs the mechanical T+3/trail exit). Uses the Claude subscription
# (no per-token API cost). All output -> logs/news_reeval_<date>.log.
$ErrorActionPreference = 'Continue'
$ROOT = 'C:\Users\macie\VSC\Trading'
Set-Location $ROOT
$env:CI = 'true'   # tell the CLI not to wait on stdin

$stamp = (Get-Date).ToString('yyyy-MM-dd')
$logDir = Join-Path $ROOT 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir "news_reeval_$stamp.log"

$prompt = Get-Content -Raw (Join-Path $ROOT 'experiments\news_edge\reeval_prompt.md')
$claude = Join-Path $env:APPDATA 'npm\claude.cmd'
if (-not (Test-Path $claude)) { $claude = 'claude' }   # fall back to PATH

"=== headless news re-eval START $((Get-Date).ToString('o')) ===" | Add-Content -Path $log -Encoding utf8
# Scoped allow-list: web + shell (python/git) + file r/w + Alpaca READ tools only. No order/
# position-modifying tools are granted, so even a misfire cannot trade. Model pinned to Opus.
& $claude -p $prompt --max-turns 50 --model claude-opus-4-8 `
  --allowedTools WebSearch WebFetch Bash Read Write Edit Glob Grep `
  "mcp__alpaca__get_clock" "mcp__alpaca__get_news" `
  *>> $log
"=== agent done (exit $LASTEXITCODE); wrapper pushing to origin ===" | Add-Content -Path $log -Encoding utf8
# git push is gated for the headless AGENT, so the WRAPPER does it. Pull-rebase BEFORE push:
# the VM pushes data commits daily, so a plain push is rejected as non-fast-forward (this is
# the bug that stranded the 08-12..14 boards). Integrate first (disjoint files = clean rebase).
& git -C $ROOT pull --rebase --autostash origin main *>> $log 2>&1
& git -C $ROOT push origin main *>> $log 2>&1
"=== headless news re-eval END (push exit $LASTEXITCODE) $((Get-Date).ToString('o')) ===" | Add-Content -Path $log -Encoding utf8
