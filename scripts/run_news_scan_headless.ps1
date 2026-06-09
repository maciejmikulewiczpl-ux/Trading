# Headless morning news-edge scan — launched by the Windows scheduled task "NewsEdgeScan"
# at ~6:33am PDT (9:33 ET) on weekdays. Runs Claude Code non-interactively to do the scan
# (read news, score sources, log picks, git push) so the VM can trade them at 9:40 ET.
# Fires regardless of whether the interactive VSCode chat is open/active. Uses the existing
# Claude Code subscription (no per-token API cost). All output -> logs/news_scan_<date>.log.
$ErrorActionPreference = 'Continue'
$ROOT = 'C:\Users\macie\VSC\Trading'
Set-Location $ROOT
$env:CI = 'true'   # tell the CLI not to wait on stdin

$stamp = (Get-Date).ToString('yyyy-MM-dd')
$logDir = Join-Path $ROOT 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir "news_scan_$stamp.log"

$prompt = Get-Content -Raw (Join-Path $ROOT 'scripts\news_scan_prompt.md')
# Full path to the npm-global Claude Code CLI (Task Scheduler may not have user PATH).
$claude = Join-Path $env:APPDATA 'npm\claude.cmd'
if (-not (Test-Path $claude)) { $claude = 'claude' }   # fall back to PATH

"=== headless news scan START $((Get-Date).ToString('o')) ===" | Add-Content -Path $log -Encoding utf8
& $claude -p $prompt --dangerously-skip-permissions --max-turns 60 *>> $log
"=== headless news scan END (exit $LASTEXITCODE) $((Get-Date).ToString('o')) ===" | Add-Content -Path $log -Encoding utf8
