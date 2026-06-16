# Weekly accumulation+chatter scan — launched by the Windows task "AccumulationScan"
# Sundays 10:30am PT (right after LotteryShortInterest at 10am). MUST use .venv-openbb
# (yfinance lives there only). 13F institutional accumulation (artifact-filtered) +
# Reddit/StockTwits chatter overlap -> CSV + phone push (ntfy). ~4-5 min. Plain python,
# no LLM. Research watchlist only. Output -> logs/accumulation_scan_<date>.log.
$ErrorActionPreference = 'Continue'
$ROOT = 'C:\Users\macie\VSC\Trading'
Set-Location $ROOT
$stamp = (Get-Date).ToString('yyyy-MM-dd')
$log = Join-Path $ROOT "logs\accumulation_scan_$stamp.log"
New-Item -ItemType Directory -Force -Path (Join-Path $ROOT 'logs') | Out-Null
"=== accumulation scan START $((Get-Date).ToString('o')) ===" | Add-Content -Path $log -Encoding utf8
& (Join-Path $ROOT '.venv-openbb\Scripts\python.exe') (Join-Path $ROOT 'scripts\weekly_accumulation_scan.py') *>> $log 2>&1
"=== accumulation scan END (exit $LASTEXITCODE) $((Get-Date).ToString('o')) ===" | Add-Content -Path $log -Encoding utf8
