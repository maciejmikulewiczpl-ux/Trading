# Weekly short-interest cache builder — launched by the Windows task "LotteryShortInterest"
# Sundays 10am PT. MUST use .venv-openbb (yfinance lives there only). Fills the squeeze
# signal's cache (.short_interest_cache.json). Output -> logs/lottery_si_<date>.log.
$ErrorActionPreference = 'Continue'
$ROOT = 'C:\Users\macie\VSC\Trading'
Set-Location $ROOT
$stamp = (Get-Date).ToString('yyyy-MM-dd')
$log = Join-Path $ROOT "logs\lottery_si_$stamp.log"
New-Item -ItemType Directory -Force -Path (Join-Path $ROOT 'logs') | Out-Null
"=== lottery short-interest START $((Get-Date).ToString('o')) ===" | Add-Content -Path $log -Encoding utf8
& (Join-Path $ROOT '.venv-openbb\Scripts\python.exe') (Join-Path $ROOT 'experiments\lottery\update_short_interest.py') *>> $log 2>&1
"=== lottery short-interest END (exit $LASTEXITCODE) $((Get-Date).ToString('o')) ===" | Add-Content -Path $log -Encoding utf8
