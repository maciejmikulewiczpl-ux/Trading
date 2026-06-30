# Daily biotech surge radar -- Windows task "BiotechRadar" ~6:15am PT (pre-market).
# Scans the XBI universe for heating-up biotechs (volume-building + momentum + short
# interest) + upcoming trial catalysts, prints trade cards, and pushes the top names to
# the phone (ntfy). Read-only watchlist -- SPECULATIVE, not auto-trading. Plain python.
$ErrorActionPreference = 'Continue'
$ROOT = 'C:\Users\macie\VSC\Trading'
Set-Location $ROOT
$stamp = (Get-Date).ToString('yyyy-MM-dd')
$log = Join-Path $ROOT "logs\biotech_radar_$stamp.log"
New-Item -ItemType Directory -Force -Path (Join-Path $ROOT 'logs') | Out-Null
"=== biotech radar START $((Get-Date).ToString('o')) ===" | Add-Content -Path $log -Encoding utf8
& (Join-Path $ROOT '.venv-openbb\Scripts\python.exe') (Join-Path $ROOT 'scripts\biotech_radar.py') *>> $log 2>&1
# commit the snapshot + push so the VM status page can show it
& git -C $ROOT add live/biotech_radar_latest.json *>> $log 2>&1
& git -C $ROOT commit -q -m "biotech radar $stamp" *>> $log 2>&1
& git -C $ROOT push origin main *>> $log 2>&1
& ssh -o BatchMode=yes -o ConnectTimeout=15 trading-vm "cd /home/ubuntu/trading && git pull --ff-only" *>> $log 2>&1
"=== biotech radar END (exit $LASTEXITCODE) $((Get-Date).ToString('o')) ===" | Add-Content -Path $log -Encoding utf8
