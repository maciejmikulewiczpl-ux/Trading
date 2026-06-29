# After-close lottery scorer — launched by the Windows task "LotteryScore" ~1:10pm PT
# (just after the 12:55 PT / 15:55 ET measurement close). Scores today's board
# (ret_945_close), backfills prior days' 1d/3d, pushes the hit-rate summary to the phone,
# commits the scored picks, and nudges the VM to pull for the status page.
# Plain python (no agent), deterministic. Output -> logs/lottery_score_<date>.log.
$ErrorActionPreference = 'Continue'
$ROOT = 'C:\Users\macie\VSC\Trading'
Set-Location $ROOT
$stamp = (Get-Date).ToString('yyyy-MM-dd')
$log = Join-Path $ROOT "logs\lottery_score_$stamp.log"
New-Item -ItemType Directory -Force -Path (Join-Path $ROOT 'logs') | Out-Null
"=== lottery score START $((Get-Date).ToString('o')) ===" | Add-Content -Path $log -Encoding utf8
& (Join-Path $ROOT '.venv\Scripts\python.exe') (Join-Path $ROOT 'experiments\lottery\notify_score.py') *>> $log 2>&1
# rebuild the per-trade outcome ledger (logs/lottery_trade_ledger.csv) for deep-dive analysis
& (Join-Path $ROOT '.venv\Scripts\python.exe') (Join-Path $ROOT 'experiments\lottery\trade_ledger.py') *>> $log 2>&1
& git -C $ROOT add experiments/lottery/picks *>> $log 2>&1
& git -C $ROOT commit -q -m "lottery outcomes $stamp" *>> $log 2>&1
& git -C $ROOT push origin main *>> $log 2>&1
& ssh -o BatchMode=yes -o ConnectTimeout=15 trading-vm "cd /home/ubuntu/trading && git pull --ff-only" *>> $log 2>&1
"=== lottery score END (exit $LASTEXITCODE) $((Get-Date).ToString('o')) ===" | Add-Content -Path $log -Encoding utf8
