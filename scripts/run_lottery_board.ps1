# Morning lottery board builder — launched by the Windows task "LotteryBoard" ~6:24am PT.
# Builds today's hype board (signals 1-3+6 live; squeeze/uoa come online days 3-4), pushes
# a summary to the phone via ntfy, commits the immutable picks JSON, and nudges the VM to
# pull so the lottery-bot timer (09:44 ET) sees today's board.
# Plain python (no agent), deterministic. Output -> logs/lottery_board_<date>.log.
$ErrorActionPreference = 'Continue'
$ROOT = 'C:\Users\macie\VSC\Trading'
Set-Location $ROOT
$stamp = (Get-Date).ToString('yyyy-MM-dd')
$log = Join-Path $ROOT "logs\lottery_board_$stamp.log"
New-Item -ItemType Directory -Force -Path (Join-Path $ROOT 'logs') | Out-Null
"=== lottery board START $((Get-Date).ToString('o')) ===" | Add-Content -Path $log -Encoding utf8
& (Join-Path $ROOT '.venv\Scripts\python.exe') (Join-Path $ROOT 'experiments\lottery\notify_board.py') *>> $log 2>&1
# commit the immutable picks JSON so the bot on the VM can read today's board.
& git -C $ROOT add experiments/lottery/picks *>> $log 2>&1
& git -C $ROOT commit -q -m "lottery board $stamp" *>> $log 2>&1
& git -C $ROOT push origin main *>> $log 2>&1
# nudge the VM to pull so launch_lottery_bot.sh finds today's board immediately.
& ssh -o BatchMode=yes -o ConnectTimeout=15 trading-vm "cd /home/ubuntu/trading && git pull --ff-only" *>> $log 2>&1
"=== lottery board END (exit $LASTEXITCODE) $((Get-Date).ToString('o')) ===" | Add-Content -Path $log -Encoding utf8
