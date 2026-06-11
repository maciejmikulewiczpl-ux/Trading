# After-close scorer — launched by the Windows task "NewsEdgeScore" ~13:05 PT weekdays
# (just after the 12:55 PT / 15:55 ET measurement close). Scores today's news-edge picks and
# pushes the head-to-head (my read vs the StockTwits crowd) to your phone via ntfy.
# Plain python (no agent), deterministic. Output -> logs/news_score_<date>.log.
$ErrorActionPreference = 'Continue'
$ROOT = 'C:\Users\macie\VSC\Trading'
Set-Location $ROOT
$stamp = (Get-Date).ToString('yyyy-MM-dd')
$log = Join-Path $ROOT "logs\news_score_$stamp.log"
New-Item -ItemType Directory -Force -Path (Join-Path $ROOT 'logs') | Out-Null
"=== news-edge score START $((Get-Date).ToString('o')) ===" | Add-Content -Path $log -Encoding utf8
& (Join-Path $ROOT '.venv\Scripts\python.exe') (Join-Path $ROOT 'experiments\news_edge\notify_score.py') *>> $log 2>&1
# commit the scored picks so the record (with outcomes) syncs to the repo / VM status page.
& git -C $ROOT add experiments/news_edge/picks *>> $log 2>&1
& git -C $ROOT commit -q -m "news-edge outcomes $stamp" *>> $log 2>&1
& git -C $ROOT push origin main *>> $log 2>&1
# nudge the VM to pull so its status page shows today's outcomes immediately —
# the VM otherwise only pulls at the 9:40 ET news-orb launch, leaving the
# News-Edge tab's today-table blank until the next morning (seen 2026-06-11).
& ssh -o BatchMode=yes -o ConnectTimeout=15 trading-vm "cd /home/ubuntu/trading && git pull --ff-only" *>> $log 2>&1
"=== news-edge score END (exit $LASTEXITCODE) $((Get-Date).ToString('o')) ===" | Add-Content -Path $log -Encoding utf8
