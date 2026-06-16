# ONE-OFF travel-day helper: run BOTH morning scans NOW, regardless of schedule.
#
# Use this when you'll be OFFLINE (e.g. on a plane) at the normal scan time
# (~9:24-9:33 ET). Run it while you still have wifi: it generates TODAY's ET-dated
# news-edge + hype picks, commits, pushes, and nudges the VM. The VM bots pull and
# trade those picks at 9:40 / 9:44 ET *even if this laptop is then offline* (their
# launch wrappers git-pull on their own). Entries still execute at the normal time on
# live prices — only the SELECTION uses the thinner early-premarket data (reduced but
# tradeable, exactly the one-off trade-off).
#
# IMPORTANT: run this TOMORROW MORNING (after midnight ET, ideally after ~6:00 ET so
# premarket data exists), NOT tonight — picks are keyed to the ET date at run time.
#
# This is purely ADDITIVE: it does not touch the scheduled tasks. If you DON'T run it,
# NewsEdgeScan / LotteryBoard fire on their normal schedule as usual. (If you run it but
# then stay online, the scheduled scan will also run later and simply refresh the picks —
# harmless.)
#
# Double-click run_morning_scans_now.cmd, or run this .ps1 directly.
$ErrorActionPreference = 'Continue'
$ROOT = 'C:\Users\macie\VSC\Trading'
Set-Location $ROOT

$etTz  = [System.TimeZoneInfo]::FindSystemTimeZoneById('Eastern Standard Time')
$etNow = [System.TimeZoneInfo]::ConvertTimeFromUtc([DateTime]::UtcNow, $etTz)
$stamp = $etNow.ToString('yyyy-MM-dd')

Write-Host "================================================================"
Write-Host " EARLY MORNING SCANS  -  ET date $stamp  ($($etNow.ToString('HH:mm')) ET)" -ForegroundColor Cyan
Write-Host " Generating today's news-edge + hype picks now, then pushing so the"
Write-Host " VM trades them at 9:40 / 9:44 ET even if this laptop goes offline."
Write-Host " Takes ~10 min total (news ~7, board ~2.5). Keep this window open +"
Write-Host " stay online until you see 'SAFE TO GO OFFLINE'."
Write-Host "================================================================`n"

Write-Host "[1/2] News-edge scan (Claude headless - the slow one, ~7 min) ..." -ForegroundColor Yellow
& (Join-Path $ROOT 'scripts\run_news_scan_headless.ps1')
Write-Host "      news scan finished.`n"

Write-Host "[2/2] Hype board ..." -ForegroundColor Yellow
& (Join-Path $ROOT 'scripts\run_lottery_board.ps1')
Write-Host "      hype board finished.`n"

# Safety-net push (the launchers already push; this is idempotent).
& git -C $ROOT push origin main | Out-Host

# ---- verify + report ----
$newsPicks = Join-Path $ROOT "experiments\news_edge\picks\$stamp.json"
$hypePicks = Join-Path $ROOT "experiments\lottery\picks\$stamp.json"
function Show-Picks($path, $label) {
  if (Test-Path $path) {
    try {
      $j = Get-Content -Raw $path | ConvertFrom-Json
      $n = @($j.picks).Count
      $syms = (@($j.picks) | Select-Object -First 6 | ForEach-Object { $_.symbol }) -join ', '
      Write-Host ("  OK  {0,-10}: {1} picks  [{2}]" -f $label, $n, $syms) -ForegroundColor Green
    } catch {
      Write-Host ("  OK  {0,-10}: file present" -f $label) -ForegroundColor Green
    }
    return $true
  }
  Write-Host ("  MISSING  {0}: no picks file for $stamp" -f $label) -ForegroundColor Red
  return $false
}

Write-Host "---------------------------- RESULT ----------------------------"
$okN = Show-Picks $newsPicks 'news-edge'
$okH = Show-Picks $hypePicks 'hype'
$unpushed = & git -C $ROOT rev-list 'origin/main..main' --count
if ([string]::IsNullOrWhiteSpace($unpushed)) { $unpushed = '0' }
Write-Host ""
if ($okN -and $okH -and $unpushed -eq '0') {
  Write-Host " SAFE TO GO OFFLINE - both picks pushed; the VM pulls + trades at 9:40/9:44 ET." -ForegroundColor Green
} elseif ($okN -or $okH) {
  Write-Host " PARTIAL - a picks set is missing or unpushed ($unpushed unpushed commit(s))." -ForegroundColor Yellow
  Write-Host "           Check the lines above before going offline." -ForegroundColor Yellow
} else {
  Write-Host " FAILED - no picks generated. The bots will be IDLE today." -ForegroundColor Red
}
Write-Host "================================================================`n"
Read-Host "Press Enter to close"
