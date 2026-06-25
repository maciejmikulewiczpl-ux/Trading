# One-time phone reminder — Windows task "PortfolioVerdictCheck" (~2026-07-28).
# The Hype experiment hits its pre-registered 30-trading-day verdict around now and the
# news-edge measurement is mature. Time to review the scoreboards and build the combined
# ORB + validated-experiments portfolio (risk/Sharpe-weighted, regime-throttled). That
# build needs a LIVE Claude session — this just pings the phone so it doesn't slip.
$ErrorActionPreference = 'Continue'
$ROOT = 'C:\Users\macie\VSC\Trading'
$topic = $null
if (Test-Path (Join-Path $ROOT '.env')) {
  foreach ($line in Get-Content (Join-Path $ROOT '.env')) {
    if ($line -match '^\s*NTFY_TOPIC\s*=\s*(.+)$') { $topic = $matches[1].Trim().Trim('"').Trim("'") }
  }
}
$msg = "Hype 30-trading-day verdict due + news-edge measurement mature. Open Claude to review the scoreboards and build the combined ORB+experiments portfolio (risk-weighted, regime-throttled). Context in memory: orb_optimization_jun25 / lottery_experiment."
if ($topic) {
  try {
    Invoke-RestMethod -Uri "https://ntfy.sh/$topic" -Method Post -Body $msg `
      -Headers @{ Title = 'Portfolio build: verdicts due'; Tags = 'bar_chart'; Priority = '4' }
  } catch {}
}
$log = Join-Path $ROOT 'logs\portfolio_reminder.log'
"$((Get-Date).ToString('o'))  reminder fired (topic set: $([bool]$topic))" | Add-Content -Path $log -Encoding utf8
