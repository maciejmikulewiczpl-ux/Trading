# Wrapper used by Windows Task Scheduler to launch the monthly dual-momentum runner.
# Fires every weekday morning; the runner self-gates to the first trading day of the
# month (no-ops otherwise) and refuses live orders unless a dedicated DUALMOM account
# key is set. Captures output to logs/dualmom_task_<date>.log.

$ErrorActionPreference = 'Continue'
$ROOT = 'C:\Users\macie\VSC\Trading'
$logDir = Join-Path $ROOT 'logs'
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }

$stamp = Get-Date -Format 'yyyy-MM-dd'
$wrapLog = Join-Path $logDir "dualmom_task_$stamp.log"

"=== DualMom launch at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz') ===" |
    Out-File -FilePath $wrapLog -Append -Encoding utf8

Set-Location $ROOT

$py = Join-Path $ROOT '.venv\Scripts\python.exe'
$script = Join-Path $ROOT 'live\run_dualmom.py'

& $py $script @args 2>&1 | Out-File -FilePath $wrapLog -Append -Encoding utf8
$exitCode = $LASTEXITCODE

"=== DualMom end at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz') (exit=$exitCode) ===" |
    Out-File -FilePath $wrapLog -Append -Encoding utf8

exit $exitCode
