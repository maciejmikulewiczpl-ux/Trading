# Wrapper used by Windows Task Scheduler to launch the ORB paper runner.
# Captures stdout + stderr (including any pre-startup Python errors) to logs/task_<date>.log.
# The runner itself also writes its own detailed log to logs/orb_<date>.log.

$ErrorActionPreference = 'Continue'
$ROOT = 'C:\Users\macie\VSC\Trading'
$logDir = Join-Path $ROOT 'logs'
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }

$stamp = Get-Date -Format 'yyyy-MM-dd'
$wrapLog = Join-Path $logDir "task_$stamp.log"

"=== Task launch at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz') ===" |
    Out-File -FilePath $wrapLog -Append -Encoding utf8

Set-Location $ROOT

$py = Join-Path $ROOT '.venv\Scripts\python.exe'
$script = Join-Path $ROOT 'live\paper_orb.py'

# Pass through any additional args the task was registered with.
# Pipe through Out-File -Encoding utf8 because PS 5.1's native redirection writes UTF-16
# and corrupts the wrapper log when interleaved with the UTF-8 banner lines below.
& $py $script @args 2>&1 | Out-File -FilePath $wrapLog -Append -Encoding utf8
$exitCode = $LASTEXITCODE

"=== Task end at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz') (exit=$exitCode) ===" |
    Out-File -FilePath $wrapLog -Append -Encoding utf8

exit $exitCode
