$ts = (Get-Date -Format 'yyyyMMdd-HHmmss')
$log = Join-Path (Get-Location) "tools\ci_artifacts\poller_run_$ts.log"
Set-Location 'C:\intelligent-trading-bot-master'
# Run the poller and redirect all output (stdout+stderr) to the log
# Use the PowerShell redirection operator *> which works in PowerShell 5.1
./tmp_poll_jobs.ps1 -JobIds ci_dummy_1,ci_dummy_2,ci_dummy_3 -MaxAttempts 30 -PollInterval 1 *> $log
Write-Output "Poller finished, log: $log"