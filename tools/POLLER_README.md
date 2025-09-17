Quick guide — Poller usage and troubleshooting

Purpose
- The project provides two poller scripts to collect job summaries:
  1. `tmp_poll_jobs.ps1` — legacy poller with looping behaviour. It can hang if the dashboard API is unreachable. It now includes a file-only fallback but is considered higher-risk for local interactive runs.
  2. `tmp_poll_jobs_once.ps1` — recommended single-pass poller. Tries the API once per job; if API is down or returns 404 it falls back to local metadata files under `logs/jobs/`. Always completes quickly.

Recommended commands (from repository root)

# Run single-pass for specific jobs and print JSON summary
Set-Location 'C:\intelligent-trading-bot-master'
./tmp_poll_jobs_once.ps1 -JobIds ci_dummy_1,ci_dummy_2,ci_dummy_3
Get-Content tools\ci_artifacts\ci_ui_runs_summary.json -Raw

# If you need to run the full poller (not recommended interactively), make sure the server is up
Set-Location 'C:\intelligent-trading-bot-master'
./tmp_poll_jobs.ps1 -JobIds ci_dummy_1,ci_dummy_2,ci_dummy_3 -MaxAttempts 30 -PollInterval 1

Troubleshooting
- If a poll appears to hang: stop it and use `tmp_poll_jobs_once.ps1` instead.
- To find a stuck poller process:
  Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and $_.CommandLine -like '*tmp_poll_jobs.ps1*' } | Select-Object ProcessId,CommandLine,CreationDate
- To force-stop by PID:
  Stop-Process -Id <PID> -Force

Notes
- CI should use `tmp_poll_jobs_once.ps1` or the Python helper `tools/build_ci_runs_summary.py` to produce deterministic artifacts.
- I recommend leaving `tmp_poll_jobs.ps1` in place for backward compatibility but avoid running it interactively.
- Server retains logs/jobs with a simple policy and cleans old items opportunistically. You can control limits via env vars:
  - ITB_LOGS_MAX_JOBS (default 200)
  - ITB_LOGS_MAX_TOTAL_MB (default 500)

If you want, I can now:
- Remove the long poller from the repo and keep only the single-pass version, or
- Keep both and add a CI step documentation. 

Tell me which and I'll finish it now.