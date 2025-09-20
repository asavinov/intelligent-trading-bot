# PR: Minimal CI + Stable Pollers + Job Env Snapshot + Log Retention

This PR brings a pragmatic, low-flakiness foundation to iterate on the dashboard automation and debugging workflows.

What changed
- Minimal CI workflow (`.github/workflows/artifacts.yml`): runs deterministic dummy jobs, builds a concise summary, captures env snapshot, and uploads artifacts. No server/UI in CI to avoid flakiness.
- Stable pollers:
  - `tmp_poll_jobs_once.ps1`: one-shot, file/API hybrid with file fallback; supports comma-separated JobIds; resolves relative paths from repo root.
  - `tmp_poll_jobs.ps1`: same hardening for the legacy looping poller; adds quick server health-check and file-only summary when API is down.
- Server enhancements (`dashboard/api/scripts.py`):
  - Per-job env snapshot (`logs/jobs/{job_id}.env.json`) and `env_snapshot_path` in metadata, `/status`, and `/history` responses; included in logs ZIP.
  - Opportunistic log retention for `logs/jobs` with env vars:
    - `ITB_LOGS_MAX_JOBS` (default 200)
    - `ITB_LOGS_MAX_TOTAL_MB` (default 500)
- CI summary builder (`tools/build_ci_runs_summary.py`): now includes `env_snapshot_path` if present.
- Docs: `tools/POLLER_README.md` mentions retention env vars; minor clarifications.
- Small cleanup: fix Python 3.13 UTC deprecation warning in `scripts/ci_dummy.py`.

Why
- Make automation stable and fast. Avoid flaky UI runs and focus on deterministic artifacts for debugging (Issue #1/#6/#7).
- Ensure all UI/CLI-launched jobs persist full logs and minimal environment context.
- Prevent unbounded growth of `logs/jobs`.

How to verify
- Locally:
  1. Run 2-3 dummy jobs:
     - `python scripts/ci_dummy.py --job-id ci_dummy_1 --exit-code 0`
     - `python scripts/ci_dummy.py --job-id ci_dummy_2 --exit-code 0`
     - `python scripts/ci_dummy.py --job-id ci_dummy_3 --exit-code 1` (allowed to fail)
  2. Build summary: `python tools/build_ci_runs_summary.py`.
  3. One-shot poller: `./tmp_poll_jobs_once.ps1 -JobIds ci_dummy_1,ci_dummy_2,ci_dummy_3` and inspect `tools/ci_artifacts/ci_ui_runs_summary.json`.
  4. (Optional) Start dashboard, run a real script, then download logs ZIP from `/api/scripts/logs/{job_id}/download` and confirm `*.env.json` is included.

Notes / Future work
- If/when we re-enable UI tests in CI, keep `reload=false` and isolate runtime tmp dirs to avoid WatchFiles restarts.
- Enhance Jobs History UI to display env snapshot link and filters/pagination (Issue #5) and add a small env viewer (Issue #7).
- Add simple unit tests for retention and metadata (Issue #8).
