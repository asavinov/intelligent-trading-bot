(OUTDATED)

# Server

#### General system functions:

* Ping the server: https://python-binance.readthedocs.io/en/latest/general.html#id1
* Get system status: https://python-binance.readthedocs.io/en/latest/general.html#id3
* Check server time and compare with local time: https://python-binance.readthedocs.io/en/latest/general.html#id2

#### Time synchronization in OS and time zones

https://www.digitalocean.com/community/tutorials/how-to-set-up-time-synchronization-on-ubuntu-16-04

````markdown
# Dashboard Server notes (updated)

This document summarizes the dashboard server endpoints and feature gates that are relevant for local development and safe enablement in production.

## Feature gates (ENV)

- DASHBOARD_PIPELINE_ENABLED: When set to 1/true, enables the pipeline API/UI. The server enforces this gate defensively. The client reads it via /api/system/settings.
- RUN_PLAYWRIGHT_UI_TESTS: When set to 1, enables running the UI E2E test locally; defaults to off in CI to remain deterministic.
- ITB_USE_TF_NN: When set to 1, enables TensorFlow/Keras backend for NN models in tests; otherwise uses a lightweight sklearn fallback.

## System settings endpoint

- GET /api/system/settings → { pipeline_enabled: boolean }
  - The frontend reads this to decide whether to show/enable Pipeline UI controls.

## Pipeline API (feature-gated)

Prefix: /api/pipeline

- POST /api/pipeline/run
  - Body: { steps?: string[], config_file?: string, timeout_per_step?: number, extra_args?: { [step]: string[] } }
  - Returns: { pipeline_id, status, steps, config }
  - Requires DASHBOARD_PIPELINE_ENABLED=1.

- GET /api/pipeline/status/{pipeline_id}
  - Returns pipeline meta including per-step status, job_ids, and timing.

- GET /api/pipeline/stream/{pipeline_id}
  - SSE stream of pipeline log lines, finishing with a [FINISHED] marker.

- GET /api/pipeline/artifacts/{pipeline_id}
  - Returns a ZIP containing pipeline log/meta and each step job's stdout/stderr/meta/env when available.

### Local smoke test

1) Start the dashboard server with DASHBOARD_PIPELINE_ENABLED=1
2) Run the local smoke helper:

   python -m tools.pipeline_smoke_local --server http://127.0.0.1:8000 --config configs/config-quick-1d-ci.jsonc --steps download,merge

You should see the pipeline progress to completed; logs and meta are written under logs/pipelines/.

To download all artifacts for a run:

  Open: http://127.0.0.1:8000/api/pipeline/artifacts/{pipeline_id}

```diff
Note:
- The CI intentionally avoids exercising the full pipeline to keep runs deterministic and fast.
- Use the smoke script locally or add a manual (workflow_dispatch) GitHub Actions workflow if needed.
```
````

