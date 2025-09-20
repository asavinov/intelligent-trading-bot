import os
import json
import time
import requests

"""
Local smoke test for the Pipeline API (run while the dashboard server is up).
- Uses a tiny config and a short step list.
- Requires: set DASHBOARD_PIPELINE_ENABLED=1 in the server environment.

Run:
  python -m tools.pipeline_smoke_local --server http://127.0.0.1:8000 --config configs/config-quick-1d-ci.jsonc --steps download,merge
"""

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8000", help="Dashboard server base URL")
    parser.add_argument("--config", default="configs/config-quick-1d-ci.jsonc", help="Path to config file")
    parser.add_argument("--steps", default="download,merge", help="Comma-separated steps")
    args = parser.parse_args()

    base = args.server.rstrip('/') + "/api/pipeline"
    steps = [s.strip() for s in args.steps.split(',') if s.strip()]

    # Start pipeline
    resp = requests.post(base + "/run", json={
        "steps": steps,
        "config_file": args.config,
        "timeout_per_step": 60,
    })
    if resp.status_code != 200:
        print("Failed to start pipeline:", resp.status_code, resp.text)
        return 1
    data = resp.json()
    pipeline_id = data.get("pipeline_id")
    print("Started pipeline:", pipeline_id)

    # Poll status
    status_url = base + f"/status/{pipeline_id}"
    while True:
        r = requests.get(status_url)
        if r.status_code != 200:
            print("Status error:", r.status_code, r.text)
            break
        st = r.json()
        print("Status:", st.get("status"), "steps:", [s.get("status") for s in st.get("steps", [])])
        if st.get("status") in ("completed", "failed", "error"):
            print("Final status:", st.get("status"))
            print(json.dumps(st, indent=2))
            break
        time.sleep(2)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
