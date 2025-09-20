import argparse
import json
import sys
import time
from pathlib import Path

try:
    import requests
except Exception as e:
    print("ERROR: The 'requests' package is required to run this helper.")
    print("Please install it (pip install requests) and try again.")
    sys.exit(2)


def post_run(base_url: str, script: str, config: str | None, timeout: int | None, extra_args: list[str] | None):
    url = f"{base_url}/api/scripts/run"
    payload = {
        "script_name": script,
    }
    if config:
        payload["config_file"] = config
    if timeout:
        payload["timeout"] = int(timeout)
    if extra_args:
        payload["extra_args"] = list(map(str, extra_args))

    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_status(base_url: str, job_id: str):
    url = f"{base_url}/api/scripts/status/{job_id}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def download_logs(base_url: str, job_id: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    url = f"{base_url}/api/scripts/logs/{job_id}/download"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    out_path = out_dir / f"{job_id}.zip"
    out_path.write_bytes(resp.content)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Run dashboard API jobs and collect artifacts")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", default="8019", help="Server port")
    parser.add_argument("--script", default="download_binance", help="Script name to run")
    parser.add_argument("--config", default="configs/config-sample-1h.jsonc", help="Config file path")
    parser.add_argument("--runs", type=int, default=3, help="How many runs to execute")
    parser.add_argument("--timeout", type=int, default=300, help="Per-job timeout seconds (server-side monitor hint)")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Seconds between status polls")
    parser.add_argument("--max-wait", type=int, default=900, help="Max seconds to wait for a job to finish")
    parser.add_argument("--out", default="logs/jobs", help="Where to save downloaded logs Zips")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, help="Extra CLI args to forward to the script (use after --)")

    args = parser.parse_args()
    base_url = f"http://{args.host}:{args.port}"
    out_dir = Path(args.out)
    print(f"Using server: {base_url}")
    print(f"Script: {args.script}; Config: {args.config}; Runs: {args.runs}")

    # Quick health check
    try:
        r = requests.get(f"{base_url}/health", timeout=10)
        r.raise_for_status()
        print("Health:", r.json())
    except Exception as e:
        print("ERROR: Server /health check failed:", e)
        sys.exit(1)

    job_ids: list[str] = []
    for i in range(1, args.runs + 1):
        print(f"\n=== Starting run {i}/{args.runs} ===")
        try:
            res = post_run(base_url, args.script, args.config, args.timeout, args.extra)
        except Exception as e:
            print(f"Run {i}: failed to start - {e}")
            break
        job_id = res.get("job_id")
        print("Started job:", json.dumps(res, ensure_ascii=False))
        if not job_id:
            print("No job_id returned; aborting")
            break
        job_ids.append(job_id)

        # Poll until completion
        start = time.time()
        last_status = None
        while True:
            try:
                st = get_status(base_url, job_id)
                status = st.get("status")
                if status != last_status:
                    print(f"Status: {status}")
                    last_status = status
                if status in ("completed", "failed", "error"):
                    rc = st.get("returncode")
                    print(f"Finished with status={status} returncode={rc}")
                    break
            except Exception as e:
                print(f"Polling error (will retry): {e}")

            if time.time() - start > args.max_wait:
                print("ERROR: Max wait exceeded; moving on")
                break
            time.sleep(args.poll_interval)

        # Download logs
        try:
            out_path = download_logs(base_url, job_id, out_dir)
            print(f"Logs saved to: {out_path}")
        except Exception as e:
            print(f"Could not download logs for {job_id}: {e}")

    print("\nSummary (job_ids):")
    for jid in job_ids:
        print(jid)


if __name__ == "__main__":
    main()
