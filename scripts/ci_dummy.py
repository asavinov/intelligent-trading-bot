#!/usr/bin/env python3
"""A tiny deterministic CI helper used by the lightweight workflow.

Usage examples:
  python scripts/ci_dummy.py --exit-code 0
  python scripts/ci_dummy.py --job-id myjob --exit-code 1

The script will print a JSON object to stdout with fields: job_id, exit_code, timestamp.
It will also write a small stdout log and an atomic meta JSON into `logs/jobs/` so CI can collect them.
"""
import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone


def atomic_write(path: str, data: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
    os.replace(tmp, path)


def main() -> int:
    p = argparse.ArgumentParser(description="CI dummy job")
    p.add_argument("--job-id", default=None, help="Optional job id to use")
    p.add_argument("--exit-code", type=int, default=0, help="Exit code to emit")
    args = p.parse_args()

    job_id = args.job_id or str(uuid.uuid4())
    exit_code = int(args.exit_code)

    # Ensure logs directory exists
    logs_dir = os.path.join("logs", "jobs")
    os.makedirs(logs_dir, exist_ok=True)

    ts = datetime.now(timezone.utc).isoformat()
    stdout_path = os.path.join(logs_dir, f"{job_id}.stdout.log")
    meta_path = os.path.join(logs_dir, f"{job_id}.meta.json")

    # Write small stdout content
    stdout_content = f"CI dummy run\njob_id={job_id}\ntimestamp={ts}\nexit_code={exit_code}\n"
    atomic_write(stdout_path, stdout_content)

    # Metadata JSON
    meta = {
        "job_id": job_id,
        "exit_code": exit_code,
        "start_ts": ts,
        "end_ts": ts,
        "stdout_path": stdout_path,
    }
    atomic_write(meta_path, json.dumps(meta, ensure_ascii=False, indent=2))

    # Print a stable JSON to stdout for CI
    out = {"job_id": job_id, "exit_code": exit_code, "timestamp": ts}
    print(json.dumps(out, ensure_ascii=False))

    return exit_code


if __name__ == "__main__":
    code = main()
    sys.exit(code)
