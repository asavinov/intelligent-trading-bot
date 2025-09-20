"""Build a simple CI runs summary from logs/jobs metadata files.

This creates tools/ci_artifacts/ci_ui_runs_summary.json with an array of objects:
{ job_id, status, returncode, stdout_snippet }
"""
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
logs_dir = project_root / 'logs' / 'jobs'
outdir = project_root / 'tools' / 'ci_artifacts'
outdir.mkdir(parents=True, exist_ok=True)

results = []
for p in sorted(logs_dir.glob('*.meta.json')):
    try:
        data = json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        continue
    job_id = data.get('job_id') or p.stem
    # Normalize return code
    rc = data.get('returncode')
    if rc is None:
        rc = data.get('exit_code')
    try:
        rc_int = int(rc) if rc is not None else None
    except Exception:
        rc_int = None
    status = data.get('status') or ('completed' if (rc_int is None or rc_int == 0) else 'failed')
    returncode = rc_int
    stdout_path = data.get('log_stdout') or data.get('stdout_path')
    env_snapshot_path = data.get('env_snapshot_path')
    stdout_snippet = ''
    if stdout_path:
        sp = Path(stdout_path)
        if not sp.is_absolute():
            sp = project_root / sp
        if sp.exists():
            try:
                txt = sp.read_text(encoding='utf-8', errors='ignore')
                lines = txt.splitlines()
                stdout_snippet = '\n'.join(lines[:5])
            except Exception:
                stdout_snippet = ''

    row = {
        'job_id': job_id,
        'status': status,
        'returncode': returncode,
        'stdout_snippet': stdout_snippet
    }
    if env_snapshot_path:
        row['env_snapshot_path'] = env_snapshot_path
    results.append(row)

(outdir / 'ci_ui_runs_summary.json').write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
print('Wrote', outdir / 'ci_ui_runs_summary.json')