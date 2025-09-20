"""
Pipeline Orchestration API
/api/pipeline/* endpoints to run multiple scripts sequentially with progress streaming.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pathlib import Path
from typing import List, Optional, Dict, Any
import asyncio
import json
import os
import time

# Reuse the existing script runner endpoints internally
from .scripts import run_script, get_script_status, ScriptRunRequest  # type: ignore

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


class PipelineRunRequest(BaseModel):
    steps: Optional[List[str]] = Field(
        default=None,
        description="List of steps or script names. Defaults to a standard pipeline if omitted."
    )
    config_file: Optional[str] = Field(
        default=None,
        description="Path to config file (relative to project root or configs/)."
    )
    timeout_per_step: Optional[int] = Field(
        default=None,
        description="Optional timeout in seconds for each step."
    )
    extra_args: Optional[Dict[str, List[str]]] = Field(
        default=None,
        description="Optional mapping step->extra CLI args list"
    )


def _pipeline_dirs(project_root: Path):
    base = project_root / 'logs' / 'pipelines'
    base.mkdir(parents=True, exist_ok=True)
    return base


def _pipeline_paths(project_root: Path, pipeline_id: str):
    base = _pipeline_dirs(project_root)
    return {
        'log': base / f"{pipeline_id}.log",
        'meta': base / f"{pipeline_id}.meta.json",
    }


def _write_pipeline_meta_atomic(path: Path, meta: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix('.meta.tmp')
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _append_line(path: Path, line: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(line.rstrip('\n') + '\n')


def _normalize_steps(steps: Optional[List[str]]) -> List[str]:
    # Accept both abstract steps and direct script names
    default_steps = ['download', 'merge', 'features', 'labels']
    s = steps or default_steps
    return s


def _step_to_script(step: str) -> str:
    mapping = {
        'download': 'download_binance',
        'merge': 'merge',
        'features': 'features',
        'labels': 'labels',
        'train': 'train',
        'signals': 'signals',
        'predict': 'predict',
        'predict_rolling': 'predict_rolling',
        'output': 'output',
        'simulate': 'simulate',
    }
    # If user provided a known abstract step, map it; otherwise assume it's a direct script name
    return mapping.get(step, step)


@router.post('/run')
async def run_pipeline(req: PipelineRunRequest):
    project_root = Path(__file__).parent.parent.parent

    # Server-side feature gate: require env flag to be enabled
    raw = os.getenv("DASHBOARD_PIPELINE_ENABLED", "false")
    enabled = str(raw).strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        raise HTTPException(status_code=403, detail=(
            "Pipeline feature is disabled by server policy. Set DASHBOARD_PIPELINE_ENABLED=1 to enable."
        ))

    # Generate pipeline id
    import uuid
    pipeline_id = str(uuid.uuid4())
    paths = _pipeline_paths(project_root, pipeline_id)

    steps = _normalize_steps(req.steps)
    steps_resolved = [{
        'name': step,
        'script': _step_to_script(step),
        'status': 'pending',
        'job_id': None,
        'start_time': None,
        'end_time': None,
        'returncode': None,
    } for step in steps]

    meta: Dict[str, Any] = {
        'pipeline_id': pipeline_id,
        'created_at': time.time(),
        'status': 'starting',
        'config': req.config_file or '',
        'steps': steps_resolved,
    }
    _write_pipeline_meta_atomic(paths['meta'], meta)
    _append_line(paths['log'], f"--- PIPELINE START {pipeline_id} ---")

    # Background task to orchestrate steps
    async def _orchestrate():
        nonlocal meta
        try:
            for idx, step in enumerate(meta['steps']):
                step['status'] = 'starting'
                step['start_time'] = time.time()
                _append_line(paths['log'], f"[STEP {idx+1}/{len(meta['steps'])}] {step['name']} -> {step['script']} starting")
                _write_pipeline_meta_atomic(paths['meta'], meta)

                # Prepare run request for script
                extra = []
                if req.extra_args and step['name'] in req.extra_args:
                    extra = [str(x) for x in req.extra_args[step['name']]]

                sr = ScriptRunRequest(
                    script_name=step['script'],
                    config_file=req.config_file,
                    timeout=req.timeout_per_step,
                    extra_args=extra or None,
                )

                try:
                    result = await run_script(sr)  # returns dict with job_id, etc.
                except HTTPException as he:
                    # Immediate failure
                    step['status'] = 'failed'
                    step['end_time'] = time.time()
                    _append_line(paths['log'], f"[STEP {idx+1}] failed to start: {he.detail}")
                    meta['status'] = 'failed'
                    _write_pipeline_meta_atomic(paths['meta'], meta)
                    _append_line(paths['log'], f"--- PIPELINE FAILED {pipeline_id} ---")
                    return

                job_id = result.get('job_id')
                step['job_id'] = job_id
                step['status'] = 'running'
                _append_line(paths['log'], f"[STEP {idx+1}] job started id={job_id}")
                _write_pipeline_meta_atomic(paths['meta'], meta)

                # Poll job status until completion
                while True:
                    try:
                        status_obj = await get_script_status(job_id)  # type: ignore
                    except HTTPException as he:
                        # If temporarily unavailable, wait and retry
                        await asyncio.sleep(1)
                        continue
                    st = (status_obj.get('status') or '').lower()
                    if st == 'running':
                        await asyncio.sleep(1)
                        continue
                    # finished
                    step['end_time'] = time.time()
                    step['returncode'] = status_obj.get('returncode')
                    step['status'] = 'completed' if st == 'completed' else 'failed'
                    _append_line(paths['log'], f"[STEP {idx+1}] finished status={step['status']} returncode={step['returncode']}")
                    _write_pipeline_meta_atomic(paths['meta'], meta)
                    if step['status'] != 'completed':
                        meta['status'] = 'failed'
                        _append_line(paths['log'], f"--- PIPELINE FAILED at step {idx+1}: {step['name']} ---")
                        _write_pipeline_meta_atomic(paths['meta'], meta)
                        return
                    break

            # All steps completed
            meta['status'] = 'completed'
            meta['ended_at'] = time.time()
            _append_line(paths['log'], f"--- PIPELINE COMPLETED {pipeline_id} ---")
            _write_pipeline_meta_atomic(paths['meta'], meta)
        except Exception as e:
            meta['status'] = 'error'
            meta['ended_at'] = time.time()
            _append_line(paths['log'], f"[ERROR] pipeline exception: {e}")
            _append_line(paths['log'], f"--- PIPELINE ERROR {pipeline_id} ---")
            try:
                _write_pipeline_meta_atomic(paths['meta'], meta)
            except Exception:
                pass

    # Fire and forget orchestration
    asyncio.create_task(_orchestrate())

    return {
        'pipeline_id': pipeline_id,
        'status': 'started',
        'steps': [s['name'] for s in steps_resolved],
        'config': req.config_file or None,
    }


@router.get('/status/{pipeline_id}')
async def get_pipeline_status(pipeline_id: str):
    project_root = Path(__file__).parent.parent.parent
    paths = _pipeline_paths(project_root, pipeline_id)
    if not paths['meta'].exists():
        raise HTTPException(status_code=404, detail='Pipeline not found')
    try:
        with open(paths['meta'], 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to read pipeline meta: {e}')


@router.get('/stream/{pipeline_id}')
async def stream_pipeline_logs(pipeline_id: str):
    project_root = Path(__file__).parent.parent.parent
    paths = _pipeline_paths(project_root, pipeline_id)
    log_path = paths['log']
    if not log_path.exists():
        # If not yet created, wait a bit then stream
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.touch(exist_ok=True)
        except Exception:
            pass

    async def generate():
        last_pos = 0
        # Start at end to stream new data
        try:
            last_pos = os.path.getsize(log_path)
        except Exception:
            last_pos = 0

        while True:
            try:
                size = os.path.getsize(log_path)
                if size > last_pos:
                    with open(log_path, 'rb') as f:
                        f.seek(last_pos)
                        data = f.read()
                        last_pos = f.tell()
                        try:
                            text = data.decode('utf-8', errors='ignore')
                        except Exception:
                            text = ''
                        for line in text.splitlines(True):
                            yield f"data: {line}\n\n"
                # Exit when pipeline reports completed/failed/error
                meta_path = paths['meta']
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text(encoding='utf-8'))
                        if (meta.get('status') or '') in ('completed', 'failed', 'error'):
                            # Flush any remaining
                            size2 = os.path.getsize(log_path)
                            if size2 > last_pos:
                                with open(log_path, 'rb') as f:
                                    f.seek(last_pos)
                                    data = f.read()
                                    try:
                                        text = data.decode('utf-8', errors='ignore')
                                    except Exception:
                                        text = ''
                                    for line in text.splitlines(True):
                                        yield f"data: {line}\n\n"
                            yield f"data: [FINISHED] pipeline {pipeline_id} status={meta.get('status')}\n\n"
                            break
                    except Exception:
                        pass
            except Exception as e:
                yield f"data: [ERROR] {e}\n\n"
                break
            await asyncio.sleep(0.3)

    return StreamingResponse(generate(), media_type='text/event-stream')
