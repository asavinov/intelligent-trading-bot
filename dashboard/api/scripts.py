"""
Script Management API Endpoints
مدیریت و اجرای اسکریپت‌های ربات
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pathlib import Path
import subprocess
import asyncio
import uuid
import time
import sys
import os
import json
from typing import Dict, Any, Optional, List
from ..core.config_validator import ConfigValidator
import psutil
import uuid
from typing import BinaryIO
from tempfile import NamedTemporaryFile
import io
import zipfile
import platform

router = APIRouter(prefix="/scripts", tags=["scripts"])

# Global process manager - now stores dict with process info
active_processes: Dict[str, Dict[str, Any]] = {}


def _meta_path_for_job(project_root: Path, job_id: str) -> Path:
    logs_dir = project_root / 'logs' / 'jobs'
    return logs_dir / f"{job_id}.meta.json"


def _write_job_metadata_atomic(project_root: Path, job_id: str):
    """Write a compact metadata JSON for the job atomically to disk.

    Uses a temporary file and os.replace to avoid partially written metadata files.
    """
    try:
        if job_id not in active_processes:
            return
        info = active_processes[job_id]
        meta = {
            "job_id": job_id,
            "script": info.get("script"),
            "config": info.get("config"),
            "pid": (info.get("process").pid if info.get("process") else None),
            "status": info.get("status"),
            "start_time": info.get("start_time"),
            "end_time": info.get("end_time"),
            "returncode": info.get("returncode"),
            "log_stdout": info.get("log_stdout"),
            "log_stderr": info.get("log_stderr"),
            "env_snapshot_path": info.get("env_snapshot_path")
        }
        meta_path = _meta_path_for_job(project_root, job_id)
        # Ensure containing dir exists
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        # Write atomically
        with NamedTemporaryFile('w', encoding='utf-8', delete=False, dir=str(meta_path.parent)) as tf:
            tf.write(json.dumps(meta, ensure_ascii=False, indent=2))
            tmp = tf.name
        os.replace(tmp, str(meta_path))
    except Exception as e:
        # Avoid raising - metadata is best-effort
        print(f"_write_job_metadata_atomic: failed for {job_id}: {e}")


def _setup_log_rotation(log_file: Path, max_size_mb: int = 10, backup_count: int = 3):
    """Setup log rotation for a single log file.
    
    Args:
        log_file: Path to the log file
        max_size_mb: Maximum size in MB before rotation
        backup_count: Number of backup files to keep
    """
    try:
        if not log_file.exists():
            return
            
        max_size_bytes = max_size_mb * 1024 * 1024
        current_size = log_file.stat().st_size
        
        if current_size <= max_size_bytes:
            return
            
        # Rotate existing files
        for i in range(backup_count - 1, 0, -1):
            old_file = log_file.with_suffix(f'.{i}.log')
            new_file = log_file.with_suffix(f'.{i + 1}.log')
            if old_file.exists():
                if i == backup_count - 1:
                    old_file.unlink()  # Delete oldest
                else:
                    old_file.rename(new_file)
        
        # Move current log to .1.log
        log_file.rename(log_file.with_suffix('.1.log'))
        
        # Create new empty log file
        log_file.touch()
        
    except Exception as e:
        print(f"Log rotation failed for {log_file}: {e}")


def _cleanup_old_job_logs(logs_dir: Path, max_jobs: int = None, max_total_mb: int = None):
    """Keep logs/jobs tidy by removing oldest jobs beyond limits.

    - max_jobs: keep at most this many jobs (triples: meta/stdout/stderr).
    - max_total_mb: keep total size under this many MB by deleting oldest.
    
    Values can be overridden via environment variables:
    - ITB_LOGS_MAX_JOBS: maximum number of jobs to keep
    - ITB_LOGS_MAX_TOTAL_MB: maximum total size in MB
    """
    # Use environment variables if available, otherwise use defaults
    if max_jobs is None:
        max_jobs = int(os.getenv('ITB_LOGS_MAX_JOBS', '200'))
    if max_total_mb is None:
        max_total_mb = int(os.getenv('ITB_LOGS_MAX_TOTAL_MB', '500'))
    try:
        if not logs_dir.exists():
            return
        # Group files by job_id (prefix before extension)
        items: Dict[str, Dict[str, Any]] = {}
        for p in logs_dir.iterdir():
            if not p.is_file():
                continue
            name = p.name
            if name.count('.') < 2:
                # expect patterns like {job_id}.stdout.log / .stderr.log / .meta.json
                # but accept any two-dot filenames as belonging to a job_id
                pass
            job_id = name.split('.')[0]
            info = items.setdefault(job_id, {"files": [], "start_time": None, "mtime": 0, "size": 0})
            info["files"].append(p)
            try:
                st = p.stat()
                info["mtime"] = max(info["mtime"], st.st_mtime)
                info["size"] += st.st_size
            except Exception:
                pass
            if name.endswith('.meta.json'):
                try:
                    data = json.loads(p.read_text(encoding='utf-8'))
                    st = float(data.get('start_time') or data.get('start_ts') or 0)
                    if st:
                        info['start_time'] = st
                except Exception:
                    pass

        # Build list and sort by start_time (fallback to mtime)
        rows = []
        total_size = 0
        for jid, info in items.items():
            ts = info.get('start_time') or info.get('mtime') or 0
            sz = info.get('size') or 0
            total_size += sz
            rows.append((ts, jid, sz, info['files']))
        rows.sort(key=lambda x: x[0])  # oldest first

        max_total_bytes = max_total_mb * 1024 * 1024
        # Delete oldest until under count limit
        while len(rows) > max_jobs:
            _, jid, sz, files = rows.pop(0)
            for f in files:
                try:
                    f.unlink(missing_ok=True)
                except Exception:
                    pass
            total_size -= sz

        # Delete oldest until under size limit
        while total_size > max_total_bytes and rows:
            _, jid, sz, files = rows.pop(0)
            for f in files:
                try:
                    f.unlink(missing_ok=True)
                except Exception:
                    pass
            total_size -= sz
    except Exception as e:
        print(f"cleanup logs/jobs failed: {e}")


def _test_log_rotation_and_cleanup():
    """Unit tests for log rotation and cleanup functionality.
    
    This function can be called during development or testing to verify
    that log rotation and cleanup work correctly.
    """
    import tempfile
    import shutil
    from datetime import datetime
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        logs_dir = temp_path / 'logs' / 'jobs'
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        print("Testing log rotation and cleanup...")
        
        # Test 1: Log rotation
        test_log = logs_dir / 'test.stdout.log'
        test_log.write_text('test content' * 1000)  # Make it larger than 10MB threshold
        
        _setup_log_rotation(test_log, max_size_mb=0.001, backup_count=2)  # Very small threshold for testing
        
        if test_log.exists():
            print("✓ Log rotation test passed")
        else:
            print("✗ Log rotation test failed")
            
        # Test 2: Cleanup with many files
        for i in range(10):
            job_id = f'test-job-{i}'
            (logs_dir / f'{job_id}.stdout.log').write_text(f'stdout content {i}')
            (logs_dir / f'{job_id}.stderr.log').write_text(f'stderr content {i}')
            (logs_dir / f'{job_id}.meta.json').write_text(json.dumps({
                'job_id': job_id,
                'start_time': time.time() - i * 3600,  # Stagger start times
                'status': 'completed'
            }))
        
        # Run cleanup with small limits
        _cleanup_old_job_logs(logs_dir, max_jobs=5, max_total_mb=1)
        
        remaining_files = list(logs_dir.glob('*'))
        if len(remaining_files) <= 15:  # 5 jobs * 3 files each
            print("✓ Cleanup test passed")
        else:
            print(f"✗ Cleanup test failed - {len(remaining_files)} files remaining")
            
        print("Log rotation and cleanup tests completed")

class ScriptRunRequest(BaseModel):
    script_name: str
    config_file: Optional[str] = None
    # Timeout in seconds for the user-requested job (optional).
    # If provided, monitor_process will use this value as max wait time.
    timeout: Optional[int] = None
    # Optional additional CLI args to pass to the script, e.g. ["--verbose"] or ["--start_time","2024-01-01"]
    extra_args: Optional[List[str]] = None

@router.post("/run")
async def run_script(request: ScriptRunRequest):
    """اجرای اسکریپت در محیط کاملاً جداگانه"""
    project_root = Path(__file__).parent.parent.parent
    
    # Determine which config to use: request value wins; otherwise try setup default
    config_to_use = None
    if getattr(request, 'config_file', None):
        config_to_use = request.config_file
    else:
        # Try to read saved setup default (support both possible locations)
        try:
            setup_path = project_root / 'setup' / 'dashboard_setup.json'
            if not setup_path.exists():
                # fallback to older location
                setup_path = project_root / 'dashboard' / 'setup_config.json'
            if setup_path.exists():
                with open(setup_path, 'r', encoding='utf-8') as sf:
                    try:
                        setup_json = json.load(sf)
                        sel = setup_json.get('selected_config') or setup_json.get('default_config')
                        if sel:
                            config_to_use = sel
                    except Exception:
                        pass
        except Exception:
            pass

    # Handle special scripts that are not in scripts directory
    if request.script_name == "service.server":
        script_path = project_root / "service" / "server.py"
    else:
        script_path = project_root / "scripts" / f"{request.script_name}.py"
    
    # Handle config file path - scripts expect relative path from project root
    config_file_arg = None
    if config_to_use:
        # If absolute path provided
        if Path(config_to_use).is_absolute():
            config_path = Path(config_to_use)
            # Convert to relative path for script
            try:
                config_file_arg = str(config_path.relative_to(project_root))
            except ValueError:
                config_file_arg = str(config_path)
        else:
            # Normalize config file argument to avoid double 'configs/configs/..' when
            # caller already passed a path prefixed with 'configs/' or similar.
            req_cf = str(config_to_use)

            # If caller already included the configs/ prefix, use as-is
            if req_cf.startswith("configs/") or req_cf.startswith("configs\\"):
                config_file_arg = req_cf
            else:
                # Prefer project_root/configs/<name> when it exists. Support cases where
                # callers provide a logical name (without extension) by searching the
                # configs directory for a matching stem.
                configs_dir = project_root / "configs"
                config_path = configs_dir / req_cf
                if config_path.exists():
                    config_file_arg = f"configs/{req_cf}"
                else:
                    # Try to resolve by stem (e.g. 'config-sample-1min' -> 'config-sample-1min.jsonc')
                    found = None
                    try:
                        if configs_dir.exists():
                            for p in configs_dir.iterdir():
                                if p.is_file() and p.stem == req_cf:
                                    found = p
                                    break
                    except Exception:
                        found = None

                    if found:
                        # Use the relative path under configs/
                        config_file_arg = f"configs/{found.name}"
                    else:
                        # Fallback to project_root/<name> if present
                        alt_path = project_root / req_cf
                        if alt_path.exists():
                            config_file_arg = req_cf
                        else:
                            # Last-resort: pass through what caller provided (script will resolve or fail)
                            config_file_arg = req_cf
    
    if not script_path.exists():
        raise HTTPException(status_code=404, detail=f"اسکریپت {request.script_name} یافت نشد")
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())

    # --- Server-side validation: ensure the resolved config file actually exists when provided ---
    # This avoids launching a job that will immediately fail with FileNotFoundError
    if config_file_arg:
        try:
            candidates = []
            cf = str(config_file_arg)
            # Absolute path candidate
            p = Path(cf)
            if p.is_absolute():
                candidates.append(p)
            else:
                # project-relative candidate
                candidates.append(project_root / cf)
                # candidate under configs/
                candidates.append(project_root / 'configs' / cf)
                # try common extensions if no suffix provided
                if Path(cf).suffix == '':
                    candidates.append(project_root / 'configs' / (cf + '.jsonc'))
                    candidates.append(project_root / 'configs' / (cf + '.json'))

            found = False
            for cand in candidates:
                try:
                    if cand.exists():
                        found = True
                        break
                except Exception:
                    # ignore any permission/path errors when probing
                    continue

            if not found:
                tried = [str(x) for x in candidates]
                raise HTTPException(status_code=400, detail=(
                    f"Config file قابل یافتن نیست: {config_file_arg}. \n"
                    f"مسیرهای امتحان‌شده: {tried}. \n"
                    "لطفاً یک config معتبر انتخاب کنید یا مسیر کامل فایل را وارد نمایید."
                ))
        except HTTPException:
            raise
        except Exception:
            # If something unexpected happens during validation, don't block execution;
            # fall through and let the job attempt to run (script may still resolve config differently).
            pass
    
    # Setup environment
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    
    # Command construction - execute script with the same interpreter in unbuffered mode
    # Use sys.executable to ensure we run with the current Python and -u to disable buffering
    cmd = [sys.executable, "-u", str(script_path)]
    # pass config using -c to match CLI scripts' click option
    if config_file_arg:
        cmd += ["-c", config_file_arg]

    # Append any additional args provided by the caller
    if getattr(request, 'extra_args', None):
        # ensure it's a list of strings
        extra = [str(x) for x in request.extra_args]
        cmd += extra
    
    # اجرای اسکریپت در محیط کاملاً ایزوله
    try:
        # Ensure child Python runs unbuffered so prints flush promptly
        env["PYTHONUNBUFFERED"] = "1"

        if sys.platform == "win32":
            # در ویندوز بجای batch file، مستقیماً با env vars اجرا کنیم
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(project_root),
                env=env,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                shell=False
            )
        else:
            # Unix/Linux systems
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=str(project_root),
                preexec_fn=os.setsid if hasattr(os, 'setsid') else None
            )

        # Prepare per-job log files so we persist output even if the server restarts
        logs_dir = Path(project_root) / 'logs' / 'jobs'
        logs_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = str(logs_dir / f"{job_id}.stdout.log")
        stderr_path = str(logs_dir / f"{job_id}.stderr.log")

        # Determine the normalized config value we'll record everywhere (UI/API consistency)
        actual_config_record = config_file_arg or (config_to_use or "")

        # Write a compact environment snapshot for this job
        try:
            env_snapshot_path = str(logs_dir / f"{job_id}.env.json")
            env_sample = {
                "timestamp": time.time(),
                "python_version": sys.version,
                "platform": {
                    "system": platform.system(),
                    "release": platform.release(),
                    "version": platform.version(),
                    "machine": platform.machine(),
                    "python_implementation": platform.python_implementation(),
                },
                "cwd": str(project_root),
                "script": request.script_name,
                "config": actual_config_record,
                "env": {
                    k: env.get(k) for k in [
                        "PYTHONPATH", "VIRTUAL_ENV", "CONDA_PREFIX", "PATH", "PROCESSOR_ARCHITECTURE"
                    ] if env.get(k) is not None
                }
            }
            with open(env_snapshot_path, 'w', encoding='utf-8') as ef:
                json.dump(env_sample, ef, ensure_ascii=False, indent=2)
        except Exception as e:
            env_snapshot_path = None
            print(f"env snapshot failed for {job_id}: {e}")

        # Store process info - record the normalized config argument we passed to the script
        active_processes[job_id] = {
            "process": process,
            "script": request.script_name,
            "config": actual_config_record,
            "start_time": time.time(),
            "status": "starting",
            "stdout": "",
            "stderr": "",
            "log_stdout": stdout_path,
            "log_stderr": stderr_path,
            "env_snapshot_path": env_snapshot_path
        }

        # Persist initial metadata
        try:
            _write_job_metadata_atomic(project_root, job_id)
        except Exception:
            pass

        # Opportunistic cleanup of old logs/jobs
        try:
            max_jobs = int(os.environ.get('ITB_LOGS_MAX_JOBS', '200'))
            max_total_mb = int(os.environ.get('ITB_LOGS_MAX_TOTAL_MB', '500'))
            _cleanup_old_job_logs(logs_dir, max_jobs=max_jobs, max_total_mb=max_total_mb)
        except Exception:
            pass

        # Start background tasks to stream subprocess stdout/stderr to files
        async def _stream_reader_to_file(reader, path: str, job_id: str, kind: str):
            """Read from an asyncio StreamReader and append bytes to a file asynchronously."""
            loop = asyncio.get_running_loop()
            try:
                # Use a blocking chunked read in an executor thread for cross-platform reliability.
                # Some Windows file objects don't behave well with readline/asyncio; reading raw chunks
                # (read) is more robust.
                with open(path, 'ab') as fh:
                    while True:
                        try:
                            chunk = await loop.run_in_executor(None, reader.read, 4096)
                        except Exception as inner_e:
                            if job_id in active_processes:
                                active_processes[job_id][f'{kind}_stream_error'] = str(inner_e)
                            break

                        if not chunk:
                            break

                        # write chunk (bytes) via executor to avoid blocking the loop
                        await loop.run_in_executor(None, fh.write, chunk)
                        await loop.run_in_executor(None, fh.flush)
            except Exception as e:
                # Record streaming error in process_info for debugging
                if job_id in active_processes:
                    active_processes[job_id][f'{kind}_stream_error'] = str(e)

        # kick off streaming tasks (don't await)
        # Write an initial marker into each log to indicate the writer started
        try:
            with open(stdout_path, 'ab') as f:
                f.write(f"---JOB START {job_id} STDOUT---\n".encode('utf-8'))
        except Exception:
            pass
        try:
            with open(stderr_path, 'ab') as f:
                f.write(f"---JOB START {job_id} STDERR---\n".encode('utf-8'))
        except Exception:
            pass

        if process.stdout:
            asyncio.create_task(_stream_reader_to_file(process.stdout, stdout_path, job_id, 'stdout'))
        if process.stderr:
            asyncio.create_task(_stream_reader_to_file(process.stderr, stderr_path, job_id, 'stderr'))
        # Log which config we decided to use (helps debugging)
        print(f"Script job {job_id} started for {request.script_name} using config: {actual_config_record}")

        # Quick startup validation - wait a bit to catch immediate failures
        await asyncio.sleep(0.5)
        if process.poll() is not None:
            # Process already terminated - likely startup error
            try:
                stdout, stderr = process.communicate(timeout=5)
                stdout_text = stdout.decode('utf-8', errors='ignore') if stdout else ""
                stderr_text = stderr.decode('utf-8', errors='ignore') if stderr else ""
                
                active_processes[job_id].update({
                    "status": "failed",
                    "stdout": stdout_text,
                    "stderr": stderr_text,
                    "returncode": process.returncode,
                    "end_time": time.time()
                })
                # update metadata
                try:
                    _write_job_metadata_atomic(project_root, job_id)
                except Exception:
                    pass
                
                error_msg = f"اسکریپت بلافاصله متوقف شد - کد خطا: {process.returncode}"
                if stderr_text:
                    error_msg += f"\nخطا: {stderr_text[:200]}..."
                
                raise HTTPException(status_code=500, detail=error_msg)

            except subprocess.TimeoutExpired:
                process.kill()
                active_processes[job_id]["status"] = "failed"
                raise HTTPException(status_code=500, detail="اسکریپت در startup timeout شد")

        # If we get here, process started successfully
        active_processes[job_id]["status"] = "running"

        # Start background monitoring, pass the requested timeout (seconds) if provided
        asyncio.create_task(monitor_process(job_id, request.timeout))

        # Return the actual config used (normalized) so clients see the same value
        return {
            "job_id": job_id,
            "status": "started",
            "script": request.script_name,
            "config": actual_config_record if actual_config_record else None,
            "pid": process.pid,
            "estimated_duration": get_estimated_duration(request.script_name)
        }

    except Exception as e:
        import traceback
        error_detail = f"خطا در اجرای اسکریپت: {str(e)}\n{traceback.format_exc()}"
        print(f"Script execution error: {error_detail}")  # Log to console
        raise HTTPException(status_code=500, detail=f"خطا در اجرای اسکریپت: {str(e)}")


async def monitor_process(job_id: str, user_timeout: Optional[int] = None):
    """Monitor process execution in background"""
    try:
        if job_id not in active_processes:
            print(f"Monitor: Job {job_id} not found in active_processes")
            return
            
        process_info = active_processes[job_id]
        process = process_info["process"]
        
        print(f"Monitor: Starting monitoring for {job_id} (PID: {process.pid})")

    # Wait for process to complete with timeout protection
        # Default max wait time is 30 minutes, but honor the user's requested timeout if provided
        default_max = 30 * 60  # 30 minutes
        max_wait_time = user_timeout if (user_timeout and user_timeout > 0) else default_max
        wait_start = time.time()
        
        while process.poll() is None:
            await asyncio.sleep(1)
            
            # Safety timeout - if process runs too long, something might be wrong
            if time.time() - wait_start > max_wait_time:
                print(f"Monitor: Process {job_id} exceeded max wait time, marking as timeout")
                try:
                    process.kill()
                except:
                    pass
                process_info["status"] = "failed"
                process_info["stderr"] = "Process timed out after 30 minutes"
                process_info["end_time"] = time.time()
                return
        
        print(f"Monitor: Process {job_id} completed with return code: {process.returncode}")

        # Read tail of per-job log files (if present) to populate process_info for API responses
        try:
            stdout_path = process_info.get('log_stdout')
            stderr_path = process_info.get('log_stderr')
            def _tail_file(path, max_bytes=2000):
                if not path:
                    return ""
                try:
                    with open(path, 'rb') as f:
                        f.seek(0, os.SEEK_END)
                        size = f.tell()
                        start = max(0, size - max_bytes)
                        f.seek(start)
                        data = f.read()
                        try:
                            return data.decode('utf-8', errors='ignore')
                        except Exception:
                            return '<binary data>'
                except Exception as e:
                    return f"<could not read log: {e}>"

            stdout_text = _tail_file(stdout_path, max_bytes=4000)
            stderr_text = _tail_file(stderr_path, max_bytes=4000)

            process_info['stdout'] = stdout_text
            process_info['stderr'] = stderr_text
            print(f"Monitor: Collected tail logs for {job_id} - stdout: {len(stdout_text)} chars, stderr: {len(stderr_text)} chars")

        except Exception as e:
            print(f"Monitor: Error collecting tail logs for {job_id}: {e}")
            process_info['stdout'] = process_info.get('stdout', '')
            process_info['stderr'] = process_info.get('stderr', '')
        
        # Update status
        process_info["status"] = "completed" if process.returncode == 0 else "failed"
        process_info["end_time"] = time.time()
        process_info["returncode"] = process.returncode

        # Persist metadata after completion
        try:
            # derive project_root
            project_root = Path(__file__).parent.parent.parent
            _write_job_metadata_atomic(project_root, job_id)
        except Exception:
            pass

        # Cleanup pass after completion as well (enforces retention)
        try:
            logs_dir = project_root / 'logs' / 'jobs'
            max_jobs = int(os.environ.get('ITB_LOGS_MAX_JOBS', '200'))
            max_total_mb = int(os.environ.get('ITB_LOGS_MAX_TOTAL_MB', '500'))
            _cleanup_old_job_logs(logs_dir, max_jobs=max_jobs, max_total_mb=max_total_mb)
        except Exception:
            pass

        print(f"Monitor: Updated status for {job_id}: {process_info['status']}")

    except Exception as e:
        print(f"Monitor: Error monitoring process {job_id}: {e}")
        if job_id in active_processes:
            active_processes[job_id]["status"] = "error"
            active_processes[job_id]["stderr"] = str(e)
            active_processes[job_id]["end_time"] = time.time()


@router.get("/status/{job_id}")
async def get_script_status(job_id: str):
    """دریافت وضعیت اجرای اسکریپت"""
    # First, prefer in-memory active process info (jobs started via this API)
    if job_id not in active_processes:
        # Fallback: if the job has been created by an external runner (CI helper)
        # and only wrote metadata files under logs/jobs, attempt to return that
        # metadata so clients (and pollers) can query status for file-backed jobs.
        try:
            project_root = Path(__file__).parent.parent.parent
            meta_path = project_root / 'logs' / 'jobs' / f"{job_id}.meta.json"
            if meta_path.exists():
                try:
                    with open(meta_path, 'r', encoding='utf-8') as mf:
                        data = json.load(mf)
                except Exception:
                    raise HTTPException(status_code=500, detail="خطا در خواندن metadata محلی")

                # Try to read stdout snippet if available
                stdout_text = ''
                stdout_path = data.get('log_stdout') or data.get('stdout_path') or ''
                if stdout_path:
                    try:
                        sp = Path(stdout_path)
                        if not sp.is_absolute():
                            sp = project_root / str(sp)
                        if sp.exists():
                            with open(sp, 'r', encoding='utf-8', errors='ignore') as sf:
                                txt = sf.read()
                                # keep only a small snippet
                                stdout_text = txt if len(txt) < 4000 else txt[-4000:]
                    except Exception:
                        stdout_text = ''

                response = {
                    'job_id': data.get('job_id', job_id),
                    'status': data.get('status') or ('completed' if data.get('exit_code') in (0, '0', None) else 'failed'),
                    'script': data.get('script'),
                    'config': data.get('config'),
                    'start_time': data.get('start_ts') or data.get('start_time'),
                    'pid': data.get('pid'),
                    'returncode': data.get('exit_code') or data.get('returncode'),
                    'stdout': stdout_text,
                    'stderr': '',
                    'env_snapshot_path': data.get('env_snapshot_path')
                }
                if data.get('end_ts') or data.get('end_time'):
                    response['end_time'] = data.get('end_ts') or data.get('end_time')
                    try:
                        response['duration'] = float(response['end_time']) - float(response.get('start_time') or 0)
                    except Exception:
                        pass

                return response

        except HTTPException:
            raise
        except Exception:
            # If fallback fails, continue to raise not found below
            pass

        raise HTTPException(status_code=404, detail="Job ID یافت نشد")

    process_info = active_processes[job_id]
    process = process_info["process"]
    
    # Check if process is still running
    poll_result = process.poll()
    if poll_result is None:
        status = "running"
        returncode = None
    else:
        # Process has finished, collect output if not already done
        if "stdout" not in process_info or process_info["stdout"] == "":
            try:
                stdout, stderr = process.communicate(timeout=1)
                process_info["stdout"] = stdout.decode('utf-8', errors='ignore')
                process_info["stderr"] = stderr.decode('utf-8', errors='ignore')
                process_info["end_time"] = time.time()
            except:
                pass
        status = "completed" if poll_result == 0 else "failed"
        returncode = poll_result
    
    response = {
        "job_id": job_id,
        "status": status,
        "script": process_info["script"],
        "config": process_info["config"],
        "start_time": process_info["start_time"],
        "pid": process.pid,
        "returncode": returncode,
        "stdout": process_info.get("stdout", ""),
        "stderr": process_info.get("stderr", ""),
        "env_snapshot_path": process_info.get("env_snapshot_path")
    }
    
    if "end_time" in process_info:
        response["end_time"] = process_info["end_time"]
        response["duration"] = process_info["end_time"] - process_info["start_time"]
    
    return response


@router.get("/active")
async def list_active_scripts():
    """لیست اسکریپت‌های در حال اجرا و خطاها"""
    active_jobs = []
    jobs_to_remove = []
    
    for job_id, process_info in active_processes.items():
        process = process_info["process"]
        
        # Check current process status
        poll_result = process.poll()
        
        # Helper to tail a log file safely
        def _tail_file_safe(path, max_chars=500):
            if not path:
                return ''
            try:
                if not os.path.exists(path):
                    return ''
                with open(path, 'rb') as f:
                    f.seek(max(0, os.path.getsize(path) - max_chars))
                    data = f.read()
                    return data.decode('utf-8', errors='ignore')
            except Exception:
                return ''

        if poll_result is None:
            # Still running
            current_status = process_info.get("status", "running")
            # Prefer small tail from the per-job stdout/stderr files (background readers write there)
            so_tail = _tail_file_safe(process_info.get('log_stdout'), max_chars=800)
            se_tail = _tail_file_safe(process_info.get('log_stderr'), max_chars=800)

            # If no file content yet, fall back to in-memory stored fields
            if not so_tail:
                so_tail = (process_info.get("stdout") or '')[:800]
            if not se_tail:
                se_tail = (process_info.get("stderr") or '')[:800]

            active_jobs.append({
                "job_id": job_id,
                "script": process_info["script"],
                "config": process_info["config"],
                "pid": process.pid,
                "status": current_status,
                "start_time": process_info["start_time"],
                "cpu_percent": get_process_cpu_usage(process.pid),
                "memory_mb": get_process_memory_usage(process.pid),
                "stdout": so_tail,
                "stderr": se_tail
            })
        else:
            # Process finished - update status if not already done by monitor_process
            if "final_processed" not in process_info:
                print(f"Active: Process {job_id} finished but not processed by monitor, handling now")
                try:
                    # Try to collect output if available
                    if hasattr(process, 'stdout') and process.stdout:
                        try:
                            stdout, stderr = process.communicate(timeout=2)
                            process_info["stdout"] = stdout.decode('utf-8', errors='ignore') if stdout else ""
                            process_info["stderr"] = stderr.decode('utf-8', errors='ignore') if stderr else ""
                        except subprocess.TimeoutExpired:
                            print(f"Active: communicate() timed out for finished process {job_id}")
                            process_info["stdout"] = "Output collection timed out"
                            process_info["stderr"] = "Process finished but output unavailable"
                        except Exception as e:
                            print(f"Active: Error collecting output for {job_id}: {e}")
                            process_info["stdout"] = ""
                            process_info["stderr"] = str(e)
                    else:
                        process_info["stdout"] = f"Process completed with return code: {poll_result}"
                        process_info["stderr"] = ""
                    
                    process_info["returncode"] = poll_result
                    process_info["end_time"] = time.time()
                    process_info["status"] = "completed" if poll_result == 0 else "failed"
                    process_info["final_processed"] = True
                    
                    print(f"Active: Updated process {job_id} status to: {process_info['status']}")
                    
                except Exception as e:
                    print(f"Active: Error processing finished job {job_id}: {e}")
                    process_info["status"] = "error"
                    process_info["stderr"] = str(e)
                    process_info["final_processed"] = True
            
            # Show failed/completed jobs for a while before cleanup
            if process_info.get("status") in ["failed", "error"] or poll_result != 0:
                active_jobs.append({
                    "job_id": job_id,
                    "script": process_info["script"],
                    "config": process_info["config"],
                    "pid": process.pid,
                    "status": "failed",
                    "start_time": process_info["start_time"],
                    "end_time": process_info.get("end_time", time.time()),
                    "returncode": poll_result,
                    "stdout": process_info.get("stdout", "")[:500],
                    "stderr": process_info.get("stderr", "")[:500],
                    "duration": process_info.get("end_time", time.time()) - process_info["start_time"]
                })
                
                # Mark for cleanup after 30 seconds
                if time.time() - process_info.get("end_time", 0) > 30:
                    jobs_to_remove.append(job_id)
            else:
                # Successful completion - show briefly then cleanup
                active_jobs.append({
                    "job_id": job_id,
                    "script": process_info["script"],
                    "config": process_info["config"],
                    "pid": process.pid,
                    "status": "completed",
                    "start_time": process_info["start_time"],
                    "end_time": process_info.get("end_time", time.time()),
                    "returncode": poll_result,
                    "stdout": process_info.get("stdout", "")[:200],
                    "stderr": process_info.get("stderr", "")[:200],
                    "duration": process_info.get("end_time", time.time()) - process_info["start_time"]
                })
                
                # Mark for cleanup after 10 seconds for successful jobs
                if time.time() - process_info.get("end_time", 0) > 10:
                    jobs_to_remove.append(job_id)
    
    # Cleanup old jobs
    for job_id in jobs_to_remove:
        if job_id in active_processes:
            del active_processes[job_id]
    
    return active_jobs  # Return array directly, not wrapped in object


@router.get("/list")
async def list_available_scripts():
    """لیست اسکریپت‌های موجود"""
    try:
        scripts_dir = Path(__file__).parent.parent.parent / "scripts"
        available_scripts = []
        
        # Define script descriptions and categories
        script_info = {
            "download_binance": {
                "description": "دانلود داده‌های تاریخی از Binance",
                "category": "data_collection",
                "duration": "2-5 دقیقه",
                "requires_api": True
            },
            "download_mt5": {
                "description": "دانلود داده‌های تاریخی از MT5",
                "category": "data_collection", 
                "duration": "2-5 دقیقه",
                "requires_api": True
            },
            "download_yahoo": {
                "description": "دانلود داده‌های تاریخی از Yahoo Finance",
                "category": "data_collection",
                "duration": "1-2 دقیقه",
                "requires_api": False
            },
            "merge": {
                "description": "ادغام و پردازش داده‌های خام",
                "category": "data_processing",
                "duration": "5-10 دقیقه",
                "requires_api": False
            },
            "features": {
                "description": "استخراج ویژگی‌های تکنیکال",
                "category": "feature_engineering",
                "duration": "10-20 دقیقه",
                "requires_api": False
            },
            "labels": {
                "description": "تولید برچسب‌های آموزشی",
                "category": "feature_engineering",
                "duration": "5-15 دقیقه",
                "requires_api": False
            },
            "train": {
                "description": "آموزش مدل‌های یادگیری ماشین",
                "category": "model_training",
                "duration": "30-120 دقیقه",
                "requires_api": False
            },
            "signals": {
                "description": "تولید سیگنال‌های معاملاتی",
                "category": "trading",
                "duration": "2-5 دقیقه",
                "requires_api": False
            },
            "predict": {
                "description": "پیش‌بینی قیمت‌های آینده",
                "category": "prediction",
                "duration": "1-3 دقیقه",
                "requires_api": False
            },
            "predict_rolling": {
                "description": "پیش‌بینی با بازآموزی دوره‌ای مدل‌ها",
                "category": "prediction",
                "duration": "60-300 دقیقه",
                "requires_api": False
            },
            "output": {
                "description": "اسکریپت output",
                "category": "output",
                "duration": "1-2 دقیقه",
                "requires_api": False
            },
            "simulate": {
                "description": "شبیه‌سازی معاملات و بک‌تست",
                "category": "backtesting",
                "duration": "10-30 دقیقه",
                "requires_api": False
            }
        }
        
        if scripts_dir.exists():
            for script_file in scripts_dir.glob("*.py"):
                if script_file.name != "__init__.py":
                    script_name = script_file.stem
                    info = script_info.get(script_name, {
                        "description": f"اسکریپت {script_name}",
                        "category": "other",
                        "duration": "نامشخص",
                        "requires_api": False
                    })
                    
                    available_scripts.append({
                        "name": script_name,
                        "file": script_file.name,
                        "path": str(script_file),
                        **info
                    })

        # Add service script from service directory
        service_dir = Path(__file__).parent.parent.parent / "service"
        service_file = service_dir / "server.py"
        if service_file.exists():
            available_scripts.append({
                "name": "service.server",
                "file": "server.py",
                "path": str(service_file),
                "description": "سرویس تولید سیگنال‌های آنلاین",
                "category": "service",
                "duration": "دائمی (سرویس)",
                "requires_api": True
            })
        
        return {"scripts": available_scripts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در دریافت لیست اسکریپت‌ها: {str(e)}")

def get_estimated_duration(script_name: str) -> str:
    """تخمین مدت زمان اجرای اسکریپت"""
    durations = {
        "download_binance": "2-5 دقیقه",
        "download_mt5": "2-5 دقیقه", 
        "download_yahoo": "1-2 دقیقه",
        "merge": "5-10 دقیقه",
        "features": "10-20 دقیقه",
        "labels": "5-15 دقیقه",
        "train": "30-120 دقیقه",
        "signals": "2-5 دقیقه",
        "predict": "1-3 دقیقه",
        "predict_rolling": "60-300 دقیقه",
        "output": "1-2 دقیقه",
        "simulate": "10-30 دقیقه",
        "service.server": "دائمی (سرویس)"
    }
    return durations.get(script_name, "نامشخص")

def get_process_cpu_usage(pid: int) -> float:
    """دریافت میزان استفاده CPU پروسه"""
    try:
        process = psutil.Process(pid)
        return process.cpu_percent()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.0

def get_process_memory_usage(pid: int) -> float:
    """دریافت میزان استفاده حافظه پروسه (MB)"""
    try:
        process = psutil.Process(pid)
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.0


@router.get('/history')
async def get_jobs_history(status: Optional[str] = None, script: Optional[str] = None, limit: int = 50, offset: int = 0):
    """لیست history jobها با استفاده از metadata اتمیک ذخیره‌شده.

    Query params:
    - status: فیلتر بر اساس status (running, completed, failed, stopped)
    - script: نام اسکریپت برای فیلتر (مثلاً download_binance)
    - limit: تعداد آیتم بازگشتی (پیش‌فرض 50)
    - offset: جا به جایی برای pagination
    """
    project_root = Path(__file__).parent.parent.parent
    logs_dir = project_root / 'logs' / 'jobs'
    results = []

    if not logs_dir.exists():
        return {"total": 0, "jobs": []}

    try:
        for p in logs_dir.glob('*.meta.json'):
            try:
                with open(p, 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
                    # normalize keys we expect
                    entry = {
                        'job_id': data.get('job_id'),
                        'script': data.get('script'),
                        'config': data.get('config'),
                        'pid': data.get('pid'),
                        'status': data.get('status'),
                        'start_time': data.get('start_time'),
                        'end_time': data.get('end_time'),
                        'returncode': data.get('returncode'),
                        'log_stdout': data.get('log_stdout'),
                        'log_stderr': data.get('log_stderr'),
                        'env_snapshot_path': data.get('env_snapshot_path')
                    }
                    results.append(entry)
            except Exception:
                # ignore malformed metadata files
                continue

        # Apply filters
        if status:
            results = [r for r in results if (r.get('status') or '').lower() == status.lower()]
        if script:
            results = [r for r in results if (r.get('script') or '') == script]

        # Sort by start_time desc (fallback to job_id if missing)
        def _sort_key(r):
            try:
                return -(float(r.get('start_time') or 0))
            except Exception:
                return 0

        results.sort(key=_sort_key)

        total = len(results)
        # Pagination
        sliced = results[offset: offset + limit]

        return {"total": total, "jobs": sliced}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در خواندن history: {e}")

@router.get("/logs/{job_id}/stream")
async def stream_script_logs(job_id: str):
    """استریم لاگ‌های زنده اسکریپت"""
    if job_id not in active_processes:
        raise HTTPException(status_code=404, detail="Job ID یافت نشد")
    # Stream by tailing the per-job stdout log file that background readers write to.
    # This avoids reading from the process pipe directly (which the background
    # reader tasks already consume) and works cross-platform.
    process_info = active_processes[job_id]
    stdout_path = process_info.get('log_stdout')

    async def generate_logs():
        last_pos = 0
        # If file exists, start at its current end to stream new data
        if stdout_path and os.path.exists(stdout_path):
            try:
                last_pos = os.path.getsize(stdout_path)
            except Exception:
                last_pos = 0

        while True:
            try:
                if stdout_path and os.path.exists(stdout_path):
                    size = os.path.getsize(stdout_path)
                    if size > last_pos:
                        with open(stdout_path, 'rb') as f:
                            f.seek(last_pos)
                            data = f.read()
                            last_pos = f.tell()
                            try:
                                text = data.decode('utf-8', errors='ignore')
                            except Exception:
                                text = '<binary data>'
                            # Yield each line as an SSE-like message
                            for line in text.splitlines(True):
                                yield f"data: {line}\n\n"

                proc = process_info.get('process')
                # If process finished, flush remaining content and exit
                if proc and proc.poll() is not None:
                    # read any remaining bytes
                    if stdout_path and os.path.exists(stdout_path):
                        size = os.path.getsize(stdout_path)
                        if size > last_pos:
                            with open(stdout_path, 'rb') as f:
                                f.seek(last_pos)
                                data = f.read()
                                try:
                                    text = data.decode('utf-8', errors='ignore')
                                except Exception:
                                    text = '<binary data>'
                                for line in text.splitlines(True):
                                    yield f"data: {line}\n\n"
                    yield f"data: [FINISHED] Process completed with code {proc.returncode}\n\n"
                    break

            except Exception as e:
                yield f"data: [ERROR] {str(e)}\n\n"
                break

            await asyncio.sleep(0.25)

    return StreamingResponse(generate_logs(), media_type="text/event-stream")


@router.get('/debug/active_processes')
async def debug_list_active_processes():
    """Debug endpoint: return a compact summary of active_processes for inspection."""
    try:
        out = {}
        for jid, info in active_processes.items():
            out[jid] = {
                'script': info.get('script'),
                'status': info.get('status'),
                'pid': info.get('process').pid if info.get('process') else None,
                'start_time': info.get('start_time'),
                'returncode': info.get('returncode', None),
                'stdout_len': None,
                'stderr_len': None,
                'stream_errors': {k: v for k, v in info.items() if k.endswith('_stream_error')}
            }
            try:
                so = info.get('log_stdout')
                se = info.get('log_stderr')
                if so and os.path.exists(so):
                    out[jid]['stdout_len'] = os.path.getsize(so)
                if se and os.path.exists(se):
                    out[jid]['stderr_len'] = os.path.getsize(se)
            except Exception:
                pass
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/debug/job/{job_id}')
async def debug_get_job(job_id: str):
    """Debug endpoint: return full stored process info for a specific job (helpful for debugging)."""
    if job_id not in active_processes:
        raise HTTPException(status_code=404, detail='Job ID not found')
    info = active_processes[job_id]
    # Avoid returning the raw process object (not JSON serializable) - replace with pid and repr
    proc = info.get('process')
    safe = dict(info)
    safe['process'] = {'pid': proc.pid if proc else None, 'repr': repr(proc) if proc else None}
    # If log files exist, include small tail previews
    try:
        so = info.get('log_stdout')
        se = info.get('log_stderr')
        def tail(path, max_chars=2000):
            if not path or not os.path.exists(path):
                return ''
            try:
                with open(path, 'rb') as f:
                    f.seek(max(0, os.path.getsize(path) - max_chars))
                    return f.read().decode('utf-8', errors='ignore')
            except Exception as e:
                return f'<could not read: {e}>'

        safe['log_stdout_tail'] = tail(so, 2000)
        safe['log_stderr_tail'] = tail(se, 2000)
    except Exception:
        pass

    return safe


@router.delete("/stop/{job_id}")
async def stop_script(job_id: str):
    """
    توقف اسکریپت در حال اجرا
    """
    print(f"API: Request to stop script {job_id}")
    
    if job_id not in active_processes:
        print(f"API: Job {job_id} not found in active processes")
        raise HTTPException(status_code=404, detail="Script not found")
    
    process_info = active_processes[job_id]
    process = process_info.get("process")
    
    if not process:
        print(f"API: No process found for job {job_id}")
        raise HTTPException(status_code=404, detail="Process not found")
    
    try:
        print(f"API: Attempting to kill process for job {job_id}")
        process.kill()
        process_info["status"] = "stopped"
        process_info["end_time"] = time.time()
        # Persist metadata for stopped job
        try:
            project_root = Path(__file__).parent.parent.parent
            _write_job_metadata_atomic(project_root, job_id)
        except Exception:
            pass
        print(f"API: Successfully killed process for job {job_id}")
        
        return {"status": "success", "message": f"Script {job_id} stopped"}
    except Exception as e:
        print(f"API: Error killing process for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop script: {str(e)}")


@router.get('/logs/{job_id}/download')
@router.post("/test-log-rotation")
async def test_log_rotation():
    """Test endpoint for log rotation and cleanup functionality.
    
    This endpoint can be used to verify that log rotation and cleanup
    work correctly in the current environment.
    """
    try:
        _test_log_rotation_and_cleanup()
        return {"success": True, "message": "Log rotation and cleanup tests completed"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/logs/{job_id}/download")
async def download_job_logs(job_id: str):
    """Download a zip archive containing stdout, stderr and meta for a job (if present)."""
    project_root = Path(__file__).parent.parent.parent
    logs_dir = project_root / 'logs' / 'jobs'
    stdout_path = logs_dir / f"{job_id}.stdout.log"
    stderr_path = logs_dir / f"{job_id}.stderr.log"
    meta_path = logs_dir / f"{job_id}.meta.json"
    env_path = logs_dir / f"{job_id}.env.json"

    # Ensure at least one file exists
    files_exist = any(p.exists() for p in [stdout_path, stderr_path, meta_path, env_path])
    if not files_exist:
        raise HTTPException(status_code=404, detail="No logs found for the given job_id")

    buf = io.BytesIO()
    try:
        with zipfile.ZipFile(buf, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
            if stdout_path.exists():
                zf.write(stdout_path, arcname=f"{job_id}.stdout.log")
            if stderr_path.exists():
                zf.write(stderr_path, arcname=f"{job_id}.stderr.log")
            if meta_path.exists():
                zf.write(meta_path, arcname=f"{job_id}.meta.json")
            if env_path.exists():
                zf.write(env_path, arcname=f"{job_id}.env.json")
        buf.seek(0)
        headers = {
            'Content-Disposition': f'attachment; filename="{job_id}-logs.zip"'
        }
        return StreamingResponse(buf, media_type='application/zip', headers=headers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build zip archive: {e}")


@router.get('/env/{job_id}')
async def get_job_env(job_id: str):
    """Return the env snapshot JSON for a job if present."""
    try:
        project_root = Path(__file__).parent.parent.parent
        env_path = project_root / 'logs' / 'jobs' / f"{job_id}.env.json"
        if not env_path.exists():
            raise HTTPException(status_code=404, detail='Env snapshot not found')
        try:
            data = json.loads(env_path.read_text(encoding='utf-8'))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Failed to read env snapshot: {e}')
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
