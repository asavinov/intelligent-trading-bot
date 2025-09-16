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

router = APIRouter(prefix="/scripts", tags=["scripts"])

# Global process manager - now stores dict with process info
active_processes: Dict[str, Dict[str, Any]] = {}

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
                # Prefer project_root/configs/<name> when it exists
                config_path = project_root / "configs" / req_cf
                if config_path.exists():
                    config_file_arg = f"configs/{req_cf}"
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
    
    # Setup environment
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    
    # Command construction - execute script directly with proper PYTHONPATH
    # Use direct script execution with proper CLI arguments
    cmd = ["python", str(script_path)]
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
        
        # Store process info - record the normalized config argument we passed to the script
        actual_config_record = config_file_arg or (config_to_use or "")
        active_processes[job_id] = {
            "process": process,
            "script": request.script_name,
            "config": actual_config_record,
            "start_time": time.time(),
            "status": "starting",
            "stdout": "",
            "stderr": ""
        }
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
        
        # Collect final output safely
        try:
            # Since we use PIPE, collect the output
            try:
                stdout, stderr = process.communicate(timeout=5)
                stdout_text = stdout.decode('utf-8', errors='ignore') if stdout else ""
                stderr_text = stderr.decode('utf-8', errors='ignore') if stderr else ""
                
                process_info["stdout"] = stdout_text
                process_info["stderr"] = stderr_text
                print(f"Monitor: Collected output for {job_id} - stdout: {len(stdout_text)} chars, stderr: {len(stderr_text)} chars")
                
            except subprocess.TimeoutExpired:
                print(f"Monitor: communicate() timed out for {job_id}")
                # If communicate times out, kill and move on
                try:
                    process.kill()
                except:
                    pass
                process_info["stdout"] = "Output collection timed out"
                process_info["stderr"] = "Process cleanup timeout"
            
            # Log detailed information for debugging
            if process.returncode != 0:
                print(f"Monitor: Script {job_id} failed with return code {process.returncode}")
                if process_info.get("stderr"):
                    print(f"Monitor: STDERR: {process_info['stderr'][:200]}...")
                
        except Exception as e:
            print(f"Monitor: Error collecting output for {job_id}: {e}")
            process_info["stdout"] = f"Output collection failed: {str(e)}"
            process_info["stderr"] = str(e)
        
        # Update status
        process_info["status"] = "completed" if process.returncode == 0 else "failed"
        process_info["end_time"] = time.time()
        process_info["returncode"] = process.returncode
        
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
    if job_id not in active_processes:
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
        "stderr": process_info.get("stderr", "")
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
        
        if poll_result is None:
            # Still running
            current_status = process_info.get("status", "running")
            active_jobs.append({
                "job_id": job_id,
                "script": process_info["script"],
                "config": process_info["config"],
                "pid": process.pid,
                "status": current_status,
                "start_time": process_info["start_time"],
                "cpu_percent": get_process_cpu_usage(process.pid),
                "memory_mb": get_process_memory_usage(process.pid),
                "stdout": process_info.get("stdout", "")[:500],  # Last 500 chars
                "stderr": process_info.get("stderr", "")[:500]   # Last 500 chars
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

@router.get("/logs/{job_id}/stream")
async def stream_script_logs(job_id: str):
    """استریم لاگ‌های زنده اسکریپت"""
    if job_id not in active_processes:
        raise HTTPException(status_code=404, detail="Job ID یافت نشد")
    
    async def generate_logs():
        process_info = active_processes[job_id]
        process = process_info["process"]
        
        while process.poll() is None:
            # Read available output
            if process.stdout:
                try:
                    line = process.stdout.readline()
                    if line:
                        yield f"data: {line.decode('utf-8', errors='ignore')}\n\n"
                except:
                    pass
            await asyncio.sleep(0.1)
        
        # Send final status
        yield f"data: [FINISHED] Process completed with code {process.returncode}\n\n"
    
    return StreamingResponse(generate_logs(), media_type="text/plain")


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
        print(f"API: Successfully killed process for job {job_id}")
        
        return {"status": "success", "message": f"Script {job_id} stopped"}
    except Exception as e:
        print(f"API: Error killing process for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop script: {str(e)}")
