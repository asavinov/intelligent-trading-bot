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
from typing import Dict, Any, Optional
from ..core.config_validator import ConfigValidator
import psutil
import uuid

router = APIRouter(prefix="/scripts", tags=["scripts"])

# Global process manager - now stores dict with process info
active_processes: Dict[str, Dict[str, Any]] = {}

class ScriptRunRequest(BaseModel):
    script_name: str
    config_file: str
    timeout: Optional[int] = 3600

@router.post("/run")
async def run_script(request: ScriptRunRequest):
    """اجرای اسکریپت در محیط کاملاً جداگانه"""
    project_root = Path(__file__).parent.parent.parent
    script_path = project_root / "scripts" / f"{request.script_name}.py"
    
    # Handle config file path - scripts expect relative path from project root
    if Path(request.config_file).is_absolute():
        config_path = Path(request.config_file)
        # Convert to relative path for script
        try:
            config_file_arg = str(config_path.relative_to(project_root))
        except ValueError:
            config_file_arg = str(config_path)
    else:
        # Try different locations
        config_path = project_root / "configs" / request.config_file
        if not config_path.exists():
            config_path = project_root / request.config_file
        # Use relative path for script
        config_file_arg = f"configs/{request.config_file}"
    
    if not script_path.exists():
        raise HTTPException(status_code=404, detail=f"اسکریپت {request.script_name} یافت نشد")
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Setup environment
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    
    # Command construction - use relative config path for script
    # Use Python with sys.path modification for proper imports
    python_code = f"import sys; sys.path.insert(0, r'{project_root}'); from scripts.{request.script_name} import main; main.callback(config_file=r'{config_file_arg}')"
    cmd = ["python", "-c", python_code]
    
    # اجرای اسکریپت در محیط کاملاً ایزوله
    try:
        if sys.platform == "win32":
            # در ویندوز از PowerShell برای اجرای کاملاً جداگانه استفاده می‌کنیم
            cmd_str = " ".join([f'"{c}"' if " " in c else c for c in cmd])
            
            # Create a batch file for isolated execution
            batch_content = f"""@echo off
cd /d "{project_root}"
set PYTHONPATH={project_root};%PYTHONPATH%
{cmd_str}
"""
            batch_file = project_root / f"temp_script_{job_id}.bat"
            with open(batch_file, 'w', encoding='utf-8') as f:
                f.write(batch_content)
            
            # Run batch file in completely detached process
            process = subprocess.Popen(
                [str(batch_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(project_root),
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
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
        
        # Store process info
        active_processes[job_id] = {
            "process": process,
            "script": request.script_name,
            "config": request.config_file,
            "start_time": time.time(),
            "status": "running",
            "stdout": "",
            "stderr": "",
            "batch_file": batch_file if sys.platform == "win32" else None
        }
        
        # Start background monitoring
        asyncio.create_task(monitor_process(job_id))
        
        return {
            "job_id": job_id,
            "status": "started",
            "script": request.script_name,
            "config": request.config_file,
            "pid": process.pid,
            "estimated_duration": get_estimated_duration(request.script_name)
        }
        
    except Exception as e:
        import traceback
        error_detail = f"خطا در اجرای اسکریپت: {str(e)}\n{traceback.format_exc()}"
        print(f"Script execution error: {error_detail}")  # Log to console
        raise HTTPException(status_code=500, detail=f"خطا در اجرای اسکریپت: {str(e)}")


async def monitor_process(job_id: str):
    """Monitor process execution in background"""
    try:
        if job_id not in active_processes:
            return
            
        process_info = active_processes[job_id]
        process = process_info["process"]
        
        # Wait for process to complete
        while process.poll() is None:
            await asyncio.sleep(1)
        
        # Collect final output
        try:
            stdout, stderr = process.communicate(timeout=5)
            process_info["stdout"] = stdout.decode('utf-8', errors='ignore')
            process_info["stderr"] = stderr.decode('utf-8', errors='ignore')
        except subprocess.TimeoutExpired:
            process_info["stdout"] = "Process timed out during output collection"
            process_info["stderr"] = "Timeout error"
        
        # Update status
        process_info["status"] = "completed" if process.returncode == 0 else "failed"
        process_info["end_time"] = time.time()
        process_info["returncode"] = process.returncode
        
        # Clean up batch file if exists
        if process_info.get("batch_file") and Path(process_info["batch_file"]).exists():
            try:
                os.remove(process_info["batch_file"])
            except:
                pass
                
    except Exception as e:
        print(f"Error monitoring process {job_id}: {e}")
        if job_id in active_processes:
            active_processes[job_id]["status"] = "error"
            active_processes[job_id]["stderr"] = str(e)


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
    """لیست اسکریپت‌های در حال اجرا"""
    active_jobs = []
    
    for job_id, process_info in active_processes.items():
        process = process_info["process"]
        if process.poll() is None:
            active_jobs.append({
                "job_id": job_id,
                "script": process_info["script"],
                "config": process_info["config"],
                "pid": process.pid,
                "status": "running",
                "start_time": process_info["start_time"],
                "cpu_percent": get_process_cpu_usage(process.pid),
                "memory_mb": get_process_memory_usage(process.pid)
            })
    
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
        "predict": "1-3 دقیقه"
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

@router.post("/run")
async def run_script(request: ScriptRunRequest):
    """اجرای اسکریپت در محیط کاملاً جداگانه"""
    project_root = Path(__file__).parent.parent.parent
    script_path = project_root / "scripts" / f"{request.script_name}.py"
    
    # Handle config file path - scripts expect relative path from project root
    if Path(request.config_file).is_absolute():
        config_path = Path(request.config_file)
        # Convert to relative path for script
        try:
            config_file_arg = str(config_path.relative_to(project_root))
        except ValueError:
            config_file_arg = str(config_path)
    else:
        # Try different locations
        config_path = project_root / "configs" / request.config_file
        if not config_path.exists():
            config_path = project_root / request.config_file
        # Use relative path for script
        config_file_arg = f"configs/{request.config_file}"
    
    if not script_path.exists():
        raise HTTPException(status_code=404, detail=f"اسکریپت {request.script_name} یافت نشد")
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Setup environment
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    
    # Command construction - use relative config path for script
    # Use Python with sys.path modification for proper imports
    python_code = f"import sys; sys.path.insert(0, r'{project_root}'); from scripts.{request.script_name} import main; main.callback(config_file=r'{config_file_arg}')"
    cmd = ["python", "-c", python_code]
    
    # اجرای اسکریپت در محیط کاملاً ایزوله
    try:
        if sys.platform == "win32":
            # در ویندوز از PowerShell برای اجرای کاملاً جداگانه استفاده می‌کنیم
            cmd_str = " ".join([f'"{c}"' if " " in c else c for c in cmd])
            
            # Create a batch file for isolated execution
            batch_content = f"""@echo off
cd /d "{project_root}"
set PYTHONPATH={project_root};%PYTHONPATH%
{cmd_str}
"""
            batch_file = project_root / f"temp_script_{job_id}.bat"
            with open(batch_file, 'w', encoding='utf-8') as f:
                f.write(batch_content)
            
            # Run batch file in completely detached process
            process = subprocess.Popen(
                [str(batch_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(project_root),
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
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
        
        # Store process info
        active_processes[job_id] = {
            "process": process,
            "script": request.script_name,
            "config": request.config_file,
            "start_time": time.time(),
            "status": "running",
            "stdout": "",
            "stderr": "",
            "batch_file": batch_file if sys.platform == "win32" else None
        }
        
        # Start background monitoring
        asyncio.create_task(monitor_process(job_id))
        
        return {
            "job_id": job_id,
            "status": "started",
            "script": request.script_name,
            "config": request.config_file,
            "pid": process.pid,
            "estimated_duration": get_estimated_duration(request.script_name)
        }
        
    except Exception as e:
        import traceback
        error_detail = f"خطا در اجرای اسکریپت: {str(e)}\n{traceback.format_exc()}"
        print(f"Script execution error: {error_detail}")  # Log to console
        raise HTTPException(status_code=500, detail=f"خطا در اجرای اسکریپت: {str(e)}")

async def monitor_process(job_id: str):
    """Monitor process execution in background"""
    try:
        if job_id not in active_processes:
            return
            
        process_info = active_processes[job_id]
        process = process_info["process"]
        
        # Wait for process to complete
        while process.poll() is None:
            await asyncio.sleep(1)
        
        # Collect final output
        try:
            stdout, stderr = process.communicate(timeout=5)
            process_info["stdout"] = stdout.decode('utf-8', errors='ignore')
            process_info["stderr"] = stderr.decode('utf-8', errors='ignore')
        except subprocess.TimeoutExpired:
            process_info["stdout"] = "Process timed out during output collection"
            process_info["stderr"] = "Timeout error"
        
        # Update status
        process_info["status"] = "completed" if process.returncode == 0 else "failed"
        process_info["end_time"] = time.time()
        process_info["returncode"] = process.returncode
        
        # Clean up batch file if exists
        if process_info.get("batch_file") and Path(process_info["batch_file"]).exists():
            try:
                os.remove(process_info["batch_file"])
            except:
                pass
                
    except Exception as e:
        print(f"Error monitoring process {job_id}: {e}")
        if job_id in active_processes:
            active_processes[job_id]["status"] = "error"
            active_processes[job_id]["stderr"] = str(e)

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

@router.delete("/stop/{job_id}")
async def stop_script(job_id: str):
    """متوقف کردن اجرای اسکریپت"""
    if job_id not in active_processes:
        raise HTTPException(status_code=404, detail="Job ID یافت نشد")
    
    try:
        process_info = active_processes[job_id]
        process = process_info["process"]
        
        if process.poll() is None:  # Process is still running
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            
            process_info["status"] = "stopped"
            process_info["end_time"] = time.time()
            
            return {"message": "اسکریپت با موفقیت متوقف شد"}
        else:
            return {
                "message": "اسکریپت قبلاً تمام شده است"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در متوقف کردن اسکریپت: {str(e)}")

@router.get("/active")
async def list_active_scripts():
    """لیست اسکریپت‌های در حال اجرا"""
    active_jobs = []
    
    for job_id, process_info in active_processes.items():
        process = process_info["process"]
        if process.poll() is None:
            active_jobs.append({
                "job_id": job_id,
                "script": process_info["script"],
                "config": process_info["config"],
                "pid": process.pid,
                "status": "running",
                "start_time": process_info["start_time"],
                "cpu_percent": get_process_cpu_usage(process.pid),
                "memory_mb": get_process_memory_usage(process.pid)
            })
    
    return {"active_scripts": active_jobs}

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
