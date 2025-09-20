"""
Safe Script Wrapper
اجرای ایمن اسکریپت‌های موجود
"""

import asyncio
import subprocess
import os
import time
import psutil
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import sys

class ScriptWrapper:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.active_processes: Dict[str, subprocess.Popen] = {}
        
    async def run_script_safe(self, script_name: str, config_path: str, 
                             timeout: int = 3600) -> Dict[str, Any]:
        """اجرای ایمن اسکریپت با error handling کامل"""
        
        # 1. Pre-execution validation
        script_path = self.project_root / "scripts" / f"{script_name}.py"
        if not script_path.exists():
            raise FileNotFoundError(f"Script {script_name} not found at {script_path}")
            
        try:
            script_path = self.project_root / "scripts" / f"{script_name}.py"
            
            if not script_path.exists():
                raise FileNotFoundError(f"اسکریپت {script_name} یافت نشد")
            
            # Validate config file
            config_file_path = Path(config_path)
            if not config_file_path.exists():
                # Try relative to project root
                config_file_path = self.project_root / config_path
                if not config_file_path.exists():
                    # Try in configs directory
                    config_file_path = self.project_root / "configs" / config_path
                    if not config_file_path.exists():
                        raise FileNotFoundError(f"فایل کانفیگ {config_path} یافت نشد")
            
            # Build command
            cmd = [
                sys.executable, str(script_path),
                "--config_file", str(config_file_path)
            ]
            
            # Set environment
            env = os.environ.copy()
            env["PYTHONPATH"] = str(self.project_root)
            
            # Start process
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=str(self.project_root)
            )
            
            # Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise TimeoutError(f"اسکریپت {script_name} بیش از {timeout} ثانیه طول کشید")
            
            return {
                "success": process.returncode == 0,
                "return_code": process.returncode,
                "stdout": stdout.decode('utf-8', errors='ignore'),
                "stderr": stderr.decode('utf-8', errors='ignore'),
                "script_name": script_name,
                "config_path": str(config_file_path)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "script_name": script_name,
                "config_path": config_path
            }
    
    def get_estimated_duration(self, script_name: str) -> str:
        """تخمین مدت زمان اجرا بر اساس نوع اسکریپت"""
        durations = {
            "download_binance": "2-5 دقیقه",
            "download_mt5": "3-8 دقیقه",
            "download_yahoo": "1-3 دقیقه",
            "merge": "1-2 دقیقه",
            "features": "5-15 دقیقه",
            "labels": "3-8 دقیقه", 
            "train": "30-120 دقیقه",
            "signals": "2-5 دقیقه",
            "predict": "1-3 دقیقه",
            "backtest": "10-30 دقیقه"
        }
        return durations.get(script_name, "نامشخص")
    
    async def get_script_status(self, job_id: str) -> Dict[str, Any]:
        """دریافت وضعیت اسکریپت"""
        if job_id not in self.active_processes:
            return {"job_id": job_id, "status": "not_found", "error": "Job ID not found"}
        
        process = self.active_processes[job_id]
        
        try:
            # Check if process is still running
            if process.returncode is None:
                # Process is still running
                return {
                    "job_id": job_id,
                    "status": "running",
                    "pid": process.pid,
                    "cpu_percent": self.get_process_cpu_usage(process.pid),
                    "memory_mb": self.get_process_memory_usage(process.pid),
                    "runtime_seconds": time.time() - getattr(process, 'start_time', time.time())
                }
            else:
                # Process completed
                status = "completed" if process.returncode == 0 else "failed"
                return {
                    "job_id": job_id,
                    "status": status,
                    "exit_code": process.returncode,
                    "pid": process.pid,
                    "runtime_seconds": time.time() - getattr(process, 'start_time', time.time())
                }
        except Exception as e:
            return {
                "job_id": job_id,
                "status": "error",
                "error": str(e)
            }
    
    def get_process_cpu_usage(self, pid: int) -> float:
        """CPU usage of process"""
        try:
            process = psutil.Process(pid)
            return process.cpu_percent()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0
    
    def get_process_memory_usage(self, pid: int) -> float:
        """Memory usage in MB"""
        try:
            process = psutil.Process(pid)
            return round(process.memory_info().rss / 1024 / 1024, 2)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0
    
    async def stop_script(self, job_id: str) -> Dict[str, Any]:
        """متوقف کردن اسکریپت"""
        if job_id not in self.active_processes:
            return {"job_id": job_id, "status": "not_found", "error": "Job ID not found"}
        
        process = self.active_processes[job_id]
        
        try:
            if process.returncode is None:
                # Graceful termination
                process.terminate()
                
                # Wait for graceful termination
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    # Force kill if doesn't terminate gracefully
                    process.kill()
                    await process.wait()
                
                return {
                    "job_id": job_id,
                    "status": "stopped",
                    "message": "اسکریپت متوقف شد"
                }
            else:
                return {
                    "job_id": job_id,
                    "status": "already_finished",
                    "message": "اسکریپت قبلاً تمام شده است"
                }
        except Exception as e:
            return {
                "job_id": job_id,
                "status": "error",
                "error": f"خطا در متوقف کردن: {str(e)}"
            }
    
    def get_active_scripts(self) -> List[Dict[str, Any]]:
        """لیست اسکریپت‌های در حال اجرا"""
        active_jobs = []
        
        for job_id, process in list(self.active_processes.items()):
            try:
                if process.returncode is None:
                    active_jobs.append({
                        "job_id": job_id,
                        "pid": process.pid,
                        "status": "running",
                        "cpu_percent": self.get_process_cpu_usage(process.pid),
                        "memory_mb": self.get_process_memory_usage(process.pid),
                        "runtime_seconds": time.time() - getattr(process, 'start_time', time.time())
                    })
                else:
                    # Remove completed processes from active list
                    del self.active_processes[job_id]
            except Exception:
                # Remove problematic processes
                if job_id in self.active_processes:
                    del self.active_processes[job_id]
        
        return active_jobs
    
    async def stream_script_logs(self, job_id: str):
        """استریم لاگ‌های اسکریپت"""
        if job_id not in self.active_processes:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Job ID یافت نشد'})}\n\n"
            return
        
        process = self.active_processes[job_id]
        
        # Send connection confirmation
        yield f"data: {json.dumps({'type': 'connected', 'message': 'اتصال برقرار شد', 'job_id': job_id})}\n\n"
        
        try:
            while process.returncode is None:
                # Read stdout
                if process.stdout:
                    try:
                        line = await asyncio.wait_for(process.stdout.readline(), timeout=0.1)
                        if line:
                            log_data = {
                                "type": "stdout",
                                "content": line.decode('utf-8', errors='ignore').strip(),
                                "timestamp": time.time(),
                                "job_id": job_id
                            }
                            yield f"data: {json.dumps(log_data, ensure_ascii=False)}\n\n"
                    except asyncio.TimeoutError:
                        pass
                
                # Read stderr
                if process.stderr:
                    try:
                        line = await asyncio.wait_for(process.stderr.readline(), timeout=0.1)
                        if line:
                            log_data = {
                                "type": "stderr",
                                "content": line.decode('utf-8', errors='ignore').strip(),
                                "timestamp": time.time(),
                                "job_id": job_id
                            }
                            yield f"data: {json.dumps(log_data, ensure_ascii=False)}\n\n"
                    except asyncio.TimeoutError:
                        pass
                
                await asyncio.sleep(0.1)
            
            # Process completed
            completion_data = {
                "type": "completed",
                "exit_code": process.returncode,
                "message": "اسکریپت با موفقیت اجرا شد" if process.returncode == 0 else "اسکریپت با خطا متوقف شد",
                "job_id": job_id,
                "timestamp": time.time()
            }
            yield f"data: {json.dumps(completion_data, ensure_ascii=False)}\n\n"
            
        except Exception as e:
            error_data = {
                "type": "error",
                "message": str(e),
                "job_id": job_id,
                "timestamp": time.time()
            }
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
    
    def validate_script_environment(self) -> Dict[str, Any]:
        """اعتبارسنجی محیط اجرای اسکریپت‌ها"""
        issues = []
        
        # Check if scripts directory exists
        scripts_dir = self.project_root / "scripts"
        if not scripts_dir.exists():
            issues.append("پوشه scripts یافت نشد")
        
        # Check if common modules exist
        common_dir = self.project_root / "common"
        if not common_dir.exists():
            issues.append("پوشه common یافت نشد")
        
        # Check if configs directory exists
        configs_dir = self.project_root / "configs"
        if not configs_dir.exists():
            issues.append("پوشه configs یافت نشد")
        
        # Check Python path
        if str(self.project_root) not in sys.path:
            issues.append("مسیر پروژه در PYTHONPATH نیست")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "project_root": str(self.project_root)
        }
