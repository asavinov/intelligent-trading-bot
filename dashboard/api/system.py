"""
System Monitoring and Health API Endpoints
مانیتورینگ سیستم و تشخیص مشکلات
"""

from fastapi import APIRouter, HTTPException
from pathlib import Path
import psutil
import sys
import shutil
import subprocess
import json
import time
import requests
from typing import Dict, Any
import asyncio
from fastapi.responses import StreamingResponse
from ..core.config_validator import ConfigValidator

router = APIRouter(prefix="/system", tags=["system"])


def get_cpu_status():
    """دریافت وضعیت CPU"""
    return {
        "percent": psutil.cpu_percent(interval=1),
        "count": psutil.cpu_count(),
        "frequency": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
    }


def get_memory_status():
    """دریافت وضعیت حافظه"""
    memory = psutil.virtual_memory()
    return {
        "total_gb": round(memory.total / (1024**3), 1),
        "available_gb": round(memory.available / (1024**3), 1),
        "percent_used": memory.percent,
        "used_gb": round(memory.used / (1024**3), 1)
    }


def get_disk_status():
    """دریافت وضعیت دیسک"""
    try:
        disk = psutil.disk_usage('C:\\')
        return {
            "total_gb": round(disk.total / (1024**3), 1),
            "free_gb": round(disk.free / (1024**3), 1),
            "percent_used": round((disk.used / disk.total) * 100, 1)
        }
    except Exception:
        return {"error": "Unable to get disk info"}

@router.get("/health")
async def get_system_health():
    """بررسی سلامت کلی سیستم"""
    try:
        # System resources
        memory_info = get_memory_status()
        disk_info = get_disk_status()
        cpu_info = get_cpu_status()
        
        health_data = {
            "status": "healthy",
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "memory": memory_info,
            "disk": disk_info,
            "cpu_count": cpu_info["count"],
            "cpu_percent": cpu_info["percent"]
        }
        
        # Check if system is healthy
        warnings = []
        errors = []
        
        if "percent_used" in memory_info and memory_info["percent_used"] > 90:
            health_data["status"] = "warning"
            warnings.append("Memory usage is high")
        
        if "percent_used" in disk_info and disk_info["percent_used"] > 95:
            health_data["status"] = "warning"
            warnings.append("Disk space is low")
        
        if warnings:
            health_data["warnings"] = warnings
        if errors:
            health_data["errors"] = errors
        
        return health_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking system health: {str(e)}")

@router.get("/resources")
async def get_system_resources():
    """دریافت منابع سیستم به صورت real-time"""
    try:
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        disk = psutil.disk_usage('C:\\' if sys.platform == 'win32' else '/')
        
        return {
            "cpu": {
                "percent": cpu_percent,
                "healthy": cpu_percent < 80
            },
            "memory": {
                "percent": memory.percent,
                "total_gb": round(memory.total / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "healthy": memory.percent < 85
            },
            "disk": {
                "percent": round((disk.used / disk.total) * 100, 1),
                "total_gb": round(disk.total / (1024**3), 1),
                "free_gb": round(disk.free / (1024**3), 1),
                "healthy": (disk.used / disk.total) * 100 < 90
            },
            "timestamp": time.time()
        }
    except Exception as e:
        return {"error": str(e), "timestamp": time.time()}

def get_disk_status() -> Dict[str, Any]:
    """وضعیت دیسک"""
    try:
        disk = psutil.disk_usage('/')
        usage_percent = (disk.used / disk.total) * 100
        
        return {
            "healthy": usage_percent < 90,
            "usage_percent": round(usage_percent, 2),
            "total_gb": round(disk.total / (1024**3), 2),
            "used_gb": round(disk.used / (1024**3), 2),
            "free_gb": round(disk.free / (1024**3), 2),
            "status": "normal" if usage_percent < 80 else "high" if usage_percent < 90 else "critical"
        }
    except Exception as e:
        return {"healthy": False, "error": str(e)}

async def check_internet_connection() -> Dict[str, Any]:
    """بررسی اتصال اینترنت"""
    try:
        response = requests.get("https://api.binance.com/api/v3/ping", timeout=5)
        response_time = response.elapsed.total_seconds()
        
        return {
            "healthy": response.status_code == 200,
            "response_time_ms": round(response_time * 1000, 2),
            "status": "connected" if response.status_code == 200 else "disconnected"
        }
    except requests.exceptions.Timeout:
        return {"healthy": False, "error": "Timeout", "status": "timeout"}
    except Exception as e:
        return {"healthy": False, "error": str(e), "status": "error"}

async def check_binance_connection() -> Dict[str, Any]:
    """بررسی اتصال به Binance API"""
    try:
        # Check if we have saved API credentials
        project_root = Path(__file__).parent.parent.parent
        setup_file = project_root / "dashboard" / "setup_config.json"
        
        if not setup_file.exists():
            return {"healthy": False, "status": "not_configured", "message": "API تنظیم نشده"}
        
        # For now, just check basic connectivity
        response = requests.get("https://api.binance.com/api/v3/exchangeInfo", timeout=10)
        
        return {
            "healthy": response.status_code == 200,
            "status": "connected" if response.status_code == 200 else "error",
            "response_time_ms": round(response.elapsed.total_seconds() * 1000, 2)
        }
    except Exception as e:
        return {"healthy": False, "error": str(e), "status": "error"}

async def check_telegram_connection() -> Dict[str, Any]:
    """بررسی اتصال Telegram Bot"""
    try:
        project_root = Path(__file__).parent.parent.parent
        setup_file = project_root / "dashboard" / "setup_config.json"
        
        if not setup_file.exists():
            return {"healthy": False, "status": "not_configured", "message": "Telegram تنظیم نشده"}
        
        # Load config and test
        with open(setup_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        if "telegram" not in config:
            return {"healthy": False, "status": "not_configured", "message": "Telegram تنظیم نشده"}
        
        bot_token = config["telegram"].get("bot_token")
        if not bot_token:
            return {"healthy": False, "status": "not_configured", "message": "Bot token موجود نیست"}
        
        # Test bot info
        url = f"https://api.telegram.org/bot{bot_token}/getMe"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            bot_info = response.json()
            return {
                "healthy": True,
                "status": "connected",
                "bot_name": bot_info.get("result", {}).get("first_name", "Unknown")
            }
        else:
            return {"healthy": False, "status": "error", "error": "Invalid bot token"}
            
    except Exception as e:
        return {"healthy": False, "error": str(e), "status": "error"}

@router.get("/resources/stream")
async def stream_system_resources():
    """استریم real-time منابع سیستم"""
    
    async def resource_generator():
        while True:
            try:
                resource_data = {
                    "timestamp": time.time(),
                    "cpu": get_cpu_status(),
                    "memory": get_memory_status(),
                    "disk": get_disk_status()
                }
                
                yield f"data: {json.dumps(resource_data, ensure_ascii=False)}\n\n"
                await asyncio.sleep(2)
                
            except Exception as e:
                error_data = {
                    "type": "error", 
                    "message": str(e),
                    "timestamp": time.time()
                }
                yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
                break
    
    return StreamingResponse(
        resource_generator(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )

@router.get("/diagnostics")
async def run_system_diagnostics():
    """تشخیص خودکار مشکلات سیستم"""
    try:
        diagnostics = {
            "timestamp": time.time(),
            "checks": {
                "internet_connection": await check_internet_connection(),
                "binance_api": await check_binance_connection(),
                "telegram_bot": await check_telegram_connection(),
                "disk_space": get_disk_status(),
                "memory_usage": get_memory_status(),
                "cpu_usage": get_cpu_status(),
                "project_files": check_project_files()
            }
        }
        
        # Identify issues and suggest solutions
        issues = []
        solutions = []
        
        for check_name, result in diagnostics["checks"].items():
            if not result.get("healthy", True):
                issue = {
                    "component": check_name,
                    "severity": get_issue_severity(check_name, result),
                    "description": result.get("error", "مشکل نامشخص"),
                    "solution": get_solution_for_issue(check_name, result)
                }
                issues.append(issue)
        
        diagnostics["issues"] = issues
        diagnostics["overall_health"] = len(issues) == 0
        diagnostics["critical_issues"] = [i for i in issues if i["severity"] == "critical"]
        
        return diagnostics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در تشخیص سیستم: {str(e)}")

def check_project_files() -> Dict[str, Any]:
    """بررسی فایل‌های پروژه"""
    try:
        project_root = Path(__file__).parent.parent.parent
        
        required_dirs = ["scripts", "configs", "common", "service"]
        missing_dirs = []
        
        for dir_name in required_dirs:
            if not (project_root / dir_name).exists():
                missing_dirs.append(dir_name)
        
        return {
            "healthy": len(missing_dirs) == 0,
            "missing_directories": missing_dirs,
            "project_root": str(project_root)
        }
    except Exception as e:
        return {"healthy": False, "error": str(e)}

def get_issue_severity(component: str, result: Dict[str, Any]) -> str:
    """تعیین شدت مشکل"""
    critical_components = ["disk", "memory"]
    
    if component in critical_components:
        if "critical" in result.get("status", ""):
            return "critical"
        elif "high" in result.get("status", ""):
            return "warning"
    
    return "info"

def get_solution_for_issue(component: str, result: Dict[str, Any]) -> str:
    """ارائه راه‌حل برای مشکل"""
    solutions = {
        "internet_connection": "بررسی کنید که اتصال اینترنت فعال باشد",
        "binance_api": "API Key و Secret را در تنظیمات بررسی کنید",
        "telegram_bot": "Bot Token و Chat ID را در تنظیمات بررسی کنید",
        "disk": "فایل‌های غیرضروری را پاک کنید یا فضای دیسک را افزایش دهید",
        "memory": "برنامه‌های غیرضروری را ببندید یا RAM را افزایش دهید",
        "cpu": "منتظر بمانید تا پردازش‌های سنگین تمام شوند",
        "project_files": "بررسی کنید که تمام فایل‌های پروژه موجود باشند"
    }
    
    return solutions.get(component, "راه‌حل مشخصی موجود نیست")

@router.post("/auto-recovery")
async def auto_recovery():
    """بازیابی خودکار مشکلات"""
    try:
        recovery_actions = []
        
        # Check and fix common issues
        diagnostics = await run_system_diagnostics()
        
        for issue in diagnostics.get("issues", []):
            component = issue["component"]
            
            if component == "disk" and issue["severity"] == "critical":
                # Try to clean up temporary files
                cleanup_result = cleanup_temp_files()
                recovery_actions.append({
                    "action": "cleanup_temp_files",
                    "status": "completed" if cleanup_result else "failed",
                    "description": "پاکسازی فایل‌های موقت"
                })
        
        return {
            "recovery_actions": recovery_actions,
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در بازیابی خودکار: {str(e)}")

def cleanup_temp_files() -> bool:
    """پاکسازی فایل‌های موقت"""
    try:
        import tempfile
        import shutil
        
        temp_dir = Path(tempfile.gettempdir())
        
        # Clean up old temporary files (older than 1 day)
        current_time = time.time()
        cleaned_files = 0
        
        for temp_file in temp_dir.glob("*"):
            try:
                if temp_file.is_file():
                    file_age = current_time - temp_file.stat().st_mtime
                    if file_age > 86400:  # 1 day in seconds
                        temp_file.unlink()
                        cleaned_files += 1
            except:
                continue
        
        return cleaned_files > 0
        
    except Exception:
        return False


@router.get("/connections")
async def get_system_connections():
    """
    بررسی وضعیت اتصالات سیستم
    """
    try:
        connections = {
            "binance": check_binance_connection(),
            "mt5": check_mt5_connection(),
            "telegram": check_telegram_connection(),
            "database": check_database_connection(),
            "apis_status": check_apis_status()
        }
        
        # محاسبه وضعیت کلی
        total_connections = len(connections)
        active_connections = sum(1 for conn in connections.values() if conn.get("status") == "connected")
        
        overall_status = {
            "status": "healthy" if active_connections >= total_connections * 0.7 else "degraded",
            "active_connections": active_connections,
            "total_connections": total_connections,
            "connections": connections,
            "last_check": time.time()
        }
        
        return overall_status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking connections: {str(e)}")


def check_binance_connection():
    """بررسی اتصال به Binance API"""
    try:
        import requests
        response = requests.get("https://api.binance.com/api/v3/ping", timeout=5)
        if response.status_code == 200:
            return {
                "name": "Binance API",
                "status": "connected",
                "response_time": response.elapsed.total_seconds(),
                "last_check": time.time()
            }
    except Exception as e:
        pass
    
    return {
        "name": "Binance API", 
        "status": "disconnected",
        "error": "اتصال برقرار نشد",
        "last_check": time.time()
    }


def check_mt5_connection():
    """بررسی اتصال به MT5"""
    try:
        # بررسی ماژول MT5
        import MetaTrader5 as mt5
        
        # تست اتصال
        if mt5.initialize():
            account_info = mt5.account_info()
            mt5.shutdown()
            
            if account_info:
                return {
                    "name": "MetaTrader 5",
                    "status": "connected", 
                    "account": account_info.login if hasattr(account_info, 'login') else "Unknown",
                    "last_check": time.time()
                }
    except ImportError:
        return {
            "name": "MetaTrader 5",
            "status": "not_installed",
            "error": "MT5 Python package not installed",
            "last_check": time.time()
        }
    except Exception as e:
        pass
    
    return {
        "name": "MetaTrader 5",
        "status": "disconnected",
        "error": "اتصال برقرار نشد",
        "last_check": time.time()
    }


def check_telegram_connection():
    """بررسی اتصال به Telegram Bot"""
    try:
        # بررسی متغیرهای محیطی
        import os
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        
        if not bot_token:
            return {
                "name": "Telegram Bot",
                "status": "not_configured",
                "error": "Bot token not found",
                "last_check": time.time()
            }
        
        # تست اتصال به API
        import requests
        response = requests.get(f"https://api.telegram.org/bot{bot_token}/getMe", timeout=5)
        
        if response.status_code == 200:
            bot_info = response.json()
            if bot_info.get("ok"):
                return {
                    "name": "Telegram Bot",
                    "status": "connected",
                    "bot_username": bot_info.get("result", {}).get("username", "Unknown"),
                    "last_check": time.time()
                }
    except Exception as e:
        pass
    
    return {
        "name": "Telegram Bot",
        "status": "disconnected", 
        "error": "اتصال برقرار نشد",
        "last_check": time.time()
    }


def check_database_connection():
    """بررسی اتصال به دیتابیس"""
    try:
        # بررسی فایل دیتابیس SQLite
        db_path = Path(__file__).parent.parent / "dashboard.db"
        
        if db_path.exists():
            import sqlite3
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            conn.close()
            
            return {
                "name": "Local Database",
                "status": "connected",
                "type": "SQLite",
                "path": str(db_path),
                "last_check": time.time()
            }
    except Exception as e:
        pass
    
    return {
        "name": "Local Database",
        "status": "disconnected",
        "error": "اتصال برقرار نشد",
        "last_check": time.time()
    }


def check_apis_status():
    """بررسی وضعیت API های داخلی"""
    try:
        # بررسی endpoint های اصلی
        endpoints_to_check = [
            "/api/scripts/active",
            "/api/configs/list", 
            "/api/system/resources"
        ]
        
        working_endpoints = 0
        for endpoint in endpoints_to_check:
            try:
                import requests
                response = requests.get(f"http://127.0.0.1:8000{endpoint}", timeout=2)
                if response.status_code == 200:
                    working_endpoints += 1
            except:
                continue
        
        return {
            "name": "Internal APIs",
            "status": "connected" if working_endpoints == len(endpoints_to_check) else "partial",
            "working_endpoints": working_endpoints,
            "total_endpoints": len(endpoints_to_check),
            "last_check": time.time()
        }
    except Exception as e:
        pass
    
    return {
        "name": "Internal APIs",
        "status": "disconnected",
        "error": "بررسی نشد",
        "last_check": time.time()
    }
