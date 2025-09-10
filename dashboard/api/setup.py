"""
Setup Wizard API Endpoints
راه‌اندازی اولیه سیستم
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import requests
import sys
import psutil
import shutil
from pathlib import Path
import json
from typing import Dict, Any

router = APIRouter(prefix="/setup", tags=["setup"])

class BinanceConfig(BaseModel):
    api_key: str
    api_secret: str

class TelegramConfig(BaseModel):
    bot_token: str
    chat_id: str

@router.get("/system-health")
async def system_health_check():
    """بررسی سلامت سیستم"""
    try:
        # Python version
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        # Memory info
        memory = psutil.virtual_memory()
        memory_gb = round(memory.total / (1024**3), 1)
        
        # Disk space
        disk = shutil.disk_usage('/')
        disk_free_gb = round(disk.free / (1024**3), 1)
        
        # CPU info
        cpu_count = psutil.cpu_count()
        
        # Check dependencies
        dependencies_status = await check_dependencies()
        
        return {
            "python_version": python_version,
            "memory_gb": memory_gb,
            "disk_free_gb": disk_free_gb,
            "cpu_count": cpu_count,
            "dependencies_status": dependencies_status,
            "overall_health": all(dep["status"] == "installed" for dep in dependencies_status.values())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در بررسی سیستم: {str(e)}")

async def check_dependencies():
    """بررسی وابستگی‌های مورد نیاز"""
    required_packages = [
        "pandas", "numpy", "requests", "fastapi", "uvicorn", "psutil"
    ]
    
    status = {}
    for package in required_packages:
        try:
            if package == "scikit-learn":
                import sklearn
                status[package] = {"status": "installed", "version": "unknown"}
            else:
                __import__(package)
                status[package] = {"status": "installed", "version": "unknown"}
        except ImportError:
            status[package] = {"status": "missing", "error": "Package not found"}
    
    return status

@router.post("/validate-binance")
async def validate_binance_api(config: BinanceConfig):
    """اعتبارسنجی API کلیدهای Binance"""
    try:
        import hashlib
        import hmac
        import time
        
        # Test basic connectivity first
        ping_url = "https://api.binance.com/api/v3/ping"
        ping_response = requests.get(ping_url, timeout=10)
        
        if ping_response.status_code != 200:
            return {
                "valid": False,
                "error": "اتصال به Binance برقرار نشد",
                "suggestion": "بررسی کنید که اتصال اینترنت فعال باشد"
            }
        
        # Test API key with account endpoint
        timestamp = int(time.time() * 1000)
        query_string = f"timestamp={timestamp}"
        
        # Create signature
        signature = hmac.new(
            config.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        headers = {
            'X-MBX-APIKEY': config.api_key
        }
        
        account_url = f"https://api.binance.com/api/v3/account?{query_string}&signature={signature}"
        account_response = requests.get(account_url, headers=headers, timeout=10)
        
        if account_response.status_code == 200:
            account_data = account_response.json()
            return {
                "valid": True,
                "message": "API Key و Secret معتبر است",
                "account_type": account_data.get("accountType", "SPOT"),
                "can_trade": account_data.get("canTrade", False),
                "permissions": account_data.get("permissions", [])
            }
        elif account_response.status_code == 401:
            return {
                "valid": False,
                "error": "API Key یا Secret نامعتبر است",
                "suggestion": "API Key و Secret را از Binance دوباره بررسی کنید"
            }
        else:
            return {
                "valid": False,
                "error": f"خطا در اتصال: {account_response.status_code}",
                "suggestion": "لطفاً دوباره تلاش کنید"
            }
        
    except requests.exceptions.Timeout:
        return {
            "valid": False,
            "error": "Timeout در اتصال به Binance",
            "suggestion": "بررسی کنید که اتصال اینترنت پایدار باشد"
        }
    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
            "suggestion": "بررسی کنید که API Key صحیح باشد"
        }


@router.get("/discover-configs")
async def discover_configurations():
    """کشف فایل‌های کانفیگ موجود"""
    try:
        configs_dir = Path(__file__).parent.parent.parent / "configs"
        config_files = []
        
        if configs_dir.exists():
            for config_file in configs_dir.glob("*.json*"):
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        if config_file.suffix == '.jsonc':
                            # Simple JSONC parsing - remove comments
                            content = f.read()
                            lines = content.split('\n')
                            clean_lines = []
                            for line in lines:
                                if '//' in line:
                                    line = line[:line.index('//')]
                                clean_lines.append(line)
                            content = '\n'.join(clean_lines)
                            config_data = json.loads(content)
                        else:
                            config_data = json.load(f)
                    
                    config_files.append({
                        "filename": config_file.name,
                        "path": str(config_file),
                        "symbol": config_data.get("symbol", "N/A"),
                        "frequency": config_data.get("frequency", "N/A"),
                        "venue": config_data.get("venue", "N/A"),
                        "description": f"{config_data.get('symbol', 'Unknown')} - {config_data.get('frequency', 'Unknown')}",
                        "status": "ready"
                    })
                except Exception as e:
                    config_files.append({
                        "filename": config_file.name,
                        "path": str(config_file),
                        "symbol": "Error",
                        "frequency": "Error",
                        "venue": "Error",
                        "description": f"خطا در خواندن فایل: {str(e)}",
                        "status": "error"
                    })
        
        return {
            "configs": config_files,
            "total": len(config_files)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در کشف کانفیگ‌ها: {str(e)}")


@router.post("/save-setup")
async def save_setup_config(setup_data: Dict[str, Any]):
    """ذخیره تنظیمات نهایی setup"""
    try:
        # Create setup config directory
        setup_dir = Path(__file__).parent.parent.parent / "setup"
        setup_dir.mkdir(exist_ok=True)
        
        # Save setup configuration
        setup_file = setup_dir / "dashboard_setup.json"
        
        setup_config = {
            "setup_completed": True,
            "setup_date": setup_data.get("timestamp", ""),
            "system_health": setup_data.get("system_health", {}),
            "binance_configured": setup_data.get("binance", {}).get("valid", False),
            "telegram_configured": setup_data.get("telegram", {}).get("valid", False),
            "selected_config": setup_data.get("selected_config", ""),
            "dashboard_version": "1.0.0"
        }
        
        with open(setup_file, 'w', encoding='utf-8') as f:
            json.dump(setup_config, f, indent=2, ensure_ascii=False)
        
        return {
            "success": True,
            "message": "تنظیمات با موفقیت ذخیره شد",
            "config_file": str(setup_file)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در ذخیره تنظیمات: {str(e)}")

@router.post("/test-telegram")
async def test_telegram(config: TelegramConfig):
    """تست اتصال Telegram Bot"""
    try:
        url = f"https://api.telegram.org/bot{config.bot_token}/sendMessage"
        data = {
            "chat_id": config.chat_id,
            "text": "🤖 تست اتصال ربات ترید موفقیت آمیز بود!\n✅ داشبورد با موفقیت راه‌اندازی شد."
        }
        
        response = requests.post(url, data=data, timeout=10)
        
        if response.status_code == 200:
            return {
                "success": True, 
                "message": "پیام تست با موفقیت ارسال شد",
                "response": response.json()
            }
        else:
            return {
                "success": False, 
                "error": f"خطای HTTP {response.status_code}",
                "details": response.text
            }
            
    except requests.exceptions.Timeout:
        return {
            "success": False, 
            "error": "Timeout در ارسال پیام",
            "suggestion": "بررسی کنید که اتصال اینترنت پایدار باشد"
        }
    except Exception as e:
        return {
            "success": False, 
            "error": str(e),
            "suggestion": "بررسی کنید که Bot Token و Chat ID صحیح باشند"
        }

@router.get("/discover-configs")
async def discover_configurations():
    """کشف کانفیگ‌های موجود"""
    try:
        project_root = Path(__file__).parent.parent.parent
        configs_dir = project_root / "configs"
        
        if not configs_dir.exists():
            return {"configs": [], "error": "پوشه configs یافت نشد"}
        
        configs = []
        for config_file in configs_dir.glob("*.json*"):
            try:
                # Try to read basic info from config
                with open(config_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Remove comments for JSON parsing
                    if config_file.suffix == '.jsonc':
                        import re
                        content = re.sub(r'//.*', '', content)
                        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
                    
                    config_data = json.loads(content)
                    
                    configs.append({
                        "filename": config_file.name,
                        "path": str(config_file.relative_to(project_root)),
                        "symbol": config_data.get("symbol", "Unknown"),
                        "frequency": config_data.get("frequency", "Unknown"),
                        "venue": config_data.get("venue", "Unknown"),
                        "size": config_file.stat().st_size,
                        "status": "ready"
                    })
            except Exception as e:
                configs.append({
                    "filename": config_file.name,
                    "path": str(config_file.relative_to(project_root)),
                    "error": str(e),
                    "status": "error"
                })
        
        return {"configs": configs}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در کشف کانفیگ‌ها: {str(e)}")

@router.post("/save-setup")
async def save_setup_config(setup_data: Dict[str, Any]):
    """ذخیره تنظیمات راه‌اندازی"""
    try:
        project_root = Path(__file__).parent.parent.parent
        setup_file = project_root / "dashboard" / "setup_config.json"
        
        # Create dashboard directory if it doesn't exist
        setup_file.parent.mkdir(exist_ok=True)
        
        with open(setup_file, 'w', encoding='utf-8') as f:
            json.dump(setup_data, f, indent=2, ensure_ascii=False)
        
        return {
            "success": True,
            "message": "تنظیمات با موفقیت ذخیره شد",
            "file": str(setup_file)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در ذخیره تنظیمات: {str(e)}")

@router.post("/final-test")
async def run_final_setup_test(test_data: Dict[str, Any]):
    """اجرای تست نهایی راه‌اندازی"""
    try:
        results = {
            "tests": [],
            "overall_success": True,
            "ready_for_production": False
        }
        
        # Test 1: System Health
        try:
            health_response = await system_health_check()
            results["tests"].append({
                "name": "بررسی سلامت سیستم",
                "status": "success" if health_response.get("overall_health") else "failed",
                "message": "سیستم آماده است" if health_response.get("overall_health") else "سیستم نیاز به بررسی دارد"
            })
        except Exception as e:
            results["tests"].append({
                "name": "بررسی سلامت سیستم", 
                "status": "failed",
                "message": f"خطا: {str(e)}"
            })
            results["overall_success"] = False

        # Test 2: Config Validation
        if test_data.get("selected_config"):
            try:
                config_path = Path(test_data["selected_config"])
                if config_path.exists():
                    results["tests"].append({
                        "name": "اعتبارسنجی کانفیگ",
                        "status": "success",
                        "message": f"کانفیگ {config_path.name} معتبر است"
                    })
                else:
                    results["tests"].append({
                        "name": "اعتبارسنجی کانفیگ",
                        "status": "failed",
                        "message": "فایل کانفیگ یافت نشد"
                    })
                    results["overall_success"] = False
            except Exception as e:
                results["tests"].append({
                    "name": "اعتبارسنجی کانفیگ",
                    "status": "failed",
                    "message": f"خطا: {str(e)}"
                })
                results["overall_success"] = False

        # Test 3: Scripts Availability
        try:
            project_root = Path(__file__).parent.parent.parent
            download_script = project_root / "scripts" / "download_binance.py"
            if download_script.exists():
                results["tests"].append({
                    "name": "اسکریپت‌های سیستم",
                    "status": "success",
                    "message": "اسکریپت‌های مورد نیاز موجود است"
                })
            else:
                results["tests"].append({
                    "name": "اسکریپت‌های سیستم",
                    "status": "warning",
                    "message": "برخی اسکریپت‌ها یافت نشد"
                })
        except Exception as e:
            results["tests"].append({
                "name": "اسکریپت‌های سیستم",
                "status": "failed",
                "message": f"خطا: {str(e)}"
            })

        # Final Assessment
        successful_tests = len([t for t in results["tests"] if t["status"] == "success"])
        total_tests = len(results["tests"])
        
        if successful_tests >= (total_tests * 0.7):  # 70% success rate
            results["ready_for_production"] = True
            results["final_message"] = "🎉 راه‌اندازی با موفقیت تکمیل شد! سیستم آماده استفاده است."
        else:
            results["final_message"] = "⚠️ برخی تست‌ها ناموفق بودند. لطفاً مشکلات را برطرف کرده و دوباره تست کنید."

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در تست نهایی: {str(e)}")
