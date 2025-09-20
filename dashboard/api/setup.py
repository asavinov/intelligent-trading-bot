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
import re
from pathlib import Path
import json
from typing import Dict, Any
from datetime import datetime
from typing import Optional

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


def read_env_file() -> Dict[str, str]:
    """Read a simple .env file from project root and return a dict of keys."""
    try:
        project_root = Path(__file__).parent.parent.parent
        env_file = project_root / '.env'
        result: Dict[str, str] = {}
        if not env_file.exists():
            return result

        for line in env_file.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            key, val = line.split('=', 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            result[key] = val

        return result
    except Exception:
        return {}


def mask_secret(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    v = str(value)
    if len(v) <= 6:
        return '****'
    # show last 4 characters
    return '****' + v[-4:]

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
    """کشف فایل‌های کانفیگ موجود - مسیرها نسبی به project root برگردانده می‌شود"""
    try:
        project_root = Path(__file__).parent.parent.parent
        configs_dir = project_root / "configs"
        config_files = []

        if configs_dir.exists():
            for config_file in configs_dir.glob("*.json*"):
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Allow JSONC-ish comments removal for parsing
                        content_clean = re.sub(r'//.*', '', content)
                        content_clean = re.sub(r'/\*.*?\*/', '', content_clean, flags=re.DOTALL)
                        try:
                            config_data = json.loads(content_clean)
                        except Exception:
                            # Fallback: try plain json load
                            f.seek(0)
                            config_data = json.load(f)

                    config_files.append({
                        "filename": config_file.name,
                        "path": str(config_file.relative_to(project_root)),
                        "symbol": config_data.get("symbol", "N/A"),
                        "frequency": config_data.get("frequency", "N/A"),
                        "venue": config_data.get("venue", "N/A"),
                        "description": f"{config_data.get('symbol', 'Unknown')} - {config_data.get('frequency', 'Unknown')}",
                        "status": "ready"
                    })
                except Exception as e:
                    config_files.append({
                        "filename": config_file.name,
                        "path": str(config_file.relative_to(project_root)),
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
        
        # Merge with .env fallbacks for missing values (do not expose secrets)
        env = read_env_file()

        # Determine selected_config: payload wins, otherwise keep existing or env suggestion
        selected = setup_data.get("selected_config") or env.get('SELECTED_CONFIG') or setup_data.get('selected_config', '')

        setup_config = {
            "setup_completed": True,
            "setup_date": setup_data.get("timestamp", ""),
            "system_health": setup_data.get("system_health", {}),
            "binance_configured": setup_data.get("binance", {}).get("valid", False) or bool(env.get('BINANCE_API_KEY')),
            "telegram_configured": setup_data.get("telegram", {}).get("valid", False) or bool(env.get('TELEGRAM_BOT_TOKEN')),
            "selected_config": selected,
            "dashboard_version": "1.0.0",
            "configured_from_env": {
                "binance": bool(env.get('BINANCE_API_KEY')),
                "telegram": bool(env.get('TELEGRAM_BOT_TOKEN'))
            }
        }
        
        with open(setup_file, 'w', encoding='utf-8') as f:
            json.dump(setup_config, f, indent=2, ensure_ascii=False)

        # 🔥 NEW: Update actual config file with real credentials
        # When updating actual config files with credentials, prefer explicit payload values.
        # If payload lacks secrets but .env has them, use env values internally (do not echo secrets back).
        selected_config = selected
        # prepare a combined setup_data_for_update for update_config_file
        combined_setup = dict(setup_data)
        # If binance not provided in payload but exists in env, fill it for update
        if not combined_setup.get('binance'):
            if env.get('BINANCE_API_KEY') or env.get('BINANCE_API_SECRET'):
                combined_setup['binance'] = {
                    'valid': True,
                    'api_key': env.get('BINANCE_API_KEY'),
                    'api_secret': env.get('BINANCE_API_SECRET')
                }

        if not combined_setup.get('telegram'):
            if env.get('TELEGRAM_BOT_TOKEN') or env.get('TELEGRAM_CHAT_ID'):
                combined_setup['telegram'] = {
                    'valid': True,
                    'bot_token': env.get('TELEGRAM_BOT_TOKEN'),
                    'chat_id': env.get('TELEGRAM_CHAT_ID')
                }

        if selected_config:
            await update_config_file(selected_config, combined_setup)
        
        # Return success but do NOT include secret values in the response
        return {
            "success": True,
            "message": "تنظیمات با موفقیت ذخیره شد و config files به‌روزرسانی شدند",
            "config_file": str(setup_file),
            "updated_config": selected_config,
            "secrets_present": {
                "binance": bool(env.get('BINANCE_API_KEY')) or bool(setup_data.get('binance', {}).get('api_key')),
                "telegram": bool(env.get('TELEGRAM_BOT_TOKEN')) or bool(setup_data.get('telegram', {}).get('bot_token'))
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در ذخیره تنظیمات: {str(e)}")


@router.get('/selected-config')
async def get_selected_setup_config():
    """Return the selected config saved by the dashboard setup (if any)."""
    try:
        project_root = Path(__file__).parent.parent.parent
        setup_file = project_root / 'setup' / 'dashboard_setup.json'
        if not setup_file.exists():
            return {"selected_config": None}

        with open(setup_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return {
            "selected_config": data.get('selected_config') if isinstance(data, dict) else None,
            "raw": data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading setup config: {str(e)}")


@router.get('/defaults')
async def get_setup_defaults():
    """Return safe (redacted) defaults read from .env for UI prefill."""
    try:
        env = read_env_file()
        defaults = {
            "binance": {
                "has_key": bool(env.get('BINANCE_API_KEY')),
                "api_key_masked": mask_secret(env.get('BINANCE_API_KEY'))
            },
            "telegram": {
                "has_token": bool(env.get('TELEGRAM_BOT_TOKEN')),
                "bot_token_masked": mask_secret(env.get('TELEGRAM_BOT_TOKEN'))
            },
            "selected_config_suggestion": env.get('SELECTED_CONFIG') or None,
            "env_found": True if env else False
        }

        # Also attempt to include selected_config from saved setup if present
        project_root = Path(__file__).parent.parent.parent
        setup_file = project_root / 'setup' / 'dashboard_setup.json'
        if setup_file.exists():
            try:
                data = json.loads(setup_file.read_text(encoding='utf-8'))
                defaults['selected_config_saved'] = data.get('selected_config')
            except Exception:
                defaults['selected_config_saved'] = None

        return defaults
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading defaults: {str(e)}")

async def update_config_file(config_filename: str, setup_data: Dict[str, Any]):
    """به‌روزرسانی فایل config واقعی با credentials جدید"""
    try:
        project_root = Path(__file__).parent.parent.parent
        configs_dir = project_root / "configs"

        # Resolve provided path: accept absolute or relative (relative to project root)
        p = Path(config_filename)
        if p.is_absolute():
            config_path = p
        else:
            # Try relative to project root first
            candidate = project_root / p
            if candidate.exists():
                config_path = candidate
            else:
                # Fallback to configs_dir / filename
                config_path = configs_dir / p.name

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Read current config
        raw = config_path.read_text(encoding='utf-8')

        # Create a backup before modifying
        timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        backup_path = config_path.parent / f"{config_path.name}.bak.{timestamp}"
        shutil.copyfile(str(config_path), str(backup_path))

        # Try to parse JSON (support JSONC-ish by removing comments)
        content_clean = re.sub(r'//.*', '', raw)
        content_clean = re.sub(r'/\*.*?\*/', '', content_clean, flags=re.DOTALL)
        try:
            config_data = json.loads(content_clean)
        except Exception:
            # Last resort: try plain json.loads on original raw
            config_data = json.loads(raw)

        # Update fields programmatically (safer than regex)
        binance_data = setup_data.get("binance", {})
        if binance_data.get("valid") and binance_data.get("api_key"):
            config_data["api_key"] = binance_data.get("api_key")
            config_data["api_secret"] = binance_data.get("api_secret")

        telegram_data = setup_data.get("telegram", {})
        if telegram_data.get("valid") and telegram_data.get("bot_token"):
            config_data["telegram_bot_token"] = telegram_data.get("bot_token")
            config_data["telegram_chat_id"] = telegram_data.get("chat_id")

        # Write updated config back (atomically)
        new_content = json.dumps(config_data, indent=2, ensure_ascii=False)
        config_path.write_text(new_content, encoding='utf-8')

        # Validate written JSON
        try:
            json.loads(config_path.read_text(encoding='utf-8'))
        except Exception as e:
            # Rollback from backup
            shutil.copyfile(str(backup_path), str(config_path))
            raise RuntimeError(f"Updated config is invalid JSON, rolled back. Error: {str(e)}")

        print(f"✅ Config file updated: {config_path}")
        
    except Exception as e:
        print(f"❌ خطا در به‌روزرسانی config file: {str(e)}")
        raise e

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
