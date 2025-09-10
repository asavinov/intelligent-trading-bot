"""
Configuration Management API Endpoints
مدیریت فایل‌های کانفیگ
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import json
import re
from pathlib import Path
from typing import Dict, Any, List

router = APIRouter(prefix="/configs", tags=["configs"])

class ConfigValidationRequest(BaseModel):
    config_data: Dict[str, Any]

@router.get("/list")
async def list_configurations():
    """لیست کانفیگ‌های موجود"""
    try:
        project_root = Path(__file__).parent.parent.parent
        configs_dir = project_root / "configs"
        configs = []
        
        if configs_dir.exists():
            for config_file in configs_dir.glob("*.json*"):
                # Parse config to get metadata
                try:
                    config_data = parse_config_file(config_file)
                    symbol = config_data.get("symbol", "نامشخص")
                    venue = config_data.get("venue", "نامشخص")
                    freq = config_data.get("freq", "نامشخص")
                    description = config_data.get("description", "")
                    
                    configs.append({
                        "name": config_file.stem,
                        "path": str(config_file),
                        "size": config_file.stat().st_size,
                        "modified": config_file.stat().st_mtime,
                        "symbol": symbol,
                        "venue": venue,
                        "frequency": freq,
                        "description": description,
                        "has_api_keys": bool(config_data.get("api_key")) and config_data.get("api_key") != "<binance-key>",
                        "has_telegram": bool(config_data.get("telegram_bot_token")) and config_data.get("telegram_bot_token") != "<token>"
                    })
                except Exception as parse_error:
                    # If parsing fails, add basic info
                    configs.append({
                        "name": config_file.stem,
                        "path": str(config_file),
                        "size": config_file.stat().st_size,
                        "modified": config_file.stat().st_mtime,
                        "symbol": "خطا در خواندن",
                        "venue": "نامشخص",
                        "frequency": "نامشخص",
                        "description": f"خطا: {str(parse_error)}",
                        "has_api_keys": False,
                        "has_telegram": False
                    })
        
        return configs
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در لیست کانفیگ‌ها: {str(e)}")


def parse_config_file(config_path: Path) -> dict:
    """Parse JSON/JSONC config file"""
    import json
    import re
    
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove comments for JSONC files
    if config_path.suffix == '.jsonc':
        # Remove single line comments
        content = re.sub(r'//.*', '', content)
        # Remove multi-line comments
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    
    return json.loads(content)

async def get_config_info(config_file: Path, project_root: Path) -> Dict[str, Any]:
    """اطلاعات یک فایل کانفیگ"""
    with open(config_file, 'r', encoding='utf-8') as f:
        content = f.read()
        
        # Handle JSONC (JSON with comments)
        if config_file.suffix == '.jsonc':
            content = remove_json_comments(content)
        
        config_data = json.loads(content)
        
        return {
            "filename": config_file.name,
            "path": str(config_file.relative_to(project_root)),
            "symbol": config_data.get("symbol", "Unknown"),
            "frequency": config_data.get("frequency", "Unknown"),
            "venue": config_data.get("venue", "Unknown"),
            "size": config_file.stat().st_size,
            "modified": config_file.stat().st_mtime,
            "status": "ready",
            "data": config_data
        }

def remove_json_comments(content: str) -> str:
    """حذف کامنت‌ها از JSONC"""
    # Remove single line comments
    content = re.sub(r'//.*', '', content)
    # Remove multi-line comments
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    return content

@router.get("/{config_name}")
async def get_configuration(config_name: str):
    """دریافت یک کانفیگ خاص"""
    try:
        project_root = Path(__file__).parent.parent.parent
        config_file = project_root / "configs" / config_name
        
        if not config_file.exists():
            raise HTTPException(status_code=404, detail=f"کانفیگ {config_name} یافت نشد")
        
        config_info = await get_config_info(config_file, project_root)
        return config_info
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در خواندن کانفیگ: {str(e)}")

@router.post("/validate")
async def validate_configuration(request: ConfigValidationRequest):
    """اعتبارسنجی کانفیگ"""
    try:
        config_data = request.config_data
        validation_results = []
        
        # Required fields validation
        required_fields = ["symbol", "frequency", "venue"]
        for field in required_fields:
            if field not in config_data:
                validation_results.append({
                    "field": field,
                    "status": "error",
                    "message": f"فیلد {field} الزامی است"
                })
            else:
                validation_results.append({
                    "field": field,
                    "status": "valid",
                    "value": config_data[field]
                })
        
        # Symbol validation
        if "symbol" in config_data:
            symbol_validation = validate_symbol(config_data["symbol"])
            validation_results.append(symbol_validation)
        
        # Frequency validation
        if "frequency" in config_data:
            freq_validation = validate_frequency(config_data["frequency"])
            validation_results.append(freq_validation)
        
        # Venue validation
        if "venue" in config_data:
            venue_validation = validate_venue(config_data["venue"])
            validation_results.append(venue_validation)
        
        # Overall validation status
        has_errors = any(result["status"] == "error" for result in validation_results)
        
        return {
            "valid": not has_errors,
            "validation_results": validation_results,
            "summary": {
                "total_checks": len(validation_results),
                "passed": len([r for r in validation_results if r["status"] == "valid"]),
                "failed": len([r for r in validation_results if r["status"] == "error"]),
                "warnings": len([r for r in validation_results if r["status"] == "warning"])
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در اعتبارسنجی: {str(e)}")

def validate_symbol(symbol: str) -> Dict[str, Any]:
    """اعتبارسنجی نماد"""
    if not symbol:
        return {"field": "symbol", "status": "error", "message": "نماد نمی‌تواند خالی باشد"}
    
    if not symbol.isupper():
        return {"field": "symbol", "status": "warning", "message": "نماد باید با حروف بزرگ باشد"}
    
    if len(symbol) < 6:
        return {"field": "symbol", "status": "warning", "message": "نماد کوتاه است"}
    
    return {"field": "symbol", "status": "valid", "value": symbol}

def validate_frequency(frequency: str) -> Dict[str, Any]:
    """اعتبارسنجی فرکانس"""
    valid_frequencies = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]
    
    if frequency not in valid_frequencies:
        return {
            "field": "frequency", 
            "status": "error", 
            "message": f"فرکانس نامعتبر. مقادیر مجاز: {', '.join(valid_frequencies)}"
        }
    
    return {"field": "frequency", "status": "valid", "value": frequency}

def validate_venue(venue: str) -> Dict[str, Any]:
    """اعتبارسنجی صرافی"""
    valid_venues = ["BINANCE", "MT5"]
    
    if venue not in valid_venues:
        return {
            "field": "venue", 
            "status": "error", 
            "message": f"صرافی نامعتبر. مقادیر مجاز: {', '.join(valid_venues)}"
        }
    
    return {"field": "venue", "status": "valid", "value": venue}

@router.post("/{config_name}")
async def save_configuration(config_name: str, config_data: Dict[str, Any]):
    """ذخیره کانفیگ"""
    try:
        project_root = Path(__file__).parent.parent.parent
        configs_dir = project_root / "configs"
        configs_dir.mkdir(exist_ok=True)
        
        config_file = configs_dir / config_name
        
        # Validate before saving
        validation_request = ConfigValidationRequest(config_data=config_data)
        validation_result = await validate_configuration(validation_request)
        
        if not validation_result["valid"]:
            return {
                "success": False,
                "message": "کانفیگ نامعتبر است",
                "validation": validation_result
            }
        
        # Save configuration
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        return {
            "success": True,
            "message": "کانفیگ با موفقیت ذخیره شد",
            "file": str(config_file.relative_to(project_root)),
            "validation": validation_result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در ذخیره کانفیگ: {str(e)}")

@router.delete("/{config_name}")
async def delete_configuration(config_name: str):
    """حذف کانفیگ"""
    try:
        project_root = Path(__file__).parent.parent.parent
        config_file = project_root / "configs" / config_name
        
        if not config_file.exists():
            raise HTTPException(status_code=404, detail=f"کانفیگ {config_name} یافت نشد")
        
        config_file.unlink()
        
        return {
            "success": True,
            "message": f"کانفیگ {config_name} حذف شد"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در حذف کانفیگ: {str(e)}")

@router.get("/templates/list")
async def list_config_templates():
    """لیست قالب‌های کانفیگ"""
    templates = {
        "binance_1min": {
            "name": "Binance 1 Minute",
            "description": "کانفیگ پایه برای Binance با فرکانس 1 دقیقه",
            "template": {
                "symbol": "BTCUSDT",
                "frequency": "1m",
                "venue": "BINANCE",
                "features": {
                    "technical_indicators": True,
                    "volume_indicators": True,
                    "price_action": True
                },
                "training": {
                    "test_size": 0.2,
                    "validation_size": 0.1
                }
            }
        },
        "binance_1hour": {
            "name": "Binance 1 Hour", 
            "description": "کانفیگ پایه برای Binance با فرکانس 1 ساعت",
            "template": {
                "symbol": "BTCUSDT",
                "frequency": "1h",
                "venue": "BINANCE",
                "features": {
                    "technical_indicators": True,
                    "volume_indicators": True,
                    "price_action": True
                },
                "training": {
                    "test_size": 0.2,
                    "validation_size": 0.1
                }
            }
        },
        "mt5_1hour": {
            "name": "MetaTrader 5 1 Hour",
            "description": "کانفیگ پایه برای MT5 با فرکانس 1 ساعت", 
            "template": {
                "symbol": "EURUSD",
                "frequency": "1h",
                "venue": "MT5",
                "features": {
                    "technical_indicators": True,
                    "volume_indicators": False,
                    "price_action": True
                },
                "training": {
                    "test_size": 0.2,
                    "validation_size": 0.1
                }
            }
        }
    }
    
    return {"templates": templates}
