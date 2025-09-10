"""
Config Validator - اعتبارسنجی فایل‌های کانفیگ
"""

import json
import jsonc_parser
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class ConfigValidator:
    """اعتبارسنجی و validation فایل‌های کانفیگ"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.configs_dir = project_root / "configs"
        
    def validate_config_file(self, config_file: str) -> Dict[str, Any]:
        """اعتبارسنجی کامل یک فایل کانفیگ"""
        
        result = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "config_data": None,
            "missing_fields": [],
            "venue_compatible": False
        }
        
        try:
            # 1. بررسی وجود فایل
            config_path = self.configs_dir / config_file
            if not config_path.exists():
                result["errors"].append(f"فایل کانفیگ {config_file} یافت نشد")
                return result
            
            # 2. پارس کردن فایل JSON/JSONC
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    if config_file.endswith('.jsonc'):
                        config_data = jsonc_parser.loads(f.read())
                    else:
                        config_data = json.load(f)
                result["config_data"] = config_data
            except Exception as e:
                result["errors"].append(f"خطا در پارس فایل: {str(e)}")
                return result
            
            # 3. بررسی فیلدهای ضروری
            required_fields = [
                "symbol", "venue", "start_time", "end_time", 
                "freq", "model", "features", "labels"
            ]
            
            for field in required_fields:
                if field not in config_data:
                    result["missing_fields"].append(field)
            
            # 4. اعتبارسنجی venue
            venue = config_data.get("venue", "")
            supported_venues = ["binance", "yahoo", "mt5"]
            
            if venue not in supported_venues:
                result["errors"].append(f"venue '{venue}' پشتیبانی نمی‌شود. venues مجاز: {supported_venues}")
            else:
                result["venue_compatible"] = True
            
            # 5. بررسی symbol format
            symbol = config_data.get("symbol", "")
            if venue == "binance" and not symbol.endswith("USDT"):
                result["warnings"].append("برای Binance، symbol معمولاً باید با USDT تمام شود")
            
            # 6. بررسی تاریخ‌ها
            start_time = config_data.get("start_time")
            end_time = config_data.get("end_time")
            
            if start_time and end_time:
                if start_time >= end_time:
                    result["errors"].append("start_time باید کمتر از end_time باشد")
            
            # 7. بررسی frequency
            freq = config_data.get("freq", "")
            valid_freqs = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
            if freq not in valid_freqs:
                result["warnings"].append(f"frequency '{freq}' ممکن است پشتیبانی نشود")
            
            # 8. تعیین وضعیت نهایی
            if not result["errors"] and not result["missing_fields"]:
                result["valid"] = True
            
            return result
            
        except Exception as e:
            result["errors"].append(f"خطای غیرمنتظره: {str(e)}")
            return result
    
    def get_available_configs(self) -> List[Dict[str, Any]]:
        """دریافت لیست کانفیگ‌های موجود با metadata"""
        
        configs = []
        
        if not self.configs_dir.exists():
            return configs
        
        for config_file in self.configs_dir.glob("*.json*"):
            config_info = {
                "filename": config_file.name,
                "path": str(config_file),
                "size_kb": round(config_file.stat().st_size / 1024, 2),
                "modified": config_file.stat().st_mtime
            }
            
            # اعتبارسنجی سریع
            validation = self.validate_config_file(config_file.name)
            config_info.update({
                "valid": validation["valid"],
                "venue": validation.get("config_data", {}).get("venue", "unknown"),
                "symbol": validation.get("config_data", {}).get("symbol", "unknown"),
                "errors_count": len(validation["errors"]),
                "warnings_count": len(validation["warnings"])
            })
            
            configs.append(config_info)
        
        return sorted(configs, key=lambda x: x["filename"])
    
    def create_config_template(self, venue: str, symbol: str) -> Dict[str, Any]:
        """ایجاد template کانفیگ برای venue مشخص"""
        
        templates = {
            "binance": {
                "symbol": symbol or "BTCUSDT",
                "venue": "binance",
                "start_time": "2024-01-01",
                "end_time": "2024-12-31",
                "freq": "1h",
                "model": {
                    "class": "LGBMClassifier",
                    "params": {
                        "n_estimators": 100,
                        "max_depth": 5,
                        "learning_rate": 0.1
                    }
                },
                "features": {
                    "highlow": {"period": 20},
                    "ta": {"indicators": ["rsi", "macd", "bb"]}
                },
                "labels": {
                    "highlow": {"period": 10, "threshold": 0.02}
                }
            },
            "yahoo": {
                "symbol": symbol or "BTC-USD",
                "venue": "yahoo",
                "start_time": "2024-01-01",
                "end_time": "2024-12-31",
                "freq": "1d",
                "model": {
                    "class": "RandomForestClassifier",
                    "params": {
                        "n_estimators": 50,
                        "max_depth": 10
                    }
                },
                "features": {
                    "ta": {"indicators": ["sma", "ema", "rsi"]}
                },
                "labels": {
                    "simple": {"threshold": 0.01}
                }
            }
        }
        
        return templates.get(venue, templates["binance"])
    
    def fix_common_issues(self, config_file: str) -> Dict[str, Any]:
        """اصلاح خودکار مشکلات رایج"""
        
        result = {
            "fixed": False,
            "changes": [],
            "errors": []
        }
        
        try:
            validation = self.validate_config_file(config_file)
            
            if not validation["config_data"]:
                result["errors"].append("نمی‌توان فایل کانفیگ را خواند")
                return result
            
            config_data = validation["config_data"].copy()
            changes_made = False
            
            # اصلاح venue مشکل‌دار
            if "venue" in config_data:
                venue = config_data["venue"].lower()
                if venue not in ["binance", "yahoo", "mt5"]:
                    if "binance" in venue or "btc" in config_data.get("symbol", "").lower():
                        config_data["venue"] = "binance"
                        result["changes"].append("venue به binance تغییر یافت")
                        changes_made = True
            
            # اصلاح symbol format
            if config_data.get("venue") == "binance":
                symbol = config_data.get("symbol", "")
                if symbol and not symbol.endswith("USDT"):
                    config_data["symbol"] = symbol + "USDT"
                    result["changes"].append(f"symbol به {config_data['symbol']} تغییر یافت")
                    changes_made = True
            
            # ذخیره تغییرات
            if changes_made:
                config_path = self.configs_dir / config_file
                backup_path = self.configs_dir / f"{config_file}.backup"
                
                # ایجاد backup
                import shutil
                shutil.copy2(config_path, backup_path)
                
                # ذخیره فایل اصلاح شده
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
                
                result["fixed"] = True
                result["changes"].append(f"backup در {backup_path.name} ایجاد شد")
            
            return result
            
        except Exception as e:
            result["errors"].append(f"خطا در اصلاح: {str(e)}")
            return result
