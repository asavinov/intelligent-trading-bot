"""
Trading Signals API Endpoints
API برای مدیریت سیگنال‌های معاملاتی
"""

from fastapi import APIRouter, HTTPException
from pathlib import Path
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import glob

router = APIRouter(prefix="/signals", tags=["signals"])


def get_latest_signal_from_files():
    """
    آخرین سیگنال را از فایل‌های output پیدا می‌کند
    """
    try:
        # جستجو در پوشه‌های مختلف برای فایل‌های سیگنال
        search_paths = [
            Path(__file__).parent.parent.parent / "outputs",
            Path(__file__).parent.parent.parent / "data" / "signals",
            Path(__file__).parent.parent.parent / "results",
        ]
        
        latest_signal = None
        latest_time = datetime.min
        
        for search_path in search_paths:
            if not search_path.exists():
                continue
                
            # جستجو برای فایل‌های JSON با نام‌های مختلف
            signal_patterns = [
                "signals*.json",
                "signal*.json", 
                "trading_signals*.json",
                "latest_signal*.json"
            ]
            
            for pattern in signal_patterns:
                signal_files = list(search_path.glob(pattern))
                
                for signal_file in signal_files:
                    try:
                        # بررسی زمان آخرین تغییر فایل
                        file_time = datetime.fromtimestamp(signal_file.stat().st_mtime)
                        
                        if file_time > latest_time:
                            with open(signal_file, 'r', encoding='utf-8') as f:
                                signal_data = json.load(f)
                                
                            # اگر داده‌ای موجود است، آن را ذخیره کن
                            if signal_data:
                                latest_signal = {
                                    "signal": signal_data,
                                    "file_path": str(signal_file),
                                    "timestamp": file_time.isoformat(),
                                    "source": "file"
                                }
                                latest_time = file_time
                                
                    except (json.JSONDecodeError, PermissionError) as e:
                        continue
        
        return latest_signal
        
    except Exception as e:
        return None


def get_sample_signal():
    """
    نمونه سیگنال برای تست
    """
    return {
        "symbol": "BTCUSDT",
        "action": "BUY",
        "price": 45250.50,
        "confidence": 0.85,
        "timestamp": datetime.now().isoformat(),
        "indicators": {
            "rsi": 35.2,
            "macd": "bullish",
            "volume": "high"
        },
        "source": "sample"
    }


@router.get("/latest")
async def get_latest_signal():
    """
    آخرین سیگنال معاملاتی تولید شده
    """
    try:
        # ابتدا سعی کن آخرین سیگنال را از فایل‌ها پیدا کن
        latest_signal = get_latest_signal_from_files()
        
        if latest_signal:
            return {
                "status": "success",
                "signal": latest_signal["signal"],
                "timestamp": latest_signal["timestamp"],
                "source": latest_signal["source"],
                "file_path": latest_signal.get("file_path")
            }
        
        # اگر فایلی پیدا نشد، نمونه سیگنال برگردان
        sample_signal = get_sample_signal()
        return {
            "status": "no_recent_signals",
            "signal": sample_signal,
            "timestamp": sample_signal["timestamp"],
            "source": "sample",
            "message": "هیچ سیگنال اخیری یافت نشد - نمونه سیگنال نمایش داده شده"
        }
        
    except Exception as e:
        # در صورت خطا، پیام مناسب برگردان
        return {
            "status": "error",
            "signal": None,
            "timestamp": datetime.now().isoformat(),
            "source": "error",
            "message": f"خطا در بارگذاری سیگنال: {str(e)}"
        }


@router.get("/history")
async def get_signals_history(limit: int = 10):
    """
    تاریخچه سیگنال‌های معاملاتی
    """
    try:
        # جستجو برای فایل‌های تاریخی سیگنال
        search_paths = [
            Path(__file__).parent.parent.parent / "outputs",
            Path(__file__).parent.parent.parent / "data" / "signals",
        ]
        
        signals_history = []
        
        for search_path in search_paths:
            if not search_path.exists():
                continue
                
            signal_files = list(search_path.glob("signals*.json"))
            signal_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            for signal_file in signal_files[:limit]:
                try:
                    with open(signal_file, 'r', encoding='utf-8') as f:
                        signal_data = json.load(f)
                    
                    signals_history.append({
                        "signal": signal_data,
                        "file_path": str(signal_file),
                        "timestamp": datetime.fromtimestamp(signal_file.stat().st_mtime).isoformat()
                    })
                    
                except (json.JSONDecodeError, PermissionError):
                    continue
        
        return {
            "status": "success",
            "signals": signals_history,
            "count": len(signals_history)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading signals history: {str(e)}")


@router.get("/status")
async def get_signals_status():
    """
    وضعیت سیستم تولید سیگنال
    """
    try:
        # بررسی آخرین فایل سیگنال
        latest_signal = get_latest_signal_from_files()
        
        status = {
            "signal_generator_active": latest_signal is not None,
            "last_signal_time": latest_signal["timestamp"] if latest_signal else None,
            "signals_today": 0,  # می‌تواند از دیتابیس یا فایل‌ها محاسبه شود
            "system_status": "active" if latest_signal else "inactive"
        }
        
        return status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking signals status: {str(e)}")
