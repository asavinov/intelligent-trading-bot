# راهنمای نهایی پیاده‌سازی داشبورد ربات ترید هوشمند

## 🎯 **اصل طلایی: "چیزی را که نتوانیم ببینیم نمیتوانیم کنترل کنیم"**

## 📋 **خلاصه اجرایی**

این راهنما طرح کاملی برای پیاده‌سازی داشبورد وب ارائه می‌دهد که بر اساس تحلیل عمیق کدبیس موجود و ورکفلوهای کاربری طراحی شده است.

## 🏗️ **معماری نهایی**

### **Technology Stack**
```
Frontend: HTML5 + Tailwind CSS + Vanilla JavaScript
Backend: FastAPI + SQLModel + Pydantic  
Database: SQLite
Process Management: asyncio + subprocess
Real-time: Server-Sent Events (SSE)
```

### **Architecture Overview**
```
┌─────────────────────────────────────────────────────────┐
│                 Web Dashboard                           │
├─────────────────────────────────────────────────────────┤
│  Frontend Layer                                         │
│  ├── Setup Wizard (اولین راه‌اندازی)                    │
│  ├── Script Runner Interface                            │
│  ├── Real-time Monitoring                               │
│  ├── Config Editor & Validator                          │
│  ├── API Discovery & Testing                            │
│  └── System Health Dashboard                            │
├─────────────────────────────────────────────────────────┤
│  FastAPI Backend                                        │
│  ├── User Workflow APIs                                 │
│  ├── Script Management APIs                             │
│  ├── Real-time SSE Endpoints                            │
│  ├── System Health & Diagnostics                        │
│  └── Configuration Management                           │
├─────────────────────────────────────────────────────────┤
│  Integration Layer                                       │
│  ├── Safe Script Wrapper                                │
│  ├── Config Validator                                   │
│  ├── API Connection Manager                             │
│  ├── Telegram Bot Interface                             │
│  └── Resource Monitor                                   │
├─────────────────────────────────────────────────────────┤
│  Existing Codebase (بدون تغییر)                        │
│  ├── scripts/ (download, merge, train, etc.)            │
│  ├── service/ (App.py, server.py)                       │
│  ├── inputs/ (collector_binance, collector_mt5)         │
│  ├── outputs/ (notifiers, traders)                      │
│  └── configs/ (JSON configuration files)               │
└─────────────────────────────────────────────────────────┘
```

## 📁 **ساختار پروژه**

```
intelligent-trading-bot/
├── dashboard/                          # جدید - Dashboard application
│   ├── main.py                        # FastAPI entry point
│   ├── api/                           # API endpoints
│   │   ├── setup.py                   # Setup wizard APIs
│   │   ├── scripts.py                 # Script management
│   │   ├── configs.py                 # Configuration management
│   │   ├── system.py                  # Health & diagnostics
│   │   └── notifications.py           # Telegram management
│   ├── core/                          # Core functionality
│   │   ├── script_wrapper.py          # Safe script execution
│   │   ├── config_validator.py        # Config validation
│   │   ├── api_manager.py             # API connections
│   │   ├── telegram_manager.py        # Telegram integration
│   │   └── resource_monitor.py        # System monitoring
│   ├── frontend/                      # Static files
│   │   ├── index.html                 # Main dashboard
│   │   ├── setup.html                 # Setup wizard
│   │   ├── css/                       # Tailwind CSS
│   │   └── js/                        # JavaScript modules
│   └── models/                        # Data models
│       ├── database.py                # SQLite setup
│       └── schemas.py                 # Pydantic models
├── [existing folders unchanged]        # کدبیس موجود بدون تغییر
```

## 🚀 **فازهای پیاده‌سازی**

### **فاز 0: System Discovery & Setup (روز 1-2)**
```
Day 1: Environment Analysis
├── ✅ System health check
├── ✅ Dependencies validation  
├── ✅ Config files discovery
├── ✅ API capabilities assessment
└── ✅ User requirements gathering

Day 2: Initial Setup Wizard
├── ✅ API keys configuration (Binance/MT5)
├── ✅ Telegram bot setup & testing
├── ✅ First validation run
├── ✅ Error detection & recovery
└── ✅ User walkthrough & training
```

### **فاز 1: Core MVP (هفته 1)**
```
Week 1: Essential User Experience
├── Setup wizard با validation کامل
├── Script runner با real-time monitoring
├── Config editor با error checking
├── System health dashboard
├── Basic error handling & recovery
└── User documentation
```

### **فاز 2: Advanced Features (هفته 2)**
```
Week 2: Enhanced Functionality  
├── Advanced monitoring & alerts
├── Model management interface
├── Performance analytics
├── Backup & restore capabilities
├── Security enhancements
└── API rate limiting
```

### **فاز 3: Production Ready (هفته 3+)**
```
Week 3+: Production Features
├── Advanced security (JWT auth)
├── Multi-user support
├── Advanced analytics & reporting
├── Automated testing suite
├── Deployment automation
└── Comprehensive documentation
```

## 🔧 **پیاده‌سازی کلیدی**

### **1. Safe Script Wrapper**
```python
from pathlib import Path
import asyncio
import subprocess
import os
from typing import Dict, Any, Optional

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
            raise FileNotFoundError(f"Script {script_name} not found")
            
        config_full_path = self.project_root / config_path
        if not config_full_path.exists():
            raise FileNotFoundError(f"Config {config_path} not found")
        
        # 2. Environment setup
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.project_root)
        
        # 3. Command construction
        cmd = [
            "python", str(script_path),
            "--config", str(config_full_path)
        ]
        
        # 4. Process execution with monitoring
        job_id = f"{script_name}_{int(time.time())}"
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=str(self.project_root)
            )
            
            self.active_processes[job_id] = process
            
            # 5. Real-time output streaming
            async def stream_output():
                async for line in process.stdout:
                    yield f"data: {line.decode().strip()}\n\n"
                    
            return {
                "job_id": job_id,
                "status": "running",
                "stream": stream_output(),
                "process": process
            }
            
        except Exception as e:
            return {
                "job_id": job_id,
                "status": "failed",
                "error": str(e)
            }
```

### **2. Setup Wizard API**
```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import requests

router = APIRouter(prefix="/api/setup", tags=["setup"])

class BinanceConfig(BaseModel):
    api_key: str
    api_secret: str

class TelegramConfig(BaseModel):
    bot_token: str
    chat_id: str

@router.post("/validate-binance")
async def validate_binance_api(config: BinanceConfig):
    """اعتبارسنجی API کلیدهای Binance"""
    try:
        from binance.client import Client
        client = Client(api_key=config.api_key, api_secret=config.api_secret)
        account_info = client.get_account()
        
        return {
            "valid": True,
            "account_type": account_info.get("accountType"),
            "permissions": account_info.get("permissions"),
            "can_trade": "TRADING" in account_info.get("permissions", [])
        }
    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
            "suggestion": "بررسی کنید که API Key و Secret صحیح باشند"
        }

@router.post("/test-telegram")
async def test_telegram(config: TelegramConfig):
    """تست اتصال Telegram Bot"""
    try:
        url = f"https://api.telegram.org/bot{config.bot_token}/sendMessage"
        data = {
            "chat_id": config.chat_id,
            "text": "🤖 تست اتصال ربات ترید موفقیت آمیز بود!"
        }
        
        response = requests.post(url, data=data, timeout=10)
        if response.status_code == 200:
            return {"success": True, "message": "پیام تست ارسال شد"}
        else:
            return {"success": False, "error": response.text}
            
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.get("/system-health")
async def system_health_check():
    """بررسی سلامت سیستم"""
    import sys
    import psutil
    import shutil
    
    return {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "memory_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        "disk_free_gb": round(shutil.disk_usage('/').free / (1024**3), 1),
        "cpu_count": psutil.cpu_count(),
        "dependencies_status": await check_dependencies()
    }

async def check_dependencies():
    """بررسی وابستگی‌های مورد نیاز"""
    required_packages = [
        "pandas", "numpy", "scikit-learn", 
        "binance", "python-telegram-bot", "fastapi"
    ]
    
    status = {}
    for package in required_packages:
        try:
            __import__(package)
            status[package] = "installed"
        except ImportError:
            status[package] = "missing"
    
    return status
```

### **3. Real-time Monitoring**
```python
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import asyncio
import json

router = APIRouter(prefix="/api/monitor", tags=["monitoring"])

@router.get("/script-logs/{job_id}")
async def stream_script_logs(job_id: str):
    """استریم real-time لاگ‌های اسکریپت"""
    
    async def log_generator():
        # Header for SSE
        yield "data: {\"type\": \"connected\", \"message\": \"اتصال برقرار شد\"}\n\n"
        
        # Monitor active process
        if job_id in script_wrapper.active_processes:
            process = script_wrapper.active_processes[job_id]
            
            try:
                while process.returncode is None:
                    # Read stdout
                    if process.stdout:
                        line = await process.stdout.readline()
                        if line:
                            log_data = {
                                "type": "stdout",
                                "content": line.decode().strip(),
                                "timestamp": time.time()
                            }
                            yield f"data: {json.dumps(log_data)}\n\n"
                    
                    await asyncio.sleep(0.1)
                    
                # Process completed
                completion_data = {
                    "type": "completed",
                    "exit_code": process.returncode,
                    "message": "اسکریپت با موفقیت اجرا شد" if process.returncode == 0 else "اسکریپت با خطا متوقف شد"
                }
                yield f"data: {json.dumps(completion_data)}\n\n"
                
            except Exception as e:
                error_data = {
                    "type": "error",
                    "message": str(e)
                }
                yield f"data: {json.dumps(error_data)}\n\n"
        else:
            yield "data: {\"type\": \"error\", \"message\": \"Job ID یافت نشد\"}\n\n"
    
    return StreamingResponse(
        log_generator(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache"}
    )

@router.get("/system-resources")
async def stream_system_resources():
    """استریم real-time منابع سیستم"""
    
    async def resource_generator():
        while True:
            try:
                import psutil
                
                resource_data = {
                    "cpu_percent": psutil.cpu_percent(interval=1),
                    "memory_percent": psutil.virtual_memory().percent,
                    "memory_used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
                    "disk_usage_percent": psutil.disk_usage('/').percent,
                    "timestamp": time.time()
                }
                
                yield f"data: {json.dumps(resource_data)}\n\n"
                await asyncio.sleep(2)
                
            except Exception as e:
                error_data = {"type": "error", "message": str(e)}
                yield f"data: {json.dumps(error_data)}\n\n"
                break
    
    return StreamingResponse(
        resource_generator(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache"}
    )
```

## 🎨 **Frontend Components**

### **Setup Wizard Interface**
```html
<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>راه‌اندازی اولیه - ربات ترید هوشمند</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100 font-sans">
    <div class="container mx-auto px-4 py-8">
        <div class="max-w-4xl mx-auto">
            <h1 class="text-3xl font-bold text-center mb-8">راه‌اندازی اولیه سیستم</h1>
            
            <!-- Progress Steps -->
            <div class="flex justify-between mb-8">
                <div class="step active" data-step="1">
                    <div class="step-circle">1</div>
                    <div class="step-label">بررسی سیستم</div>
                </div>
                <div class="step" data-step="2">
                    <div class="step-circle">2</div>
                    <div class="step-label">تنظیم API</div>
                </div>
                <div class="step" data-step="3">
                    <div class="step-circle">3</div>
                    <div class="step-label">Telegram</div>
                </div>
                <div class="step" data-step="4">
                    <div class="step-circle">4</div>
                    <div class="step-label">تست نهایی</div>
                </div>
            </div>
            
            <!-- Step 1: System Health -->
            <div id="step-1" class="step-content">
                <div class="bg-white rounded-lg shadow-md p-6">
                    <h2 class="text-xl font-semibold mb-4">بررسی سلامت سیستم</h2>
                    
                    <div id="health-check-results">
                        <div class="loading">در حال بررسی...</div>
                    </div>
                    
                    <div class="mt-6">
                        <button id="run-health-check" class="btn-primary">
                            شروع بررسی
                        </button>
                    </div>
                </div>
            </div>
            
            <!-- Step 2: API Configuration -->
            <div id="step-2" class="step-content hidden">
                <div class="bg-white rounded-lg shadow-md p-6">
                    <h2 class="text-xl font-semibold mb-4">تنظیم API کلیدها</h2>
                    
                    <!-- Binance API -->
                    <div class="mb-6">
                        <h3 class="text-lg font-medium mb-3">🔑 Binance API</h3>
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div>
                                <label class="block text-sm font-medium mb-2">API Key:</label>
                                <input type="text" id="binance-api-key" 
                                       class="w-full px-3 py-2 border rounded-md"
                                       placeholder="API Key خود را وارد کنید">
                            </div>
                            <div>
                                <label class="block text-sm font-medium mb-2">API Secret:</label>
                                <input type="password" id="binance-api-secret" 
                                       class="w-full px-3 py-2 border rounded-md"
                                       placeholder="API Secret خود را وارد کنید">
                            </div>
                        </div>
                        <div class="mt-3">
                            <button id="test-binance" class="btn-secondary">
                                تست اتصال Binance
                            </button>
                            <div id="binance-status" class="mt-2"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="/static/js/setup-wizard.js"></script>
</body>
</html>
```

### **Main Dashboard Interface**
```html
<div class="dashboard-container">
    <!-- Header -->
    <header class="bg-blue-600 text-white p-4">
        <div class="flex justify-between items-center">
            <h1 class="text-2xl font-bold">داشبورد ربات ترید هوشمند</h1>
            <div class="flex items-center space-x-4">
                <div class="status-indicator" id="system-status">
                    <span class="status-dot bg-green-500"></span>
                    سیستم فعال
                </div>
            </div>
        </div>
    </header>
    
    <!-- Main Content -->
    <div class="flex">
        <!-- Sidebar -->
        <aside class="w-64 bg-gray-800 text-white min-h-screen">
            <nav class="p-4">
                <ul class="space-y-2">
                    <li><a href="#dashboard" class="nav-link active">داشبورد</a></li>
                    <li><a href="#scripts" class="nav-link">مدیریت اسکریپت‌ها</a></li>
                    <li><a href="#configs" class="nav-link">تنظیمات</a></li>
                    <li><a href="#monitoring" class="nav-link">مانیتورینگ</a></li>
                    <li><a href="#notifications" class="nav-link">اعلان‌ها</a></li>
                </ul>
            </nav>
        </aside>
        
        <!-- Main Panel -->
        <main class="flex-1 p-6">
            <!-- Quick Actions -->
            <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
                <div class="quick-action-card">
                    <h3>بروزرسانی داده‌ها</h3>
                    <button class="btn-primary mt-2" onclick="runScript('download_binance')">
                        دانلود جدید
                    </button>
                </div>
                
                <div class="quick-action-card">
                    <h3>اجرای تحلیل</h3>
                    <button class="btn-primary mt-2" onclick="runFullAnalysis()">
                        تحلیل کامل
                    </button>
                </div>
                
                <div class="quick-action-card">
                    <h3>آخرین سیگنال</h3>
                    <div id="latest-signal" class="mt-2">
                        <span class="signal-score">+0.23</span>
                        <span class="signal-time">11:15 AM</span>
                    </div>
                </div>
                
                <div class="quick-action-card">
                    <h3>وضعیت اتصالات</h3>
                    <div class="connection-status mt-2">
                        <div class="status-item">
                            <span class="status-dot bg-green-500"></span>
                            Binance: متصل
                        </div>
                        <div class="status-item">
                            <span class="status-dot bg-green-500"></span>
                            Telegram: فعال
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Real-time Logs -->
            <div class="bg-white rounded-lg shadow-md p-6">
                <h2 class="text-xl font-semibold mb-4">لاگ‌های زنده سیستم</h2>
                <div id="live-logs" class="bg-gray-900 text-green-400 p-4 rounded font-mono text-sm h-64 overflow-y-auto">
                    <!-- Real-time logs will appear here -->
                </div>
            </div>
        </main>
    </div>
</div>
```

## 📊 **مانیتورینگ و عیب‌یابی**

### **System Health Monitoring**
```python
@router.get("/system/diagnostics")
async def run_system_diagnostics():
    """تشخیص خودکار مشکلات سیستم"""
    
    diagnostics = {
        "internet_connection": await check_internet(),
        "binance_api": await check_binance_connection(),
        "telegram_bot": await check_telegram_connection(),
        "disk_space": await check_disk_space(),
        "dependencies": await check_python_dependencies(),
        "config_files": await validate_config_files(),
        "data_files": await check_data_integrity()
    }
    
    # تشخیص مشکلات و ارائه راه‌حل
    issues = []
    solutions = []
    
    for check, status in diagnostics.items():
        if not status.get("healthy", True):
            issues.append({
                "component": check,
                "issue": status.get("error"),
                "severity": status.get("severity", "medium"),
                "solution": get_solution_for_issue(check, status)
            })
    
    return {
        "overall_health": len(issues) == 0,
        "diagnostics": diagnostics,
        "issues": issues,
        "auto_fix_available": any(issue.get("auto_fixable") for issue in issues)
    }

async def check_internet():
    """بررسی اتصال اینترنت"""
    try:
        response = requests.get("https://api.binance.com/api/v3/ping", timeout=5)
        return {
            "healthy": response.status_code == 200,
            "response_time": response.elapsed.total_seconds()
        }
    except Exception as e:
        return {
            "healthy": False,
            "error": str(e),
            "severity": "high"
        }
```

## 🔐 **امنیت و بهترین روش‌ها**

### **API Key Management**
```python
import os
from cryptography.fernet import Fernet

class SecureConfigManager:
    def __init__(self):
        self.key = self._get_or_create_key()
        self.cipher = Fernet(self.key)
    
    def _get_or_create_key(self):
        """دریافت یا ایجاد کلید رمزنگاری"""
        key_file = Path("dashboard/.secret_key")
        if key_file.exists():
            return key_file.read_bytes()
        else:
            key = Fernet.generate_key()
            key_file.write_bytes(key)
            return key
    
    def encrypt_api_key(self, api_key: str) -> str:
        """رمزنگاری API Key"""
        return self.cipher.encrypt(api_key.encode()).decode()
    
    def decrypt_api_key(self, encrypted_key: str) -> str:
        """رمزگشایی API Key"""
        return self.cipher.decrypt(encrypted_key.encode()).decode()
    
    def store_secure_config(self, config_data: dict):
        """ذخیره ایمن تنظیمات"""
        secure_config = {}
        
        # رمزنگاری فیلدهای حساس
        sensitive_fields = ["api_key", "api_secret", "bot_token"]
        
        for key, value in config_data.items():
            if key in sensitive_fields:
                secure_config[key] = self.encrypt_api_key(value)
            else:
                secure_config[key] = value
        
        # ذخیره در فایل محافظت شده
        config_file = Path("dashboard/secure_config.json")
        with open(config_file, 'w') as f:
            json.dump(secure_config, f, indent=2)
        
        # تنظیم مجوزهای فایل (فقط خواندن برای مالک)
        os.chmod(config_file, 0o600)
```

## 🎯 **نتیجه‌گیری و مراحل بعدی**

### **آماده برای شروع:**
1. ✅ طرح کامل و جزئیات فنی تعریف شده
2. ✅ ورکفلوهای کاربری مشخص شده  
3. ✅ معماری و تکنولوژی انتخاب شده
4. ✅ کدهای کلیدی نوشته شده
5. ✅ فازهای پیاده‌سازی تعریف شده

### **اولین قدم:**
```bash
# ایجاد محیط داشبورد
mkdir dashboard
cd dashboard

# نصب وابستگی‌ها
pip install fastapi uvicorn sqlmodel aiofiles python-multipart

# شروع پیاده‌سازی
python -c "print('🚀 شروع پیاده‌سازی داشبورد')"
```

### **اصل راهنما:**
**"هر قدم باید قابل مشاهده، قابل تست و قابل کنترل باشد"**

این راهنما تمام جزئیات لازم برای شروع پیاده‌سازی را فراهم می‌کند و تضمین می‌کند که کاربر از ابتدا تا انتها کنترل کامل بر فرآیند داشته باشد.
