# ورکفلوهای کاربری و راه‌اندازی اولیه سیستم

## 🎯 **اصل طلایی: "چیزی را که نتوانیم ببینیم نمیتوانیم کنترل کنیم"**

این سند ورکفلوهای کاربری کامل را تعریف می‌کند تا کاربر از لحظه اول تا استفاده پیشرفته، کنترل کامل بر سیستم داشته باشد.

## 📋 **ورکفلو 1: راه‌اندازی اولیه سیستم (First-Time Setup)**

### **مرحله 1.1: بررسی پیش‌نیازها**
```
Dashboard Startup Checklist:
┌─────────────────────────────────────────┐
│ ✓ Python 3.8+ installed                │
│ ✓ Required packages available           │
│ ✓ Minimum 8GB RAM                       │
│ ✓ 10GB+ free disk space                │
│ ✓ Internet connection active            │
└─────────────────────────────────────────┘
```

**API Endpoint:**
```python
@router.get("/system/health-check")
async def system_health_check():
    return {
        "python_version": "3.9.7",
        "memory_gb": 16.0,
        "disk_free_gb": 45.2,
        "internet": True,
        "dependencies": {
            "pandas": "2.0.3",
            "numpy": "1.24.3",
            "missing": ["ta-lib"]  # نیاز به نصب
        }
    }
```

### **مرحله 1.2: تشخیص کانفیگ‌های موجود**
```
Available Configurations Discovery:
┌─────────────────────────────────────────┐
│ 📁 configs/                             │
│   ├── config-sample-1min.jsonc ✓       │
│   ├── config-sample-1h.jsonc ✓         │
│   ├── config-mt5-sample-1h.jsonc ✓     │
│   └── user-custom.jsonc (create new)   │
└─────────────────────────────────────────┘
```

**Frontend Component:**
```html
<div class="setup-wizard">
    <h2>انتخاب یا ایجاد کانفیگ</h2>
    <div class="config-options">
        <div class="existing-configs">
            <h3>کانفیگ‌های موجود:</h3>
            <div class="config-card" data-config="config-sample-1min.jsonc">
                <h4>نمونه 1 دقیقه</h4>
                <p>Symbol: BTCUSDT | Frequency: 1min</p>
                <span class="status ready">آماده استفاده</span>
            </div>
        </div>
        <div class="create-new">
            <button class="btn-primary">ایجاد کانفیگ جدید</button>
        </div>
    </div>
</div>
```

### **مرحله 1.3: تنظیم API Keys**
```
API Configuration Wizard:
┌─────────────────────────────────────────┐
│ 🔑 Binance API Setup                   │
│   ├── API Key: [_______________] Test   │
│   ├── Secret: [_______________] Test    │
│   └── Status: ❌ Not Connected         │
│                                         │
│ 🔑 MetaTrader 5 Setup (Optional)       │
│   ├── Account: [_______________]        │
│   ├── Password: [_______________]       │
│   ├── Server: [_______________]         │
│   └── Status: ⚠️ Not Configured        │
└─────────────────────────────────────────┘
```

**API Validation:**
```python
@router.post("/setup/validate-binance")
async def validate_binance_api(api_key: str, api_secret: str):
    try:
        client = Client(api_key=api_key, api_secret=api_secret)
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
```

### **مرحله 1.4: تنظیم Telegram Bot**
```
Telegram Bot Setup Wizard:
┌─────────────────────────────────────────┐
│ 📱 Telegram Notification Setup         │
│                                         │
│ Step 1: Create Bot                      │
│ ├── Go to @BotFather                    │
│ ├── Send: /newbot                       │
│ ├── Choose name: Trading Bot            │
│ └── Get Token: [_______________] Test   │
│                                         │
│ Step 2: Get Chat ID                     │
│ ├── Send message to your bot            │
│ ├── Visit: api.telegram.org/bot<token>/getUpdates │
│ ├── Find "chat":{"id": XXXXXXX}        │
│ └── Chat ID: [_______________] Test     │
│                                         │
│ Step 3: Test Connection                 │
│ └── [Send Test Message] ✓ Success!     │
└─────────────────────────────────────────┘
```

**Telegram Test API:**
```python
@router.post("/setup/test-telegram")
async def test_telegram(bot_token: str, chat_id: str):
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = {
            "chat_id": chat_id,
            "text": "🤖 تست اتصال ربات ترید موفقیت آمیز بود!"
        }
        
        response = requests.post(url, data=data)
        if response.status_code == 200:
            return {"success": True, "message": "پیام تست ارسال شد"}
        else:
            return {"success": False, "error": response.text}
            
    except Exception as e:
        return {"success": False, "error": str(e)}
```

## 📋 **ورکفلو 2: اجرای اولین تحلیل (First Analysis Run)**

### **مرحله 2.1: راهنمای گام به گام**
```
First Analysis Workflow:
┌─────────────────────────────────────────┐
│ 📊 اجرای اولین تحلیل                   │
│                                         │
│ Step 1: Download Data ⏱️ 2-5 min       │
│ ├── دانلود داده‌های BTCUSDT             │
│ ├── Progress: [████████░░] 80%         │
│ └── Status: Downloading...              │
│                                         │
│ Step 2: Merge Data ⏱️ 1-2 min          │
│ ├── ادغام داده‌های مختلف                │
│ └── Status: Waiting...                  │
│                                         │
│ Step 3: Generate Features ⏱️ 5-15 min  │
│ ├── تولید ویژگی‌های فنی                │
│ └── Status: Waiting...                  │
│                                         │
│ Step 4: Generate Labels ⏱️ 3-8 min     │
│ ├── تولید برچسب‌های ML                 │
│ └── Status: Waiting...                  │
│                                         │
│ Step 5: Train Model ⏱️ 30-120 min      │
│ ├── آموزش مدل یادگیری ماشین            │
│ └── Status: Waiting...                  │
│                                         │
│ [Start Analysis Pipeline] [Pause] [Stop] │
└─────────────────────────────────────────┘
```

### **مرحله 2.2: مانیتورینگ Real-time**
```html
<div class="analysis-monitor">
    <div class="current-step">
        <h3>مرحله فعلی: دانلود داده‌ها</h3>
        <div class="progress-bar">
            <div class="progress" style="width: 65%"></div>
        </div>
        <p>در حال دانلود BTCUSDT از Binance...</p>
    </div>
    
    <div class="live-logs">
        <h4>لاگ‌های زنده:</h4>
        <div class="log-container">
            <div class="log-line">2025-01-09 11:20:15 - شروع دانلود BTCUSDT</div>
            <div class="log-line">2025-01-09 11:20:16 - دریافت 1000 کندل</div>
            <div class="log-line">2025-01-09 11:20:17 - ذخیره در data.csv</div>
        </div>
    </div>
    
    <div class="system-resources">
        <h4>منابع سیستم:</h4>
        <div class="resource-item">
            <span>CPU:</span>
            <div class="meter"><div style="width: 45%"></div></div>
            <span>45%</span>
        </div>
        <div class="resource-item">
            <span>RAM:</span>
            <div class="meter"><div style="width: 62%"></div></div>
            <span>5.2GB / 8GB</span>
        </div>
    </div>
</div>
```

## 📋 **ورکفلو 3: مدیریت روزانه سیستم**

### **مرحله 3.1: Dashboard اصلی**
```
Daily Operations Dashboard:
┌─────────────────────────────────────────┐
│ 🎛️ کنترل پنل اصلی                      │
│                                         │
│ 📊 وضعیت کلی سیستم                     │
│ ├── آخرین تحلیل: 2 ساعت پیش ✓          │
│ ├── اتصال Binance: متصل ✓               │
│ ├── اعلان‌های Telegram: فعال ✓          │
│ └── معاملات خودکار: غیرفعال ⚠️          │
│                                         │
│ 🔄 عملیات سریع                         │
│ ├── [بروزرسانی داده‌ها]                │
│ ├── [اجرای تحلیل جدید]                 │
│ ├── [ارسال گزارش]                      │
│ └── [تنظیمات اعلان‌ها]                 │
│                                         │
│ 📈 آخرین سیگنال‌ها                     │
│ ├── BTC: Score +0.23 (خرید ضعیف)       │
│ ├── زمان: 11:15 AM                     │
│ └── [مشاهده جزییات]                    │
└─────────────────────────────────────────┘
```

### **مرحله 3.2: کنترل اعلان‌ها**
```html
<div class="notification-control">
    <h3>مدیریت اعلان‌ها</h3>
    
    <div class="notification-types">
        <div class="notification-item">
            <label>
                <input type="checkbox" checked> 
                اعلان‌های سیگنال (Score)
            </label>
            <div class="settings">
                <label>حد آستانه: <input type="number" value="0.15" step="0.01"></label>
                <label>فرکانس حداکثر: <select><option>هر 5 دقیقه</option></select></label>
            </div>
        </div>
        
        <div class="notification-item">
            <label>
                <input type="checkbox"> 
                اعلان‌های معاملات
            </label>
            <div class="settings disabled">
                <p>⚠️ معاملات خودکار غیرفعال است</p>
            </div>
        </div>
        
        <div class="notification-item">
            <label>
                <input type="checkbox" checked> 
                گزارش‌های روزانه
            </label>
            <div class="settings">
                <label>زمان ارسال: <input type="time" value="08:00"></label>
            </div>
        </div>
    </div>
    
    <div class="test-section">
        <button class="btn-secondary">ارسال پیام تست</button>
        <div class="last-sent">آخرین ارسال: 2 ساعت پیش ✓</div>
    </div>
</div>
```

## 📋 **ورکفلو 4: عیب‌یابی و بازیابی**

### **مرحله 4.1: تشخیص مشکلات**
```
System Diagnostics:
┌─────────────────────────────────────────┐
│ 🔍 تشخیص خودکار مشکلات                │
│                                         │
│ ✓ اتصال اینترنت: OK                    │
│ ✓ API Binance: OK (Response: 120ms)    │
│ ❌ Telegram Bot: FAILED                 │
│   └── Error: Unauthorized (401)        │
│   └── Fix: بررسی Bot Token             │
│                                         │
│ ⚠️ فضای دیسک: 85% پر                   │
│   └── Warning: کمتر از 2GB باقی مانده  │
│   └── Action: پاکسازی فایل‌های قدیمی   │
│                                         │
│ ✓ Python Dependencies: OK              │
│ ✓ Config Files: OK                     │
│                                         │
│ [Run Full Diagnostic] [Auto Fix]       │
└─────────────────────────────────────────┘
```

### **مرحله 4.2: بازیابی خودکار**
```python
@router.post("/system/auto-recovery")
async def auto_recovery():
    recovery_actions = []
    
    # بررسی و تعمیر مشکلات رایج
    if not check_telegram_connection():
        recovery_actions.append({
            "action": "telegram_reconnect",
            "status": "attempting",
            "description": "تلاش برای اتصال مجدد به Telegram"
        })
    
    if get_disk_usage() > 0.9:  # بیش از 90% پر
        recovery_actions.append({
            "action": "cleanup_old_files", 
            "status": "completed",
            "description": "پاکسازی فایل‌های قدیمی انجام شد"
        })
    
    return {"recovery_actions": recovery_actions}
```

## 📋 **ورکفلو 5: API Discovery و مستندات**

### **مرحله 5.1: کشف API های موجود**
```
Available APIs Discovery:
┌─────────────────────────────────────────┐
│ 🌐 API های در دسترس                    │
│                                         │
│ 📊 Script Management                    │
│ ├── POST /api/scripts/run/{name}        │
│ ├── GET /api/scripts/status/{job_id}    │
│ ├── GET /api/scripts/logs/{name}/stream │
│ └── GET /api/scripts/list               │
│                                         │
│ ⚙️ Configuration                        │
│ ├── GET /api/configs/list               │
│ ├── GET /api/configs/{name}             │
│ ├── POST /api/configs/{name}            │
│ └── POST /api/configs/validate          │
│                                         │
│ 🔔 Notifications                        │
│ ├── GET /api/notifications/status       │
│ ├── POST /api/notifications/toggle      │
│ ├── POST /api/notifications/test        │
│ └── GET /api/notifications/history      │
│                                         │
│ 📈 System Monitoring                    │
│ ├── GET /api/system/health              │
│ ├── GET /api/system/resources           │
│ └── GET /api/system/diagnostics         │
│                                         │
│ [View Full API Docs] [Test APIs]       │
└─────────────────────────────────────────┘
```

### **مرحله 5.2: Interactive API Testing**
```html
<div class="api-explorer">
    <h3>تست API ها</h3>
    
    <div class="api-endpoint">
        <div class="endpoint-header">
            <span class="method POST">POST</span>
            <span class="url">/api/scripts/run/download_binance</span>
        </div>
        
        <div class="endpoint-body">
            <h4>Request Body:</h4>
            <textarea class="json-input">{
  "config_file": "config-sample-1min.jsonc"
}</textarea>
            
            <button class="btn-test">Test API</button>
            
            <div class="response-section">
                <h4>Response:</h4>
                <pre class="response-output">{
  "job_id": "download_binance_1641234567",
  "status": "started",
  "estimated_duration": "2-5 minutes"
}</pre>
            </div>
        </div>
    </div>
</div>
```

## 🎯 **تکمیل طرح با ورکفلوهای کاربری**

### **فاز 0 اصلاح شده: System Discovery (روز 1-2)**
```
Day 1: Environment & Discovery
├── System health check
├── Config files discovery  
├── API capabilities mapping
├── Existing data assessment
└── User requirements gathering

Day 2: Initial Setup Wizard
├── API keys configuration
├── Telegram bot setup
├── First test run
├── Validation & troubleshooting
└── User training/walkthrough
```

### **فاز 1 اصلاح شده: User-Centric MVP (هفته 1)**
```
Week 1: Complete User Experience
├── Setup wizard (Day 1-2)
├── Guided first analysis (Day 3-4)  
├── Dashboard with full visibility (Day 5-6)
├── Error handling & recovery (Day 7)
└── User documentation & training
```

### **Integration Points:**
- هر مرحله دارای **Setup Wizard** مخصوص
- **Real-time Monitoring** در تمام عملیات
- **Auto-Discovery** برای تنظیمات و API ها
- **Interactive Testing** برای validation
- **Recovery Workflows** برای مشکلات رایج

این ورکفلوهای کاربری **کنترل کامل** را از لحظه اول تا استفاده پیشرفته فراهم می‌کنند و اصل طلایی "دیدن = کنترل کردن" را محقق می‌سازند.
