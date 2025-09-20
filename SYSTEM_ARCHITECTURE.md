# 🤖 سند کامل معماری سیستم ربات ترید هوشمند

## 📁 ساختار کلی پروژه

```
intelligent-trading-bot/
├── 📂 scripts/           # اسکریپت‌های اصلی پردازش (Batch Processing)
├── 📂 service/           # سرویس آنلاین (Online Service)  
├── 📂 inputs/            # ماژول‌های جمع‌آوری داده (Data Collectors)
├── 📂 outputs/           # ماژول‌های خروجی (Notifiers & Traders)
├── 📂 common/            # کتابخانه‌های مشترک (مشاهده: [COMMON_MODULES.md](./COMMON_MODULES.md))
├── 📂 configs/           # فایل‌های پیکربندی
└── 📂 dashboard/         # رابط وب مدیریت
```

---

## 🔄 Workflow کامل تولید سیگنال Bitcoin

### 🎯 **فاز آماده‌سازی (Offline Batch Processing)**

#### 1️⃣ **مرحله جمع‌آوری داده**
```bash
# دانلود داده‌های تاریخی از Binance
python scripts/download_binance.py -c configs/config-sample-1h.jsonc

# دانلود داده‌های تاریخی از MT5  
python scripts/download_mt5.py -c configs/config-sample-1h.jsonc

# دانلود داده‌های تاریخی از Yahoo Finance
python scripts/download_yahoo.py -c configs/config-sample-1h.jsonc
```

#### 2️⃣ **مرحله پردازش داده**
```bash
# ادغام داده‌ها از منابع مختلف
python scripts/merge.py -c configs/config-sample-1h.jsonc
```

#### 3️⃣ **مرحله مهندسی ویژگی**
```bash
# استخراج ویژگی‌های تکنیکال (RSI, MACD, Moving Averages, etc.)
python scripts/features.py -c configs/config-sample-1h.jsonc

# تولید برچسب‌های آموزشی (Labels)
python scripts/labels.py -c configs/config-sample-1h.jsonc
```

#### 4️⃣ **مرحله آموزش مدل**
```bash
# آموزش مدل‌های یادگیری ماشین
python scripts/train.py -c configs/config-sample-1h.jsonc
```

#### 5️⃣ **مرحله پیش‌بینی**
```bash
# پیش‌بینی قیمت‌های آینده
python scripts/predict.py -c configs/config-sample-1h.jsonc

# پیش‌بینی با بازآموزی دوره‌ای (برای پروداکشن)
python scripts/predict_rolling.py -c configs/config-sample-1h.jsonc
```

#### 6️⃣ **مرحله تولید سیگنال**
```bash
# تولید سیگنال‌های معاملاتی نهایی
python scripts/signals.py -c configs/config-sample-1h.jsonc
```

#### 7️⃣ **مرحله ارزیابی (اختیاری)**
```bash
# شبیه‌سازی معاملات و بک‌تست
python scripts/simulate.py -c configs/config-sample-1h.jsonc

# تولید گزارش‌های خروجی  
python scripts/output.py -c configs/config-sample-1h.jsonc
```

### 🚀 **فاز تولید زنده (Online Service)**

#### 8️⃣ **سرویس آنلاین**
```bash
# راه‌اندازی سرویس تولید سیگنال‌های زنده
python service/server.py -c configs/config-sample-1h.jsonc
```

---

## 📋 شرح تفصیلی هر فایل

### 🎯 **Scripts Directory** (پردازش دسته‌ای)

| فایل | نقش | CLI Arguments | مدت اجرا | وابستگی‌ها |
|------|-----|---------------|-----------|-------------|
| **`download_binance.py`** | دانلود داده‌های OHLCV از Binance API | `-c config_file` | 2-5 دقیقه | Binance API, service.App |
| **`download_mt5.py`** | دانلود داده‌های OHLCV از MetaTrader 5 | `-c config_file` | 2-5 دقیقه | MT5 Terminal, service.App |
| **`download_yahoo.py`** | دانلود داده‌های OHLCV از Yahoo Finance | `-c config_file` | 1-2 دقیقه | yfinance, service.App |
| **`merge.py`** | ادغام و همگام‌سازی داده‌ها از منابع مختلف | `-c config_file` | 5-10 دقیقه | pandas, service.App |
| **`features.py`** | استخراج ویژگی‌های تکنیکال (RSI, MACD, etc.) | `-c config_file` | 10-20 دقیقه | common.generators, ModelStore |
| **`labels.py`** | تولید برچسب‌های آموزشی برای ML | `-c config_file` | 5-15 دقیقه | common.generators |
| **`train.py`** | آموزش مدل‌های یادگیری ماشین | `-c config_file` | 30-120 دقیقه | sklearn, common.classifiers |
| **`predict.py`** | پیش‌بینی با مدل‌های آموزش‌دیده | `-c config_file` | 1-3 دقیقه | ModelStore, ML models |
| **`predict_rolling.py`** | پیش‌بینی با بازآموزی دوره‌ای مدل‌ها | `-c config_file` | 60-300 دقیقه | ModelStore, تمام pipeline |
| **`signals.py`** | تولید سیگنال‌های معاملاتی نهایی | `-c config_file` | 2-5 دقیقه | ModelStore, predictions |
| **`simulate.py`** | شبیه‌سازی معاملات و بک‌تست | `-c config_file` | 10-30 دقیقه | signals, portfolio logic |
| **`output.py`** | تولید گزارش‌های خروجی و تحلیل | `-c config_file` | 1-2 دقیقه | analysis results |

### 🔧 **Service Directory** (سرویس آنلاین)

| فایل | نقش | نحوه استفاده | ویژگی‌ها |
|------|-----|-------------|----------|
| **`App.py`** | کلاس مرکزی global state management | `from service.App import *` | Variables, config loader, state |
| **`server.py`** | سرویس اصلی تولید سیگنال زنده | `python server.py -c config` | AsyncIO, Scheduler, Real-time |
| **`analyzer.py`** | آنالیز داده‌ها و تولید insights | Called by server.py | Data processing, ML inference |
| **`mt5.py`** | اتصال و مدیریت MetaTrader 5 | Integration module | MT5 Terminal connection |

### 📥 **Inputs Directory** (جمع‌آورکننده‌های داده)

| فایل | نقش | نحوه فراخوانی | عملکرد |
|------|-----|-------------|---------|
| **`collector_binance.py`** | جمع‌آوری real-time از Binance | `await main_collector_task()` | WebSocket, REST API, Live data |
| **`collector_mt5.py`** | جمع‌آوری real-time از MT5 | `await main_collector_task()` | MT5 Terminal, Live quotes |

### 📤 **Outputs Directory** (خروجی‌ها و اعلان‌گرها)

| فایل | نقش | نحوه فراخوانی | عملکرد |
|------|-----|-------------|---------|
| **`notifier_trades.py`** | ارسال اعلان معاملات به Telegram | Called by service | Trade notifications, Telegram Bot |
| **`notifier_scores.py`** | ارسال نمرات و آمار به کانال | Called by service | Score notifications, Analytics |
| **`notifier_diagram.py`** | ارسال نمودارها و چارت‌ها | Called by service | Chart generation, Image sending |
| **`trader_binance.py`** | اجرای معاملات روی Binance | Called by service | Auto-trading, Order execution |
| **`trader_mt5.py`** | اجرای معاملات روی MT5 | Called by service | Auto-trading via MT5 |

### 🌐 **Frontend**

| فایل | نقش | دسترسی | ویژگی‌ها |
|------|-----|---------|----------|
| **`simple-test.html`** | صفحه تست ساده dashboard | Browser interface | Quick testing, Debug |

---

## 🔗 **روابط و وابستگی‌ها**

### **Data Flow:**
```
Raw Data → Merge → Features → Labels → Train → Predict → Signals → Output
    ↓
Service.server (Real-time) → Collectors → Analyzer → Notifiers/Traders
```

### **Key Dependencies:**
- **service.App**: مرکز مدیریت کانفیگ و state - همه فایل‌ها از آن استفاده می‌کنند
- **common.model_store**: مدیریت مدل‌های ML - features, train, predict, signals استفاده می‌کنند
- **common.generators**: تولیدکننده‌های feature و signal - features, labels, signals استفاده می‌کنند

---

## ⚙️ **نحوه اجرا و فراخوانی**

### **🖥️ از Command Line:**
```bash
# همه اسکریپت‌ها از همین pattern استفاده می‌کنند
python scripts/[script_name].py -c configs/[config_file].jsonc
python service/server.py -c configs/[config_file].jsonc
```

### **🌐 از Dashboard:**
1. به آدرس `http://127.0.0.1:8000` بروید
2. قسمت "⚙️ مدیریت اسکریپت‌ها" را انتخاب کنید  
3. اسکریپت مورد نظر و config file را انتخاب کنید
4. دکمه "🚀 اجرا" را بزنید

### **📋 از کد Python:**
```python
# برای استفاده در کدهای سفارشی
from service.App import *
load_config('configs/config-sample-1h.jsonc')

# برای collector ها
from inputs.collector_binance import main_collector_task
await main_collector_task()

# برای notifier ها  
from outputs.notifier_trades import trader_simulation
await trader_simulation(df, model, config, model_store)
```

---

## 🎯 **سناریوهای کاربرد**

### **🔄 تولید سیگنال دوره‌ای (Offline)**
برای تحلیل‌های دوره‌ای و بروزرسانی مدل‌ها:
1. Download → Merge → Features → Labels → Train → Signals

### **⚡ تولید سیگنال زنده (Online)**  
برای سیگنال‌های real-time:
1. `service.server` (دائمی اجرا شود)
2. Collectors جمع‌آوری کنند
3. Analyzer تحلیل کند
4. Notifiers ارسال کنند

### **📊 بک‌تست و ارزیابی**
برای بررسی عملکرد:
1. Complete workflow → Simulate → Output

---

## ⚠️ **نکات مهم**

1. **ترتیب اجرا**: Scripts باید به ترتیب خاص اجرا شوند (وابستگی data pipeline)
2. **Config files**: همه اسکریپت‌ها نیاز به فایل کانفیگ معتبر دارند
3. **API Keys**: download_binance و service.server نیاز به Binance API keys دارند
4. **MT5 Terminal**: download_mt5 و collector_mt5 نیاز به MT5 terminal فعال دارند
5. **Telegram Bot**: notifiers نیاز به Telegram bot token دارند

---

## 🏁 **Quick Start برای Bitcoin Signals**

```bash
# 1. آماده‌سازی داده‌ها (یکبار)
python scripts/download_binance.py -c configs/config-sample-1h.jsonc
python scripts/merge.py -c configs/config-sample-1h.jsonc
python scripts/features.py -c configs/config-sample-1h.jsonc
python scripts/labels.py -c configs/config-sample-1h.jsonc
python scripts/train.py -c configs/config-sample-1h.jsonc

# 2. تولید سیگنال (دوره‌ای)
python scripts/predict.py -c configs/config-sample-1h.jsonc
python scripts/signals.py -c configs/config-sample-1h.jsonc

# 3. سرویس زنده (دائمی)
python service/server.py -c configs/config-sample-1h.jsonc
```

یا از **Dashboard** به آدرس: `http://127.0.0.1:8000`
