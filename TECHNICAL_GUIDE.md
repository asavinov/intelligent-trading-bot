# 🔧 راهنمای تکنیکی تفصیلی سیستم ربات ترید

## 📚 معرفی

این سند راهنمای فنی سیستم ربات ترید هوشمند است. برای اطلاعات کامل‌تر در مورد ماژول‌های مشترک، به فایل [COMMON_MODULES.md](./COMMON_MODULES.md) مراجعه کنید.

## 📋 **جزئیات فایل‌های کانفیگ**

### **🔧 ساختار فایل Config**
```jsonc
{
    // اطلاعات کلی
    "symbol": "BTCUSDT",              // نماد معاملاتی
    "freq": "1h",                     // بازه زمانی (1m, 5m, 15m, 1h, 4h, 1d)
    "time_column": "timestamp",       // نام ستون زمان
    
    // مسیرها
    "data_folder": "./data",          // مسیر پوشه داده‌ها
    "model_folder": "./models",       // مسیر پوشه مدل‌ها
    
    // منابع داده
    "data_sources": [
        {
            "folder": "BTCUSDT",       // نام پوشه
            "file": "klines",          // نام فایل
            "column_prefix": ""        // پیشوند ستون‌ها
        }
    ],
    
    // API کلیدها
    "api_key": "your_binance_api_key",
    "api_secret": "your_binance_secret",
    
    // تنظیمات آموزش
    "train": true,                    // حالت آموزش
    "train_length": 100000,           // تعداد رکورد آموزش
    "predict_length": 1000,           // تعداد رکورد پیش‌بینی
    
    // تنظیمات ویژگی
    "features_horizon": 100,          // افق ویژگی‌ها
    "feature_sets": [
        {
            "generator": "talib",      // نوع تولیدکننده
            "columns": ["close"],      // ستون‌های مورد استفاده
            "functions": [             // توابع تکنیکال
                {"talib_name": "RSI", "period": 14},
                {"talib_name": "MACD", "fastperiod": 12, "slowperiod": 26}
            ]
        }
    ],
    
    // تنظیمات Telegram
    "telegram_bot_token": "your_bot_token",
    "telegram_chat_id": "your_chat_id"
}
```

---

## 📊 **فرمت‌های فایل‌های داده**

### **🗂️ فایل‌های ورودی (Input Files)**

#### **1. فایل Klines (OHLCV)**
```csv
timestamp,open,high,low,close,volume
2024-01-01T00:00:00,43000.5,43500.0,42800.0,43200.0,1234.56
2024-01-01T01:00:00,43200.0,43800.0,43100.0,43650.0,987.34
```

#### **2. فایل Merged Data**
```csv
timestamp,open,high,low,close,volume,rsi_14,macd,signal_line
2024-01-01T00:00:00,43000.5,43500.0,42800.0,43200.0,1234.56,65.4,12.3,8.7
```

### **🗂️ فایل‌های خروجی (Output Files)**

#### **1. فایل Features**
```csv
timestamp,close,rsi_14,macd,bb_upper,bb_lower,ema_20,sma_50
2024-01-01T00:00:00,43200.0,65.4,12.3,44000.0,42000.0,43100.0,42900.0
```

#### **2. فایل Predictions**  
```csv
timestamp,prediction_buy_prob,prediction_sell_prob,predicted_price
2024-01-01T00:00:00,0.75,0.15,43500.0
```

#### **3. فایل Signals**
```csv
timestamp,signal,confidence,side,entry_price,stop_loss,take_profit
2024-01-01T00:00:00,BUY,0.85,LONG,43200.0,42500.0,44500.0
```

---

## 🔄 **مثال‌های کاربردی**

### **📥 استفاده از Collectors**

```python
# نمونه استفاده از Binance Collector
import asyncio
from inputs.collector_binance import main_collector_task
from service.App import *

async def collect_live_data():
    # بارگذاری کانفیگ
    load_config('configs/config-sample-1h.jsonc')
    
    # اجرای collector
    result = await main_collector_task()
    
    if result == 0:
        print("✅ داده‌ها با موفقیت جمع‌آوری شدند")
        print(f"📊 آخرین داده: {App.df.tail(1)}")
    else:
        print("❌ خطا در جمع‌آوری داده‌ها")

# اجرا
asyncio.run(collect_live_data())
```

### **📤 استفاده از Notifiers**

```python
# نمونه ارسال سیگنال به Telegram
import asyncio
from outputs.notifier_trades import send_transaction_message
from service.App import *

async def send_signal():
    load_config('configs/config-sample-1h.jsonc')
    
    # ساخت transaction نمونه
    transaction = {
        'symbol': 'BTCUSDT',
        'side': 'BUY',
        'price': 43200.0,
        'confidence': 0.85,
        'timestamp': '2024-01-01T00:00:00'
    }
    
    # ارسال به Telegram
    await send_transaction_message(transaction, App.config)
    print("📱 سیگنال به Telegram ارسال شد")

asyncio.run(send_signal())
```

### **🤖 استفاده از Service Server**

```python
# نمونه راه‌اندازی سرویس کامل
import asyncio
from service.server import main_task
from service.App import *

async def start_trading_service():
    # بارگذاری کانفیگ
    load_config('configs/config-sample-1h.jsonc')
    
    print("🚀 شروع سرویس ترید...")
    
    # اجرای تسک اصلی
    try:
        await main_task()
        print("✅ سرویس با موفقیت اجرا شد")
    except Exception as e:
        print(f"❌ خطا در سرویس: {e}")

# راه‌اندازی با scheduler
asyncio.run(start_trading_service())
```

---

## 🛠️ **ابزارهای پیشرفته**

### **📊 مانیتورینگ عملکرد**

```python
# نمونه مانیتورینگ pipeline
from pathlib import Path
import pandas as pd
from service.App import *

def monitor_pipeline_health():
    load_config('configs/config-sample-1h.jsonc')
    
    data_path = Path(App.config["data_folder"]) / App.config["symbol"]
    
    # بررسی فایل‌های موجود
    files_to_check = [
        "klines.csv",           # داده خام
        "merged.csv",           # داده ادغام شده  
        "features.csv",         # ویژگی‌ها
        "predictions.csv",      # پیش‌بینی‌ها
        "signals.csv"           # سیگنال‌ها
    ]
    
    print("🔍 بررسی سلامت Pipeline:")
    for file_name in files_to_check:
        file_path = data_path / file_name
        if file_path.exists():
            df = pd.read_csv(file_path)
            last_update = df['timestamp'].iloc[-1]
            print(f"✅ {file_name}: {len(df)} رکورد - آخرین بروزرسانی: {last_update}")
        else:
            print(f"❌ {file_name}: فایل موجود نیست")

monitor_pipeline_health()
```

### **🧪 تست A/B مدل‌ها**

```python
# نمونه مقایسه مدل‌های مختلف
import pandas as pd
from common.model_store import ModelStore
from service.App import *

def compare_models():
    load_config('configs/config-sample-1h.jsonc')
    
    # بارگذاری مدل‌ها
    model_store = ModelStore(App.config)
    model_store.load_models()
    
    # داده‌های تست
    data_path = Path(App.config["data_folder"]) / App.config["symbol"]
    df = pd.read_csv(data_path / "features.csv")
    
    print("📊 مقایسه عملکرد مدل‌ها:")
    
    for model_name, model_info in model_store.models.items():
        if model_info.get('model'):
            # اجرای پیش‌بینی
            predictions = model_info['model'].predict(df[model_info['features']])
            accuracy = calculate_accuracy(predictions, df['actual_labels'])
            print(f"🎯 {model_name}: دقت = {accuracy:.2%}")

compare_models()
```

---

## ⚡ **بهینه‌سازی عملکرد**

### **🚀 تسریع پردازش**

```python
# استفاده از پردازش موازی
import multiprocessing as mp
from functools import partial

def process_symbol_batch(symbols, config_file):
    """پردازش موازی چندین نماد"""
    
    def process_single_symbol(symbol):
        # تغییر symbol در کانفیگ
        config = load_config(config_file)
        config['symbol'] = symbol
        
        # اجرای pipeline کامل
        subprocess.run(['python', 'scripts/download_binance.py', '-c', config_file])
        subprocess.run(['python', 'scripts/features.py', '-c', config_file])
        subprocess.run(['python', 'scripts/predict.py', '-c', config_file])
        
        return f"✅ {symbol} پردازش شد"
    
    # اجرای موازی
    with mp.Pool() as pool:
        results = pool.map(process_single_symbol, symbols)
    
    return results

# مثال استفاده
symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
results = process_symbol_batch(symbols, 'configs/config-sample-1h.jsonc')
print(results)
```

### **💾 مدیریت حافظه**

```python
# بهینه‌سازی استفاده از حافظه برای داده‌های بزرگ
import pandas as pd

def optimize_dataframe_memory(df):
    """کاهش استفاده از حافظه"""
    
    # تبدیل نوع داده‌ها
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif df[col].dtype == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
    
    return df

def process_large_dataset_chunked(file_path, chunk_size=10000):
    """پردازش فایل‌های بزرگ به صورت chunk"""
    
    results = []
    
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        # پردازش هر chunk
        chunk = optimize_dataframe_memory(chunk)
        processed_chunk = apply_features(chunk)
        results.append(processed_chunk)
    
    # ترکیب نتایج
    return pd.concat(results, ignore_index=True)
```

---

## 🐛 **عیب‌یابی رایج**

### **❌ مشکلات متداول و راه‌حل**

#### **1. خطای API Key**
```bash
ERROR: Invalid API key
```
**راه‌حل:**
- بررسی صحت API key و secret در فایل کانفیگ
- اطمینان از فعال بودن API key در Binance
- بررسی permissions (Spot Trading, Futures Trading)

#### **2. خطای اتصال به MT5**
```bash
ERROR: MT5 terminal not found
```
**راه‌حل:**
- اطمینان از باز بودن MT5 Terminal  
- فعال کردن "Allow automated trading" در MT5
- بررسی نصب کتابخانه MetaTrader5

#### **3. خطای حافظه کم**
```bash
MemoryError: Unable to allocate array
```
**راه‌حل:**
- کاهش `train_length` در کانفیگ
- استفاده از chunk processing
- افزایش virtual memory

#### **4. خطای فایل موجود نیست**
```bash
FileNotFoundError: Input file does not exist
```
**راه‌حل:**
- اجرای اسکریپت‌های قبلی در workflow
- بررسی مسیر فایل‌ها در کانفیگ
- اطمینان از وجود پوشه‌های لازم

### **🔍 ابزارهای عیب‌یابی**

```python
# اسکریپت تشخیص مشکل
def diagnose_system():
    """تشخیص مشکلات سیستم"""
    
    print("🔍 تشخیص مشکلات سیستم...")
    
    # بررسی فایل‌های کانفیگ
    config_files = Path('configs').glob('*.jsonc')
    for config_file in config_files:
        try:
            load_config(str(config_file))
            print(f"✅ کانفیگ {config_file.name} معتبر است")
        except Exception as e:
            print(f"❌ کانفیگ {config_file.name} دارای خطا: {e}")
    
    # بررسی اتصال API
    try:
        from binance import Client
        client = Client(api_key="test", api_secret="test")
        server_time = client.get_server_time()
        print("✅ اتصال به Binance API سالم است")
    except Exception as e:
        print(f"❌ مشکل در اتصال به Binance API: {e}")
    
    # بررسی فضای دیسک
    import shutil
    total, used, free = shutil.disk_usage('.')
    free_gb = free // (2**30)
    if free_gb < 5:
        print(f"⚠️ فضای دیسک کم: {free_gb} GB")
    else:
        print(f"✅ فضای دیسک کافی: {free_gb} GB")

diagnose_system()
```

---

## 📈 **نظارت و گزارش‌گیری**

### **📊 Dashboard سفارشی**

```python
# نمونه dashboard ساده
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def create_monitoring_dashboard():
    st.title("📊 داشبورد مانیتورینگ ربات ترید")
    
    # بارگذاری داده‌ها
    signals_df = pd.read_csv('data/BTCUSDT/signals.csv')
    
    # نمایش آمار کلی
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_signals = len(signals_df)
        st.metric("کل سیگنال‌ها", total_signals)
    
    with col2:
        buy_signals = len(signals_df[signals_df['signal'] == 'BUY'])
        st.metric("سیگنال‌های خرید", buy_signals)
    
    with col3:
        avg_confidence = signals_df['confidence'].mean()
        st.metric("میانگین اعتماد", f"{avg_confidence:.2%}")
    
    # نمودار قیمت و سیگنال‌ها
    fig = go.Figure()
    
    # قیمت
    fig.add_trace(go.Scatter(
        x=signals_df['timestamp'],
        y=signals_df['entry_price'],
        mode='lines',
        name='قیمت Bitcoin'
    ))
    
    # سیگنال‌های خرید
    buy_df = signals_df[signals_df['signal'] == 'BUY']
    fig.add_trace(go.Scatter(
        x=buy_df['timestamp'],
        y=buy_df['entry_price'],
        mode='markers',
        marker=dict(color='green', size=10),
        name='خرید'
    ))
    
    st.plotly_chart(fig)

# اجرا: streamlit run monitoring_dashboard.py
```

---

## 🎯 **بهترین روش‌ها (Best Practices)**

### **✅ توصیه‌های مهم**

1. **🔐 امنیت API Keys:**
   - هرگز API keys را در کد commit نکنید
   - از environment variables استفاده کنید
   - API keys را محدود به permissions لازم کنید

2. **💾 Backup منظم:**
   - مدل‌های آموزش‌دیده را backup کنید
   - فایل‌های کانفیگ را version control کنید
   - داده‌های مهم را در cloud ذخیره کنید

3. **📊 مانیتورینگ مستمر:**
   - عملکرد مدل‌ها را روزانه بررسی کنید
   - Log files را منظم چک کنید
   - Alert های خطا تنظیم کنید

4. **⚡ بهینه‌سازی:**
   - از cache برای داده‌های تکراری استفاده کنید
   - پردازش را در ساعات کم‌ترافیک انجام دهید
   - Memory leaks را بررسی کنید

5. **🧪 تست مستمر:**
   - Paper trading قبل از live trading
   - A/B testing برای مدل‌های جدید
   - Backtesting منظم روی داده‌های جدید

---

یک سیستم کامل و قدرتمند برای تولید سیگنال‌های ترید هوشمند! 🚀
