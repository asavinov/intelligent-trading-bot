"""
Database Models and Setup - Simple SQLite Implementation
مدل‌های پایگاه داده و تنظیمات
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

# Simple SQLite database setup without SQLAlchemy
DATABASE_PATH = Path(__file__).parent.parent / "dashboard.db"


def get_connection():
    """دریافت اتصال به پایگاه داده"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """راه‌اندازی اولیه پایگاه داده"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Create script_runs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS script_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT UNIQUE,
            script_name TEXT,
            config_file TEXT,
            status TEXT,
            start_time TEXT,
            end_time TEXT,
            exit_code INTEGER,
            pid INTEGER,
            output_log TEXT,
            error_log TEXT,
            estimated_duration TEXT,
            actual_duration REAL
        )
    ''')
    
    # Create system_health table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_health (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            overall_status TEXT,
            cpu_percent REAL,
            memory_percent REAL,
            memory_used_gb REAL,
            memory_total_gb REAL,
            disk_percent REAL,
            disk_free_gb REAL,
            internet_healthy INTEGER,
            binance_healthy INTEGER,
            telegram_healthy INTEGER,
            details TEXT
        )
    ''')
    
    # Create configurations table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS configurations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE,
            symbol TEXT,
            frequency TEXT,
            venue TEXT,
            is_active INTEGER,
            last_used TEXT,
            validation_status TEXT,
            validation_details TEXT,
            created_at TEXT,
            updated_at TEXT
        )
    ''')
    
    # Create api_connections table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS api_connections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            service_name TEXT,
            status TEXT,
            last_test TEXT,
            response_time_ms REAL,
            error_message TEXT,
            config_details TEXT
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Database initialized successfully")


def get_db():
    """دریافت session پایگاه داده - compatibility function"""
    return get_connection()


class ScriptRunModel:
    """مدل اجرای اسکریپت"""
    
    @staticmethod
    def create(job_id: str, script_name: str, config_file: str, 
               pid: int, estimated_duration: str) -> bool:
        """ایجاد رکورد جدید"""
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO script_runs 
                (job_id, script_name, config_file, status, start_time, pid, estimated_duration)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (job_id, script_name, config_file, 'running', 
                  datetime.now().isoformat(), pid, estimated_duration))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error creating script run: {e}")
            return False
    
    @staticmethod
    def update_status(job_id: str, status: str, exit_code: int = None, 
                     end_time: str = None) -> bool:
        """بروزرسانی وضعیت"""
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE script_runs 
                SET status = ?, exit_code = ?, end_time = ?
                WHERE job_id = ?
            ''', (status, exit_code, end_time or datetime.now().isoformat(), job_id))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error updating script run: {e}")
            return False
    
    @staticmethod
    def get_active() -> List[Dict[str, Any]]:
        """دریافت اسکریپت‌های فعال"""
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM script_runs 
                WHERE status = 'running' 
                ORDER BY start_time DESC
            ''')
            rows = cursor.fetchall()
            conn.close()
            return [dict(row) for row in rows]
        except Exception as e:
            print(f"Error getting active scripts: {e}")
            return []


class SystemHealthModel:
    """مدل سلامت سیستم"""
    
    @staticmethod
    def log_health(health_data: Dict[str, Any]) -> bool:
        """ثبت وضعیت سلامت"""
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO system_health 
                (timestamp, overall_status, cpu_percent, memory_percent, 
                 memory_used_gb, memory_total_gb, disk_percent, disk_free_gb,
                 internet_healthy, binance_healthy, telegram_healthy, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                health_data.get('overall_status'),
                health_data.get('cpu', {}).get('usage_percent'),
                health_data.get('memory', {}).get('usage_percent'),
                health_data.get('memory', {}).get('used_gb'),
                health_data.get('memory', {}).get('total_gb'),
                health_data.get('disk', {}).get('usage_percent'),
                health_data.get('disk', {}).get('free_gb'),
                health_data.get('internet', {}).get('healthy', 0),
                health_data.get('binance_api', {}).get('healthy', 0),
                health_data.get('telegram', {}).get('healthy', 0),
                json.dumps(health_data)
            ))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error logging health: {e}")
            return False
