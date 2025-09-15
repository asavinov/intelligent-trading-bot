"""
Test script for the Dashboard using Playwright
"""
import asyncio
from playwright.async_api import async_playwright
import time
import json
import os

# Configuration
DASHBOARD_URL = "http://localhost:8000"
SCREENSHOT_DIR = "test_screenshots"

# Create screenshots directory if it doesn't exist
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

async def take_screenshot(page, name):
    """Helper function to take screenshots"""
    path = os.path.join(SCREENSHOT_DIR, f"{name}.png")
    await page.screenshot(path=path)
    print(f"Screenshot saved: {path}")

async def test_dashboard():
    """Main test function"""
    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch(headless=False, slow_mo=100)
        context = await browser.new_context()
        page = await context.new_page()
        
        # Set viewport size
        await page.set_viewport_size({"width": 1280, "height": 800})
        
        # Test 1: Load Dashboard
        print("Testing dashboard loading...")
        try:
            await page.goto(DASHBOARD_URL, timeout=15000)
            await page.wait_for_selector("text=داشبورد ربات ترید هوشمند", timeout=10000)
            await take_screenshot(page, "01_dashboard_loaded")
            print("✓ Dashboard loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load dashboard: {str(e)}")
            await take_screenshot(page, "error_dashboard_load")
            return
        
        # Test 2: Check System Status
        print("Checking system status...")
        try:
            # Wait for status indicators to load
            await page.wait_for_selector(".status-indicator", timeout=10000)
            await take_screenshot(page, "02_system_status")
            print("✓ System status displayed")
        except Exception as e:
            print(f"✗ Failed to load system status: {str(e)}")
        
        # Test 3: Navigate to Scripts
        print("Testing scripts page...")
        try:
            await page.click("text=مدیریت اسکریپت‌ها")
            await page.wait_for_selector("h2:has-text('مدیریت اسکریپت‌ها')", timeout=5000)
            await take_screenshot(page, "03_scripts_page")
            print("✓ Scripts page loaded")
            
            # Check if script list is populated
            await page.wait_for_selector(".script-item", timeout=5000)
            script_count = await page.locator(".script-item").count()
            print(f"✓ Found {script_count} scripts")
            
        except Exception as e:
            print(f"✗ Failed to load scripts page: {str(e)}")
        
        # Test 4: Check Configurations
        print("Testing configurations...")
        try:
            await page.click("text=تنظیمات")
            await page.wait_for_selector("h2:has-text('مدیریت تنظیمات')", timeout=5000)
            await take_screenshot(page, "04_configs_page")
            print("✓ Configurations page loaded")
        except Exception as e:
            print(f"✗ Failed to load configurations: {str(e)}")
        
        # Test 5: Check Monitoring
        print("Testing monitoring...")
        try:
            await page.click("text=مانیتورینگ")
            await page.wait_for_selector("h2:has-text('مانیتورینگ سیستم')", timeout=5000)
            await take_screenshot(page, "05_monitoring_page")
            print("✓ Monitoring page loaded")
            
            # Check if metrics are updating
            await asyncio.sleep(2)  # Wait for metrics to update
            cpu_usage = await page.locator(".cpu-usage").text_content()
            memory_usage = await page.locator(".memory-usage").text_content()
            print(f"✓ System metrics - CPU: {cpu_usage}, Memory: {memory_usage}")
            
        except Exception as e:
            print(f"✗ Failed to load monitoring: {str(e)}")
        
        # Close browser
        await browser.close()

if __name__ == "__main__":
    print("Starting dashboard tests...")
    asyncio.run(test_dashboard())
    print("Test completed. Check the screenshots in the 'test_screenshots' directory.")
