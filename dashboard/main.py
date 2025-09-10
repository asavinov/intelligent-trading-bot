"""
داشبورد مدیریت ربات ترید هوشمند
FastAPI Application Entry Point
"""

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn
from pathlib import Path
import sys
from contextlib import asynccontextmanager

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import API routers
from dashboard.api.setup import router as setup_router
from dashboard.api.scripts import router as scripts_router
from dashboard.api.system import router as system_router
from dashboard.api.configs import router as configs_router
from dashboard.api.signals import router as signals_router

# Import database initialization
from dashboard.models.database import init_database

# Import core components
from dashboard.core.script_wrapper import ScriptWrapper

# Global instances
script_wrapper = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """مدیریت چرخه حیات اپلیکیشن"""
    global script_wrapper
    
    # Startup
    print("🔧 راه‌اندازی سرویس‌ها...")
    init_database()
    script_wrapper = ScriptWrapper(project_root)
    print("✅ Database initialized successfully")
    print("🚀 Dashboard started successfully!")
    print(f"📁 Project root: {project_root}")
    print("🔗 Access dashboard at: http://127.0.0.1:8000")
    
    yield
    
    # Shutdown
    print("🔄 بستن سرویس‌ها...")


# Initialize FastAPI app
app = FastAPI(
    title="ربات ترید هوشمند - داشبورد",
    description="داشبورد مدیریت و کنترل ربات ترید هوشمند",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# Setup static files and templates
dashboard_dir = Path(__file__).parent
frontend_dir = dashboard_dir / "frontend"

# Mount static files correctly
app.mount("/static/css", StaticFiles(directory=str(frontend_dir / "css")), name="css")
app.mount("/static/js", StaticFiles(directory=str(frontend_dir / "js")), name="js")
app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

templates = Jinja2Templates(directory=str(frontend_dir))

# Include API routers
app.include_router(setup_router, prefix="/api")
app.include_router(scripts_router, prefix="/api")
app.include_router(system_router, prefix="/api")
app.include_router(configs_router, prefix="/api")
app.include_router(signals_router, prefix="/api")

@app.get("/", response_class=HTMLResponse)
async def dashboard_home(request: Request):
    """صفحه اصلی داشبورد"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/setup", response_class=HTMLResponse)
async def setup_wizard(request: Request):
    """صفحه راه‌اندازی اولیه"""
    return templates.TemplateResponse("setup.html", {"request": request})

@app.get("/debug", response_class=HTMLResponse)
async def debug_page(request: Request):
    """صفحه تست و عیب‌یابی"""
    return templates.TemplateResponse("debug.html", {"request": request})

@app.get("/simple-test", response_class=HTMLResponse)
async def simple_test_page(request: Request):
    """صفحه تست ساده JavaScript"""
    return templates.TemplateResponse("simple-test.html", {"request": request})

@app.get("/console-test", response_class=HTMLResponse)
async def console_test_page(request: Request):
    """صفحه تست کنسول و JavaScript"""
    return templates.TemplateResponse("console-test.html", {"request": request})

@app.get("/click-test", response_class=HTMLResponse)
async def click_test_page(request: Request):
    """صفحه تست کامل کلیک"""
    return templates.TemplateResponse("click-test.html", {"request": request})

@app.get("/health")
async def health_check():
    """بررسی سلامت سرویس"""
    return {"status": "healthy", "service": "trading-bot-dashboard"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info"
    )
