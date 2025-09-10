@echo off

echo.
echo =================
echo  Bot Controller
echo =================
echo.

:menu
echo 1. Start Server
echo 2. Stop Server
echo 3. Open Dashboard
echo 0. Exit
echo.
set /p choice=Select [0-3]: 

if "%choice%"=="1" goto start
if "%choice%"=="2" goto stop
if "%choice%"=="3" goto dashboard
if "%choice%"=="0" exit

goto menu

:start
start "Bot Server" cmd /k "python -m uvicorn dashboard.main:app --host 0.0.0.0 --port 8000"
echo Starting server...
goto menu

:stop
taskkill /f /im python.exe >nul 2>&1
echo Server stopped.
goto menu

:dashboard
start http://127.0.0.1:8000
goto menu
