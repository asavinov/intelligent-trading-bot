@echo off
setlocal ENABLEDELAYEDEXPANSION

set "SCRIPT_DIR=%~dp0"
rem Normalize trailing backslash
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

rem Prefer venv Python if available
set "VENV_PY=%SCRIPT_DIR%\.venv\Scripts\python.exe"
if exist "%VENV_PY%" (
	set "PYEXE=%VENV_PY%"
	set "PYDESC=venv"
) else (
	set "PYEXE=python"
	set "PYDESC=system"
)

set "TITLE_ON=ITB Dashboard (pipeline ON)"
set "TITLE_OFF=ITB Dashboard (pipeline OFF)"

echo.
echo =============================
echo  Intelligent Trading Dashboard
echo  Python: %PYDESC%
echo =============================
echo.

:menu
echo 1. Start Server (pipeline ON)
echo 2. Start Server (pipeline OFF)
echo 3. Stop Server
echo 4. Open Dashboard
echo 0. Exit
echo.
set /p choice=Select [0-4]: 

if "%choice%"=="1" goto start_on
if "%choice%"=="2" goto start_off
if "%choice%"=="3" goto stop
if "%choice%"=="4" goto dashboard
if "%choice%"=="0" goto end

goto menu

:start_on
call :start_server 1
goto menu

:start_off
call :start_server 0
goto menu

:start_server
rem %1 == 1 => pipeline ON, else OFF
set "PIPELINE_FLAG=%~1"
if "%PIPELINE_FLAG%"=="1" (
	set "TITLE=%TITLE_ON%"
) else (
	set "TITLE=%TITLE_OFF%"
)

rem Open a new console window with proper environment and run uvicorn via selected Python
rem Use 127.0.0.1 for safety; change to 0.0.0.0 if you need LAN access
start "%TITLE%" cmd /k "cd /d "%SCRIPT_DIR%" ^&^& set PYTHONPATH=%SCRIPT_DIR% ^&^& set PYTHONUNBUFFERED=1 ^&^& set DASHBOARD_PIPELINE_ENABLED=%PIPELINE_FLAG% ^&^& "%PYEXE%" -m uvicorn dashboard.main:app --host 127.0.0.1 --port 8000"
echo Started server in a new window: %TITLE%
goto :eof

:stop
rem Try to gracefully close only our dashboard windows (both ON/OFF titles)
for %%T in ("%TITLE_ON%" "%TITLE_OFF%") do (
	rem Close the console window and its child processes
	taskkill /FI "WINDOWTITLE eq %%~T" /T /F >nul 2>&1
)
echo Stop signal sent (if a server window was open, it should be closed now).
goto menu

:dashboard
start http://127.0.0.1:8000
goto menu

:end
endlocal
exit /b 0
