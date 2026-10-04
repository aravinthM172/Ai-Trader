@echo off
REM Pushes a read-only dashboard snapshot to the cloud every 60 s (DASHBOARD_URL / DASHBOARD_PUSH_TOKEN in .env).
REM Outbound only; never trades.  Keeps restarting itself.
cd /d "%~dp0"
:loop
venv\Scripts\python.exe -u -m dashboard.publisher >> logs\dashboard_publisher_console.log 2>&1
timeout /t 30 /nobreak > nul
goto loop
