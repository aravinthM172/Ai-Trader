@echo off
REM BTC H1 live trader -- keeps restarting if Python ever exits unexpectedly.
REM Real orders ONLY if .env has LIVE_TRADING=true AND MT5 "Algo Trading" is ON
REM AND state\KILL_SWITCH does not exist.  Otherwise it is a dry run.
REM Emergency stop: create the file state\KILL_SWITCH (or close this window).
cd /d "%~dp0"
:loop
venv\Scripts\python.exe -u run_live.py --loop 30 >> logs\live_console.log 2>&1
echo %date% %time% run_live exited (code %errorlevel%), restarting in 30s >> logs\live_console.log
timeout /t 30 /nobreak > nul
goto loop
