@echo off
REM Multi-symbol H1 momentum trader -- keeps restarting if Python exits.
REM Sends real orders ONLY if .env has MULTI_LIVE_TRADING=true AND LIVE_TRADING=true AND the
REM account type matches LIVE_ACCOUNT_MODE, MT5 Algo Trading is ON and state\KILL_SWITCH is absent.
cd /d "%~dp0"
:loop
venv\Scripts\python.exe -u -m execution.live_multi --loop 30 >> logs\multi_console.log 2>&1
echo %date% %time% live_multi exited (code %errorlevel%), restarting in 30s >> logs\multi_console.log
timeout /t 30 /nobreak > nul
goto loop
