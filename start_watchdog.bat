@echo off
REM Watchdog for the BTC H1 live trader: alerts (Telegram if configured), bot/MT5 restart,
REM hourly edge monitor, daily heartbeat.  Never sends orders.  Keeps restarting itself.
cd /d "%~dp0"
:loop
venv\Scripts\python.exe -u -m tools.watchdog --interval 60 >> logs\watchdog_console.log 2>&1
echo %date% %time% watchdog exited (code %errorlevel%), restarting in 30s >> logs\watchdog_console.log
timeout /t 30 /nobreak > nul
goto loop
