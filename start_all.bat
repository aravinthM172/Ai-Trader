@echo off
REM Starts everything for the challenge rehearsal, each in its own minimised window:
REM   multi-symbol trader + watchdog (alerts/restarts) + cloud dashboard publisher.
REM Does NOT start start_live.bat (BTC-only bot) -- it must not run on the same account.
cd /d "%~dp0"
start "trader"    /min cmd /c start_live_multi.bat
start "watchdog"  /min cmd /c start_watchdog.bat
start "dashboard" /min cmd /c start_dashboard_publisher.bat
echo Started: trader, watchdog, dashboard publisher.  Logs in logs\  -- close their windows to stop.
