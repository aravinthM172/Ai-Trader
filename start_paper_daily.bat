@echo off
REM Paper (dry-run) forward test of the round-3 daily strategies -- NEVER sends orders.
REM Runs as the hourly Windows scheduled task "GoldAI Paper Daily" (hidden, no window).
REM This script (re)creates that task and runs one pass now.  Results: reports\paper_daily_status.json
REM Remove it with:  schtasks /delete /tn "GoldAI Paper Daily" /f
cd /d "%~dp0"
schtasks /create /tn "GoldAI Paper Daily" /sc hourly /mo 1 /st 00:05 /tr "wscript.exe \"%~dp0tools\run_paper_daily_hidden.vbs\"" /f
schtasks /run /tn "GoldAI Paper Daily"
