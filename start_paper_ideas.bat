@echo off
REM Paper (dry-run) forward test of the BTC / ETH breakout and the gold Asian-hours rule -- NEVER sends orders.
REM Runs as the hourly Windows scheduled task "GoldAI Paper Ideas" (hidden, no window), 20 minutes past the hour.
REM This script (re)creates that task and runs one pass now.  Results: reports\paper_ideas_status.json
REM Standing:   venv\Scripts\python.exe -m execution.paper_ideas --report
REM Remove it with:  schtasks /delete /tn "GoldAI Paper Ideas" /f
cd /d "%~dp0"
schtasks /create /tn "GoldAI Paper Ideas" /sc hourly /mo 1 /st 00:20 /tr "wscript.exe \"%~dp0tools\run_paper_ideas_hidden.vbs\"" /f
schtasks /run /tn "GoldAI Paper Ideas"
