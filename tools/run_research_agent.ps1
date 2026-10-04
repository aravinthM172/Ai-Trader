# One research-agent run (headless Claude Code), tightly sandboxed.
# The agent may edit ONLY backtest/research_strategies.py and run ONLY the research gate.
# Project deny rules (.claude/settings.json) additionally block live files, .env and state/.
#   powershell -ExecutionPolicy Bypass -File tools\run_research_agent.ps1
Set-Location (Split-Path $PSScriptRoot -Parent)
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
New-Item -ItemType Directory -Force research\agent_runs | Out-Null
$prompt = Get-Content agents\research_agent.md -Raw
claude -p $prompt `
  --allowedTools "Read" "Grep" "Glob" "Edit(backtest/research_strategies.py)" "Bash(venv/Scripts/python -m backtest.research_gate)" `
  --disallowedTools "Write" "WebFetch" "WebSearch" `
  | Tee-Object -FilePath "research\agent_runs\$stamp.md"
