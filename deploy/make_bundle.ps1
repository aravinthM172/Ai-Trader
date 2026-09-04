# Run on the Windows PC.  Builds gold-ai-trader-bundle.tar.gz with exactly what
# the cloud forward test needs (code + the two H1 cutoff datasets) -- no venv,
# no reports, no big M5/M15 CSVs, and NO .env / NO broker credentials
# (deploy/oracle_bootstrap.sh writes a minimal .env with LIVE_TRADING=false).
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$items = @(
    "backtest", "common", "config", "execution", "mt5", "risk", "strategy", "tools", "tests",
    "data/__init__.py", "data/quality.py",
    "data/btcusd_vx_H1_dense.csv", "data/xauusd_vx_H1.csv",
    "deploy", "run_btc.py",
    "ORACLE_DEPLOY.md", "BTC_NEXT_STEPS.md", "XAU_H1_VALIDATION.md", "BTC_H1_VALIDATION.md"
)
$missing = $items | Where-Object { -not (Test-Path $_) }
if ($missing) { throw "missing: $($missing -join ', ')" }

$out = "gold-ai-trader-bundle.tar.gz"
tar --exclude="__pycache__" --exclude="*.pyc" --exclude="*.sqlite" -czf $out $items
$mb = [math]::Round((Get-Item $out).Length / 1MB, 1)
Write-Host "wrote $out  ($mb MB)"
Write-Host ""
Write-Host "next:"
Write-Host "  scp $out ubuntu@<PUBLIC_IP>:~/"
Write-Host "  ssh ubuntu@<PUBLIC_IP>"
Write-Host "  mkdir -p gold-ai-trader && tar -xzf $out -C gold-ai-trader"
Write-Host "  cd gold-ai-trader && bash deploy/oracle_bootstrap.sh"
