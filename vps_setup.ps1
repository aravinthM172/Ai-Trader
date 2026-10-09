# One-shot setup for the Ezerhost Windows VPS (run inside the VPS, in PowerShell as Administrator).
# Expects vps_bundle.zip (and .env) on the Desktop.  Installs Python 3.14.4 + MT5, unpacks the bot to
# C:\gold-ai-trader, builds the venv, runs the tests.  It does NOT start the bot.
$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$ProgressPreference = "SilentlyContinue"
$desk = [Environment]::GetFolderPath("Desktop")
$app  = "C:\gold-ai-trader"
$dl   = "C:\setup-downloads"
New-Item -ItemType Directory -Force $dl | Out-Null

function Step($m) { Write-Host "`n=== $m ===" -ForegroundColor Cyan }

Step "1/7 Python 3.14.4"
$py = "C:\Program Files\Python314\python.exe"
if (-not (Test-Path $py)) {
    Invoke-WebRequest "https://www.python.org/ftp/python/3.14.4/python-3.14.4-amd64.exe" -OutFile "$dl\python.exe"
    Start-Process "$dl\python.exe" -Wait -ArgumentList "/quiet InstallAllUsers=1 PrependPath=1 Include_test=0"
}
& $py --version

Step "2/7 MetaTrader 5"
Invoke-WebRequest "https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe" -OutFile "$dl\mt5setup.exe"
Start-Process "$dl\mt5setup.exe" -Wait -ArgumentList "/auto"

Step "3/7 Unpack the bot"
if (-not (Test-Path "$desk\vps_bundle.zip")) { throw "vps_bundle.zip not found on the Desktop" }
Expand-Archive "$desk\vps_bundle.zip" -DestinationPath $app -Force
New-Item -ItemType Directory -Force "$app\logs", "$app\state" | Out-Null
if (Test-Path "$desk\.env") { Copy-Item "$desk\.env" "$app\.env" -Force; Write-Host ".env copied" }
else { Write-Warning ".env not on the Desktop - copy it to $app\.env before starting the bot" }

Step "4/7 Virtual env + packages"
Set-Location $app
& $py -m venv venv
& "$app\venv\Scripts\python.exe" -m pip install --upgrade pip -q
& "$app\venv\Scripts\python.exe" -m pip install -r requirements.txt -q
# same versions as the PC
& "$app\venv\Scripts\python.exe" -m pip install -q "numba==0.67.0" "numpy==2.5.2" "pandas==3.0.5" "MetaTrader5==5.0.6147"

Step "5/7 Tests"
& "$app\venv\Scripts\python.exe" -m pytest -q
if ($LASTEXITCODE -ne 0) { Write-Warning "some tests failed - send the output to Claude" }

Step "6/7 Safety check of .env (live switches must be off on the VPS for now)"
if (Test-Path "$app\.env") {
    $live = Select-String -Path "$app\.env" -Pattern '^\s*(MULTI_)?LIVE_TRADING\s*=\s*true' -CaseSensitive:$false
    if ($live) { Write-Warning "LIVE trading is switched ON in .env - set MULTI_LIVE_TRADING=false and LIVE_TRADING=false until FundingPips OKs the VPS" }
    else { Write-Host "OK: live trading is off" -ForegroundColor Green }
}

Step "7/7 Windows tweaks for a 24/7 bot"
powercfg /change standby-timeout-ac 0
powercfg /change monitor-timeout-ac 0
tzutil /g

Write-Host "`nDONE. Next: open MetaTrader 5, log in to a DEMO account, Tools > Options > Expert Advisors > tick 'Allow algorithmic trading'," -ForegroundColor Green
Write-Host "then run:  cd $app ; venv\Scripts\python.exe -m tools.go_live_preflight --balance 5000" -ForegroundColor Green
