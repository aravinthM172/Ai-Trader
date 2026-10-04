# One command to put the dashboard on the cloud VM:
#   powershell -ExecutionPolicy Bypass -File deploy\deploy_dashboard.ps1 -Ip <PUBLIC_IP>
param([Parameter(Mandatory = $true)][string]$Ip, [string]$User = "ubuntu")
Set-Location (Split-Path $PSScriptRoot -Parent)
scp -r dashboard deploy/dashboard_setup.sh "${User}@${Ip}:~/"
if ($LASTEXITCODE -ne 0) { throw "scp failed" }
ssh "${User}@${Ip}" "bash ~/dashboard_setup.sh"
