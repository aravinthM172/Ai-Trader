"""Install the dashboard bot settings on the FundedNext VPS (2026-10-20): per-market on/off, risk % or fixed lot,
stop / target ATR and auto risk, edited from the dashboard (strategy/bot_settings.py).

1. identity: .env PROP_CHALLENGE=fundednext_2step and status balance > 10,000 (else nothing is changed)
2. the VPS live_multi.py, dashboard/data.py and dashboard/publisher.py must equal commit 07b0989
   (the version the VPS runs), so no VPS-only change is overwritten (else nothing is changed; their copies are saved here)
3. backups *.bak.before-settings, upload the 4 files, restart trader / watchdog / publisher outside minutes 0-15
   (open positions and MT5 untouched), confirm a fresh status file and no traceback.
No .env change.  Until settings are saved from the dashboard the bot trades exactly as before.
Rollback: copy the .bak.before-settings files back, delete strategy\bot_settings.py and state\bot_settings.json, restart.
Run:  VPS_PW=... python vps_bot_settings.py
"""
import base64
import hashlib
import os
import subprocess
import sys
from pathlib import Path

from winrm.protocol import Protocol

REPO = Path(r"D:\Downloads\Project\gold-ai-trader")
HERE = Path(__file__).parent      # VPS copies of mismatching files are saved here
FILES = ["execution/live_multi.py", "strategy/bot_settings.py", "dashboard/data.py", "dashboard/publisher.py"]
BASE = "07b0989"            # the commit the VPS runs (weekend-flat); the change was built on it
CHECKED = ["execution/live_multi.py", "dashboard/data.py", "dashboard/publisher.py"]

CHECK = r"""
Set-Location C:\gold-ai-trader
$env_ = Get-Content .env -Raw
$st = Get-Content reports\multi_live_status.json -Raw -ErrorAction SilentlyContinue | ConvertFrom-Json
'computer: ' + $env:COMPUTERNAME
'prop: ' + (($env_ -split "`n" | Where-Object { $_ -match '^PROP_CHALLENGE=' }) -join '')
'settings file: ' + (Test-Path state\bot_settings.json)
'status balance: ' + $st.balance + ' at ' + $st.generated_utc
if (($env_ -match '(?m)^PROP_CHALLENGE=fundednext_2step') -and ([double]$st.balance -gt 10000)) { 'IDENTITY OK' } else { 'IDENTITY WRONG' }
"""

CHANGE = r"""
Set-Location C:\gold-ai-trader
$m = (Get-Date).ToUniversalTime().Minute
while ($m -lt 16 -or $m -gt 55) { 'minute ' + $m + ' is inside the entry window -- waiting'; Start-Sleep 60; $m = (Get-Date).ToUniversalTime().Minute }
foreach ($f in @('execution\live_multi.py', 'strategy\bot_settings.py', 'dashboard\data.py', 'dashboard\publisher.py')) {
  if ((Test-Path $f) -and -not (Test-Path "$f.bak.before-settings")) { Copy-Item $f "$f.bak.before-settings" }
  if (Test-Path "$f.new") { Move-Item "$f.new" $f -Force; 'installed ' + $f }
}
$t0 = (Get-Date).ToUniversalTime()
$p = Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python' -and $_.CommandLine -match 'execution\.live_multi|tools\.watchdog|dashboard\.publisher' }
$p | ForEach-Object { 'stopping pid ' + $_.ProcessId + ' | ' + (('' + $_.CommandLine) -replace '.*python.exe"?\s*','') ; Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
'waiting 120 s for the loops to restart'
Start-Sleep 120
Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|terminal64' } | ForEach-Object {
  $_.Name + ' pid ' + $_.ProcessId + ' started ' + $_.CreationDate.ToUniversalTime().ToString('MM-dd HH:mm:ss') + ' | ' + (('' + $_.CommandLine) -replace '.*python.exe"?\s*','')
}
$st = Get-Content reports\multi_live_status.json -Raw | ConvertFrom-Json
'status: generated ' + $st.generated_utc + ' | balance ' + $st.balance + ' | kill ' + $st.kill_switch_active
if ([datetime]::Parse($st.generated_utc).ToUniversalTime() -gt $t0) { 'STATUS FRESH' } else { 'STATUS OLD' }
$log = Get-ChildItem logs\multi_live*.log -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1
if ($log) { '--- log tail ' + $log.Name; Get-Content $log.FullName -Tail 12 }
"""


def run(p, sh, s, show=True):
    enc = base64.b64encode(("$ProgressPreference='SilentlyContinue'\n" + s).encode("utf-16-le")).decode()
    cid = p.run_command(sh, "powershell", ["-NoProfile", "-NonInteractive", "-EncodedCommand", enc])
    out, err, rc = p.get_command_output(sh, cid)
    p.cleanup_command(sh, cid)
    o = out.decode("utf-8", "replace")
    if show:
        print(o, flush=True)
    if rc != 0:
        print(f"[exit {rc}] {err.decode('utf-8', 'replace')[-600:]}")
        o += f"\n[exit {rc}]"
    return o


def norm(b: bytes) -> str:
    return hashlib.sha256(b.replace(b"\r\n", b"\n")).hexdigest()


def upload(p, sh, rel: str):
    data = base64.b64encode((REPO / rel).read_bytes()).decode()
    win = "C:\\gold-ai-trader\\" + rel.replace("/", "\\")
    run(p, sh, f"Set-Content '{win}.b64' '' -NoNewline; Remove-Item '{win}.new' -ErrorAction SilentlyContinue", show=False)
    # cmd.exe caps a command line at 8191 chars and -EncodedCommand roughly triples the text: 1500 b64 chars per call
    for i in range(0, len(data), 1500):
        if "[exit" in run(p, sh, f"[IO.File]::AppendAllText('{win}.b64', '{data[i:i + 1500]}')", show=False):
            sys.exit(f"upload of {rel} failed -- nothing installed yet")
    o = run(p, sh, f"$b=[Convert]::FromBase64String([IO.File]::ReadAllText('{win}.b64')); [IO.File]::WriteAllBytes('{win}.new', $b);"
                   f" Remove-Item '{win}.b64'; (Get-FileHash '{win}.new').Hash", show=False).strip()
    want = hashlib.sha256((REPO / rel).read_bytes()).hexdigest().upper()
    if o != want:
        sys.exit(f"upload of {rel} corrupted ({o} != {want}) -- nothing installed yet")
    print(f"uploaded {rel} -> {rel}.new (hash ok)")


pw = os.getenv("VPS_PW") or sys.exit("set VPS_PW")
p = Protocol(endpoint="http://160.250.204.196:55985/wsman", transport="ntlm", username="Administrator", password=pw,
             operation_timeout_sec=3600, read_timeout_sec=3700)
sh = p.open_shell()
try:
    print("== 1. which machine is this?")
    if "IDENTITY OK" not in run(p, sh, CHECK):
        sys.exit("NOT the FundedNext machine -- nothing changed.")
    print("== 2. are the VPS files the versions the change was made from?")
    for rel in CHECKED:
        win = "C:\\gold-ai-trader\\" + rel.replace("/", "\\")
        remote = base64.b64decode(run(p, sh, f"[Convert]::ToBase64String([IO.File]::ReadAllBytes('{win}'))", show=False).strip())
        head = subprocess.run(["git", "-C", str(REPO), "show", f"{BASE}:{rel}"], capture_output=True, check=True).stdout
        if norm(remote) != norm(head):
            out = HERE / ("vps_current_" + rel.replace("/", "_"))
            out.write_bytes(remote)
            sys.exit(f"VPS {rel} differs from the laptop's committed one -- nothing changed. Its copy: {out}")
        print(f"  {rel}: same as laptop HEAD")
    print("== 3. upload")
    for f in FILES:
        upload(p, sh, f)
    print("== 4. install and restart")
    o = run(p, sh, CHANGE)
    ok = "STATUS FRESH" in o and "Traceback" not in o
    print("DONE: dashboard bot settings installed (bot unchanged until settings are saved)" if ok else "CHECK: status not fresh or a traceback in the log -- look at the dashboard")
finally:
    p.close_shell(sh)
