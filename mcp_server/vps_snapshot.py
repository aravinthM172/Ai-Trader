"""
Runs ON THE VPS (sent by mcp_server/vps.py, never imported here): prints one JSON snapshot of the live
multi trader.  Read-only: it opens the trade DB with mode=ro, reads status / state / log files and lists
processes.  Never connects to MT5, never writes a file.  Standard library only (runs in the VPS venv).
"""
import json
import sqlite3
import subprocess
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

R = Path(r"C:\gold-ai-trader")
SECRET = ("PASSWORD", "TOKEN", "SECRET", "KEY", "LOGIN", "CHAT_ID")
ENV_KEEP = ("MULTI_", "LIVE_", "PROP_", "KILL", "MT5_SERVER")
PROBLEM = ("ERROR", "CRITICAL", "Traceback", "FAILED", "EMERGENCY")


def rj(p):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def tail(p, n, max_bytes=6_000_000):
    """Last n lines of a text file, reading at most max_bytes from its end."""
    try:
        with p.open("rb") as f:
            f.seek(max(0, p.stat().st_size - max_bytes))
            return [ln.decode("utf-8", "replace").rstrip() for ln in deque(f, maxlen=n)]
    except Exception:
        return []


now = datetime.now(timezone.utc)
o = {"now_utc": now.isoformat()}
o["status"] = rj(R / "reports" / "multi_live_status.json")
o["challenge"] = rj(R / "state" / "challenge_state.json")
k = R / "state" / "KILL_SWITCH"
o["kill_switch_file"] = k.read_text(encoding="utf-8", errors="replace")[:300] if k.exists() else None

st = rj(R / "state" / "multi_live_state.json") or {}
cut = (now - timedelta(hours=72)).isoformat()
o["decisions"] = {key: v for key, v in (st.get("evaluated") or {}).items() if key.split("|", 1)[-1] >= cut}
o["why"] = {key: v for key, v in (st.get("why") or {}).items() if key.split("|", 1)[-1] >= cut}

env = {}
try:
    for ln in (R / ".env").read_text(encoding="utf-8-sig", errors="replace").splitlines():
        key, sep, val = ln.partition("=")
        key = key.strip()
        if sep and key.startswith(ENV_KEEP):
            env[key] = "<set>" if any(s in key for s in SECRET) else val.strip()
except Exception as e:
    env["error"] = str(e)
o["settings"] = env

log = tail(R / "logs" / "multi_console.log", 20000)
o["status_lines"] = [ln for ln in log if ln.startswith("[")][-400:]
o["problem_lines"] = [ln[:300] for ln in log if any(w in ln for w in PROBLEM)][-60:]
o["log_tail"] = [ln[:300] for ln in log[-60:]]
o["watchdog_tail"] = [ln[:300] for ln in tail(R / "logs" / "watchdog_console.log", 30)]

try:
    c = sqlite3.connect(f"file:{(R / 'state' / 'multi_live.sqlite').as_posix()}?mode=ro", uri=True)
    try:
        cols = [r[1] for r in c.execute("PRAGMA table_info(trades)")]
        o["trades"] = [dict(zip(cols, r)) for r in c.execute("SELECT * FROM trades ORDER BY opened_utc DESC LIMIT 200")]
    finally:
        c.close()
except Exception as e:
    o["trades"] = {"error": str(e)}

ps = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|terminal64' } | "
      "Select-Object ProcessId, Name, CommandLine, @{n='started_utc';e={$_.CreationDate.ToUniversalTime().ToString('o')}} | "
      "ConvertTo-Json -Compress")
try:
    raw = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], capture_output=True,
                         text=True, timeout=60).stdout.strip()
    procs = json.loads(raw) if raw else []
    procs = procs if isinstance(procs, list) else [procs]
    o["processes"] = [{"pid": p["ProcessId"], "name": p["Name"], "started_utc": p["started_utc"],
                       "cmd": (p.get("CommandLine") or "").split("python.exe")[-1].strip(' "')[:120]} for p in procs]
except Exception as e:
    o["processes"] = {"error": str(e)}
try:
    o["vps_boot_utc"] = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command",
         "(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToUniversalTime().ToString('o')"],
        capture_output=True, text=True, timeout=60).stdout.strip()
except Exception:
    o["vps_boot_utc"] = None

print(json.dumps(o, default=str))
