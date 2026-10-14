"""
Fetches the live trader's snapshot from the Ezerhost VPS for the MCP server: ONE WinRM login runs the fixed,
read-only mcp_server/vps_snapshot.py there and returns its JSON.  Cached for CACHE_S seconds so a burst of tool
calls costs one login (dozens of logins in a row once froze the VPS).

The password never lives in the repo: env VPS_PW, else VPS_PW=... in the untracked file .env.vps.
"""
from __future__ import annotations

import base64
import json
import os
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_SRC = Path(__file__).with_name("vps_snapshot.py")
ENV_FILE = ROOT / ".env.vps"
HOST = os.getenv("VPS_HOST", "160.250.204.196")
PORT = int(os.getenv("VPS_WINRM_PORT", "55985"))
VPS_PYTHON = r"C:\gold-ai-trader\venv\Scripts\python.exe"
CACHE_S = 60
ERROR_CACHE_S = 15             # failures are kept briefly too, so a burst of tool calls never retries the login

_cache: dict = {"t": 0.0, "data": None}


def password() -> str | None:
    if os.getenv("VPS_PW"):
        return os.environ["VPS_PW"]
    try:
        for ln in ENV_FILE.read_text(encoding="utf-8-sig").splitlines():
            k, sep, v = ln.partition("=")
            if sep and k.strip() == "VPS_PW":
                return v.strip().strip("'\"") or None
    except OSError:
        pass
    return None


def bootstrap_command() -> list[str]:
    """python -c that runs the fixed snapshot script (compressed so the command line stays short)."""
    blob = base64.b64encode(zlib.compress(SNAPSHOT_SRC.read_bytes(), 9)).decode()
    # WinRM joins the arguments with spaces and no quoting: quote the code for cmd.exe
    return ["-c", f"\"import zlib,base64;exec(zlib.decompress(base64.b64decode('{blob}')))\""]


def _fetch() -> dict:
    pw = password()
    if not pw:
        return {"error": "VPS password not set: put VPS_PW=... in .env.vps (project root, not committed) "
                         "or set the VPS_PW environment variable, then restart Claude Code"}
    from winrm.protocol import Protocol
    p = Protocol(endpoint=f"http://{HOST}:{PORT}/wsman", transport="ntlm", username="Administrator", password=pw,
                 operation_timeout_sec=90, read_timeout_sec=120)
    try:
        sh = p.open_shell(working_directory=r"C:\gold-ai-trader")
    except Exception as e:
        return {"error": f"cannot log in to the VPS: {type(e).__name__}: {str(e)[:200]}"}
    try:
        cid = p.run_command(sh, VPS_PYTHON, bootstrap_command())
        out, err, rc = p.get_command_output(sh, cid)
        p.cleanup_command(sh, cid)
    finally:
        p.close_shell(sh)
    if rc != 0:
        return {"error": f"snapshot failed (exit {rc}): {err.decode('utf-8', 'replace')[-600:]}"}
    return json.loads(out.decode("utf-8", "replace"))


def snapshot(force: bool = False) -> dict:
    """The live snapshot, from cache when younger than CACHE_S seconds (an error: ERROR_CACHE_S)."""
    d = _cache["data"]
    if not force and d is not None and time.time() - _cache["t"] < (ERROR_CACHE_S if "error" in d else CACHE_S):
        return d
    data = _fetch()
    _cache.update(t=time.time(), data=data)
    return data
