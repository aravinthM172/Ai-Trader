"""
Cloud dashboard server (runs on the Oracle Cloud VM).  Python stdlib only.

  POST /api/push    snapshot from the trading PC      (Authorization: Bearer DASHBOARD_PUSH_TOKEN)
  GET  /            the dashboard page                 (HTTP Basic auth: DASHBOARD_USER / DASHBOARD_PASSWORD)
  GET  /api/state   latest snapshot + equity history   (Basic auth)
  POST /api/tick    live bid/ask every few seconds       (Bearer token)
  GET  /api/tick    latest live prices                  (Basic auth)
  POST /api/command start / pause / emergency stop / switch MT5 account from the page (Basic auth)
  GET  /api/command status of the latest command        (Basic auth)
  GET  /api/command/next  pending command for the PC    (Bearer token)
  POST /api/command/ack   PC reports the result          (Bearer token)
  GET  /healthz     liveness                            (no auth, no data)

Binds 127.0.0.1:8080 by default; Caddy in front provides HTTPS (deploy/dashboard_setup.sh).
It only stores snapshots and queues commands for the PC to fetch -- it cannot connect to the PC.
An account-switch password is kept in memory only until the PC fetches it, and is never sent back to the page.

    DASHBOARD_PUSH_TOKEN=... DASHBOARD_USER=... DASHBOARD_PASSWORD=... python -m dashboard.cloud_app
"""
from __future__ import annotations

import base64
import gzip
import hmac
import json
import os
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = Path(os.getenv("DASHBOARD_DATA_DIR", HERE.parent / "data" / "dashboard"))
LATEST = DATA / "latest.json"
HISTORY = DATA / "equity_history.json"
MAX_BODY = 8 * 1024 * 1024
_TICK: dict = {}                                  # latest live prices (memory only)
_CMD: dict = {}                                   # latest bot command from the dashboard (memory only)
CMD_TTL = 120                                     # seconds a command waits for the PC before expiring
ACTIONS = ("start", "stop", "pause", "switch_account")
ACCOUNT_SIZES = ("5000", "10000", "25000", "50000", "100000")   # FundingPips 2-step sizes (as dashboard/account_switch)
SECRET_FIELDS = ("password",)                     # never returned to the page, wiped once the PC has it
MAX_POINTS = 50_000


def _cfg():
    return (os.getenv("DASHBOARD_PUSH_TOKEN", ""), os.getenv("DASHBOARD_USER", ""), os.getenv("DASHBOARD_PASSWORD", ""))


def bearer_ok(header: str | None, token: str) -> bool:
    if not token or not header or not header.startswith("Bearer "):
        return False
    return hmac.compare_digest(header[7:].strip().encode(), token.encode())


def basic_ok(header: str | None, user: str, password: str) -> bool:
    if not (user and password) or not header or not header.startswith("Basic "):
        return False
    try:
        u, _, p = base64.b64decode(header[6:]).decode().partition(":")
    except Exception:
        return False
    return hmac.compare_digest(u.encode(), user.encode()) & hmac.compare_digest(p.encode(), password.encode())


def store(snapshot: dict, *, data_dir: Path = DATA) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    snapshot["received_utc"] = datetime.now(timezone.utc).isoformat()
    tmp = data_dir / "latest.tmp"
    tmp.write_text(json.dumps(snapshot), encoding="utf-8")
    tmp.replace(data_dir / "latest.json")
    acct = ((snapshot.get("market") or {}).get("account") or {})
    if acct.get("equity") is not None:
        hist_p = data_dir / "equity_history.json"
        try:
            hist = json.loads(hist_p.read_text(encoding="utf-8"))
        except Exception:
            hist = []
        if not hist or hist[-1].get("login") != acct.get("login") or int(time.time()) - hist[-1]["time"] >= 55:
            hist.append({"time": int(time.time()), "equity": acct["equity"], "balance": acct.get("balance"),
                         "login": acct.get("login")})
        hist_p.write_text(json.dumps(hist[-MAX_POINTS:]), encoding="utf-8")


def state(*, data_dir: Path = DATA) -> dict:
    try:
        latest = json.loads((data_dir / "latest.json").read_text(encoding="utf-8"))
    except Exception:
        latest = {"error": "no snapshot received yet -- start dashboard.publisher on the trading PC"}
    try:
        hist = json.loads((data_dir / "equity_history.json").read_text(encoding="utf-8"))
    except Exception:
        hist = []
    login = ((latest.get("market") or {}).get("account") or {}).get("login")
    latest["equity_history"] = [h for h in hist if h.get("login") == login] if login else hist
    return latest


def new_command(body: dict) -> tuple[dict, str]:
    """Validate a command from the page -> (command, error)."""
    action, kind = body.get("action"), body.get("account_type")
    confirm = ("SWITCH REAL" if kind == "real" else "SWITCH") if action == "switch_account" else str(action).upper()
    if action not in ACTIONS or body.get("confirm") != confirm:
        return {}, f"action must be start/pause/stop/switch_account, confirmed with {confirm if action in ACTIONS else 'its word'}"
    cmd = {"id": f"{int(time.time() * 1000)}", "action": action, "force": bool(body.get("force")),
           "created": time.time(), "status": "pending", "message": "waiting for the trading PC"}
    if action == "switch_account":
        login, server, password = str(body.get("login", "")).strip(), str(body.get("server", "")).strip(), body.get("password")
        if not login.isdigit() or not 3 <= len(login) <= 15:
            return {}, "login must be the MT5 account number (digits only)"
        if not server or len(server) > 100:
            return {}, "server is required, exactly as MT5 shows it"
        if not isinstance(password, str) or not password or len(password) > 100:
            return {}, "password is required"
        if kind not in ("demo", "real"):
            return {}, "choose the account type: demo or real"
        size, phase = str(body.get("account_size", "")), str(body.get("phase", ""))
        if size not in ACCOUNT_SIZES:
            return {}, "choose the account size: " + ", ".join(f"${int(s):,}" for s in ACCOUNT_SIZES)
        if phase not in ("1", "2"):
            return {}, "choose the challenge phase: 1 or 2"
        cmd.update(login=login, server=server, password=password, account_type=kind, account_size=int(size),
                   phase=phase, close_first=bool(body.get("close_first")))
    return cmd, ""


def public_cmd() -> dict:
    """The latest command as the page may see it (no secrets)."""
    return {k: v for k, v in _CMD.items() if k not in SECRET_FIELDS}


class Handler(BaseHTTPRequestHandler):
    server_version = "dash/1"

    def log_message(self, *_):                    # keep secrets / noise out of stdout
        pass

    def _send(self, code: int, body: bytes, ctype: str = "application/json", extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _auth(self) -> bool:
        _, user, pw = _cfg()
        if basic_ok(self.headers.get("Authorization"), user, pw):
            return True
        time.sleep(0.5)                           # slow down guessing
        self._send(401, b'{"error":"auth"}', extra={"WWW-Authenticate": 'Basic realm="trader", charset="UTF-8"'})
        return False

    def do_GET(self):
        if self.path == "/healthz":
            return self._send(200, b'{"ok":true}')
        if self.path == "/api/command/next":
            token, _, _ = _cfg()
            if not bearer_ok(self.headers.get("Authorization"), token):
                time.sleep(0.5)
                return self._send(401, b'{"error":"auth"}')
            pending = _CMD.get("status") == "pending" and time.time() - _CMD.get("created", 0) <= CMD_TTL
            if _CMD.get("status") == "pending" and not pending:
                _CMD.update(status="expired", message="PC did not pick it up within 2 minutes (is the publisher running?)")
                for k in SECRET_FIELDS:
                    _CMD.pop(k, None)
            if not pending:
                return self._send(200, b"{}")
            out = dict(_CMD)
            for k in SECRET_FIELDS:
                _CMD.pop(k, None)
            _CMD.update(status="running", message="the trading PC is working on it")
            return self._send(200, json.dumps(out).encode())
        if not self._auth():
            return
        if self.path in ("/", "/index.html"):
            return self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        if self.path == "/api/command":
            return self._send(200, json.dumps(public_cmd()).encode())
        if self.path.startswith("/api/tick"):
            return self._send(200, json.dumps(_TICK or {"symbols": {}}).encode())
        if self.path.startswith("/api/state"):
            return self._send(200, json.dumps(state(), default=str).encode())
        self._send(404, b'{"error":"not found"}')

    def do_POST(self):
        if self.path == "/api/command":                       # from the logged-in page
            if not self._auth():
                return
            try:
                body = json.loads(self.rfile.read(min(int(self.headers.get("Content-Length") or 0), 4096)) or b"{}")
            except Exception:
                return self._send(400, b'{"error":"bad json"}')
            cmd, err = new_command(body)
            if err:
                return self._send(400, json.dumps({"error": err}).encode())
            _CMD.clear()
            _CMD.update(cmd)
            return self._send(200, json.dumps(public_cmd()).encode())
        if self.path == "/api/command/ack":                   # from the PC
            token, _, _ = _cfg()
            if not bearer_ok(self.headers.get("Authorization"), token):
                time.sleep(0.5)
                return self._send(401, b'{"error":"auth"}')
            try:
                body = json.loads(self.rfile.read(min(int(self.headers.get("Content-Length") or 0), 4096)) or b"{}")
            except Exception:
                return self._send(400, b'{"error":"bad json"}')
            if body.get("id") == _CMD.get("id"):
                _CMD.update(status=body.get("status", "done"), message=str(body.get("message", ""))[:500], done=time.time())
            return self._send(200, b'{"ok":true}')
        if self.path not in ("/api/push", "/api/tick"):
            return self._send(404, b'{"error":"not found"}')
        token, _, _ = _cfg()
        if not bearer_ok(self.headers.get("Authorization"), token):
            time.sleep(0.5)
            return self._send(401, b'{"error":"auth"}')
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > MAX_BODY:
            return self._send(413, b'{"error":"size"}')
        raw = self.rfile.read(n)
        try:
            if self.headers.get("Content-Encoding") == "gzip":
                raw = gzip.decompress(raw)
            data = json.loads(raw)
            if self.path == "/api/tick":
                data["received"] = time.time()
                _TICK.clear()
                _TICK.update(data)
            else:
                store(data)
        except Exception:
            return self._send(400, b'{"error":"bad snapshot"}')
        self._send(200, b'{"ok":true}')


def main() -> int:
    token, user, pw = _cfg()
    if not (token and user and pw):
        print("set DASHBOARD_PUSH_TOKEN, DASHBOARD_USER and DASHBOARD_PASSWORD")
        return 1
    host, port = os.getenv("DASHBOARD_HOST", "127.0.0.1"), int(os.getenv("DASHBOARD_PORT", 8080))
    print(f"dashboard on http://{host}:{port}  (put HTTPS in front -- deploy/dashboard_setup.sh)")
    ThreadingHTTPServer((host, port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
