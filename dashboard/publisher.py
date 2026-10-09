"""
Runs on the TRADING PC.  Every PUSH_SECONDS it collects a read-only snapshot
(dashboard/data.gather) and POSTs it to the cloud dashboard.  Outbound only -- the PC polls the
cloud for dashboard commands (start / pause / emergency stop / switch MT5 account); nothing in the
cloud can connect to this PC.  If the cloud is down, the push fails quietly and the bot is unaffected.

.env (or .env.dashboard, git-ignored) on the PC:
    DASHBOARD_URL=https://<your-host>          (e.g. https://141-1-2-3.sslip.io)
    DASHBOARD_PUSH_TOKEN=<long random secret>  (same value on the cloud server)

    python -m dashboard.publisher            # loop
    python -m dashboard.publisher --once     # one push (test)
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import time
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

from common.logging_setup import get_logger
from dashboard.data import gather

load_dotenv()
load_dotenv(Path(__file__).resolve().parents[1] / ".env.dashboard")   # dashboard-only settings (git-ignored)
log = get_logger("dashboard.pub", filename="dashboard_publisher.log")
PUSH_SECONDS = int(float(os.getenv("DASHBOARD_PUSH_SECONDS", 60)))
TICK_SECONDS = float(os.getenv("DASHBOARD_TICK_SECONDS", 3))
ROOT = Path(__file__).resolve().parents[1]
KILL = ROOT / "state" / "KILL_SWITCH"
FLATTEN = ROOT / "state" / "EMERGENCY_FLATTEN"          # read by execution/live_multi.py
PROTECTIVE = ("prop controls", "edge_monitor", "drawdown breaker")
TICK_SYMBOLS = [x.strip() for x in os.getenv("MULTI_SYMBOLS", "BTCUSD.vx,XAUUSD.vx,DAX40.vx").split(",") if x.strip()]


def push(snapshot: dict, *, url: str, token: str, timeout: float = 20.0, path: str = "/api/push") -> int:
    body = gzip.compress(json.dumps(snapshot, default=str).encode())
    req = urllib.request.Request(url.rstrip("/") + path, data=body, method="POST", headers={
        "Authorization": f"Bearer {token}", "Content-Type": "application/json", "Content-Encoding": "gzip"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    url, token = os.getenv("DASHBOARD_URL", "").strip(), os.getenv("DASHBOARD_PUSH_TOKEN", "").strip()
    if not (url and token):
        print("set DASHBOARD_URL and DASHBOARD_PUSH_TOKEN in .env")
        return 1
    gw, last_full = None, 0.0
    while True:
        if time.time() - last_full >= PUSH_SECONDS or a.once:
            last_full = time.time()
            try:
                if gw is not None:                      # gather() opens its own connection
                    gw.shutdown()
                    gw = None
                code = push(gather(), url=url, token=token)
                log.info("pushed snapshot -> %s", code)
                if a.once:
                    print(f"pushed -> HTTP {code}")
            except Exception as e:
                log.warning("push failed: %s", e)
                if a.once:
                    print(f"push failed: {e}")
            if a.once:
                return 0
        try:
            if gw is None:
                from mt5.gateway import MT5Gateway
                gw = MT5Gateway()
                if not gw.connect():
                    gw = None
            if gw is not None:
                push(ticks(gw), url=url, token=token, path="/api/tick", timeout=8.0)
            if handle_command(url, token):         # refresh the page's bot status ~10 s after a command
                last_full = time.time() - PUSH_SECONDS + 10
        except Exception as e:
            log.debug("tick push failed: %s", e)
            gw = None
        time.sleep(max(1.0, TICK_SECONDS))


def _http_json(url: str, token: str, *, data: dict | None = None, timeout: float = 8.0) -> dict:
    req = urllib.request.Request(url, data=None if data is None else json.dumps(data).encode(),
                                 method="GET" if data is None else "POST",
                                 headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode() or "{}")


def trader_running() -> bool:
    try:
        import psutil
        return any("execution.live_multi" in " ".join(p.info["cmdline"] or []) for p in psutil.process_iter(["cmdline"]))
    except Exception:
        return False


def execute_command(cmd: dict, *, kill=KILL, flatten=FLATTEN, start_task=None, running=trader_running) -> tuple[str, str]:
    """Carry out a dashboard command on this PC.  Returns (status, message)."""
    from datetime import datetime, timezone
    action = cmd.get("action")
    if action == "stop":
        kill.parent.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).isoformat()
        kill.write_text(stamp + "  website emergency stop\n", encoding="utf-8")
        flatten.write_text(stamp, encoding="utf-8")         # execution/live_multi.py closes all bot positions
        log.warning("WEBSITE EMERGENCY STOP: kill switch set, trader closing all bot positions")
        return "done", "Emergency stop: no new trades, and the trader is closing all open bot trades (within ~30 s)."
    if action == "pause":
        kill.parent.mkdir(exist_ok=True)
        kill.write_text(datetime.now(timezone.utc).isoformat() + "  website pause\n", encoding="utf-8")
        log.warning("WEBSITE PAUSE: kill switch set, open trades left to their SL/TP")
        return "done", "Paused: no new trades. Open trades keep their stop-loss / take-profit."
    if action == "switch_account":
        from dashboard.account_switch import switch_account
        status, message = switch_account(cmd, mt5=mt5_module(), magics=bot_magics())
        log.warning("WEBSITE ACCOUNT SWITCH to %s: %s -- %s", cmd.get("login"), status, message)
        return status, message
    if action == "start":
        reason = kill.read_text(encoding="utf-8").strip() if kill.exists() else ""
        if reason and any(k in reason for k in PROTECTIVE) and not cmd.get("force"):
            return "refused", f"Stopped by a safety rule ({reason[:160]}). Type FORCE to override."
        for f in (kill, flatten):
            if f.exists():
                f.unlink()
        msg = "Kill switch cleared; new entries allowed."
        if not running():
            (start_task or _start_task)()
            msg += " Trader was not running -> started it."
        log.warning("WEBSITE START: %s", msg)
        return "done", msg
    return "failed", f"unknown action {action!r}"


def _start_task() -> None:
    """Start only the trader loop (the GoldAITrader task would also start a second watchdog + publisher)."""
    from dashboard.account_switch import start_loops
    start_loops(loops=(("start_live_multi.bat", "execution.live_multi"),))


def mt5_module():
    import MetaTrader5
    return MetaTrader5


def bot_magics() -> set[int]:
    from execution.live_multi import magic_for, symbols
    return {magic_for(s) for s in symbols()}


def handle_command(url: str, token: str) -> bool:
    """Run the pending dashboard command, if any.  True when one was handled."""
    base = url.rstrip("/")
    cmd = _http_json(base + "/api/command/next", token)
    if not cmd.get("id"):
        return False
    status, message = execute_command(cmd)
    _http_json(base + "/api/command/ack", token, data={"id": cmd["id"], "status": status, "message": message})
    return True


def ticks(gw) -> dict:
    """Latest bid/ask per symbol (small, sent every few seconds for the live chart)."""
    out = {"time": time.time(), "symbols": {}}
    try:
        acct = gw.account_info()
        pos = gw.positions()
        # floating P/L as MT5 shows it: equity - balance (includes swap, which position.profit leaves out)
        out["account"] = {"equity": acct.get("equity"), "balance": acct.get("balance"),
                          "floating": round(acct["equity"] - acct["balance"], 2), "open": len(pos)}
        out["positions"] = [{"symbol": p["symbol"], "type": p["type"], "volume": p["volume"],
                             "price_open": p["price_open"], "sl": p["sl"], "tp": p["tp"],
                             "profit": round(p["profit"] + p.get("swap", 0.0), 2)} for p in pos]
    except Exception:
        pass
    for s in TICK_SYMBOLS:
        t = gw.get_tick(s)
        if t and t.get("bid"):
            out["symbols"][s] = {"bid": t["bid"], "ask": t["ask"], "time_utc": t.get("time_utc"),
                                 "age_s": t.get("age_seconds")}
    return out


if __name__ == "__main__":
    raise SystemExit(main())
