"""
Unattended-operation watchdog for the BTC H1 live trader.

Runs beside run_live.py (separate process).  Reads only the bot's own files and the
Windows process list -- it never connects to MT5 and never sends or closes orders.

Every pass (default 60 s):
  * bot alive?      reports/btc_live_status.json older than STALE_MIN -> alert;
                    no run_live.py process -> relaunch start_live.bat (AUTO_RESTART)
  * MT5 alive?      no terminal64.exe -> alert; relaunch MT5_TERMINAL_PATH if set
  * algo trading    terminal/account trade_allowed false -> alert
  * kill switch     appears -> alert with its reason
  * drawdown        balance in the last 25 % of the way to the drawdown floor -> alert
  * trades/events   new opens, closes, failed orders, SL repairs (state/btc_live_H1.sqlite)
  * edge monitor    hourly (tools/edge_monitor.py, enforced -> may write KILL_SWITCH)
  * heartbeat       daily summary at HEARTBEAT_UTC_HOUR
  * Telegram cmds   /status, /kill, /help -- from TELEGRAM_CHAT_ID only.  There is
                    deliberately no /resume: restarting after a kill is a manual step.

    python -m tools.watchdog            # loop
    python -m tools.watchdog --once     # one REPORT-ONLY pass: never restarts, launches or
                                        # writes KILL_SWITCH (restarts happen only in the loop)
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from common import notify
from common.logging_setup import get_logger
from tools import challenge_tracker, edge_monitor

load_dotenv()
log = get_logger("watchdog", filename="watchdog.log")

STATUS = ROOT / "reports" / "btc_live_status.json"
DB = ROOT / "state" / "btc_live_H1.sqlite"
KILL = ROOT / "state" / "KILL_SWITCH"
WD_STATE = ROOT / "state" / "watchdog_state.json"
START_BAT = ROOT / "start_live.bat"


def _env_f(name, default):
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


STALE_MIN = _env_f("WATCHDOG_STALE_MIN", 5)
RESTART_MIN = _env_f("WATCHDOG_RESTART_MIN", 10)
HEARTBEAT_UTC_HOUR = int(_env_f("WATCHDOG_HEARTBEAT_UTC_HOUR", 6))
AUTO_RESTART = os.getenv("WATCHDOG_AUTO_RESTART", "true").strip().lower() in ("1", "true", "yes", "on")
EVENT_KINDS = {"closed": "TRADE CLOSED", "order_send": "ORDER SENT", "order_check_failed": "ORDER CHECK FAILED",
               "sltp_repair": "SL/TP REPAIR", "close_send": "CLOSE ORDER"}


# -- process checks (psutil; injectable for tests) ---------------------------
def btc_bot_watched() -> bool:
    """Watch/restart the BTC-only bot (execution/live.py) ONLY when it is the active bot.
    Never when the multi-symbol trader already trades BTCUSD.vx on the same account --
    running both would double the BTC exposure.  Override with WATCHDOG_WATCH_BTC=true/false."""
    v = os.getenv("WATCHDOG_WATCH_BTC", "auto").strip().lower()
    if v != "auto":
        return v in ("1", "true", "yes", "on")
    multi = [x.strip() for x in os.getenv("MULTI_SYMBOLS", "").split(",") if x.strip()]
    return not (multi and "BTCUSD.vx" in multi)


def processes() -> list[tuple[str, str]]:
    import psutil
    out = []
    for p in psutil.process_iter(["name", "cmdline"]):
        try:
            out.append(((p.info["name"] or "").lower(), " ".join(p.info["cmdline"] or []).lower()))
        except Exception:
            continue
    return out


def bot_running(procs) -> bool:
    return any("python" in n and "run_live.py" in c for n, c in procs)


def mt5_running(procs) -> bool:
    return any(n == "terminal64.exe" for n, _ in procs)


def launch_bot() -> None:
    subprocess.Popen(["cmd", "/c", "start", "", "/min", str(START_BAT)], cwd=str(ROOT),
                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def launch_mt5() -> bool:
    path = os.getenv("MT5_TERMINAL_PATH", "").strip()
    if path and Path(path).exists():
        subprocess.Popen([path], cwd=str(Path(path).parent))
        return True
    return False


# -- state -----------------------------------------------------------------
def load_state(path: Path = WD_STATE) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(s: dict, path: Path = WD_STATE) -> None:
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(s, indent=2, default=str), encoding="utf-8")


def read_status(path: Path = STATUS) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def status_line(st: dict | None) -> str:
    if not st:
        return "no status file"
    a = st.get("account") or {}
    pos = a.get("open_positions") or []
    p = (f"{pos[0].get('type')} {pos[0].get('volume')} @ {pos[0].get('price_open')}" if pos else "flat")
    return (f"{st.get('mode')} | bal ${a.get('balance')} (peak ${a.get('peak_balance')}, floor ${a.get('drawdown_floor')})"
            f" | {p} | closed {st.get('closed_trades')} net ${st.get('net_pl_usd')} expR {st.get('expectancy_R')}"
            f" | kill={'ON' if st.get('kill_switch_active') else 'off'}")


# -- one pass ----------------------------------------------------------------
def check(now: datetime, ws: dict, *, procs, status_path=STATUS, db=DB, kill=KILL,
          restart=launch_bot, start_mt5=launch_mt5, run_edge=edge_monitor.run,
          allow_actions: bool = True) -> list[str]:
    """Return alert messages for this pass (state changes only); mutates ws."""
    alerts: list[str] = []
    active = set(ws.get("active", []))

    def raise_(key: str, msg: str) -> None:
        if key not in active:
            alerts.append(msg)
            active.add(key)

    def clear(key: str, msg: str | None = None) -> None:
        if key in active:
            active.discard(key)
            if msg:
                alerts.append(msg)

    st = read_status(status_path)
    age_min = None
    if st:
        try:
            age_min = (now - datetime.fromisoformat(st["generated_utc"])).total_seconds() / 60.0
        except Exception:
            age_min = None

    # bot alive (BTC-only bot: only when it is the active bot -- see btc_bot_watched)
    if not btc_bot_watched():
        clear("stale")
    elif age_min is None or age_min > STALE_MIN:
        raise_("stale", f"BOT NOT UPDATING: status is {'missing' if age_min is None else f'{age_min:.0f} min old'}")
        if (age_min is None or age_min > RESTART_MIN) and AUTO_RESTART and allow_actions and not bot_running(procs):
            last = ws.get("last_restart")
            if not last or (now - datetime.fromisoformat(last)).total_seconds() > RESTART_MIN * 60:
                restart()
                ws["last_restart"] = now.isoformat()
                alerts.append("bot process not found -> relaunched start_live.bat")
    else:
        clear("stale", "bot updating again")

    # MT5 alive
    if not mt5_running(procs):
        if "mt5_down" not in active:
            started = start_mt5() if allow_actions else False
            raise_("mt5_down", "MT5 TERMINAL NOT RUNNING" + (" -> relaunched" if started else
                                                              " (set MT5_TERMINAL_PATH in .env to auto-start)"))
    else:
        clear("mt5_down", "MT5 terminal running again")

    if st:
        a = st.get("account") or {}
        if a.get("error"):
            raise_("acct_error", f"BOT ERROR: {a['error']}")
        else:
            clear("acct_error")
        if a.get("terminal_trade_allowed") is False or a.get("account_trade_allowed") is False:
            raise_("algo_off", "ALGO TRADING IS OFF in MT5 (terminal or account) -- no orders can be sent")
        else:
            clear("algo_off", "algo trading allowed again")
        bal, peak, floor = a.get("balance"), a.get("peak_balance"), a.get("drawdown_floor")
        if bal and peak and floor and peak > floor:
            if (bal - floor) / (peak - floor) <= 0.25:
                raise_("dd_near", f"DRAWDOWN WARNING: balance ${bal} is close to the floor ${floor} (peak ${peak})")
            else:
                clear("dd_near")

    # kill switch
    if kill.exists():
        try:
            why = kill.read_text(encoding="utf-8").strip()[:300]
        except Exception:
            why = "?"
        raise_("kill", f"KILL SWITCH ACTIVE -- no new entries. {why}")
    else:
        clear("kill", "kill switch cleared")

    # new trades / events
    if db.exists():
        c = sqlite3.connect(db)
        try:
            last_id = int(ws.get("last_event_id", 0))
            rows = c.execute("SELECT id, kind, payload FROM events WHERE id > ? ORDER BY id", (last_id,)).fetchall()
            if "last_event_id" not in ws:                     # first run: don't replay history
                rows = []
                mx = c.execute("SELECT COALESCE(MAX(id),0) FROM events").fetchone()[0]
                ws["last_event_id"] = int(mx)
            for eid, kind, payload in rows:
                ws["last_event_id"] = int(eid)
                if kind in EVENT_KINDS:
                    alerts.append(f"{EVENT_KINDS[kind]}: {payload[:300]}")
            known = set(ws.get("known_tickets", []))
            tickets = c.execute("SELECT ticket, direction, volume, entry, sl, tp, risk_usd FROM trades").fetchall()
            if "known_tickets" not in ws:
                known = {t[0] for t in tickets}
            for t in tickets:
                if t[0] not in known:
                    alerts.append(f"LIVE OPEN #{t[0]} {t[1]} {t[2]} @ {t[3]} SL {t[4]} TP {t[5]} risk ${t[6]}")
                    known.add(t[0])
            ws["known_tickets"] = sorted(known)
        finally:
            c.close()

    # edge monitor, hourly
    hour_key = now.strftime("%Y-%m-%dT%H")
    if ws.get("edge_checked") != hour_key:
        ws["edge_checked"] = hour_key
        try:
            r = run_edge(enforce=allow_actions)
            if r.get("enforced"):
                alerts.append("EDGE MONITOR RETIRED THE STRATEGY: " + "; ".join(r.get("trips", [])))
        except Exception as e:
            log.exception("edge monitor failed: %s", e)

    # daily heartbeat
    day = now.strftime("%Y-%m-%d")
    if now.hour >= HEARTBEAT_UTC_HOUR and ws.get("heartbeat_day") != day:
        ws["heartbeat_day"] = day
        alerts.append("daily heartbeat: " + status_line(st))

    ws["active"] = sorted(active)
    return alerts


MULTI_STATUS = ROOT / "reports" / "multi_live_status.json"
MULTI_DB = ROOT / "state" / "multi_live.sqlite"
MULTI_REF = ROOT / "reports" / "multi_reference_trades.csv"
MULTI_BAT = ROOT / "start_live_multi.bat"


def multi_enabled() -> bool:
    v = os.getenv("WATCHDOG_WATCH_MULTI", "auto").strip().lower()
    return MULTI_STATUS.exists() if v == "auto" else v in ("1", "true", "yes", "on")


def launch_multi() -> None:
    subprocess.Popen(["cmd", "/c", "start", "", "/min", str(MULTI_BAT)], cwd=str(ROOT),
                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def check_multi(now: datetime, ws: dict, *, procs, status_path=MULTI_STATUS, db=MULTI_DB, ref=MULTI_REF,
                kill=KILL, restart=launch_multi, run_edge=edge_monitor.run, allow_actions: bool = True) -> list[str]:
    """Same checks for the multi-symbol trader (execution/live_multi.py); state kept in ws['multi']."""
    m = ws.setdefault("multi", {})
    alerts: list[str] = []
    active = set(m.get("active", []))

    def raise_(key, msg):
        if key not in active:
            alerts.append("[multi] " + msg)
            active.add(key)

    def clear(key, msg=None):
        if key in active:
            active.discard(key)
            if msg:
                alerts.append("[multi] " + msg)

    st = read_status(status_path)
    age = None
    if st:
        try:
            age = (now - datetime.fromisoformat(st["generated_utc"])).total_seconds() / 60.0
        except Exception:
            age = None
    if age is None or age > STALE_MIN:
        raise_("stale", f"TRADER NOT UPDATING: status is {'missing' if age is None else f'{age:.0f} min old'}")
        running = any("python" in n and "live_multi" in c for n, c in procs)
        if (age is None or age > RESTART_MIN) and AUTO_RESTART and allow_actions and not running:
            last = m.get("last_restart")
            if not last or (now - datetime.fromisoformat(last)).total_seconds() > RESTART_MIN * 60:
                restart()
                m["last_restart"] = now.isoformat()
                alerts.append("[multi] trader process not found -> relaunched start_live_multi.bat")
    else:
        clear("stale", "trader updating again")
    if st and st.get("error"):
        raise_("error", f"ERROR: {st['error']}")
    elif st:
        clear("error")
    if st and st.get("peak_equity") and st.get("equity"):
        dd = 1 - st["equity"] / st["peak_equity"]
        if dd >= 0.06:
            raise_("dd", f"DRAWDOWN {dd:.1%} from peak ${st['peak_equity']:.2f} (guard pauses entries at 8 %)")
        else:
            clear("dd")

    if db.exists():
        c = sqlite3.connect(db)
        try:
            if "last_event_id" not in m:
                m["last_event_id"] = int(c.execute("SELECT COALESCE(MAX(id),0) FROM events").fetchone()[0])
                m["known_tickets"] = [t[0] for t in c.execute("SELECT ticket FROM trades")]
            for eid, kind, payload in c.execute("SELECT id, kind, payload FROM events WHERE id > ? ORDER BY id",
                                                (int(m["last_event_id"]),)).fetchall():
                m["last_event_id"] = int(eid)
                if kind in EVENT_KINDS:
                    alerts.append(f"[multi] {EVENT_KINDS[kind]}: {payload[:300]}")
            known = set(m.get("known_tickets", []))
            for t in c.execute("SELECT ticket, symbol, direction, volume, entry, sl, tp, risk_usd FROM trades"):
                if t[0] not in known:
                    alerts.append(f"[multi] LIVE OPEN #{t[0]} {t[1]} {t[2]} {t[3]} @ {t[4]} SL {t[5]} TP {t[6]} risk ${t[7]}")
                    known.add(t[0])
            m["known_tickets"] = sorted(known)
        finally:
            c.close()

    hour_key = now.strftime("%Y-%m-%dT%H")
    if ref.exists() and db.exists() and m.get("edge_checked") != hour_key:
        m["edge_checked"] = hour_key
        try:
            r = run_edge(enforce=allow_actions, db=db, ref=ref, kill_file=kill,
                         out=ROOT / "reports" / "edge_monitor_multi.json")
            if r.get("enforced"):
                alerts.append("[multi] EDGE MONITOR RETIRED THE STRATEGY: " + "; ".join(r.get("trips", [])))
        except Exception as e:
            log.exception("multi edge monitor failed: %s", e)
    m["active"] = sorted(active)
    return alerts


def handle_commands(cmds: list[str], kill=KILL, status_path=STATUS) -> list[str]:
    replies = []
    for cmd in cmds:
        if cmd == "/status":
            replies.append(status_line(read_status(status_path)))
            ms = read_status(MULTI_STATUS) if multi_enabled() else None
            ch = read_status(challenge_tracker.OUT) if challenge_tracker.enabled() else None
            if ch:
                replies.append(f"[challenge] phase {ch.get('phase')} {ch.get('result')} | profit {ch.get('profit_pct')}% "
                               f"(target ${ch.get('target_usd')}) | days {len(ch.get('trading_days') or [])} | "
                               f"headroom today ${ch.get('daily_headroom_usd')}, total ${ch.get('total_headroom_usd')}")
            if ms:
                replies.append(f"[multi] {ms.get('mode')} | equity ${ms.get('equity')} (peak ${ms.get('peak_equity')}) | "
                               f"open {len(ms.get('open_positions') or [])} | closed {ms.get('closed_trades')} "
                               f"net ${ms.get('net_pl_usd')} expR {ms.get('expectancy_R')}")
        elif cmd == "/kill":
            kill.parent.mkdir(exist_ok=True)
            kill.write_text(f"{datetime.now(timezone.utc).isoformat()}  telegram /kill\n", encoding="utf-8")
            replies.append("KILL SWITCH SET -- no new entries. Open positions keep broker SL/TP. "
                           "To resume, delete state/KILL_SWITCH on the trading machine.")
        elif cmd in ("/help", "/start"):
            replies.append("/status -- bot status\n/kill -- stop new entries (manual resume only)")
    return replies


def keep_awake() -> None:
    """Windows: block idle sleep while the watchdog runs (released automatically on exit).
    The display may still turn off; MT5 and the bots keep running."""
    if sys.platform != "win32":
        return
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    if not ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED):
        log.warning("could not block idle sleep -- set Windows sleep to 'Never'")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=int, default=60)
    a = ap.parse_args()
    ws = load_state()
    if not a.once:
        keep_awake()
    if a.once:
        now, procs, tmp = datetime.now(timezone.utc), processes(), dict(ws)
        msgs = check(now, tmp, procs=procs, allow_actions=False)
        if multi_enabled():
            msgs += check_multi(now, tmp, procs=procs, allow_actions=False)
        for msg in msgs:
            print(msg)
        return 0
    notify.send("watchdog started: " + status_line(read_status()))
    while True:
        try:
            now, procs = datetime.now(timezone.utc), processes()
            msgs = check(now, ws, procs=procs)
            if multi_enabled():
                msgs += check_multi(now, ws, procs=procs)
            if challenge_tracker.enabled():
                msgs += ["[challenge] " + e for e in challenge_tracker.run_from_files()]
            for msg in msgs:
                notify.send(msg)
            if notify.enabled():
                cmds, ws["tg_offset"] = notify.poll_commands(int(ws.get("tg_offset", 0)))
                for reply in handle_commands(cmds):
                    notify.send(reply)
            save_state(ws)
        except KeyboardInterrupt:
            return 0
        except Exception as e:
            log.exception("watchdog pass failed: %s", e)
        if a.once:
            return 0
        try:
            time.sleep(max(15, a.interval))
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
