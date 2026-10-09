"""
Switch the MT5 terminal to another trading account from the dashboard (run by dashboard/publisher.py on the
trading PC).  The password arrives in the command, is passed to mt5.login() and is never written anywhere.

  1. refuse while the bot has open trades, unless the command says close_first (then flatten and wait)
  2. set state/KILL_SWITCH ("account switch") -- nothing new trades until Start is pressed on the dashboard
  3. stop the trader + watchdog windows (both keep per-account state; the watchdog keeps it in memory)
  4. mt5.login(new); on failure log back in to the old account (terminal's saved password)
  5. the account type MT5 reports must match the one chosen on the page (demo / real); if not, log back out
  6. set LIVE_ACCOUNT_MODE in .env to that type (the trader only sends orders to that kind of account)
  7. archive the old account's bot state to state/archive_<old login>_<UTC stamp>/
  8. start the challenge tracker from the chosen account size and phase (the MT5 balance must be within 90-115 % of it)
  9. start the trader + watchdog windows again (they begin with fresh state and re-read .env)
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "state"
KILL = STATE / "KILL_SWITCH"
FLATTEN = STATE / "EMERGENCY_FLATTEN"
ACCOUNT = STATE / "account.json"                  # login + server only (no password)
ENV = ROOT / ".env"
ACCOUNT_TYPES = ("demo", "real")
MODES = {0: "demo", 1: "contest", 2: "real"}       # MT5 account_info().trade_mode
PER_ACCOUNT = ("multi_live.sqlite", "multi_live_state.json", "challenge_state.json", "watchdog_state.json")
CHALLENGE = STATE / "challenge_state.json"
ACCOUNT_SIZES = (5_000, 10_000, 25_000, 50_000, 100_000)     # FundingPips 2-step account sizes
PHASES = ("1", "2")
# the MT5 balance must be within this band of the chosen size: below 90 % the account has already hit the 10 % max loss,
# above 115 % the chosen size is almost certainly wrong (a smaller size would loosen every loss limit the bot enforces)
SIZE_BAND = (0.90, 1.15)
# (batch file, python module) for each loop that must restart on the new account
LOOPS = (("start_live_multi.bat", "execution.live_multi"), ("start_watchdog.bat", "tools.watchdog"))


def set_env_value(key: str, value: str, *, env: Path = ENV) -> None:
    """Set KEY=value in .env (replacing the active line, or appending), leaving every other line untouched."""
    lines = env.read_text(encoding="utf-8").splitlines() if env.exists() else []
    out, found = [], False
    for line in lines:
        if not line.lstrip().startswith("#") and line.split("=", 1)[0].strip() == key:
            if not found:
                out.append(f"{key}={value}")
            found = True
        else:
            out.append(line)
    if not found:
        out.append(f"{key}={value}")
    tmp = env.with_name(env.name + ".tmp")
    tmp.write_text("\n".join(out) + "\n", encoding="utf-8")
    tmp.replace(env)


def validate(cmd: dict) -> tuple[int | None, str, str, str]:
    """-> (login, server, password, error)."""
    raw = str(cmd.get("login", "")).strip()
    server, password = str(cmd.get("server", "")).strip(), str(cmd.get("password", ""))
    if not raw.isdigit() or not 3 <= len(raw) <= 15:
        return None, server, password, "login must be the account number (digits only)"
    if not server or len(server) > 100:
        return None, server, password, "server is required (as shown in MT5, e.g. FundingPips-Trial)"
    if not password or len(password) > 100:
        return None, server, password, "password is required"
    if cmd.get("account_type") not in ACCOUNT_TYPES:
        return None, server, password, "choose the account type: demo or real"
    size, phase = account_size(cmd), cmd.get("phase")
    if size is None:
        return None, server, password, f"choose the account size ({', '.join(f'${s:,}' for s in ACCOUNT_SIZES)})"
    if phase not in PHASES:
        return None, server, password, "choose the challenge phase: 1 or 2"
    return int(raw), server, password, ""


def account_size(cmd: dict) -> int | None:
    try:
        size = int(cmd.get("account_size"))
    except (TypeError, ValueError):
        return None
    return size if size in ACCOUNT_SIZES else None


def seed_challenge(size: int, phase: int, *, state: Path = STATE, now: datetime | None = None) -> None:
    """Start the challenge tracker from the account's real size and phase, not from whatever balance it shows today
    (an account part-way through a challenge would otherwise get its loss limits measured from the wrong base)."""
    from tools.challenge_tracker import current_rules, new_state
    st = new_state(float(size), now or datetime.now(timezone.utc), current_rules())
    st["phase"] = phase
    (state / CHALLENGE.name).write_text(json.dumps(st, indent=2), encoding="utf-8")


def bot_positions(mt5, magics: set[int]) -> list:
    return [p for p in (mt5.positions_get() or []) if int(getattr(p, "magic", 0)) in magics]


def _wait(cond, timeout: float, step: float, sleep) -> bool:
    end = time.monotonic() + timeout
    while True:
        if cond():
            return True
        if time.monotonic() >= end:
            return False
        sleep(step)


def archive_state(old_login, *, state: Path = STATE, now: datetime | None = None) -> Path | None:
    now = now or datetime.now(timezone.utc)
    files = [state / f for f in PER_ACCOUNT if (state / f).exists()]
    if not files:
        return None
    dest = state / f"archive_{old_login or 'unknown'}_{now:%Y%m%dT%H%M%SZ}"
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.move(str(f), str(dest / f.name))
    return dest


def stop_loops(loops=LOOPS) -> int:
    """Kill each loop's cmd window first (so it cannot relaunch Python), then its Python process."""
    import psutil
    victims = []
    for p in psutil.process_iter(["name", "cmdline"]):
        cmd = " ".join(p.info["cmdline"] or [])
        name = (p.info["name"] or "").lower()
        for bat, module in loops:
            if (name == "cmd.exe" and bat in cmd) or ("python" in name and f"-m {module}" in cmd):
                victims.append((0 if name == "cmd.exe" else 1, p))
    for _, p in sorted(victims, key=lambda v: v[0]):
        try:
            p.kill()
        except psutil.Error:
            pass
    psutil.wait_procs([p for _, p in victims], timeout=15)
    return len(victims)


def start_loops(loops=LOOPS, root: Path = ROOT) -> None:
    for bat, _ in loops:
        subprocess.Popen(["cmd", "/c", "start", "", "/min", str(root / bat)], cwd=str(root),
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def switch_account(cmd: dict, *, mt5, magics: set[int], state: Path = STATE, env: Path = ENV, sleep=time.sleep,
                   stop=stop_loops, start=start_loops, flatten_timeout: float = 150.0) -> tuple[str, str]:
    """Carry out a switch_account command.  Returns (status, message) for the dashboard."""
    login, server, password, err = validate(cmd)
    if err:
        return "failed", err
    want = cmd["account_type"]
    kill, flatten = state / KILL.name, state / FLATTEN.name
    if not mt5.initialize():
        return "failed", f"MT5 not reachable: {mt5.last_error()}"
    old = mt5.account_info()
    if old and int(old.login) == login and old.server == server:
        return "done", f"Already logged in to {login} on {server}."

    mine = bot_positions(mt5, magics)
    if mine and not cmd.get("close_first"):
        return "refused", (f"The bot has {len(mine)} open trade(s) on the current account. Close them first "
                           "(tick 'close the bot's open trades first') or wait until they finish.")
    stamp = datetime.now(timezone.utc).isoformat()
    state.mkdir(parents=True, exist_ok=True)
    kill.write_text(f"{stamp}  account switch to {login} -- press Start on the dashboard\n", encoding="utf-8")
    if mine:
        flatten.write_text(stamp, encoding="utf-8")       # the running trader closes all bot positions
        if not _wait(lambda: not bot_positions(mt5, magics), flatten_timeout, 5, sleep):
            return "failed", ("Could not close the bot's trades within 2.5 min, so the account was NOT switched. "
                              "New trades are paused; check MT5.")
    if flatten.exists():
        flatten.unlink()

    stop()
    try:
        if not mt5.login(login, password=password, server=server, timeout=60_000):
            why = mt5.last_error()
            back = bool(old) and mt5.login(int(old.login), server=old.server, timeout=60_000)
            return "failed", (f"Login to {login} on {server} failed ({why}). "
                              + (f"Still on {old.login}; trading is paused, press Start to resume."
                                 if back else "Could NOT log back in to the old account -- check MT5 on the VPS."))
        new = mt5.account_info()
        if not new or int(new.login) != login:
            return "failed", "MT5 accepted the login but reports a different account -- check MT5 on the VPS."
        got = MODES.get(int(getattr(new, "trade_mode", -1)), "unknown")
        if got != want:
            back = bool(old) and mt5.login(int(old.login), server=old.server, timeout=60_000)
            return "failed", (f"You chose {want.upper()} but MT5 says account {login} is {got.upper()}, so the switch "
                              "was undone and no setting changed. "
                              + (f"Still on {old.login}; trading is paused, press Start to resume."
                                 if back else "Could NOT log back in to the old account -- check MT5 on the VPS."))
        size, phase = account_size(cmd), int(cmd["phase"])
        bal = float(getattr(new, "balance", 0) or 0)
        if not SIZE_BAND[0] * size <= bal <= SIZE_BAND[1] * size:
            back = bool(old) and mt5.login(int(old.login), server=old.server, timeout=60_000)
            return "failed", (f"You chose a ${size:,} account but MT5 says account {login} has a balance of "
                              f"{bal:,.2f} {new.currency}, so the switch was undone and no setting changed. "
                              + (f"Still on {old.login}; trading is paused, press Start to resume."
                                 if back else "Could NOT log back in to the old account -- check MT5 on the VPS."))
        set_env_value("LIVE_ACCOUNT_MODE", want, env=env)
        dest = archive_state(getattr(old, "login", None), state=state)
        (state / ACCOUNT.name).write_text(json.dumps({"login": login, "server": server, "switched_utc": stamp,
                                                      "account_size": size, "phase": phase}), encoding="utf-8")
        seed_challenge(size, phase, state=state)
        return "done", (f"Now on {login} ({server}, {want.upper()}; bot set to trade {want} accounts), "
                        f"balance {new.balance} {new.currency}; loss limits measured from ${size:,} (phase {phase}). "
                        + (f"Old bot history archived to state/{dest.name}. " if dest else "")
                        + "Trading is paused -- press Start when ready.")
    finally:
        start()
