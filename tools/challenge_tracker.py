"""
Prop-challenge rehearsal tracker: follows a DEMO account exactly the way the prop firm
would judge it, so the real challenge is bought only after the demo has passed.

Default rules = FundingPips 2-Step Standard (verified 2026-10-04 from their checkout page
and help-centre excerpts):
  Phase 1: +8 % target, Phase 2: +5 % target, min 3 trading days per phase,
  daily loss 3 % or 5 % (bought option; PROP_DAILY_LOSS_LIMIT in .env, default 0.03 = the stricter one)
  of max(day-start balance, day-start equity), max loss 10 % of the phase's
  starting balance (static).  Phase 2 restarts from the balance at which phase 1 passed
  (the firm gives a fresh account of the same size; on the demo we rebase instead).

Read-only: it reads reports/multi_live_status.json (equity/balance every bot pass) and the
trade DB (trading days).  It never trades.  Called every watchdog pass; run by hand with
    python -m tools.challenge_tracker            # print current state
    python -m tools.challenge_tracker --reset    # start a new rehearsal from the current balance

Limits are checked on each status update (every ~30 s), so a very fast spike between two
updates could be missed -- the prop guard's buffers (block at 70 %, flatten at 85 % of the daily
limit) exist for that.  The limit is written to the status file as "daily_loss_limit" for the trader.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STATE = ROOT / "state" / "challenge_state.json"
STATUS = ROOT / "reports" / "multi_live_status.json"
DB = ROOT / "state" / "multi_live.sqlite"
OUT = ROOT / "reports" / "challenge_status.json"


@dataclass(frozen=True)
class ChallengeRules:
    name: str = "FundingPips 2-Step Standard"
    targets: tuple = (0.08, 0.05)
    min_days: tuple = (3, 3)
    daily_loss: float = 0.03            # 3 % or 5 % (bought option); runtime value: current_rules()
    max_loss: float = 0.10
    day_reset_utc_hour: int = 21        # FundingPips server time is UTC+3 -> trading day starts 21:00 UTC
    inactivity_days: int = 30           # hard breach: no completed trade within 30 consecutive days


FUNDINGPIPS_2STEP_STANDARD = ChallengeRules()


def current_rules() -> ChallengeRules:
    """Rules with the daily loss limit from PROP_DAILY_LOSS_LIMIT (read at call time, after .env is loaded)."""
    return replace(FUNDINGPIPS_2STEP_STANDARD, daily_loss=float(os.getenv("PROP_DAILY_LOSS_LIMIT", "0.03")))


def trading_day(ts: datetime, reset_hour: int) -> str:
    """The firm's trading-day label for a UTC time (day rolls at reset_hour UTC)."""
    return (ts - timedelta(hours=reset_hour) + timedelta(days=1)).strftime("%Y-%m-%d")


def new_state(balance: float, now: datetime, rules: ChallengeRules = FUNDINGPIPS_2STEP_STANDARD) -> dict:
    return {"rules": rules.name, "started_utc": now.isoformat(), "phase": 1, "result": "in_progress",
            "phase_start_balance": balance, "phase_started_utc": now.isoformat(), "trading_days": [],
            "day": None, "day_ref": balance, "worst_daily_loss": 0.0, "worst_total_loss": 0.0,
            "daily_loss_limit": rules.daily_loss, "history": []}


def update(st: dict, *, now: datetime, equity: float, balance: float, traded_days: set[str],
           rules: ChallengeRules = FUNDINGPIPS_2STEP_STANDARD, last_closed: datetime | None = None) -> list[str]:
    """Advance the challenge state with one observation; returns event messages."""
    events: list[str] = []
    st["daily_loss_limit"] = rules.daily_loss                 # read by risk/prop_controls (the trader's buffers)
    if st["result"] in ("passed", "failed"):
        return events
    day = trading_day(now, rules.day_reset_utc_hour)
    if day != st["day"]:
        st["day"] = day
        st["day_ref"] = max(balance, equity)                  # firm: higher of balance / equity at day start
    phase_start = datetime.fromisoformat(st["phase_started_utc"])
    st["trading_days"] = sorted({d for d in traded_days if d >= trading_day(phase_start, rules.day_reset_utc_hour)}
                                | set(st["trading_days"]))
    base = st["phase_start_balance"]
    daily = (st["day_ref"] - equity) / st["day_ref"] if st["day_ref"] > 0 else 0.0
    total = (base - equity) / base
    st["worst_daily_loss"] = max(st["worst_daily_loss"], daily)
    st["worst_total_loss"] = max(st["worst_total_loss"], total)
    st["profit_pct"] = round(100 * (equity / base - 1), 2)
    st["daily_loss_pct"] = round(100 * daily, 2)
    st["daily_headroom_usd"] = round(st["day_ref"] * rules.daily_loss - (st["day_ref"] - equity), 2)
    st["total_headroom_usd"] = round(base * rules.max_loss - (base - equity), 2)
    i = st["phase"] - 1
    st["target_usd"] = round(base * (1 + rules.targets[i]), 2)

    since = last_closed or datetime.fromisoformat(st["started_utc"])
    st["days_since_last_closed_trade"] = round((now - since).total_seconds() / 86400, 1)
    if st["days_since_last_closed_trade"] >= rules.inactivity_days:
        st["result"] = "failed"
        events.append(f"CHALLENGE FAILED (phase {st['phase']}): no completed trade for {rules.inactivity_days} days (inactivity)")
        return events
    if st["days_since_last_closed_trade"] >= rules.inactivity_days - 10 and st.get("warn_inactive") != st["day"]:
        st["warn_inactive"] = st["day"]
        events.append(f"challenge warning: no completed trade for {st['days_since_last_closed_trade']:.0f} days "
                      f"(breach at {rules.inactivity_days})")
    if daily >= rules.daily_loss:
        st["result"] = "failed"
        events.append(f"CHALLENGE FAILED (phase {st['phase']}): daily loss {daily:.2%} >= {rules.daily_loss:.0%}")
    elif total >= rules.max_loss:
        st["result"] = "failed"
        events.append(f"CHALLENGE FAILED (phase {st['phase']}): total loss {total:.2%} >= {rules.max_loss:.0%}")
    elif equity >= base * (1 + rules.targets[i]) and len(st["trading_days"]) >= rules.min_days[i]:
        st["history"].append({"phase": st["phase"], "passed_utc": now.isoformat(), "balance": balance,
                              "days": len(st["trading_days"]), "worst_daily_loss": round(st["worst_daily_loss"], 4),
                              "worst_total_loss": round(st["worst_total_loss"], 4)})
        if st["phase"] < len(rules.targets):
            events.append(f"PHASE {st['phase']} PASSED: +{rules.targets[i]:.0%} in {len(st['trading_days'])} trading days "
                          f"-> phase {st['phase'] + 1} starts from ${balance:,.2f}")
            st.update(phase=st["phase"] + 1, phase_start_balance=balance, phase_started_utc=now.isoformat(),
                      trading_days=[], worst_daily_loss=0.0, worst_total_loss=0.0, day_ref=max(balance, equity))
        else:
            st["result"] = "passed"
            events.append(f"CHALLENGE PASSED on demo ({rules.name}) -- ready to buy the real challenge")
    else:
        for key, frac, lim, label in (("warn_daily", daily, rules.daily_loss, "daily"),
                                      ("warn_total", total, rules.max_loss, "total")):
            if frac >= 0.7 * lim and st.get(key) != st["day"]:
                st[key] = st["day"]
                events.append(f"challenge warning: {label} loss {frac:.2%} is 70 % of the {lim:.0%} limit")
    return events


# -- file wiring -------------------------------------------------------------------
def _traded_days(db: Path, reset_hour: int) -> set[str]:
    if not db.exists():
        return set()
    c = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        rows = c.execute("SELECT opened_utc FROM trades UNION SELECT closed_utc FROM trades WHERE closed_utc IS NOT NULL")
        return {trading_day(datetime.fromisoformat(r[0]), reset_hour) for r in rows if r[0]}
    finally:
        c.close()


def _last_closed(db: Path) -> datetime | None:
    if not db.exists():
        return None
    c = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        r = c.execute("SELECT MAX(closed_utc) FROM trades WHERE status='CLOSED'").fetchone()
        return datetime.fromisoformat(r[0]) if r and r[0] else None
    finally:
        c.close()


def enabled() -> bool:
    return os.getenv("PROP_CHALLENGE", "").strip().lower() in ("fundingpips_2step_standard", "1", "true", "on")


def run_from_files(*, state_path: Path = STATE, status_path: Path = STATUS, db: Path = DB,
                   out: Path = OUT, rules: ChallengeRules | None = None) -> list[str]:
    rules = rules or current_rules()
    try:
        stat = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    equity = float(stat.get("equity") or 0.0)
    balance = float(stat.get("balance") or equity)
    if equity <= 0:
        return []
    now = datetime.fromisoformat(stat["generated_utc"])
    try:
        st = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        st = new_state(balance, now, rules)
    ev = update(st, now=now, equity=equity, balance=balance,
                traded_days=_traded_days(db, rules.day_reset_utc_hour), rules=rules, last_closed=_last_closed(db))
    state_path.parent.mkdir(exist_ok=True)
    state_path.write_text(json.dumps(st, indent=2), encoding="utf-8")
    out.write_text(json.dumps(st, indent=2), encoding="utf-8")
    return ev


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="start a new rehearsal from the current balance")
    a = ap.parse_args()
    if a.reset and STATE.exists():
        STATE.unlink()
    for e in run_from_files():
        print(e)
    try:
        st = json.loads(STATE.read_text(encoding="utf-8"))
        print(json.dumps({k: st.get(k) for k in ("rules", "phase", "result", "phase_start_balance", "profit_pct",
                                                 "target_usd", "trading_days", "daily_headroom_usd",
                                                 "total_headroom_usd", "history")}, indent=2))
    except Exception:
        print("no challenge state yet (needs reports/multi_live_status.json with equity)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
