"""Tiny JSON state store: per-day realised P/L, trade counters, last-entry times."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_STATE_DIR = _ROOT / "state"
_STATE_DIR.mkdir(exist_ok=True)
_PATH = _STATE_DIR / "bot_state.json"


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def load() -> dict[str, Any]:
    if not _PATH.exists():
        return {}
    try:
        return json.loads(_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save(state: dict[str, Any]) -> None:
    _PATH.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def day_bucket(state: dict) -> dict:
    d = _today()
    day = state.setdefault("days", {}).setdefault(d, {
        "realised_pl": 0.0, "trades": 0, "start_balance": None,
    })
    return day


def set_start_balance(state: dict, balance: float) -> None:
    day = day_bucket(state)
    if day.get("start_balance") is None:
        day["start_balance"] = float(balance)


def record_intent(state: dict, symbol: str) -> None:
    """Called when a (dry-run) entry decision is APPROVED, for cooldown tracking."""
    state.setdefault("last_entry", {})[symbol] = datetime.now(timezone.utc).timestamp()
    day = day_bucket(state)
    day["trades"] = int(day.get("trades", 0)) + 1


def seconds_since_last_entry(state: dict, symbol: str) -> float | None:
    ts = state.get("last_entry", {}).get(symbol)
    if ts is None:
        return None
    return datetime.now(timezone.utc).timestamp() - float(ts)
