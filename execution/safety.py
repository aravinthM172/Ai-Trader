"""
Live-order safety controller.

Every guard here must pass before an entry is even *validated* for live sending.
None of these can be bypassed by config -- they can only be made stricter.

Guards:
  * emergency kill switch      (state/KILL_SWITCH file OR env KILL_SWITCH=1)
  * live-trading master switch  (env LIVE_TRADING, default false)
  * max open positions         (account-wide and per-symbol)
  * max daily loss             (realised, vs start-of-day balance)
  * cooldown between entries
  * duplicate-position protection (same symbol + same direction)
  * spread protection          (delegates to strategy.stops.spread_filter result)
  * stale-price protection     (tick age)
  * insufficient-margin protection (delegates to sizing result)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from common.logging_setup import get_logger
from execution import state as state_store

log = get_logger("btc.safety")

_KILL_FILE = Path(__file__).resolve().parents[1] / "state" / "KILL_SWITCH"


def live_trading_enabled() -> bool:
    return os.getenv("LIVE_TRADING", "false").strip().lower() in ("1", "true", "yes", "on")


def kill_switch_active() -> bool:
    return _KILL_FILE.exists() or os.getenv("KILL_SWITCH", "0").strip() in ("1", "true", "yes", "on")


@dataclass
class SafetyDecision:
    allow_validation: bool = True
    allow_live_send: bool = False
    checks: dict = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def evaluate(
    *,
    symbol: str,
    direction: str,
    cfg,
    account: dict,
    open_positions: list[dict],
    tick_age_seconds: float | None,
    spread_ok: bool,
    sizing_ok: bool,
    state: dict,
) -> SafetyDecision:
    d = SafetyDecision()
    c = d.checks

    # -- kill switch / master switch ---------------------------------
    kill = kill_switch_active()
    live = live_trading_enabled()
    c["kill_switch_active"] = kill
    c["live_trading_enabled"] = live
    if kill:
        d.allow_validation = False
        d.reasons.append("EMERGENCY KILL SWITCH active")

    # -- position limits -----------------------------------------
    same_symbol = [p for p in open_positions if p["symbol"] == symbol]
    same_dir = [p for p in same_symbol if p["type"] == direction]
    c["open_positions_total"] = len(open_positions)
    c["open_positions_symbol"] = len(same_symbol)
    c["max_total_positions"] = cfg.max_total_positions
    c["max_open_positions_symbol"] = cfg.max_open_positions
    if len(open_positions) >= cfg.max_total_positions:
        d.allow_validation = False
        d.reasons.append(f"account at max total positions ({cfg.max_total_positions})")
    if len(same_symbol) >= cfg.max_open_positions:
        d.allow_validation = False
        d.reasons.append(f"{symbol} at max positions ({cfg.max_open_positions})")
    if same_dir:
        d.allow_validation = False
        d.reasons.append(f"duplicate {direction} position on {symbol} (ticket {same_dir[0]['ticket']})")

    # -- daily loss --------------------------------------------
    day = state_store.day_bucket(state)
    start_bal = day.get("start_balance") or account.get("balance", 0.0)
    realised = float(day.get("realised_pl", 0.0))
    max_loss = -abs(cfg.max_daily_loss_frac * start_bal)
    c["daily_realised_pl"] = round(realised, 2)
    c["daily_loss_limit"] = round(max_loss, 2)
    if realised <= max_loss:
        d.allow_validation = False
        d.reasons.append(f"daily loss limit hit ({realised:.2f} <= {max_loss:.2f})")

    # -- cooldown ---------------------------------------------
    since = state_store.seconds_since_last_entry(state, symbol)
    c["seconds_since_last_entry"] = None if since is None else round(since, 1)
    c["cooldown_seconds"] = cfg.cooldown_seconds
    if since is not None and since < cfg.cooldown_seconds:
        d.allow_validation = False
        d.reasons.append(f"cooldown active ({since:.0f}s < {cfg.cooldown_seconds}s)")

    # -- stale price -----------------------------------------
    c["tick_age_seconds"] = tick_age_seconds
    c["stale_seconds"] = cfg.stale_seconds
    if tick_age_seconds is None or tick_age_seconds > cfg.stale_seconds:
        d.allow_validation = False
        d.reasons.append(f"stale / missing tick (age={tick_age_seconds})")

    # -- spread / margin (delegated results) -----------------
    c["spread_ok"] = bool(spread_ok)
    c["sizing_ok"] = bool(sizing_ok)
    if not spread_ok:
        d.allow_validation = False
        d.reasons.append("spread filter rejected")
    if not sizing_ok:
        d.allow_validation = False
        d.reasons.append("position sizing rejected")

    # -- trading session -------------------------------------
    if cfg.trading_hours_utc:
        import datetime as _dt
        hr = _dt.datetime.now(_dt.timezone.utc).hour
        c["hour_utc"] = hr
        c["trading_hours_utc"] = list(cfg.trading_hours_utc)
        if hr not in cfg.trading_hours_utc:
            d.allow_validation = False
            d.reasons.append(f"outside trading session (hour {hr} UTC)")

    d.allow_live_send = bool(d.allow_validation and live and not kill
                             and account.get("trade_allowed", False))
    c["account_trade_allowed"] = account.get("trade_allowed", False)

    if not d.allow_validation:
        log.warning("[%s] safety BLOCKED: %s", symbol, "; ".join(d.reasons))
    return d
