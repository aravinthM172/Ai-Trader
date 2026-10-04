"""
Portfolio-level entry guard for trading many symbols from one account.

Ideas taken from freqtrade "protections" (StoplossGuard, MaxDrawdown, CooldownPeriod)
and its max_open_trades limit, rewritten as pure functions so the SAME rules run in
backtest/portfolio_sim.py and, once approved, in live execution.

A guard only decides whether a NEW entry may open.  It never closes positions.

Rules (all configurable; a value of 0 / None disables that rule):
  max_open_positions      account-wide cap on simultaneous positions
  max_total_open_risk     sum of open risk as a fraction of equity (e.g. 0.03 = 3 %)
  max_per_group           simultaneous positions per market group (FX Majors, Indexes, ...)
  stoploss_guard          pause a symbol for `sl_pause_hours` after `sl_count` stop-outs
                          within `sl_lookback_hours`
  cooldown_hours          after any exit on a symbol, wait this long before re-entering it
  daily_loss_limit        no new entries for the rest of the UTC day once realised day P/L
                          <= -limit x start-of-day equity
  max_drawdown_pause      no new entries for `dd_pause_hours` once equity is more than this
                          fraction below its peak
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class GuardConfig:
    max_open_positions: int = 6
    max_total_open_risk: float = 0.03
    max_per_group: int = 2
    sl_count: int = 0                 # stoploss guard off by default (must be backtested first)
    sl_lookback_hours: float = 48
    sl_pause_hours: float = 24
    cooldown_hours: float = 0
    daily_loss_limit: float = 0.0
    max_drawdown_pause: float = 0.0
    dd_pause_hours: float = 72


@dataclass
class OpenPos:
    symbol: str
    group: str
    risk_frac: float                  # risk at entry as a fraction of equity at entry


@dataclass
class GuardState:
    open: list[OpenPos] = field(default_factory=list)
    stops: dict[str, list[datetime]] = field(default_factory=dict)      # symbol -> stop-out times
    last_exit: dict[str, datetime] = field(default_factory=dict)
    paused_until: dict[str, datetime] = field(default_factory=dict)     # symbol or "*" -> time
    day: str = ""
    day_start_equity: float = 0.0
    day_realised: float = 0.0
    peak_equity: float = 0.0


def new_day_if_needed(st: GuardState, now: datetime, equity: float) -> None:
    d = now.strftime("%Y-%m-%d")
    if d != st.day:
        st.day, st.day_start_equity, st.day_realised = d, equity, 0.0
    st.peak_equity = max(st.peak_equity, equity)


def allow_entry(cfg: GuardConfig, st: GuardState, *, symbol: str, group: str, risk_frac: float,
                now: datetime, equity: float) -> tuple[bool, str]:
    new_day_if_needed(st, now, equity)
    for key in ("*", symbol):
        until = st.paused_until.get(key)
        if until and now < until:
            return False, f"paused until {until.isoformat()} ({'portfolio' if key == '*' else symbol})"
    if any(p.symbol == symbol for p in st.open):
        return False, "symbol already has a position"
    if cfg.max_open_positions and len(st.open) >= cfg.max_open_positions:
        return False, f"max_open_positions {cfg.max_open_positions}"
    if cfg.max_total_open_risk and sum(p.risk_frac for p in st.open) + risk_frac > cfg.max_total_open_risk + 1e-12:
        return False, f"total open risk would exceed {cfg.max_total_open_risk:.1%}"
    if cfg.max_per_group and sum(p.group == group for p in st.open) >= cfg.max_per_group:
        return False, f"max_per_group {cfg.max_per_group} in {group}"
    if cfg.cooldown_hours and symbol in st.last_exit and now - st.last_exit[symbol] < timedelta(hours=cfg.cooldown_hours):
        return False, "cooldown after last exit"
    if cfg.daily_loss_limit and st.day_start_equity > 0 and \
            st.day_realised <= -cfg.daily_loss_limit * st.day_start_equity:
        return False, "daily loss limit reached"
    if cfg.max_drawdown_pause and st.peak_equity > 0 and equity < (1 - cfg.max_drawdown_pause) * st.peak_equity:
        st.paused_until["*"] = now + timedelta(hours=cfg.dd_pause_hours)
        st.peak_equity = equity                       # restart the drawdown clock after the pause
        return False, f"drawdown > {cfg.max_drawdown_pause:.0%} -> portfolio paused {cfg.dd_pause_hours:.0f}h"
    return True, "ok"


def on_open(st: GuardState, *, symbol: str, group: str, risk_frac: float) -> None:
    st.open.append(OpenPos(symbol, group, risk_frac))


def on_close(cfg: GuardConfig, st: GuardState, *, symbol: str, now: datetime, pnl: float,
             equity_after: float, stopped: bool) -> None:
    st.open = [p for p in st.open if p.symbol != symbol]
    new_day_if_needed(st, now, equity_after - pnl)
    st.day_realised += pnl
    st.peak_equity = max(st.peak_equity, equity_after)
    st.last_exit[symbol] = now
    if stopped and cfg.sl_count:
        lst = [t for t in st.stops.get(symbol, []) if now - t <= timedelta(hours=cfg.sl_lookback_hours)] + [now]
        st.stops[symbol] = lst
        if len(lst) >= cfg.sl_count:
            st.paused_until[symbol] = now + timedelta(hours=cfg.sl_pause_hours)
            st.stops[symbol] = []
