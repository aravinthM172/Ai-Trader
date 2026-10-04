"""
Prop-firm controls for the live trader (FundingPips 2-Step rules, checked 2026-10-04 on the
official help centre).  Pure decision logic; execution/live_multi.py acts on the result.

Combines:
  1. prop_guard.evaluate      -- daily / max-loss buffers and the phase profit target
                                 (block entries -> flatten -> kill, before the firm's limits)
  2. trade-idea loss cap      -- close any position whose floating loss reaches 0.9 % of the
                                 account size, before it can count as a 1 % "strike"
                                 (Striking System on Master accounts; also limits gap losses)
  3. post-loss cooldown       -- no new entry within 10 minutes after a losing trade closes,
                                 so separate trades are never grouped into one "trade idea"
  4. daily halt               -- after a daily flatten, no new entries until the next trading day
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from risk.prop_guard import PropRules, evaluate, max_risk_per_trade

IDEA_LOSS_CLOSE = 0.009          # fraction of account size
LOSS_COOLDOWN_MIN = 10
TARGETS = {1: 0.08, 2: 0.05}     # phase targets; funded (phase 3+) has none


@dataclass
class PropDecisionAll:
    allow_entries: bool
    flatten_all: bool
    kill: bool
    halt_today: bool
    close_tickets: list[int] = field(default_factory=list)
    risk_cap: float = 0.005
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def rules_from_challenge(ch: dict) -> PropRules | None:
    """Prop rules for the current phase from tools/challenge_tracker state."""
    if not ch or ch.get("result") in ("passed", "failed"):
        return None
    base = float(ch.get("phase_start_balance") or 0)
    if base <= 0:
        return None
    return PropRules(initial_balance=base, profit_target=TARGETS.get(int(ch.get("phase") or 0)))


def ideas_over_limit(positions: list[dict], account_size: float, limit: float = IDEA_LOSS_CLOSE) -> list[int]:
    """Tickets of trade ideas (same symbol + direction) whose combined floating loss >= limit x size."""
    groups: dict[tuple, list[dict]] = {}
    for p in positions:
        groups.setdefault((p["symbol"], p["type"]), []).append(p)
    out = []
    for ps in groups.values():
        if sum(p.get("profit") or 0.0 for p in ps) <= -limit * account_size:
            out += [p["ticket"] for p in ps]
    return out


def cooldown_active(last_losing_close: datetime | None, now: datetime, minutes: int = LOSS_COOLDOWN_MIN) -> bool:
    return last_losing_close is not None and now - last_losing_close < timedelta(minutes=minutes)


def decide(*, challenge: dict | None, equity: float, positions: list[dict], last_losing_close: datetime | None,
           now: datetime, trading_day: str, halted_day: str | None, base_risk: float = 0.005) -> PropDecisionAll:
    d = PropDecisionAll(allow_entries=True, flatten_all=False, kill=False, halt_today=False, risk_cap=base_risk)
    rules = rules_from_challenge(challenge or {})
    if rules is None:
        d.reasons.append("no active challenge state -- prop controls idle")
        return d
    day_start = float((challenge or {}).get("day_ref") or rules.initial_balance)
    pg = evaluate(rules, equity=equity, day_start=day_start)
    d.allow_entries, d.flatten_all, d.kill, d.halt_today = pg.allow_entries, pg.flatten, pg.kill, pg.day_halt and pg.flatten
    if pg.reason != "ok":
        d.reasons.append(pg.reason)
    if halted_day == trading_day:
        d.allow_entries = False
        d.reasons.append("halted for the rest of the trading day after a daily flatten")
    if not d.flatten_all:
        d.close_tickets = ideas_over_limit(positions, rules.initial_balance)
        if d.close_tickets:
            d.reasons.append(f"trade idea loss >= {IDEA_LOSS_CLOSE:.1%} of account -> close {d.close_tickets}")
    if cooldown_active(last_losing_close, now):
        d.allow_entries = False
        d.reasons.append(f"cooldown: losing trade closed < {LOSS_COOLDOWN_MIN} min ago")
    d.risk_cap = max_risk_per_trade(rules, equity=equity, day_start=day_start, n_open=len(positions), base_risk=base_risk)
    if d.risk_cap <= 0:
        d.allow_entries = False
    return d
