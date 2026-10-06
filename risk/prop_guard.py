"""
Prop-firm rule guard (funded-account challenges).

Prop firms fail an account the moment EQUITY (including floating P/L) touches:
  * the daily loss limit  -- measured from the day's starting balance/equity, and
  * the max loss limit    -- measured from the INITIAL balance (static), not the peak.
So entries must stop well before the limits, and open positions must be closed before
equity can reach them.  This module decides; the caller acts.

Decision levels (fractions of the reference; defaults = 5 % daily / 10 % max firm rules):
  daily:  equity loss >= DAILY_BLOCK  (70 % of the firm's daily limit: 5 % -> 3.5 %, 3 % -> 2.1 %)
                                       -> no new entries today
          equity loss >= DAILY_FLATTEN(85 % of the limit: 5 % -> 4.25 %, 3 % -> 2.55 %)
                                       -> close everything, halt for the day
  max:    equity <= initial x (1 - MAX_BLOCK)   (default 7 %)  -> no new entries
          equity <= initial x (1 - MAX_FLATTEN) (default 8.5 %) -> close everything, KILL_SWITCH
  profit target reached -> optional stop-trading (lock in the pass)

Pure functions; tested in tests/test_prop_guard.py.
"""
from __future__ import annotations

from dataclasses import dataclass

DAILY_BLOCK_SHARE = 0.70        # stop new entries at 70 % of the firm's daily loss limit
DAILY_FLATTEN_SHARE = 0.85      # close everything at 85 %

@dataclass
class PropRules:
    initial_balance: float
    daily_loss_limit: float = 0.05          # firm rule
    max_loss_limit: float = 0.10            # firm rule (static, from initial balance)
    daily_block: float | None = None        # our buffers; None = scale with daily_loss_limit
    daily_flatten: float | None = None
    max_block: float = 0.07
    max_flatten: float = 0.085
    profit_target: float | None = 0.10      # phase target; None = funded (no target)
    stop_at_target: bool = True

    def __post_init__(self):
        if self.daily_block is None:
            self.daily_block = round(DAILY_BLOCK_SHARE * self.daily_loss_limit, 6)
        if self.daily_flatten is None:
            self.daily_flatten = round(DAILY_FLATTEN_SHARE * self.daily_loss_limit, 6)


@dataclass
class PropDecision:
    allow_entries: bool
    flatten: bool
    kill: bool
    day_halt: bool
    reason: str
    daily_loss_frac: float
    total_loss_frac: float


def evaluate(rules: PropRules, *, equity: float, day_start: float) -> PropDecision:
    """day_start: the firm's daily reference (usually max(balance, equity) at the server-day rollover)."""
    dl = (day_start - equity) / day_start if day_start > 0 else 0.0
    tl = (rules.initial_balance - equity) / rules.initial_balance
    gain = -tl
    if tl >= rules.max_flatten:
        return PropDecision(False, True, True, False, f"equity {tl:.2%} below initial >= {rules.max_flatten:.2%}: flatten + kill",
                            dl, tl)
    if dl >= rules.daily_flatten:
        return PropDecision(False, True, False, True, f"daily equity loss {dl:.2%} >= {rules.daily_flatten:.2%}: flatten, halt today",
                            dl, tl)
    if tl >= rules.max_block:
        return PropDecision(False, False, False, False, f"equity {tl:.2%} below initial >= {rules.max_block:.2%}: no entries", dl, tl)
    if dl >= rules.daily_block:
        return PropDecision(False, False, False, True, f"daily equity loss {dl:.2%} >= {rules.daily_block:.2%}: no entries today",
                            dl, tl)
    if rules.profit_target is not None and rules.stop_at_target and gain >= rules.profit_target:
        return PropDecision(False, False, False, False, f"profit target {rules.profit_target:.0%} reached: stop trading", dl, tl)
    return PropDecision(True, False, False, False, "ok", dl, tl)


def max_risk_per_trade(rules: PropRules, *, equity: float, day_start: float, n_open: int,
                       base_risk: float = 0.005) -> float:
    """Shrink risk so that the remaining daily room covers every open stop plus this one."""
    room = rules.daily_block - max(0.0, (day_start - equity) / day_start if day_start > 0 else 0.0)
    if room <= 0:
        return 0.0
    return max(0.0, min(base_risk, room / (n_open + 1)))
