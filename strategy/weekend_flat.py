"""
Weekend-flat rule for the multi trader (added 2026-10-17).

FundedNext trades crypto Monday to Friday only, so a BTCUSD / ETHUSD position left open on Friday cannot be closed
while bitcoin keeps moving all weekend; a stop passed on Saturday fills at the Monday open.  For the symbols in
MULTI_WEEKEND_FLAT_SYMBOLS (default: none = off) every open position is closed from FLAT_FROM (Friday 20:00 UTC,
before the earliest Friday crypto close at 21:00 UTC) and no new trade is opened until REOPEN (Sunday 21:00 UTC).
Markets that are shut at the weekend (gold, USDJPY, NDX100) are not listed: their price does not move while held.

Evidence (scratchpad weekend_crypto.py, 2026-10-17, FundedNext $15K live set 2018-03 -> 2026-09, crypto fills
blocked Fri 21:00 -> Sun 21:00 UTC): holding through the weekend $3,762 / yr, worst day -$523 (Monday gaps);
closing crypto on Friday $3,701 / yr, worst day -$318, max drawdown $1,250 -> $1,169, last 12 months $2,154 ->
$2,420; challenge pass rate 100 % either way.
"""
from __future__ import annotations

from datetime import datetime

FRIDAY, SATURDAY, SUNDAY = 4, 5, 6
FLAT_FROM_HOUR = 20          # Friday, UTC
REOPEN_HOUR = 21             # Sunday, UTC


def close_due(now: datetime) -> bool:
    """True on Friday from FLAT_FROM_HOUR:00 UTC: time to close.  Not at the weekend itself, when the market is shut
    and a close order would only be rejected."""
    return now.weekday() == FRIDAY and now.hour >= FLAT_FROM_HOUR


def in_flat_window(now: datetime) -> bool:
    """True from Friday FLAT_FROM_HOUR:00 UTC until Sunday REOPEN_HOUR:00 UTC (now must be timezone-aware UTC)."""
    d, h = now.weekday(), now.hour
    return (d == FRIDAY and h >= FLAT_FROM_HOUR) or d == SATURDAY or (d == SUNDAY and h < REOPEN_HOUR)
