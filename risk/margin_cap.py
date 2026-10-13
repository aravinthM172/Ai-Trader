"""
Position-value cap for symbols a prop firm gives little leverage on (added 2026-10-13).

FundedNext trades crypto at 1:1 leverage and lists "excessive margin usage (70 % or more cumulative)" as gambling
behaviour.  With a tight stop a risk-sized BTC position can be worth two thirds of the account on its own
(0.35 % risk and a 0.53 % stop = 66 %), so the trader caps the total value of the open positions in the symbols
named in MULTI_NOTIONAL_CAP_SYMBOLS at MULTI_NOTIONAL_CAP (a share of equity, e.g. 0.6).  A capped trade is
smaller than its risk budget, never bigger.  Off when either setting is empty.
"""
from __future__ import annotations

import math


def notional(volume: float, price: float, contract_size: float) -> float:
    """Value of a position in the quote currency (USD for BTCUSD / ETHUSD)."""
    return abs(volume) * price * contract_size


def used_notional(positions: list[dict], contract_sizes: dict[str, float], symbols: set[str]) -> float:
    """Value of the open positions in the capped symbols, at their entry prices."""
    return sum(notional(p["volume"], p["price_open"], contract_sizes.get(p["symbol"], 1.0))
               for p in positions if p["symbol"] in symbols)


def cap_volume(vol: float, *, price: float, contract_size: float, equity: float, used: float, max_frac: float,
               vstep: float, vmin: float) -> tuple[float, str]:
    """(volume, note).  The largest volume <= vol whose value fits in what is left of max_frac x equity;
    0 when even the smallest lot does not fit."""
    per_lot = price * contract_size
    if vol <= 0 or per_lot <= 0 or equity <= 0 or max_frac <= 0:
        return vol, "ok"
    room = max_frac * equity - used
    if vol * per_lot <= room:
        return vol, "ok"
    capped = math.floor(room / per_lot / vstep + 1e-9) * vstep if room > 0 else 0.0
    if capped < vmin:
        return 0.0, f"position value cap: ${max(room, 0):,.0f} left of {max_frac:.0%} of equity, min lot is ${vmin * per_lot:,.0f}"
    return round(capped, 8), f"position value capped at {max_frac:.0%} of equity: {vol:g} -> {capped:g} lots"
