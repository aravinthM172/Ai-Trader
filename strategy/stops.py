"""
Dynamic ATR-based stop / target system + spread filter + trade-quality gate.

Pure functions -- take numbers in, return a decision.  No MT5, no globals.

Broker minimum stop distance (BTCUSD.vx: 2976 points = $29.76) is ALWAYS
enforced.  The ATR multiplier is configurable (never a single hard-coded stop).
A trade is rejected when market conditions make it mathematically unattractive
(stop dominated by cost, R/R too low, stop absurdly wide, spread too large).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Direction = Literal["BUY", "SELL"]


@dataclass
class StopPlan:
    direction: Direction
    entry: float
    sl: float
    tp: float
    sl_distance: float
    tp_distance: float
    reward_risk: float
    atr: float
    broker_min_distance: float
    spread_price: float
    round_trip_cost: float
    cost_to_sl_ratio: float
    checks: dict = field(default_factory=dict)
    accepted: bool = True
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        return d


def spread_filter(spread_price: float, atr: float, price: float, cfg) -> tuple[bool, dict]:
    ratio = spread_price / atr if atr > 0 else float("inf")
    frac = spread_price / price if price > 0 else float("inf")
    ok = ratio <= cfg.max_spread_atr_ratio and frac <= cfg.max_spread_frac_of_price
    ctx = {
        "spread_price": round(spread_price, 4),
        "atr": round(atr, 4),
        "spread_to_atr": round(ratio, 4),
        "max_spread_atr_ratio": cfg.max_spread_atr_ratio,
        "spread_frac_of_price": round(frac, 6),
        "max_spread_frac_of_price": cfg.max_spread_frac_of_price,
        "ok": bool(ok),
    }
    return ok, ctx


def build_stop_plan(
    *,
    direction: Direction,
    bid: float,
    ask: float,
    atr: float,
    cfg,
    broker_stops_level_price: float,
    slippage_price: float = 0.0,
) -> StopPlan:
    """
    Reference prices (verified against Valetax order_check):
      BUY  -> entry=ask ; SL & TP distances measured from BID
      SELL -> entry=bid ; SL & TP distances measured from ASK
    """
    entry = ask if direction == "BUY" else bid
    ref = bid if direction == "BUY" else ask
    price = (bid + ask) / 2.0
    spread_price = max(0.0, ask - bid)

    broker_min = max(0.0, broker_stops_level_price) * cfg.broker_stop_buffer

    raw_sl = atr * cfg.sl_atr_multiplier
    raw_tp = atr * cfg.tp_atr_multiplier

    # distances measured from the broker reference price (bid for BUY, ask for SELL)
    sl_ref_dist = max(raw_sl, broker_min)

    if direction == "BUY":
        sl = ref - sl_ref_dist
    else:
        sl = ref + sl_ref_dist

    # TRUE risk is measured from the actual fill (entry), which is the spread away from ref
    sl_distance = abs(entry - sl)                       # real risk in price
    round_trip_cost = spread_price + 2.0 * slippage_price
    cost_to_sl = round_trip_cost / sl_distance if sl_distance > 0 else float("inf")

    # size TP so the ENTRY-referenced reward/risk meets the minimum
    tp_entry_dist = max(raw_tp, broker_min, sl_distance * cfg.min_reward_risk)
    if direction == "BUY":
        tp = entry + tp_entry_dist
        # tp must also clear the broker min measured from ref
        if tp - ref < broker_min:
            tp = ref + broker_min
    else:
        tp = entry - tp_entry_dist
        if ref - tp < broker_min:
            tp = ref - broker_min

    tp_distance = abs(tp - entry)
    reward_risk = tp_distance / sl_distance if sl_distance > 0 else 0.0
    max_stop = cfg.max_stop_frac_of_price * price

    checks = {
        "broker_min_distance": round(broker_min, 4),
        "atr_sl_distance": round(raw_sl, 4),
        "sl_distance_from_ref": round(sl_ref_dist, 4),
        "sl_distance_from_entry": round(sl_distance, 4),
        "tp_distance_from_entry": round(tp_distance, 4),
        "reward_risk_from_entry": round(reward_risk, 3),
        "round_trip_cost": round(round_trip_cost, 4),
        "cost_to_sl_ratio": round(cost_to_sl, 3),
        "max_stop_distance": round(max_stop, 2),
        "sl_ref_meets_broker_min": (abs(sl - ref)) >= broker_min - 1e-9,
        "tp_ref_meets_broker_min": (abs(tp - ref)) >= broker_min - 1e-9,
    }

    plan = StopPlan(
        direction=direction, entry=round(entry, 2),
        sl=round(sl, 2), tp=round(tp, 2),
        sl_distance=round(sl_distance, 4), tp_distance=round(tp_distance, 4),
        reward_risk=round(reward_risk, 3), atr=round(atr, 4),
        broker_min_distance=round(broker_min, 4), spread_price=round(spread_price, 4),
        round_trip_cost=round(round_trip_cost, 4), cost_to_sl_ratio=round(cost_to_sl, 3),
        checks=checks,
    )

    if atr <= 0:
        plan.accepted = False
        plan.reasons.append("ATR not available")
    if reward_risk < cfg.min_reward_risk - 1e-9:
        plan.accepted = False
        plan.reasons.append(f"reward/risk {reward_risk:.2f} < {cfg.min_reward_risk}")
    if cost_to_sl > 0.5:
        plan.accepted = False
        plan.reasons.append(f"round-trip cost is {cost_to_sl:.0%} of stop distance (>50%)")
    if sl_distance > max_stop:
        plan.accepted = False
        plan.reasons.append(f"stop distance {sl_distance:.2f} > {cfg.max_stop_frac_of_price:.1%} of price")
    if broker_stops_level_price > 0 and abs(sl - ref) < broker_stops_level_price - 1e-9:
        plan.accepted = False
        plan.reasons.append("SL below raw broker stops level (from market price)")
    if broker_stops_level_price > 0 and abs(tp - ref) < broker_stops_level_price - 1e-9:
        plan.accepted = False
        plan.reasons.append("TP below raw broker stops level (from market price)")

    return plan
