"""
Order construction + validation.  DRY RUN by default.

build_request()   -> a fully-formed MT5 TRADE_ACTION_DEAL dict (not sent)
validate()        -> runs mt5.order_check() (READ-ONLY) and returns a report

order_send() is NEVER called here.  Even when LIVE_TRADING=true this module only
*validates*; sending is a separate, explicit step that is intentionally not
implemented during this integration phase.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from common.logging_setup import get_logger

log = get_logger("btc.order")

# MT5 retcodes we treat as "ok to send"
_RET_OK = {0, 10009}          # 0 = check done, 10009 = TRADE_RETCODE_DONE
_RET_CHECK_DONE = 0


@dataclass
class OrderValidation:
    symbol: str
    direction: str
    entry: float
    volume: float
    sl: float
    tp: float
    sl_distance: float
    tp_distance: float
    reward_risk: float
    spread_price: float
    atr: float
    risk_dollars: float
    risk_pct: float
    required_margin: Optional[float]
    free_margin: Optional[float]
    filling_mode: Optional[str]
    order_check_retcode: Optional[int] = None
    order_check_comment: Optional[str] = None
    request: dict = field(default_factory=dict)
    checks: dict = field(default_factory=dict)
    valid: bool = False
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    def render(self) -> str:
        L = []
        A = L.append
        A("## BTC TRADE VALIDATION")
        A(f"Symbol          : {self.symbol}")
        A(f"Direction       : {self.direction}")
        A(f"Entry           : {self.entry}")
        A(f"Volume          : {self.volume}")
        A(f"SL              : {self.sl}")
        A(f"TP              : {self.tp}")
        A(f"Risk            : ${self.risk_dollars:.2f}")
        A(f"Risk %          : {self.risk_pct:.2f}%")
        A(f"Spread          : ${self.spread_price:.2f}")
        A(f"ATR             : {self.atr:.2f}")
        A(f"Stop distance   : {self.sl_distance:.2f}")
        A(f"TP distance     : {self.tp_distance:.2f}")
        A(f"Reward/Risk     : {self.reward_risk:.2f}")
        A(f"Required margin : {self.required_margin}")
        A(f"Free margin     : {self.free_margin}")
        A(f"Filling mode    : {self.filling_mode}")
        A(f"Order check     : {self.order_check_retcode} ({self.order_check_comment})")
        A(f"Decision        : {'VALID (dry-run)' if self.valid else 'REJECTED'}")
        if self.reasons:
            A(f"Reasons         : {'; '.join(self.reasons)}")
        return "\n".join(L)


def build_request(*, gateway, spec, direction: str, entry: float,
                  volume: float, sl: float, tp: float,
                  magic: int = 8737, comment: str = "btc-dry-run",
                  deviation: int = 50) -> dict:
    mt5 = gateway.raw()
    ot = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
    return {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": spec.symbol,
        "volume": float(round(volume, 8)),
        "type": ot,
        "price": float(round(entry, spec.digits)),
        "sl": float(round(sl, spec.digits)),
        "tp": float(round(tp, spec.digits)),
        "deviation": int(deviation),
        "magic": int(magic),
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": gateway.preferred_filling(spec),
    }


def validate(*, gateway, spec, direction: str, entry: float, volume: float,
             sl: float, tp: float, atr: float, spread_price: float,
             risk_dollars: float, balance: float,
             required_margin: Optional[float], free_margin: Optional[float],
             stop_plan_checks: dict | None = None) -> OrderValidation:
    mt5 = gateway.raw()
    sl_distance = abs(entry - sl)
    tp_distance = abs(tp - entry)
    rr = tp_distance / sl_distance if sl_distance > 0 else 0.0
    risk_pct = 100.0 * risk_dollars / balance if balance > 0 else 0.0

    fill_map = {mt5.ORDER_FILLING_FOK: "FOK", mt5.ORDER_FILLING_IOC: "IOC",
                mt5.ORDER_FILLING_RETURN: "RETURN"}
    req = build_request(gateway=gateway, spec=spec, direction=direction, entry=entry,
                        volume=volume, sl=sl, tp=tp)

    ov = OrderValidation(
        symbol=spec.symbol, direction=direction, entry=round(entry, 2), volume=volume,
        sl=round(sl, 2), tp=round(tp, 2), sl_distance=round(sl_distance, 2),
        tp_distance=round(tp_distance, 2), reward_risk=round(rr, 3),
        spread_price=round(spread_price, 2), atr=round(atr, 2),
        risk_dollars=round(risk_dollars, 2), risk_pct=round(risk_pct, 3),
        required_margin=required_margin, free_margin=free_margin,
        filling_mode=fill_map.get(req["type_filling"], str(req["type_filling"])),
        request=req,
    )

    # local structural checks first
    broker_min = spec.stops_level_price
    ov.checks = {
        "broker_stops_level_price": round(broker_min, 4),
        "sl_distance_ok": sl_distance >= broker_min - 1e-9,
        "tp_distance_ok": tp_distance >= broker_min - 1e-9,
        "volume_in_range": spec.volume_min - 1e-12 <= volume <= (spec.volume_max or 1e18),
        "volume_step_ok": abs((volume / spec.volume_step) - round(volume / spec.volume_step)) < 1e-6,
        "sl_side_ok": (direction == "BUY" and sl < entry) or (direction == "SELL" and sl > entry),
        "tp_side_ok": (direction == "BUY" and tp > entry) or (direction == "SELL" and tp < entry),
        "trade_mode_full": spec.trade_mode == 4,
    }
    if stop_plan_checks:
        ov.checks["stop_plan"] = stop_plan_checks

    for k in ("sl_distance_ok", "tp_distance_ok", "volume_in_range", "volume_step_ok",
              "sl_side_ok", "tp_side_ok", "trade_mode_full"):
        if not ov.checks[k]:
            ov.reasons.append(f"local check failed: {k}")

    # broker validation (read-only)
    chk = gateway.order_check(req)
    if chk is None:
        ov.reasons.append("order_check returned None (terminal not ready?)")
    else:
        ov.order_check_retcode = chk["retcode"]
        ov.order_check_comment = chk["comment"]
        ov.checks["order_check"] = chk
        if chk["retcode"] != _RET_CHECK_DONE:
            ov.reasons.append(f"order_check retcode {chk['retcode']}: {chk['comment']}")
        if chk.get("margin", 0.0) and chk.get("margin_free", 0.0) < 0:
            ov.reasons.append("order_check reports negative free margin")

    ov.valid = len(ov.reasons) == 0
    (log.info if ov.valid else log.warning)("[%s] order validation %s | %s",
                                            spec.symbol,
                                            "VALID" if ov.valid else "REJECTED",
                                            "; ".join(ov.reasons) or "clean")
    return ov
