"""
Risk-based position sizing for a small account.

Volume comes from RISK, never from leverage.  All contract maths use the
broker's real specs; the exact loss-at-SL and margin are taken from MT5's own
``order_calc_profit`` / ``order_calc_margin`` when a gateway is supplied, with a
spec-based fallback otherwise.

Volume is normalised to volume_step and clamped to [volume_min, volume_max].
The trade is rejected if the honest size is below volume_min, if it would breach
the risk ceiling, or if free margin after entry drops below the configured floor.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SizingResult:
    direction: str
    entry: float
    sl: float
    sl_distance: float
    balance: float
    risk_fraction: float
    risk_dollars: float
    value_per_price_unit_per_lot: float
    raw_volume: float
    volume: float
    est_loss_at_sl: float
    est_loss_fraction: float
    required_margin: Optional[float]
    free_margin_before: Optional[float]
    free_margin_after: Optional[float]
    checks: dict = field(default_factory=dict)
    accepted: bool = True
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _normalise_volume(vol: float, step: float, vmin: float, vmax: float) -> float:
    if step <= 0:
        step = 0.01
    v = math.floor(vol / step + 1e-9) * step
    v = round(v, 8)
    if vmax and v > vmax:
        v = vmax
    return v


def size_position(
    *,
    direction: str,
    entry: float,
    sl: float,
    balance: float,
    spec,
    cfg,
    gateway=None,
    sim_free_margin: float | None = None,   # PAPER ONLY: use this instead of the live free margin
) -> SizingResult:
    risk_fraction = min(cfg.risk_per_trade, cfg.max_risk_per_trade)
    risk_dollars = balance * risk_fraction
    sl_distance = abs(entry - sl)
    vpp = spec.value_per_price_unit_per_lot  # $ per 1.0 price unit per 1.0 lot

    # loss per 1.0 lot at the stop
    loss_per_lot = None
    if gateway is not None:
        loss_per_lot = gateway.calc_profit(spec.symbol, direction, 1.0, entry, sl)
        if loss_per_lot is not None:
            loss_per_lot = abs(loss_per_lot)
    if not loss_per_lot or loss_per_lot <= 0:
        loss_per_lot = sl_distance * vpp

    raw_volume = risk_dollars / loss_per_lot if loss_per_lot > 0 else 0.0
    volume = _normalise_volume(raw_volume, spec.volume_step, spec.volume_min, spec.volume_max)

    # small-account allowance: if the risk-derived size rounds below volume_min but the
    # broker minimum lot would still risk <= max_risk_per_trade, take the minimum lot.
    used_min_lot_allowance = False
    if volume < spec.volume_min - 1e-12:
        min_loss = spec.volume_min * loss_per_lot
        if getattr(cfg, "allow_min_lot_over_target", True) and balance > 0 \
                and (min_loss / balance) <= cfg.max_risk_per_trade + 1e-9:
            volume = spec.volume_min
            used_min_lot_allowance = True

    # recompute the actual loss/margin at the normalised volume
    est_loss = None
    if gateway is not None and volume > 0:
        p = gateway.calc_profit(spec.symbol, direction, volume, entry, sl)
        est_loss = abs(p) if p is not None else None
    if est_loss is None:
        est_loss = sl_distance * vpp * volume

    required_margin = None
    free_before = None
    free_after = None
    if gateway is not None:
        acc = gateway.account_info()
        free_before = acc.get("margin_free")
        if sim_free_margin is not None:                 # paper: assume free margin ~= sim balance
            free_before = float(sim_free_margin)
        if volume > 0:
            required_margin = gateway.calc_margin(spec.symbol, direction, volume, entry)
        if required_margin is not None and free_before is not None:
            free_after = free_before - required_margin

    est_loss_fraction = est_loss / balance if balance > 0 else float("inf")

    checks = {
        "risk_fraction": round(risk_fraction, 5),
        "risk_dollars": round(risk_dollars, 4),
        "loss_per_lot_at_sl": round(loss_per_lot, 4),
        "used_min_lot_allowance": used_min_lot_allowance,
        "raw_volume": round(raw_volume, 6),
        "volume_min": spec.volume_min,
        "volume_step": spec.volume_step,
        "volume_max": spec.volume_max,
        "normalised_volume": volume,
        "est_loss_at_sl": round(est_loss, 4),
        "est_loss_fraction_of_balance": round(est_loss_fraction, 5),
        "required_margin": None if required_margin is None else round(required_margin, 4),
        "free_margin_before": None if free_before is None else round(free_before, 4),
        "free_margin_after": None if free_after is None else round(free_after, 4),
        "min_free_margin_frac": cfg.min_free_margin_frac,
    }

    res = SizingResult(
        direction=direction, entry=round(entry, 2), sl=round(sl, 2),
        sl_distance=round(sl_distance, 4), balance=round(balance, 2),
        risk_fraction=risk_fraction, risk_dollars=round(risk_dollars, 4),
        value_per_price_unit_per_lot=vpp,
        raw_volume=round(raw_volume, 6), volume=volume,
        est_loss_at_sl=round(est_loss, 4), est_loss_fraction=round(est_loss_fraction, 5),
        required_margin=None if required_margin is None else round(required_margin, 4),
        free_margin_before=None if free_before is None else round(free_before, 4),
        free_margin_after=None if free_after is None else round(free_after, 4),
        checks=checks,
    )

    if volume < spec.volume_min - 1e-12:
        res.accepted = False
        res.reasons.append(
            f"honest size {raw_volume:.4f} < volume_min {spec.volume_min} "
            f"(risk ${risk_dollars:.2f} too small for a {sl_distance:.2f} stop)")
    elif used_min_lot_allowance:
        res.reasons.append(
            f"NOTE: risk-derived size {raw_volume:.4f} < volume_min; using volume_min "
            f"{spec.volume_min} at {est_loss_fraction:.2%} risk (<= max {cfg.max_risk_per_trade:.0%})")
    if est_loss_fraction > cfg.max_risk_per_trade + 1e-9:
        res.accepted = False
        res.reasons.append(
            f"est loss {est_loss_fraction:.1%} of balance > max_risk_per_trade {cfg.max_risk_per_trade:.1%}")
    if spec.volume_max and volume > spec.volume_max:
        res.accepted = False
        res.reasons.append(f"volume {volume} > broker volume_max {spec.volume_max}")
    if free_after is not None and balance > 0 and free_after < cfg.min_free_margin_frac * balance:
        res.accepted = False
        res.reasons.append(
            f"free margin after ${free_after:.2f} < {cfg.min_free_margin_frac:.0%} of balance")
    if required_margin is not None and free_before is not None and required_margin > free_before:
        res.accepted = False
        res.reasons.append(f"required margin ${required_margin:.2f} > free margin ${free_before:.2f}")

    return res
