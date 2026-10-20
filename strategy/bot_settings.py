"""
Per-market bot settings edited from the dashboard (added 2026-10-20).

The owner sets, for each market: on / off, sizing mode ("risk" = lot from risk % and the stop distance, "lots" = a
fixed lot), risk %, fixed lot, stop and target in ATR.  Plus one account-wide "auto risk" switch: risk shrinks as the
account nears its total-loss floor and grows (up to 1.5x) as profit builds the cushion (Grossman-Zhou drawdown
control; backtest 2026-10-19: 0 % challenge fails vs 2 % at fixed risk, research_tests_b.py).

The dashboard sends a command, dashboard/publisher.py validates it here and writes state/bot_settings.json, and
execution/live_multi.py reads the file on every pass (no restart).  A market missing from the file keeps the .env
behaviour.  Hard limits below cannot be raised from the dashboard: no trade may risk more than HARD_MAX_RISK of
equity, whatever the risk % or fixed lot says.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "state" / "bot_settings.json"
MARKETS = ("BTCUSD", "ETHUSD", "XAUUSD", "USDJPY", "NDX100")
LIMITS = {"risk_pct": (0.05, 1.0), "lots": (0.0, 50.0), "sl_atr": (0.5, 5.0), "tp_atr": (1.0, 15.0)}
HARD_MAX_RISK = 0.01                 # 1 % of equity per trade, the ceiling for every mode
CUSHION_MIN, CUSHION_MAX = 0.3, 1.5  # auto-risk multiplier range
DEFAULT = {"enabled": True, "mode": "risk", "risk_pct": 0.35, "lots": 0.0, "sl_atr": 2.0, "tp_atr": 6.0}


def _num(v, key: str, errors: list[str], where: str) -> float | None:
    lo, hi = LIMITS[key]
    try:
        x = float(v)
    except (TypeError, ValueError):
        errors.append(f"{where}: {key} must be a number")
        return None
    if not math.isfinite(x) or not lo <= x <= hi:
        errors.append(f"{where}: {key} {x:g} outside {lo:g}..{hi:g}")
        return None
    return x


def validate(raw: dict) -> tuple[dict, list[str]]:
    """Clean settings from the dashboard -> (settings, errors).  Any error means: do not save."""
    errors: list[str] = []
    if not isinstance(raw, dict):
        return {}, ["settings must be an object"]
    markets = {}
    for sym, m in (raw.get("markets") or {}).items():
        if sym not in MARKETS:
            errors.append(f"unknown market {sym!r}")
            continue
        if not isinstance(m, dict):
            errors.append(f"{sym}: settings must be an object")
            continue
        mode = m.get("mode", "risk")
        if mode not in ("risk", "lots"):
            errors.append(f"{sym}: mode must be risk or lots")
        out = {"enabled": bool(m.get("enabled", True)), "mode": mode}
        for k in ("risk_pct", "lots", "sl_atr", "tp_atr"):
            out[k] = _num(m.get(k, DEFAULT[k]), k, errors, sym)
        if mode == "lots" and not (out["lots"] or 0) > 0:
            errors.append(f"{sym}: a fixed lot must be above 0")
        if out["sl_atr"] and out["tp_atr"] and out["tp_atr"] < out["sl_atr"]:
            errors.append(f"{sym}: target ({out['tp_atr']:g} ATR) below stop ({out['sl_atr']:g} ATR)")
        markets[sym] = out
    a = raw.get("auto_risk") or {}
    auto = {"enabled": bool(a.get("enabled", False))}
    try:
        auto["start_balance"] = float(a.get("start_balance", 0) or 0)
        auto["max_loss_pct"] = float(a.get("max_loss_pct", 10) or 10)
    except (TypeError, ValueError):
        errors.append("auto risk: start balance and max loss must be numbers")
    if auto["enabled"] and not (auto.get("start_balance", 0) > 0 and 1 <= auto.get("max_loss_pct", 0) <= 20):
        errors.append("auto risk: needs the challenge start balance and a max loss of 1-20 %")
    return {"version": 1, "markets": markets, "auto_risk": auto}, errors


def save(settings: dict, path: Path = PATH, *, who: str = "dashboard") -> None:
    """Write atomically, keep the previous file as .bak."""
    path.parent.mkdir(exist_ok=True)
    data = {**settings, "updated_utc": datetime.now(timezone.utc).isoformat(), "updated_by": who}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    if path.exists():
        path.replace(path.with_suffix(".json.bak"))
    os.replace(tmp, path)


def load(path: Path = PATH) -> dict:
    """The saved settings, re-validated; {} (= use .env) when missing or broken."""
    try:
        clean, errors = validate(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return {}
    return {} if errors else clean


def for_symbol(settings: dict, symbol: str) -> dict | None:
    return ((settings or {}).get("markets") or {}).get(symbol.replace(".vx", ""))


def cushion_multiplier(settings: dict, equity: float) -> float:
    """Auto risk: 1.0 at the start balance, down to 0.3 near the loss floor, up to 1.5 with profit."""
    a = (settings or {}).get("auto_risk") or {}
    if not a.get("enabled") or not a.get("start_balance") or equity <= 0:
        return 1.0
    room = a["start_balance"] * a["max_loss_pct"] / 100
    floor = a["start_balance"] - room
    return float(min(CUSHION_MAX, max(CUSHION_MIN, (equity - floor) / room)))


def size(m: dict, *, equity: float, risk_frac: float, sl_dist: float, value_per_unit: float,
         vmin: float, vstep: float, vmax: float) -> tuple[float, float, str]:
    """(volume, risk $, note) for a market with dashboard settings.  volume 0 = skip.
    risk mode: lot from risk_frac (already includes auto risk / prop caps); fixed-lot mode: the owner's lot.
    Either way the lot is cut so the trade never risks more than HARD_MAX_RISK of equity."""
    if sl_dist <= 0 or value_per_unit <= 0 or equity <= 0:
        return 0.0, 0.0, "bad inputs"
    per_lot = sl_dist * value_per_unit
    step = lambda v: math.floor(v / vstep + 1e-9) * vstep          # noqa: E731
    if m.get("mode") == "lots":
        vol, note = step(float(m["lots"])), f"fixed lot {m['lots']:g}"
    else:
        vol, note = step(equity * min(risk_frac, HARD_MAX_RISK) / per_lot), "ok"
    cap = step(equity * HARD_MAX_RISK / per_lot)
    if vol > cap:
        vol, note = cap, f"lot cut to {cap:g}: {note} would risk more than {HARD_MAX_RISK:.0%} of equity"
    if vol < vmin:
        if vmin * per_lot <= equity * HARD_MAX_RISK and (m.get("mode") == "lots" or vmin * per_lot <= equity * max(risk_frac, 0.0035)):
            vol = vmin
        else:
            return 0.0, vmin * per_lot, f"min lot risks ${vmin * per_lot:.2f} -- above the limit"
    vol = min(vol, vmax)
    return round(vol, 8), vol * per_lot, note
