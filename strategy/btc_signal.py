"""
BTC directional signal -- transparent, rule-based, NOT optimised, NOT an ML model.

Research (backtest/btc_signal_discovery.py, btc_phase3_confirmation.py) found NO
robust directional edge in BTC M5 OHLC.  This module therefore exists only so the
execution pipeline is end-to-end testable; it is deliberately conservative and
returns HOLD unless a clean multi-factor trend alignment is present.  Do not
treat its output as a validated edge.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Signal:
    symbol: str
    decision: str          # BUY | SELL | HOLD
    confidence: float      # 0..1
    reason: str
    features: dict

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def generate(symbol: str, feat: dict, *, min_confidence: float = 0.70) -> Signal:
    need = ("ema_fast", "ema_slow", "ema_trend", "rsi", "trend_strength_20", "macd_hist", "natr")
    if not feat or any(feat.get(k) is None for k in need):
        return Signal(symbol, "HOLD", 0.0, "features incomplete", feat)

    ef, es, et = feat["ema_fast"], feat["ema_slow"], feat["ema_trend"]
    rsi = feat["rsi"]
    ts = feat["trend_strength_20"]
    mh = feat["macd_hist"]

    up = ef > es > et
    dn = ef < es < et
    score = 0.0
    parts = []

    if up:
        score += 0.35; parts.append("EMA stacked up")
    elif dn:
        score += 0.35; parts.append("EMA stacked down")

    if up and rsi >= 55:
        score += 0.20; parts.append(f"RSI {rsi:.0f}>=55")
    elif dn and rsi <= 45:
        score += 0.20; parts.append(f"RSI {rsi:.0f}<=45")

    if up and ts > 0.3:
        score += 0.20; parts.append(f"trend_strength {ts:+.2f}")
    elif dn and ts < -0.3:
        score += 0.20; parts.append(f"trend_strength {ts:+.2f}")

    if up and mh > 0:
        score += 0.15; parts.append("MACD hist +")
    elif dn and mh < 0:
        score += 0.15; parts.append("MACD hist -")

    if up and score >= min_confidence:
        return Signal(symbol, "BUY", round(score, 3), "; ".join(parts), feat)
    if dn and score >= min_confidence:
        return Signal(symbol, "SELL", round(score, 3), "; ".join(parts), feat)
    return Signal(symbol, "HOLD", round(score, 3),
                  "no clean alignment" if not (up or dn) else f"alignment weak ({score:.2f})", feat)
