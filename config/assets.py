"""
Per-asset configuration.

Each asset (XAUUSD, BTC) gets an independent AssetConfig so they can differ in
symbol, point size, contract size, spread tolerance, ATR behaviour, stop
distances, risk limits, sessions and position sizing.

Values come from environment variables with a per-asset prefix
(``XAU_`` / ``BTC_``); anything not set falls back to the defaults below.
Nothing here reads MT5 -- live contract specs are read dynamically at runtime
(mt5/gateway.py) and *override* the static hints in this file.

Backward compatible: the legacy ``ASSETS`` dict is still exported.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def _f(prefix: str, key: str, default: float) -> float:
    return float(os.getenv(f"{prefix}_{key}", os.getenv(key, str(default))))


def _i(prefix: str, key: str, default: int) -> int:
    return int(float(os.getenv(f"{prefix}_{key}", os.getenv(key, str(default)))))


def _s(prefix: str, key: str, default: str) -> str:
    return os.getenv(f"{prefix}_{key}", os.getenv(key, default))


def _b(prefix: str, key: str, default: bool) -> bool:
    return _s(prefix, key, str(default)).strip().lower() in ("1", "true", "yes", "on")


@dataclass
class AssetConfig:
    name: str
    symbol: str
    enabled: bool = True
    timeframe: str = "M5"
    # decision source: "rule_ema_rsi" (M5 placeholder) or "momentum_rsi_mtf" (validated H1)
    strategy: str = "rule_ema_rsi"

    # --- indicators / volatility -------------------------------------------
    atr_period: int = 14
    rsi_period: int = 14
    ema_fast: int = 20
    ema_slow: int = 50
    ema_trend: int = 200

    # --- stop / target system --------------------------------------------
    sl_atr_multiplier: float = 2.0
    tp_atr_multiplier: float = 3.0
    min_reward_risk: float = 1.3
    # extra safety factor applied on top of the broker's trade_stops_level
    broker_stop_buffer: float = 1.15
    # hard cap on stop distance as a fraction of price (sanity, not a tune knob)
    max_stop_frac_of_price: float = 0.05

    # --- spread filter --------------------------------------------------
    # reject when spread / ATR exceeds this
    max_spread_atr_ratio: float = 0.35
    # absolute hard ceiling on spread as a fraction of price
    max_spread_frac_of_price: float = 0.01

    # --- risk / sizing ------------------------------------------------
    risk_per_trade: float = 0.01           # fraction of balance risked per trade
    max_risk_per_trade: float = 0.02       # hard ceiling
    allow_min_lot_over_target: bool = True  # take volume_min if raw<min AND min risk <= ceiling
    max_daily_loss_frac: float = 0.05      # of start-of-day balance
    max_open_positions: int = 1            # for THIS asset
    max_total_positions: int = 2           # account-wide
    min_free_margin_frac: float = 0.30     # keep >=30% of balance free after entry
    cooldown_seconds: int = 900            # min gap between entries for this asset

    # --- data quality --------------------------------------------------
    max_candle_gap_multiplier: float = 3.0   # flag gaps > N * timeframe
    max_bar_jump_atr: float = 12.0           # flag |ret| > N * ATR as impossible
    stale_seconds: int = 900                 # tick / last-bar older than this = stale
    min_history_bars: int = 400

    # --- sessions (UTC hours the asset may trade); empty = 24/7 --------
    trading_hours_utc: tuple[int, ...] = ()

    # --- live spec hints (overridden by MT5 at runtime) --------------
    point: float = 0.01
    contract_size: float = 1.0
    tick_size: float = 0.01
    tick_value: float = 0.01
    volume_min: float = 0.01
    volume_step: float = 0.01
    volume_max: float = 50.0
    broker_stops_level_points: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _load(name: str, prefix: str, default_symbol: str, **over) -> AssetConfig:
    tf = _s(prefix, "TIMEFRAME", over.pop("timeframe", "M5"))
    hours_raw = _s(prefix, "TRADING_HOURS_UTC", "").strip()
    hours = tuple(int(x) for x in hours_raw.replace(" ", "").split(",") if x != "") if hours_raw else ()
    cfg = AssetConfig(
        name=name,
        symbol=_s(prefix, "SYMBOL", default_symbol),
        enabled=_b(prefix, "ENABLED", over.pop("enabled", True)),
        timeframe=tf,
        strategy=_s(prefix, "STRATEGY", over.pop("strategy", "rule_ema_rsi")),
        atr_period=_i(prefix, "ATR_PERIOD", over.pop("atr_period", 14)),
        rsi_period=_i(prefix, "RSI_PERIOD", over.pop("rsi_period", 14)),
        ema_fast=_i(prefix, "EMA_FAST", over.pop("ema_fast", 20)),
        ema_slow=_i(prefix, "EMA_SLOW", over.pop("ema_slow", 50)),
        ema_trend=_i(prefix, "EMA_TREND", over.pop("ema_trend", 200)),
        sl_atr_multiplier=_f(prefix, "SL_ATR_MULTIPLIER", over.pop("sl_atr_multiplier", 2.0)),
        tp_atr_multiplier=_f(prefix, "TP_ATR_MULTIPLIER", over.pop("tp_atr_multiplier", 3.0)),
        min_reward_risk=_f(prefix, "MIN_REWARD_RISK", over.pop("min_reward_risk", 1.3)),
        broker_stop_buffer=_f(prefix, "BROKER_STOP_BUFFER", over.pop("broker_stop_buffer", 1.15)),
        max_stop_frac_of_price=_f(prefix, "MAX_STOP_FRAC_OF_PRICE", over.pop("max_stop_frac_of_price", 0.05)),
        max_spread_atr_ratio=_f(prefix, "MAX_SPREAD_ATR_RATIO", over.pop("max_spread_atr_ratio", 0.35)),
        max_spread_frac_of_price=_f(prefix, "MAX_SPREAD_FRAC_OF_PRICE", over.pop("max_spread_frac_of_price", 0.01)),
        risk_per_trade=_f(prefix, "RISK_PER_TRADE", over.pop("risk_per_trade", 0.01)),
        max_risk_per_trade=_f(prefix, "MAX_RISK_PER_TRADE", over.pop("max_risk_per_trade", 0.02)),
        allow_min_lot_over_target=_b(prefix, "ALLOW_MIN_LOT_OVER_TARGET", over.pop("allow_min_lot_over_target", True)),
        max_daily_loss_frac=_f(prefix, "MAX_DAILY_LOSS_FRAC", over.pop("max_daily_loss_frac", 0.05)),
        max_open_positions=_i(prefix, "MAX_OPEN_POSITIONS", over.pop("max_open_positions", 1)),
        max_total_positions=_i(prefix, "MAX_TOTAL_POSITIONS", over.pop("max_total_positions", 2)),
        min_free_margin_frac=_f(prefix, "MIN_FREE_MARGIN_FRAC", over.pop("min_free_margin_frac", 0.30)),
        cooldown_seconds=_i(prefix, "COOLDOWN_SECONDS", over.pop("cooldown_seconds", 900)),
        max_candle_gap_multiplier=_f(prefix, "MAX_CANDLE_GAP_MULTIPLIER", over.pop("max_candle_gap_multiplier", 3.0)),
        max_bar_jump_atr=_f(prefix, "MAX_BAR_JUMP_ATR", over.pop("max_bar_jump_atr", 12.0)),
        stale_seconds=_i(prefix, "STALE_SECONDS", over.pop("stale_seconds", 900)),
        min_history_bars=_i(prefix, "MIN_HISTORY_BARS", over.pop("min_history_bars", 400)),
        trading_hours_utc=hours,
        **over,
    )
    return cfg


def btc_config() -> AssetConfig:
    # Valetax BTCUSD.vx defaults; ATR is large in $ terms, stops_level is 2976pt = $29.76.
    return _load(
        "BTC", "BTC", "BTCUSD.vx",
        atr_period=14, rsi_period=14, ema_fast=20, ema_slow=50, ema_trend=200,
        sl_atr_multiplier=2.0, tp_atr_multiplier=3.0, min_reward_risk=1.3,
        max_spread_atr_ratio=0.35, risk_per_trade=0.01, max_open_positions=1,
        cooldown_seconds=1800, stale_seconds=1200, min_history_bars=400,
        point=0.01, contract_size=1.0, tick_size=0.01, tick_value=0.01,
        volume_min=0.01, volume_step=0.01, volume_max=50.0,
        broker_stops_level_points=2976.0,
    )


def xau_config() -> AssetConfig:
    return _load(
        "XAUUSD", "XAU", os.getenv("MT5_GOLD_SYMBOL", "XAUUSD.vx"),
        atr_period=14, rsi_period=14, sl_atr_multiplier=1.5, tp_atr_multiplier=2.0,
        max_spread_atr_ratio=0.30, risk_per_trade=0.005, max_open_positions=1,
        point=0.01, contract_size=100.0, tick_size=0.01, tick_value=1.0,
        volume_min=0.01, volume_step=0.01, volume_max=50.0,
        broker_stops_level_points=0.0,
    )


def _apply_timeframe_profile(cfg: AssetConfig, timeframe: str) -> AssetConfig:
    """Timeframe-specific overrides.  Strategy parameters are NOT changed here --
    only the timeframe, the strategy *selection*, and cadence-related gates."""
    tf = timeframe.strip().upper()
    cfg.timeframe = tf
    # momentum_rsi_mtf is validated cross-regime on H1 for BOTH BTCUSD.vx and
    # XAUUSD.vx (backtest/btc_h1_validation.py, backtest/xau_h1_validation.py).
    if cfg.name in ("BTC", "XAUUSD") and tf in ("H1", "H4"):
        pfx = "BTC" if cfg.name == "BTC" else "XAU"
        cfg.strategy = _s(pfx, "H1_STRATEGY", "momentum_rsi_mtf")
        step = 3600 if tf == "H1" else 14400
        cfg.stale_seconds = _i(pfx, "H1_STALE_SECONDS", step * 3)     # a bar can be up to ~3 steps old
        cfg.cooldown_seconds = _i(pfx, "H1_COOLDOWN_SECONDS", step)   # >= one bar between entries
        cfg.min_history_bars = _i(pfx, "H1_MIN_HISTORY_BARS", 500)    # EMA96 + momentum warmup
        # the validated momentum_rsi_mtf H1 stop/target/sizing setup
        # (backtest/btc_lab.py RiskCfg defaults -- how the edge was measured).
        # max_risk_per_trade (the hard ceiling) is deliberately left untouched.
        cfg.sl_atr_multiplier = _f(pfx, "H1_SL_ATR_MULT", 2.0)
        cfg.tp_atr_multiplier = _f(pfx, "H1_TP_ATR_MULT", 3.0)
        cfg.min_reward_risk = _f(pfx, "H1_MIN_REWARD_RISK", 1.3)
        cfg.risk_per_trade = _f(pfx, "H1_RISK_PER_TRADE", 0.01)
    return cfg


def get_asset_config(name: str, *, timeframe: str | None = None) -> AssetConfig:
    n = name.strip().upper()
    if n in ("BTC", "BTCUSD", "BTCUSD.VX"):
        cfg = btc_config()
    elif n in ("XAU", "XAUUSD", "GOLD", "XAUUSD.VX"):
        cfg = xau_config()
    else:
        raise KeyError(f"unknown asset '{name}'")
    if timeframe:
        cfg = _apply_timeframe_profile(cfg, timeframe)
    return cfg


# ---- legacy compatibility -------------------------------------------------
ASSETS = {
    "XAUUSD": {"type": "gold", "timeframe": xau_config().timeframe, "risk_multiplier": 1.0},
    "BTCUSD": {"type": "bitcoin", "timeframe": btc_config().timeframe, "risk_multiplier": 0.75},
}
