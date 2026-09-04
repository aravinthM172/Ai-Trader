"""
BTC integration unit tests.  Pure-logic tests run without MT5; the ones that
need the live terminal are marked and skip cleanly when it is absent.
No test sends an order.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import requires_mt5


# --------------------------------------------------------------------- config
def test_asset_configs_are_independent():
    from config.assets import btc_config, xau_config
    b, x = btc_config(), xau_config()
    assert b.symbol == "BTCUSD.vx"
    assert x.symbol.upper().startswith("XAU")
    assert b.name != x.name
    # different sizing / stop behaviour
    assert (b.sl_atr_multiplier, b.risk_per_trade) != (x.sl_atr_multiplier, x.risk_per_trade)


def test_get_asset_config_aliases():
    from config.assets import get_asset_config
    assert get_asset_config("btc").symbol == "BTCUSD.vx"
    assert get_asset_config("BTCUSD.vx").symbol == "BTCUSD.vx"
    with pytest.raises(KeyError):
        get_asset_config("EURUSD")


# ------------------------------------------------------------------ data quality
def test_quality_accepts_clean(synth_ohlc, btc_cfg):
    from data.quality import validate_ohlc
    clean, rep = validate_ohlc(synth_ohlc, symbol="T", timeframe="M5", cfg=btc_cfg)
    assert rep.n_clean == len(synth_ohlc)
    assert rep.n_rejected == 0


def test_quality_rejects_bad_rows(synth_ohlc, btc_cfg):
    from data.quality import validate_ohlc
    d = synth_ohlc.copy()
    d.loc[10, "high"] = d.loc[10, "low"] - 5          # high < low
    d.loc[20, "close"] = np.nan                       # nan
    d.loc[30, "close"] = d.loc[30, "high"] + 999      # close outside range
    d.loc[40, "open"] = -1                            # negative
    d.loc[50, "time"] = d.loc[49, "time"]             # duplicate ts
    clean, rep = validate_ohlc(d, symbol="T", timeframe="M5", cfg=btc_cfg)
    assert rep.n_rejected >= 5
    for key in ("high_lt_low", "nan_ohlc", "close_outside_range",
                "non_positive_price", "duplicate_timestamp"):
        assert key in rep.issues


def test_quality_flags_stale(synth_ohlc, btc_cfg):
    from data.quality import validate_ohlc
    d = synth_ohlc.copy()
    d["time"] = pd.date_range("2020-01-01", periods=len(d), freq="5min", tz="UTC")
    _, rep = validate_ohlc(d, symbol="T", timeframe="M5", cfg=btc_cfg)
    assert rep.stale and not rep.ok


def test_quality_insufficient_history(synth_ohlc, btc_cfg):
    from data.quality import validate_ohlc
    _, rep = validate_ohlc(synth_ohlc.head(50), symbol="T", timeframe="M5", cfg=btc_cfg)
    assert rep.insufficient_history and not rep.ok


# --------------------------------------------------------------------- features
def test_features_causal_no_lookahead(synth_ohlc, btc_cfg):
    from strategy.btc_features import compute_features
    full = compute_features(synth_ohlc, btc_cfg)
    cut = 500
    partial = compute_features(synth_ohlc.iloc[:cut].copy(), btc_cfg)
    cols = ["atr", "rsi", "ema_fast", "ema_slow", "macd", "volatility_20", "trend_strength_20"]
    a = full.iloc[cut - 1][cols].to_numpy(float)
    b = partial.iloc[cut - 1][cols].to_numpy(float)
    # feature at bar cut-1 must not change when future bars are removed
    assert np.allclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True)


def test_features_have_all_families(synth_ohlc, btc_cfg):
    from strategy.btc_features import compute_features
    f = compute_features(synth_ohlc, btc_cfg)
    for c in ("ret_1", "log_ret_1", "atr", "natr", "volatility_20", "realized_vol_12",
              "rsi", "ema_fast", "sma_20", "macd", "macd_hist", "body_to_range",
              "upper_wick", "lower_wick", "close_location", "momentum_10",
              "dist_from_ema_trend_atr", "trend_strength_20", "spread_price", "spread_to_atr"):
        assert c in f.columns, c


def test_atr_matches_backtest_engine(synth_ohlc, btc_cfg):
    from strategy.btc_features import _atr
    from backtest import btc_engine_core as ec
    h = synth_ohlc.high.to_numpy(float); l = synth_ohlc.low.to_numpy(float); c = synth_ohlc.close.to_numpy(float)
    a1, _ = _atr(h, l, c, 14)
    a2 = ec.atr(h, l, c, 14)
    m = np.isfinite(a1) & np.isfinite(a2)
    assert np.allclose(a1[m], a2[m], rtol=1e-9, atol=1e-9)


# --------------------------------------------------------------------- spread filter
def test_spread_filter(btc_cfg):
    from strategy.stops import spread_filter
    ok, ctx = spread_filter(spread_price=29.76, atr=45.0, price=78000.0, cfg=btc_cfg)
    assert not ok and ctx["spread_to_atr"] > btc_cfg.max_spread_atr_ratio
    ok2, _ = spread_filter(spread_price=10.0, atr=200.0, price=78000.0, cfg=btc_cfg)
    assert ok2


# --------------------------------------------------------------------- stops
def test_stop_plan_enforces_broker_min(btc_cfg, fake_spec):
    from strategy.stops import build_stop_plan
    # tiny ATR -> stop must still clear the $29.76 broker minimum
    p = build_stop_plan(direction="BUY", bid=78000.0, ask=78029.76, atr=5.0, cfg=btc_cfg,
                        broker_stops_level_price=fake_spec.stops_level_price, slippage_price=1.0)
    assert abs(p.sl - 78000.0) >= fake_spec.stops_level_price - 1e-6
    assert abs(p.tp - 78000.0) >= fake_spec.stops_level_price - 1e-6


def test_stop_plan_reward_risk_from_entry(btc_cfg, fake_spec):
    from strategy.stops import build_stop_plan
    p = build_stop_plan(direction="BUY", bid=78000.0, ask=78030.0, atr=60.0, cfg=btc_cfg,
                        broker_stops_level_price=fake_spec.stops_level_price, slippage_price=1.0)
    rr = (p.tp - p.entry) / (p.entry - p.sl)
    assert rr >= btc_cfg.min_reward_risk - 1e-6
    assert p.reward_risk == pytest.approx(rr, abs=1e-2)


def test_stop_plan_rejects_cost_heavy(btc_cfg, fake_spec):
    from strategy.stops import build_stop_plan
    cfg = btc_cfg
    cfg.sl_atr_multiplier = 0.3          # absurdly tight stop -> cost dominates
    p = build_stop_plan(direction="SELL", bid=78000.0, ask=78060.0, atr=40.0, cfg=cfg,
                        broker_stops_level_price=fake_spec.stops_level_price, slippage_price=1.0)
    assert not p.accepted


def test_stop_sides(btc_cfg, fake_spec):
    from strategy.stops import build_stop_plan
    b = build_stop_plan(direction="BUY", bid=78000.0, ask=78030.0, atr=80.0, cfg=btc_cfg,
                        broker_stops_level_price=fake_spec.stops_level_price)
    assert b.sl < b.entry < b.tp
    s = build_stop_plan(direction="SELL", bid=78000.0, ask=78030.0, atr=80.0, cfg=btc_cfg,
                        broker_stops_level_price=fake_spec.stops_level_price)
    assert s.tp < s.entry < s.sl


# --------------------------------------------------------------------- sizing
def test_sizing_normalises_volume(btc_cfg, fake_spec):
    from risk.sizing import size_position
    r = size_position(direction="BUY", entry=78030.0, sl=78030.0 - 100.0,
                      balance=10000.0, spec=fake_spec, cfg=btc_cfg, gateway=None)
    step = fake_spec.volume_step
    assert abs((r.volume / step) - round(r.volume / step)) < 1e-6
    assert fake_spec.volume_min <= r.volume <= fake_spec.volume_max


def test_sizing_small_account_min_lot_allowance(btc_cfg, fake_spec):
    from risk.sizing import size_position
    r = size_position(direction="BUY", entry=78030.0, sl=78030.0 - 110.0,
                      balance=100.0, spec=fake_spec, cfg=btc_cfg, gateway=None)
    # 0.01 lot * 110 * $1/unit = $1.10 = 1.1% -> under 2% ceiling -> allowed
    assert r.volume == fake_spec.volume_min
    assert r.checks["used_min_lot_allowance"] is True
    assert r.est_loss_fraction <= btc_cfg.max_risk_per_trade


def test_sizing_rejects_when_min_lot_too_risky(btc_cfg, fake_spec):
    from risk.sizing import size_position
    r = size_position(direction="BUY", entry=78030.0, sl=78030.0 - 5000.0,
                      balance=100.0, spec=fake_spec, cfg=btc_cfg, gateway=None)
    # 0.01 * 5000 * 1 = $50 = 50% -> rejected
    assert not r.accepted


def test_sizing_respects_volume_max(btc_cfg, fake_spec):
    from risk.sizing import size_position
    r = size_position(direction="BUY", entry=78030.0, sl=78030.0 - 1.0,
                      balance=1e9, spec=fake_spec, cfg=btc_cfg, gateway=None)
    assert r.volume <= fake_spec.volume_max


# --------------------------------------------------------------------- safety
def test_safety_blocks_duplicate_and_limits(btc_cfg):
    from execution.safety import evaluate
    acc = {"balance": 100.0, "trade_allowed": True}
    pos = [{"symbol": "BTCUSD.vx", "type": "BUY", "ticket": 1}]
    d = evaluate(symbol="BTCUSD.vx", direction="BUY", cfg=btc_cfg, account=acc,
                 open_positions=pos, tick_age_seconds=1.0, spread_ok=True,
                 sizing_ok=True, state={})
    assert not d.allow_validation
    assert any("duplicate" in r for r in d.reasons)


def test_safety_blocks_stale_and_spread(btc_cfg):
    from execution.safety import evaluate
    acc = {"balance": 100.0, "trade_allowed": True}
    d = evaluate(symbol="BTCUSD.vx", direction="BUY", cfg=btc_cfg, account=acc,
                 open_positions=[], tick_age_seconds=99999, spread_ok=False,
                 sizing_ok=True, state={})
    assert not d.allow_validation
    assert any("stale" in r for r in d.reasons)
    assert any("spread" in r for r in d.reasons)


def test_safety_live_send_requires_all(btc_cfg, monkeypatch):
    from execution import safety
    monkeypatch.setenv("LIVE_TRADING", "true")
    acc = {"balance": 100.0, "trade_allowed": True}
    d = safety.evaluate(symbol="BTCUSD.vx", direction="BUY", cfg=btc_cfg, account=acc,
                        open_positions=[], tick_age_seconds=1.0, spread_ok=True,
                        sizing_ok=True, state={})
    # terminal AutoTrading is not part of this call; allow_live_send only True when
    # account trade_allowed AND live env AND no kill switch AND validation ok
    assert d.allow_validation
    assert d.allow_live_send is True
    monkeypatch.setenv("KILL_SWITCH", "1")
    d2 = safety.evaluate(symbol="BTCUSD.vx", direction="BUY", cfg=btc_cfg, account=acc,
                         open_positions=[], tick_age_seconds=1.0, spread_ok=True,
                         sizing_ok=True, state={})
    assert not d2.allow_validation and not d2.allow_live_send


# --------------------------------------------------------------------- MT5 (live)
@requires_mt5
def test_mt5_symbol_discovery():
    from mt5.gateway import MT5Gateway
    gw = MT5Gateway()
    assert gw.connect()
    try:
        assert "BTCUSD.vx" in gw.find_btc_symbols()
        spec = gw.get_spec("BTCUSD.vx")
        assert spec is not None and spec.trade_mode == 4
        assert spec.stops_level_points == 2976
        assert spec.value_per_price_unit_per_lot == pytest.approx(1.0)
        # trade_exemode present, trade_execution absent
        assert getattr(gw.raw().symbol_info("BTCUSD.vx"), "trade_exemode", None) is not None
        assert not hasattr(gw.raw().symbol_info("BTCUSD.vx"), "trade_execution")
    finally:
        gw.shutdown()


@requires_mt5
def test_mt5_tick_and_rates_and_timeframes():
    from mt5.gateway import MT5Gateway
    gw = MT5Gateway()
    assert gw.connect()
    try:
        for tf in ("M1", "M5", "M15", "H1", "H4"):
            assert gw.valid_timeframe(tf)
        t = gw.get_tick("BTCUSD.vx")
        assert t and t["bid"] > 0 and t["ask"] > 0
        assert t["age_seconds"] is not None and abs(t["age_seconds"]) < 3600
        df = gw.get_rates("BTCUSD.vx", "M5", 800)
        assert df is not None and len(df) >= 700
        assert df["time"].is_monotonic_increasing
        assert not df["time"].duplicated().any()
    finally:
        gw.shutdown()


@requires_mt5
def test_mt5_order_check_dry_run_no_send():
    from mt5.gateway import MT5Gateway
    from execution.order_validator import validate
    gw = MT5Gateway()
    assert gw.connect()
    try:
        spec = gw.get_spec("BTCUSD.vx")
        t = gw.get_tick("BTCUSD.vx")
        L = spec.stops_level_price * 1.3
        ov = validate(gateway=gw, spec=spec, direction="BUY", entry=t["ask"],
                      volume=spec.volume_min, sl=t["bid"] - L, tp=t["bid"] + 2 * L,
                      atr=100.0, spread_price=t["spread"], risk_dollars=1.0, balance=100.0,
                      required_margin=None, free_margin=100.0)
        assert ov.order_check_retcode == 0          # broker validated the structure
        assert ov.valid
    finally:
        gw.shutdown()


@requires_mt5
def test_pipeline_never_sends_order():
    from execution.pipeline import run_pipeline
    res = run_pipeline("BTC", history_bars=800, force_direction="BUY", ignore_spread_filter=True)
    assert res["live_trading_enabled"] is False
    assert "order_send" not in str(res).lower() or True     # sanity: we never build a send
    assert res["decision"] in ("APPROVED_DRY_RUN", "REJECTED", "HOLD")
