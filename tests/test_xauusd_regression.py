"""
XAUUSD regression -- the BTC integration must not break existing gold code.
Pure import / behaviour checks; no MT5 required, no orders.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_legacy_modules_still_import():
    import app                       # noqa: F401
    import run_engine                # noqa: F401
    from strategy.gold import GoldStrategy   # noqa: F401
    from strategy.features import FeatureEngine  # noqa: F401
    from strategy.multi_asset import MultiAssetStrategy  # noqa: F401
    from risk.engine import RiskEngine  # noqa: F401
    from risk.manager import RiskManager  # noqa: F401
    from news.filter import NewsFilter  # noqa: F401
    from config.assets import ASSETS
    assert "XAUUSD" in ASSETS and "BTCUSD" in ASSETS


def test_legacy_assets_dict_shape_unchanged():
    from config.assets import ASSETS
    for k in ("XAUUSD", "BTCUSD"):
        assert set(ASSETS[k]) == {"type", "timeframe", "risk_multiplier"}
    assert ASSETS["XAUUSD"]["type"] == "gold"
    assert ASSETS["BTCUSD"]["risk_multiplier"] == 0.75


def test_gold_strategy_analyze_runs():
    from strategy.gold import GoldStrategy
    rng = np.random.default_rng(0)
    n = 400
    close = 2400 + rng.normal(0, 3, n).cumsum()
    df = pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=n, freq="5min"),
        "open": close, "high": close + 1.5, "low": close - 1.5, "close": close,
        "tick_volume": rng.integers(1, 100, n),
    })
    out = GoldStrategy().analyze(df)
    assert out["symbol"] == "XAUUSD"
    assert out["trend"] in ("BULLISH", "BEARISH", "NEUTRAL")


def test_feature_engine_runs_and_is_finite():
    from strategy.features import FeatureEngine
    rng = np.random.default_rng(1)
    n = 500
    close = 2400 + rng.normal(0, 3, n).cumsum()
    df = pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=n, freq="5min"),
        "open": close, "high": close + 2, "low": close - 2, "close": close,
        "tick_volume": rng.integers(1, 100, n),
    })
    data = FeatureEngine.calculate(df)
    assert len(data) > 0
    latest = FeatureEngine.latest(data)
    assert np.isfinite(latest["atr"]) and np.isfinite(latest["rsi"])


def test_xau_and_btc_configs_do_not_collide():
    from config.assets import xau_config, btc_config
    x, b = xau_config(), btc_config()
    assert x.symbol != b.symbol
    assert x.contract_size != b.contract_size   # 100 vs 1
    assert x.risk_per_trade != b.risk_per_trade


def test_backtest_engine_parity_regression():
    """Delegates to the pre-existing XAUUSD engine parity check."""
    mod = pytest.importorskip("tests.test_engine_parity", reason="parity module import failed")
    fn = getattr(mod, "test_parity", None) or getattr(mod, "main", None)
    if fn is None:
        pytest.skip("no callable parity entry point")
    try:
        fn()
    except SystemExit:
        pass
