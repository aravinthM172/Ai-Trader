import os
import sys
from pathlib import Path

# Tests must NEVER trade, even when .env has LIVE_TRADING=true for production.
# load_dotenv() does not override variables that are already set.
os.environ["LIVE_TRADING"] = "false"

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _mt5_live() -> bool:
    try:
        import MetaTrader5 as mt5
    except Exception:
        return False
    try:
        if not mt5.initialize():
            return False
        ok = mt5.account_info() is not None
        return ok
    except Exception:
        return False


MT5_LIVE = _mt5_live()
requires_mt5 = pytest.mark.skipif(not MT5_LIVE, reason="MT5 terminal not connected")


class FakeSpec:
    symbol = "BTCUSD.vx"
    description = "Bitcoin vs US Dollar"
    digits = 2
    point = 0.01
    tick_size = 0.01
    tick_value = 0.01
    tick_value_profit = 0.01
    tick_value_loss = 0.01
    contract_size = 1.0
    volume_min = 0.01
    volume_max = 50.0
    volume_step = 0.01
    stops_level_points = 2976.0
    freeze_level_points = 0.0
    filling_mode = 1
    order_mode = 63
    trade_mode = 4
    trade_calc_mode = 2
    trade_exemode = 2
    spread_points = 2976.0
    spread_float = True
    currency_base = "USD"
    currency_profit = "USD"
    currency_margin = "USD"

    @property
    def stops_level_price(self):
        return self.stops_level_points * self.point

    @property
    def value_per_price_unit_per_lot(self):
        return self.tick_value / self.tick_size

    def to_dict(self):
        return {k: getattr(self, k) for k in dir(self) if not k.startswith("_") and not callable(getattr(self, k))}


@pytest.fixture
def fake_spec():
    return FakeSpec()


@pytest.fixture
def btc_cfg():
    from config.assets import btc_config
    c = btc_config()
    c.min_history_bars = 200
    return c


@pytest.fixture
def synth_ohlc():
    rng = np.random.default_rng(42)
    n = 700
    t = pd.date_range("2026-08-01", periods=n, freq="5min", tz="UTC")
    steps = rng.normal(0, 80, n).cumsum()
    close = 78000 + steps
    high = close + np.abs(rng.normal(0, 40, n))
    low = close - np.abs(rng.normal(0, 40, n))
    open_ = close - rng.normal(0, 20, n)
    high = np.maximum.reduce([high, open_, close])
    low = np.minimum.reduce([low, open_, close])
    return pd.DataFrame({
        "time": t, "open": open_, "high": high, "low": low, "close": close,
        "tick_volume": rng.integers(1, 200, n),
    })
