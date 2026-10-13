import numpy as np
import pandas as pd

from backtest.btc_strategies import base_atr
from strategy import calm_filter as cf


def _bars(ranges, price=100.0):
    """Flat price with the given bar ranges (high - low), so ATR is close to the recent range."""
    r = np.asarray(ranges, float)
    t = pd.date_range("2026-01-01", periods=len(r), freq="h", tz="UTC")
    return pd.DataFrame({"time": t, "open": price, "high": price + r / 2, "low": price - r / 2, "close": price})


def test_steady_market_has_ratio_one_and_is_allowed():
    df = _bars(np.full(800, 1.0))
    assert abs(cf.ratio(df) - 1.0) < 1e-6
    ok, why = cf.allowed(df)
    assert ok is True and "1.00" in why


def test_a_burst_of_big_bars_blocks_and_a_quiet_spell_allows():
    jumpy = _bars(np.r_[np.full(780, 1.0), np.full(20, 3.0)])
    ok, why = cf.allowed(jumpy)
    assert ok is False and cf.ratio(jumpy) > 2 and "limit 1.2" in why
    quiet = _bars(np.r_[np.full(780, 1.0), np.full(20, 0.4)])
    assert cf.allowed(quiet)[0] is True and cf.ratio(quiet) < 0.6


def test_limit_is_exclusive_and_adjustable():
    df = _bars(np.r_[np.full(780, 1.0), np.full(20, 3.0)])
    r = cf.ratio(df)
    assert cf.allowed(df, max_ratio=r + 0.01)[0] is True
    assert cf.allowed(df, max_ratio=r)[0] is False               # at the limit = jumpy


def test_too_little_history_gives_no_decision():
    assert cf.ratio(_bars(np.full(150, 1.0))) is None
    assert cf.allowed(_bars(np.full(150, 1.0))) == (None, "not enough H1 bars")
    assert cf.allowed(None)[0] is None


def test_matches_the_backtest_definition():
    """Same number as the study: ATR14 / close on the signal bar over the rolling 720-bar median of that series."""
    rng = np.random.default_rng(3)
    c = 100 + np.cumsum(rng.normal(0, 0.5, 2000))
    spread = np.abs(rng.normal(0.6, 0.3, 2000)) * np.r_[np.ones(1500), np.full(500, 1.8)]
    t = pd.date_range("2025-01-01", periods=2000, freq="h", tz="UTC")
    df = pd.DataFrame({"time": t, "open": c, "high": c + spread, "low": c - spread, "close": c})
    rel = pd.Series(base_atr(df.high.to_numpy(), df.low.to_numpy(), df.close.to_numpy(), 14) / df.close.to_numpy())
    study = (rel / rel.rolling(720, min_periods=200).median()).to_numpy()
    for j in (900, 1400, 1600, 1999):
        live = cf.ratio(df.iloc[max(0, j + 1 - cf.H1_BARS):j + 1].reset_index(drop=True))   # what the trader fetches
        assert abs(live - study[j]) / study[j] < 0.02
