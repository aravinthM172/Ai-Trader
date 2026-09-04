"""
BTC feature engineering -- strictly causal.

Every value at row i uses ONLY rows <= i.  No .shift(-k), no centered windows,
no full-sample statistics.  EMA/RSI/ATR are SMA-seeded Wilder implementations
matching backtest/btc_engine_core.py so live and backtest features agree.

Feature families: returns, log returns, ATR + normalised ATR, rolling
volatility, RSI, EMA/SMA, MACD, candle body / range / wicks / close-location,
momentum, distance from EMA, trend strength, spread, spread/ATR ratio.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ema(x: np.ndarray, period: int) -> np.ndarray:
    n = x.shape[0]
    out = np.full(n, np.nan)
    if n < period:
        return out
    prev = x[:period].mean()
    out[period - 1] = prev
    alpha = 2.0 / (period + 1.0)
    for i in range(period, n):
        prev = alpha * x[i] + (1.0 - alpha) * prev
        out[i] = prev
    return out


def _rsi(x: np.ndarray, period: int) -> np.ndarray:
    n = x.shape[0]
    out = np.full(n, np.nan)
    if n < period + 1:
        return out
    d = np.diff(x)
    gain = np.where(d > 0, d, 0.0)
    loss = np.where(d < 0, -d, 0.0)
    g = gain[:period].mean()
    l = loss[:period].mean()
    out[period] = 100.0 if l == 0 else 100.0 - 100.0 / (1.0 + g / l)
    for i in range(period + 1, n):
        g = (g * (period - 1) + gain[i - 1]) / period
        l = (l * (period - 1) + loss[i - 1]) / period
        out[i] = 100.0 if l == 0 else 100.0 - 100.0 / (1.0 + g / l)
    return out


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    n = close.shape[0]
    out = np.full(n, np.nan)
    if n < period + 1:
        return out
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    prev = tr[1:period + 1].mean()
    out[period] = prev
    for i in range(period + 1, n):
        prev = (prev * (period - 1) + tr[i]) / period
        out[i] = prev
    return out, tr


def compute_features(df: pd.DataFrame, cfg, *, spread_price: float | None = None) -> pd.DataFrame:
    """Return df + causal feature columns.  Rows with warmup NaNs are dropped."""
    d = df.copy().reset_index(drop=True)
    o = d["open"].to_numpy(float)
    h = d["high"].to_numpy(float)
    l = d["low"].to_numpy(float)
    c = d["close"].to_numpy(float)

    # returns
    d["ret_1"] = pd.Series(c).pct_change(1)
    d["ret_3"] = pd.Series(c).pct_change(3)
    d["ret_6"] = pd.Series(c).pct_change(6)
    d["ret_12"] = pd.Series(c).pct_change(12)
    d["log_ret_1"] = np.log(c / np.concatenate([[np.nan], c[:-1]]))

    # ATR + true range
    atr, tr = _atr(h, l, c, cfg.atr_period)
    d["true_range"] = tr
    d["atr"] = atr
    d["natr"] = np.where(c > 0, atr / c, np.nan)          # normalised ATR (fraction of price)
    d["atr_pct_change_12"] = pd.Series(atr).pct_change(12).to_numpy()

    # volatility
    d["volatility_20"] = d["ret_1"].rolling(20, min_periods=20).std()
    d["volatility_50"] = d["ret_1"].rolling(50, min_periods=50).std()
    d["realized_vol_12"] = d["log_ret_1"].rolling(12, min_periods=12).std()

    # RSI
    d["rsi"] = _rsi(c, cfg.rsi_period)
    d["rsi_fast"] = _rsi(c, max(2, cfg.rsi_period // 2))

    # EMA / SMA
    ef = _ema(c, cfg.ema_fast)
    es = _ema(c, cfg.ema_slow)
    et = _ema(c, cfg.ema_trend)
    d["ema_fast"] = ef
    d["ema_slow"] = es
    d["ema_trend"] = et
    d["sma_20"] = pd.Series(c).rolling(20, min_periods=20).mean().to_numpy()
    d["sma_50"] = pd.Series(c).rolling(50, min_periods=50).mean().to_numpy()

    # MACD (12/26/9), causal
    macd_line = _ema(c, 12) - _ema(c, 26)
    macd_sig = _ema(macd_line[~np.isnan(macd_line)], 9)
    sig_full = np.full_like(macd_line, np.nan)
    sig_full[np.where(~np.isnan(macd_line))[0][:len(macd_sig)]] = macd_sig
    d["macd"] = macd_line
    d["macd_signal"] = sig_full
    d["macd_hist"] = macd_line - sig_full

    # candle structure
    rng = np.where((h - l) > 0, h - l, np.nan)
    d["candle_range"] = h - l
    d["candle_body"] = np.abs(c - o)
    d["body_to_range"] = np.abs(c - o) / rng
    d["upper_wick"] = (h - np.maximum(o, c)) / rng
    d["lower_wick"] = (np.minimum(o, c) - l) / rng
    d["close_location"] = (c - l) / rng
    d["bull"] = np.sign(c - o)

    # momentum / distance from EMA / trend strength
    d["momentum_10"] = pd.Series(c).diff(10).to_numpy()
    d["dist_from_ema_fast_atr"] = (c - ef) / np.where(atr > 0, atr, np.nan)
    d["dist_from_ema_trend_atr"] = (c - et) / np.where(atr > 0, atr, np.nan)
    d["ema_fast_slow_sep_atr"] = (ef - es) / np.where(atr > 0, atr, np.nan)
    d["ema_slope_fast_6"] = (ef - np.concatenate([[np.nan] * 6, ef[:-6]])) / np.where(atr > 0, atr, np.nan)
    # trend strength: fraction of last 20 closes on the trending side of the fast EMA
    above = (pd.Series(c) > pd.Series(ef)).astype(float)
    d["trend_strength_20"] = above.rolling(20, min_periods=20).mean().to_numpy() * 2.0 - 1.0

    # spread context (live spread passed in; historical uses the 'spread' column if present)
    if spread_price is not None:
        sp = float(spread_price)
        d["spread_price"] = sp
    elif "spread" in d.columns:
        d["spread_price"] = d["spread"].to_numpy(float) * cfg.point
    else:
        d["spread_price"] = np.nan
    d["spread_to_atr"] = d["spread_price"] / np.where(atr > 0, atr, np.nan)
    d["spread_frac_price"] = d["spread_price"] / np.where(c > 0, c, np.nan)

    return d


def latest_feature_row(df_feat: pd.DataFrame) -> dict:
    """Most recent fully-formed feature row as a plain dict (NaN -> None)."""
    core = ["atr", "natr", "rsi", "ema_fast", "ema_slow", "ema_trend", "macd",
            "volatility_20", "trend_strength_20", "dist_from_ema_trend_atr"]
    valid = df_feat.dropna(subset=[c for c in core if c in df_feat.columns])
    if valid.empty:
        return {}
    row = valid.iloc[-1]
    out = {}
    for k, v in row.items():
        if isinstance(v, (int, float, np.integer, np.floating)):
            out[k] = None if (isinstance(v, float) and not np.isfinite(v)) else float(v)
        else:
            out[k] = str(v)
    out["n_valid_rows"] = int(len(valid))
    return out
