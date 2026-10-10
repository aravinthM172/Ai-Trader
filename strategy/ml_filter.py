"""
Signal-quality score for the live momentum rule (added 2026-10-10) -- PAPER ONLY, nothing trades on it.

A linear (ridge) model scores a signal from what is known when the signal bar closes; a signal is "taken" when
its score is above the model's threshold (the median score of the training signals).  Study and numbers:
backtest/ml_filter_study.py, ML_FILTER_STUDY.md.  The fitted model is the JSON file next to this module; the
same feature code is used by the study and by execution/paper_ml_filter.py, so both see identical inputs.

Features (direction-signed where that makes sense):
  rsi_s       dir x (RSI14 - 50)
  mom_atr     dir x 8-bar momentum / ATR14
  ema_dist    dir x (close - EMA96) / ATR14
  vol_rel     (ATR14 / close) relative to its own median over the last VOL_WIN closed H1 bars
  trend_dist  dir x (last completed daily close - its EMA200) / that close
  trend_ok    1 when the trade is on the daily-EMA200 side, else 0
  is_buy      1 for BUY, 0 for SELL
  hour_sin / hour_cos   entry hour (UTC) on a 24 h circle
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.btc_features import _atr, _ema, _rsi
from strategy.btc_h1_signal import PARAMS

MODEL_PATH = Path(__file__).with_name("ml_filter_model.json")
FEATS = ["rsi_s", "mom_atr", "ema_dist", "vol_rel", "trend_dist", "trend_ok", "is_buy", "hour_sin", "hour_cos"]
VOL_WIN, VOL_MIN = 2000, 500             # closed H1 bars for the volatility reference (about 4 months)
EMA_SIG = PARAMS["ema_htf"] * 4          # 96 -- the trend EMA of the live rule
H1_BARS = VOL_WIN + 300                  # bars a live caller must load before the signal bar
CLIP = {"rsi_s": (-50, 50), "mom_atr": (-15, 15), "ema_dist": (-15, 15), "vol_rel": (0.2, 5.0), "trend_dist": (-0.6, 0.6)}


def bar_frame(h1: pd.DataFrame) -> pd.DataFrame:
    """Per closed H1 bar: the indicators the features are built from (index = position in h1)."""
    h, l, c = (h1[k].to_numpy(float) for k in ("high", "low", "close"))
    atr = _atr(h, l, c, 14)
    atr = atr[0] if isinstance(atr, tuple) else atr
    vol = pd.Series(atr / c)
    return pd.DataFrame({"time": h1["time"].to_numpy(), "close": c, "rsi": _rsi(c, 14),
                         "mom": pd.Series(c).diff(PARAMS["mom_win"]).to_numpy(), "ema": _ema(c, EMA_SIG), "atr": atr,
                         "vol_rel": (vol / vol.rolling(VOL_WIN, min_periods=VOL_MIN).median()).to_numpy()})


def features(bar, direction: int, d_close: float, d_ema200: float, entry_hour_utc: int) -> dict | None:
    """Feature dict for one signal.  bar = one row of bar_frame (the signal bar).  None when something is missing."""
    d = 1.0 if direction > 0 else -1.0
    atr = float(bar["atr"])
    raw = {"rsi_s": d * (float(bar["rsi"]) - 50.0), "mom_atr": d * float(bar["mom"]) / atr if atr > 0 else np.nan,
           "ema_dist": d * (float(bar["close"]) - float(bar["ema"])) / atr if atr > 0 else np.nan,
           "vol_rel": float(bar["vol_rel"]), "trend_dist": d * (d_close - d_ema200) / d_close if d_close else np.nan,
           "trend_ok": 1.0 if d * (d_close - d_ema200) > 0 else 0.0, "is_buy": 1.0 if d > 0 else 0.0,
           "hour_sin": float(np.sin(2 * np.pi * entry_hour_utc / 24)), "hour_cos": float(np.cos(2 * np.pi * entry_hour_utc / 24))}
    if not all(np.isfinite(v) for v in raw.values()):
        return None
    return {k: float(np.clip(v, *CLIP[k])) if k in CLIP else v for k, v in raw.items()}


def clip_frame(x: pd.DataFrame) -> pd.DataFrame:
    """The same clipping as features(), for a whole table (used by the study)."""
    x = x.copy()
    for k, (lo, hi) in CLIP.items():
        x[k] = x[k].clip(lo, hi)
    return x


# -- model ---------------------------------------------------------------------------------------
def fit(x: np.ndarray, y: np.ndarray, alpha: float = 100.0) -> dict:
    """Ridge regression on standardised features, closed form.  Returns a JSON-ready model."""
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    z = (x - mean) / scale
    coef = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (y - y.mean()))
    model = {"features": FEATS, "mean": mean.tolist(), "scale": scale.tolist(), "coef": coef.tolist(),
             "intercept": float(y.mean()), "alpha": alpha}
    model["threshold"] = float(np.median(predict(model, x)))
    return model


def predict(model: dict, x: np.ndarray) -> np.ndarray:
    z = (np.asarray(x, float) - np.asarray(model["mean"])) / np.asarray(model["scale"])
    return z @ np.asarray(model["coef"]) + model["intercept"]


def load_model(path: Path = MODEL_PATH) -> dict | None:
    try:
        m = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return m if m.get("features") == FEATS else None


def score(model: dict, feats: dict) -> tuple[float, bool]:
    """(expected R, taken?) for one signal."""
    s = float(predict(model, np.array([[feats[k] for k in FEATS]]))[0])
    return s, s > model["threshold"]
