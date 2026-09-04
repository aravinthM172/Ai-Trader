"""
ML model interface -- placeholder.

STATE OF THE PROJECT:
    There is NO trained ML model in this repository (models/ is empty).  The
    "intelligence" in the legacy path is claude/analyzer.py (an LLM call), not a
    learned model.  All BTC research (backtest/btc_signal_discovery.py,
    btc_phase3_confirmation.py) found NO robust predictive edge in BTC M5 OHLC,
    so no model was trained.

WHEN A MODEL IS ADDED:
    * keep XAUUSD and BTC models SEPARATE -- a model trained on XAUUSD is NOT
      valid for BTC (different price scale, volatility, microstructure).
    * pin the exact feature list + order + scaler alongside the model file.
    * this module is the single inference entry point; the pipeline calls
      ``predict(asset, feature_row)`` and must get back a dict with at least
      ``{"decision": "BUY"|"SELL"|"HOLD", "confidence": float}``.
    * reject inference on NaN / wrong-shape / feature-name-mismatch input.
"""
from __future__ import annotations

from pathlib import Path

_MODELS = Path(__file__).resolve().parents[1] / "models"

# {asset: {"path": ..., "features": [...], "scaler": ..., "format": ...}}
REGISTRY: dict[str, dict] = {}


def model_available(asset: str) -> bool:
    return asset.upper() in REGISTRY and Path(REGISTRY[asset.upper()]["path"]).exists()


def expected_features(asset: str) -> list[str] | None:
    m = REGISTRY.get(asset.upper())
    return list(m["features"]) if m else None


def predict(asset: str, feature_row: dict) -> dict:
    """No model wired up yet -> always HOLD.  Never fabricates a signal."""
    if not model_available(asset):
        return {"decision": "HOLD", "confidence": 0.0,
                "reason": f"no ML model registered for {asset}; using rule-based signal instead"}
    # --- validation that WILL run once a model exists -------------------
    feats = expected_features(asset) or []
    missing = [f for f in feats if f not in feature_row]
    if missing:
        return {"decision": "HOLD", "confidence": 0.0,
                "reason": f"feature mismatch, missing {missing[:5]}"}
    if any(feature_row.get(f) is None for f in feats):
        return {"decision": "HOLD", "confidence": 0.0, "reason": "NaN in model input"}
    raise NotImplementedError("model inference not implemented; register a model in REGISTRY first")
