"""
Public-feed H1 fetcher (Oracle-Cloud, MT5-free).  Network is mocked -- these
tests exercise the assembly logic only: level-matching, forward split, synthetic
spread column, sidecar metadata.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _fake_public(dense_tail_close: float, cutoff: pd.Timestamp, n_fwd: int, basis: float):
    """A public series: 6 overlap bars ending at `cutoff` + n_fwd bars after,
    offset from the broker by a constant `basis`.  The bar *at* the cutoff has
    close == dense_tail_close + basis exactly (ramp is measured from the cutoff),
    so the level-match offset must come out to exactly -basis."""
    t = pd.date_range(cutoff - pd.Timedelta(hours=6), periods=7 + n_fwd, freq="h", tz="UTC")
    hrs_from_cut = ((t - cutoff) / pd.Timedelta(hours=1)).to_numpy(float)
    close = dense_tail_close + basis + 0.5 * hrs_from_cut
    return pd.DataFrame({"time": t, "open": close - 1, "high": close + 2,
                         "low": close - 2, "close": close, "tick_volume": 100})


def test_import_chain_is_mt5_free():
    src = (ROOT / "tools" / "fetch_h1_public.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        mod = node.module if isinstance(node, ast.ImportFrom) else (
            node.names[0].name if isinstance(node, ast.Import) else None)
        if mod:
            assert "MetaTrader5" not in mod and "mt5" not in mod.split(".")


@pytest.mark.parametrize("asset,basis", [("BTC", 120.0), ("XAU", -7.0)])
def test_build_export_level_matches_and_splits_forward(asset, basis, tmp_path, monkeypatch):
    import tools.fetch_h1_public as F

    dense_p = ROOT / F._ASSETS[asset]["dense"]
    if not dense_p.exists():
        pytest.skip(f"{F._ASSETS[asset]['dense']} not present")
    dense = pd.read_csv(dense_p, usecols=lambda c: c in ("time", "open", "high", "low", "close"))
    dense["time"] = pd.to_datetime(dense["time"], utc=True)
    cutoff = dense["time"].max()
    last_close = float(dense.sort_values("time")["close"].iloc[-1])

    monkeypatch.setattr(F, "OUT", tmp_path)
    monkeypatch.setattr(F, "_fetch", lambda src, start_ms: _fake_public(last_close, cutoff, 40, basis))

    meta = F.build_export(asset)
    assert meta["forward_bars"] == 40
    assert meta["spread_synthetic"] is True
    assert abs(meta["level_offset_applied"] + basis) < 1e-6      # offset cancels the venue basis

    out = pd.read_csv(tmp_path / f"{F._ASSETS[asset]['slug']}_H1_export.csv")
    out["time"] = pd.to_datetime(out["time"], utc=True)
    assert (out["spread"] == F._ASSETS[asset]["spread_points"]).all()
    # seam continuity: first forward close within a normal bar move of the last real close
    fwd = out[out["time"] > cutoff].sort_values("time")
    assert len(fwd) == 40
    assert abs(float(fwd["close"].iloc[0]) - last_close) < 0.05 * last_close
    # in-sample prefix untouched (still the real Valetax bar count)
    assert (out["time"] <= cutoff).sum() == len(dense)

    sidecar = json.loads((tmp_path / f"{F._ASSETS[asset]['slug']}_H1_export.json").read_text())
    assert sidecar["asset"] == asset and "PUBLIC FEED" in sidecar["source"]


def test_replay_surfaces_synthetic_spread_note(tmp_path, monkeypatch):
    """A public-fed export must make the replay status say the spread is synthetic."""
    import tools.fetch_h1_public as F
    import execution.paper_replay as R

    if not (ROOT / F._ASSETS["XAU"]["dense"]).exists():
        pytest.skip("xau dense set not present")
    dense = pd.read_csv(ROOT / F._ASSETS["XAU"]["dense"],
                        usecols=lambda c: c in ("time", "close"))
    dense["time"] = pd.to_datetime(dense["time"], utc=True)
    cutoff = dense["time"].max()
    last_close = float(dense.sort_values("time")["close"].iloc[-1])

    monkeypatch.setattr(F, "OUT", tmp_path)
    monkeypatch.setattr(F, "_fetch", lambda s, m: _fake_public(last_close, cutoff, 60, -7.0))
    F.build_export("XAU")

    xau = R._AssetSpec(**{**R._SPECS["XAU"].__dict__,
                          "db": str(tmp_path / "x.sqlite"), "status": str(tmp_path / "x.json"),
                          "trades_csv": str(tmp_path / "x.csv"), "dense_csv": "nope.csv",
                          "fallback_cutoff": "2000-01-01T00:00:00Z"})
    monkeypatch.setattr(R, "_SPECS", {**R._SPECS, "XAU": xau})
    st = R.run_from_csv(str(tmp_path / "xauusd_vx_H1_export.csv"), asset="XAU", sim_balance=1500.0)
    assert "public feed has no spread" in st["spread_source"].lower()
    assert "broker floor" in st["spread_source"].lower()
    assert "PUBLIC FEED" in st["data_source"]
    assert st["live_trading_enabled"] is False
