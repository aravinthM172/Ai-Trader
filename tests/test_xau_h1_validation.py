"""
XAUUSD.vx H1 validation-pipeline + MT5-free replay tests.  No MT5, no orders.
Uses the committed data/xauusd_vx_H1.csv when present, else a synthetic H1 series.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
from backtest import btc_lab as lab
from backtest import btc_strategies as strat

XAU_CSV = ROOT / "data" / "xauusd_vx_H1.csv"


@pytest.fixture
def xau_df():
    if XAU_CSV.exists():
        d = pd.read_csv(XAU_CSV)
        d["time"] = pd.to_datetime(d["time"], utc=True)
        return d[d["time"] >= pd.Timestamp("2015-01-01", tz="UTC")].reset_index(drop=True)
    rng = np.random.default_rng(13)
    n = 8000
    t = pd.date_range("2018-01-01", periods=n, freq="h", tz="UTC")
    close = 1800 + rng.normal(0, 3, n).cumsum()
    o = close - rng.normal(0, 1.5, n)
    high = np.maximum.reduce([close + np.abs(rng.normal(0, 2, n)), o, close])
    low = np.minimum.reduce([close - np.abs(rng.normal(0, 2, n)), o, close])
    return pd.DataFrame({"time": t, "open": o, "high": high, "low": low, "close": close,
                         "tick_volume": rng.integers(1, 500, n), "spread": 30})


# ---------------------------------------------------------------- contract
def test_xau_contract_matches_broker_facts():
    x = lab.XAU_CONTRACT
    assert x.value_per_unit_per_lot == 100.0        # $100 per $1 move per 1.0 lot
    assert x.symbol == "XAUUSD.vx"
    assert abs(x.broker_stops_level - 0.31) < 1e-9
    assert lab.BTC_CONTRACT.value_per_unit_per_lot == 1.0   # BTC default unchanged


# ---------------------------------------------------------------- causality
@pytest.mark.parametrize("builder,kw", [
    (strat.build_trend, dict(fast=20, slow=50, trend=200, adx_min=20.0, atr_rank_max=0.85)),
    (strat.build_breakout, dict(lookback=20, expansion=1.3, atr_rank_min=0.20)),
    (strat.build_momentum, dict(rsi_buy=60.0, rsi_sell=40.0, mom_win=8, ema_htf=24)),
    (strat.build_mean_reversion, dict(z_win=20, z_entry=2.0, adx_max=18.0)),
])
def test_every_xau_strategy_is_causal(xau_df, builder, kw):
    c = xau_df.close.to_numpy(float); h = xau_df.high.to_numpy(float); l = xau_df.low.to_numpy(float)
    full, _ = builder(c, h, l, **kw)
    cut = int(len(c) * 0.6)
    part, _ = builder(c[:cut], h[:cut], l[:cut], **kw)
    assert np.array_equal(full[:cut - 1], part[:cut - 1]), "look-ahead: signal changed when future removed"


# ---------------------------------------------------------------- report shape
def test_xau_report_exists_and_shapes():
    p = ROOT / "reports" / "xau_h1_validation.json"
    if not p.exists():
        pytest.skip("run python -m backtest.xau_h1_validation")
    r = json.loads(p.read_text(encoding="utf-8"))
    assert r["meta"]["timeframe"] == "H1"
    assert r["meta"]["symbol"] == "XAUUSD.vx"
    assert r["leakage_probe"]["signal_identical_pre_truncation"] is True
    assert "edge_authenticity" in r and "regime_breakdown" in r and "walk_forward" in r
    # the cross-regime selector must not pick a single-trend rider
    assert r["best_strategy"]["name"] == "momentum_rsi_mtf"
    # every macro regime in the breakdown must be positive if edge_found
    if r["verdict"]["edge_found"]:
        assert all(v["expectancy_R"] > 0 for v in r["regime_breakdown"].values())
        assert r["verdict"]["all_regimes_positive"] is True


def test_xau_markdown_written():
    p = ROOT / "XAU_H1_VALIDATION.md"
    if not (ROOT / "reports" / "xau_h1_validation.json").exists():
        pytest.skip("run python -m backtest.xau_h1_validation")
    assert p.exists()
    txt = p.read_text(encoding="utf-8")
    assert "XAUUSD.vx H1 Validation" in txt
    assert "LIVE_TRADING` stays **false**" in txt


# ---------------------------------------------------------------- gate safety
def test_xau_live_gate_never_candidate_without_paper_and_account():
    p = ROOT / "reports" / "xau_h1_live_gate.json"
    if not p.exists():
        pytest.skip("run python -m tools.xau_h1_live_gate")
    g = json.loads(p.read_text(encoding="utf-8"))
    if g["LIVE_CANDIDATE"]:
        conds = {c["condition"]: c["pass"] for c in g["checks"]}
        assert conds.get("10. paper-trading confirmation (>= 20 forward XAU trades, net > 0)")
        assert conds.get("9. $100-account risk acceptable at 0.01 lot (p90 vol <= 2%)")
    assert g["LIVE_TRADING_env"].startswith("false")
    assert g["order_send_implemented"] is False


# ---------------------------------------------------------------- MT5-free replay
@pytest.fixture
def xau_export_csv(tmp_path, xau_df):
    d = xau_df.tail(6000).reset_index(drop=True)
    p = tmp_path / "xauusd_vx_H1_export.csv"
    d.to_csv(p, index=False)
    (tmp_path / "xauusd_vx_H1_export.json").write_text(json.dumps({
        "export_utc": "2026-08-30T16:00:00+00:00", "asset": "XAU", "symbol": "XAUUSD.vx",
        "bars": len(d), "spec": {"stops_level_price": 0.31, "value_per_unit_per_lot": 100.0}}),
        encoding="utf-8")
    return p


def test_xau_replay_is_asset_aware_and_records_xau_physics(xau_export_csv, tmp_path, monkeypatch):
    import execution.paper_replay as R
    xau = R._AssetSpec(**{**R._SPECS["XAU"].__dict__,
                          "db": str(tmp_path / "xau.sqlite"),
                          "status": str(tmp_path / "xau_status.json"),
                          "trades_csv": str(tmp_path / "xau_trades.csv"),
                          "dense_csv": "does_not_exist.csv",
                          "fallback_cutoff": "2000-01-01T00:00:00Z"})   # -> everything forward
    monkeypatch.setattr(R, "_SPECS", {**R._SPECS, "XAU": xau})
    st = R.run_from_csv(str(xau_export_csv), asset="XAU", sim_balance=5000.0)
    assert "error" not in st
    assert st["asset"] == "XAU" and st["symbol"] == "XAUUSD.vx"
    assert st["timeframe"] == "H1" and st["strategy"] == "momentum_rsi_mtf"
    assert "$0.30" in st["spread_source"] and "NOT fabricated" in st["spread_source"]
    total = st["in_sample"]["trades"] + st["forward_out_of_sample"]["trades"]
    assert total >= 1

    c = sqlite3.connect(tmp_path / "xau.sqlite")
    rows = c.execute("SELECT spread FROM trades").fetchall()
    c.close()
    for (sp,) in rows:
        assert sp >= 0.30 - 1e-9          # XAU broker spread floor, not BTC's $29.76


def test_btc_replay_still_defaults_to_btc(tmp_path, monkeypatch):
    """The asset generalisation must not change the BTC default path."""
    import execution.paper_replay as R
    assert R._SPEC.asset == "BTC"
    assert R.BROKER_MIN_SPREAD == 29.76 and R.VALUE_PER_UNIT == 1.0
    spec = R._spec("BTC")
    assert spec.value_per_unit == 1.0 and spec.broker_stops_level == 29.76
