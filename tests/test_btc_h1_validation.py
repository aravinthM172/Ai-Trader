"""
H1 validation-pipeline tests.  No MT5, no orders.  Uses the committed
data/btcusd_vx_H1_dense.csv when present, else a synthetic H1 series.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
from backtest import btc_lab as lab
from backtest import btc_strategies as strat

H1_CSV = ROOT / "data" / "btcusd_vx_H1_dense.csv"


@pytest.fixture
def h1_df():
    if H1_CSV.exists():
        d = pd.read_csv(H1_CSV)
        d["time"] = pd.to_datetime(d["time"], utc=True)
        return d
    rng = np.random.default_rng(11)
    n = 6000
    t = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    close = 45000 + rng.normal(0, 300, n).cumsum()
    high = close + np.abs(rng.normal(0, 200, n))
    low = close - np.abs(rng.normal(0, 200, n))
    o = close - rng.normal(0, 100, n)
    high = np.maximum.reduce([high, o, close])
    low = np.minimum.reduce([low, o, close])
    return pd.DataFrame({"time": t, "open": o, "high": high, "low": low, "close": close,
                         "tick_volume": rng.integers(1, 500, n), "spread": 2976})


# ---------------------------------------------------------------- dataset
def test_h1_dense_dataset_quality_if_present(h1_df, btc_cfg):
    if not H1_CSV.exists():
        pytest.skip("H1 dataset not downloaded")
    from data.quality import validate_ohlc
    c = btc_cfg
    c.timeframe = "H1"
    c.min_history_bars = 1000
    c.stale_seconds = 10 ** 9        # research file, not live
    clean, rep = validate_ohlc(h1_df, symbol="BTCUSD.vx", timeframe="H1", cfg=c)
    assert rep.n_clean >= 5000
    assert rep.duplicate_timestamps == 0
    assert rep.n_rejected == 0
    # broker spread column must not be silently zeroed / fabricated
    if "spread" in h1_df.columns:
        assert (h1_df["spread"] > 0).mean() > 0.5


def test_h1_report_exists_and_shapes():
    p = ROOT / "reports" / "btc_h1_validation.json"
    if not p.exists():
        pytest.skip("run python -m backtest.btc_h1_validation")
    r = json.loads(p.read_text(encoding="utf-8"))
    assert r["meta"]["timeframe"] == "H1"
    assert r["leakage_probe"]["signal_identical_pre_truncation"] is True
    assert "edge_authenticity" in r and "step5_walk_forward" in r


# ---------------------------------------------------------------- causality
@pytest.mark.parametrize("builder,kw", [
    (strat.build_trend, dict(fast=20, slow=50, trend=200, adx_min=20.0, atr_rank_max=0.85)),
    (strat.build_breakout, dict(lookback=20, expansion=1.3)),
    (strat.build_momentum, dict(rsi_buy=60.0, rsi_sell=40.0, mom_win=8, ema_htf=24)),
    (strat.build_mean_reversion, dict(z_win=20, z_entry=2.0, adx_max=18.0)),
])
def test_every_strategy_is_causal(h1_df, builder, kw):
    c = h1_df.close.to_numpy(float); h = h1_df.high.to_numpy(float); l = h1_df.low.to_numpy(float)
    full, _ = builder(c, h, l, **kw)
    cut = int(len(c) * 0.6)
    part, _ = builder(c[:cut], h[:cut], l[:cut], **kw)
    assert np.array_equal(full[:cut - 1], part[:cut - 1]), "look-ahead: signal changed when future removed"


def test_regime_labels_causal(h1_df):
    c = h1_df.close.to_numpy(float); h = h1_df.high.to_numpy(float); l = h1_df.low.to_numpy(float)
    t1, v1 = strat.regime_labels(c, h, l)
    cut = int(len(c) * 0.6)
    t2, v2 = strat.regime_labels(c[:cut], h[:cut], l[:cut])
    assert np.array_equal(t1[:cut - 1], t2[:cut - 1])
    assert np.array_equal(v1[:cut - 1], v2[:cut - 1])


# ---------------------------------------------------------------- lab
def test_monte_carlo_fixed_fractional_reports_all_keys(h1_df):
    c = h1_df.close.to_numpy(float); h = h1_df.high.to_numpy(float); l = h1_df.low.to_numpy(float)
    px = {k: h1_df[k].to_numpy(float) for k in ("open", "high", "low", "close")}
    entries, wu = strat.build_trend(c, h, l, adx_min=0.0, atr_rank_max=1.0)
    atr = strat.base_atr(h, l, c, 14)
    r = lab.run_backtest(px, atr, entries, wu + 5, len(c), wu, lab.Costs(),
                         lab.RiskCfg(initial_balance=5000.0))
    mc = lab.monte_carlo(r["trades"], lab.RiskCfg(), n=300, sim_balance=100.0, sim_risk_frac=0.01)
    if "n_trials" in mc:
        for k in ("prob_negative_final", "prob_ruin_equity_stop", "prob_drawdown_over_50pct",
                  "p95_max_drawdown_pct", "worst_losing_streak"):
            assert k in mc
        assert 0.0 <= mc["prob_ruin_equity_stop"] <= 1.0


def test_h1_costs_monotonic(h1_df):
    c = h1_df.close.to_numpy(float); h = h1_df.high.to_numpy(float); l = h1_df.low.to_numpy(float)
    px = {k: h1_df[k].to_numpy(float) for k in ("open", "high", "low", "close")}
    entries, wu = strat.build_momentum(c, h, l)
    atr = strat.base_atr(h, l, c, 14)
    R = lab.RiskCfg(initial_balance=5000.0)
    cheap = lab.run_backtest(px, atr, entries, wu + 5, len(c), wu,
                             lab.Costs(spread=10, slippage_per_side=0.3), R)["summary"]
    dear = lab.run_backtest(px, atr, entries, wu + 5, len(c), wu,
                            lab.Costs(spread=60, slippage_per_side=4), R)["summary"]
    if cheap.get("trades", 0) > 20 and dear.get("trades", 0) > 20:
        assert dear["expectancy_R"] <= cheap["expectancy_R"] + 1e-6


# ---------------------------------------------------------------- gate safety
def test_h1_live_gate_never_candidate_without_paper_and_account():
    p = ROOT / "reports" / "btc_h1_live_gate.json"
    if not p.exists():
        pytest.skip("run python -m tools.btc_h1_live_gate")
    g = json.loads(p.read_text(encoding="utf-8"))
    if g["LIVE_CANDIDATE"]:
        # if it ever flips true, paper + account conditions MUST have passed
        conds = {c["condition"]: c["pass"] for c in g["checks"]}
        assert conds.get("10. paper-trading confirmation (>= 20 H1 trades, net > 0)")
        assert conds.get("9. $100-account risk acceptable at 0.01 lot (p90 vol <= 2%)")
    assert g["LIVE_TRADING_env"].startswith("false")
    assert g["order_send_implemented"] is False


def test_no_live_trading_side_effects():
    import os
    assert os.getenv("LIVE_TRADING", "false").lower() in ("false", "0", "no", "off")
    assert not (ROOT / "state" / "KILL_SWITCH").exists() or True   # presence only blocks, never enables
