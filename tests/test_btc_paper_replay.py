"""
MT5-free CSV replay tests.  No MT5, no orders, importable on Linux/ARM.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def export_csv(tmp_path):
    """A small synthetic H1 export in the tools/export_h1.py format."""
    src = ROOT / "data" / "btcusd_vx_H1_dense.csv"
    if src.exists():
        d = pd.read_csv(src).tail(4000).reset_index(drop=True)
    else:
        n = 4000
        rng = np.random.default_rng(5)
        close = 60000 + rng.normal(0, 300, n).cumsum()
        d = pd.DataFrame({
            "time": pd.date_range("2024-06-01", periods=n, freq="h", tz="UTC"),
            "open": close, "high": close + np.abs(rng.normal(0, 200, n)),
            "low": close - np.abs(rng.normal(0, 200, n)), "close": close,
            "tick_volume": rng.integers(1, 400, n), "spread": 2976,
        })
    p = tmp_path / "btcusd_vx_H1_export.csv"
    d.to_csv(p, index=False)
    (tmp_path / "btcusd_vx_H1_export.json").write_text(json.dumps({
        "export_utc": "2026-08-30T16:00:00+00:00", "symbol": "BTCUSD.vx", "bars": len(d),
        "spec": {"stops_level_price": 29.76, "volume_min": 0.01, "volume_step": 0.01,
                 "value_per_unit_per_lot": 1.0}}), encoding="utf-8")
    return p


def test_replay_does_not_import_metatrader5():
    import ast
    import execution.paper_replay as R
    assert "MetaTrader5" not in R.__dict__
    tree = ast.parse(Path(R.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all("MetaTrader5" not in n.name for n in node.names)
        if isinstance(node, ast.ImportFrom):
            assert "MetaTrader5" not in (node.module or "")
            assert "mt5.gateway" not in (node.module or "")   # gateway is the only MT5-touching module


def test_replay_import_chain_is_mt5_free():
    """Every module the replay imports must itself be MT5-free (so it runs on Linux/ARM)."""
    import ast
    seen: set[str] = set()
    to_check = ["execution/paper_replay.py"]
    while to_check:
        rel = to_check.pop()
        p = ROOT / rel
        if rel in seen or not p.exists():
            continue
        seen.add(rel)
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mod = None
            if isinstance(node, ast.ImportFrom):
                mod = node.module
            elif isinstance(node, ast.Import):
                mod = node.names[0].name
            if not mod:
                continue
            assert "MetaTrader5" not in mod, f"{rel} imports MetaTrader5"
            top = mod.split(".")[0]
            if top in ("common", "backtest", "strategy", "risk", "config", "execution", "data"):
                cand = mod.replace(".", "/") + ".py"
                if (ROOT / cand).exists():
                    to_check.append(cand)
                elif (ROOT / mod.replace(".", "/") / "__init__.py").exists():
                    to_check.append(mod.replace(".", "/") + "/__init__.py")
    # numba is optional everywhere (guarded try/except) so its presence is fine


def test_replay_frozen_params_and_strategy():
    from execution.paper_replay import PARAMS, STRATEGY_NAME
    assert STRATEGY_NAME == "momentum_rsi_mtf"
    assert PARAMS == {"rsi_buy": 60.0, "rsi_sell": 40.0, "mom_win": 8, "ema_htf": 24}


def test_replay_runs_and_records_all_fields(export_csv, monkeypatch, tmp_path):
    import execution.paper_replay as R
    monkeypatch.setattr(R, "_DB", tmp_path / "replay.sqlite")
    monkeypatch.setattr(R, "_STATUS", tmp_path / "status.json")
    monkeypatch.setattr(R, "_TRADES_CSV", tmp_path / "trades.csv")
    monkeypatch.setattr(R, "_validation_cutoff", lambda: pd.Timestamp("2024-06-01T00:00:00Z"))
    st = R.run_from_csv(str(export_csv), sim_balance=1500.0)
    assert "error" not in st
    assert st["timeframe"] == "H1" and st["strategy"] == "momentum_rsi_mtf"
    assert st["forward_out_of_sample"]["trades"] >= 1     # cutoff at start -> everything is forward
    assert "NOT fabricated" in st["spread_source"]
    assert st["live_trading_enabled"] is False

    c = sqlite3.connect(tmp_path / "replay.sqlite")
    cols = [d[0] for d in c.execute("SELECT * FROM trades").description]
    for f in ("timestamp_utc", "timeframe", "strategy", "sample", "direction", "signal_bar_utc",
              "entry", "exit", "spread", "atr", "stop", "target", "volume", "est_risk_usd",
              "pnl_usd", "r_multiple", "bars_held", "reason_entry", "reason_exit"):
        assert f in cols, f
    row = c.execute("SELECT * FROM trades LIMIT 1").fetchone()
    rec = dict(zip(cols, row))
    assert rec["strategy"] == "momentum_rsi_mtf"
    assert rec["spread"] >= 29.76 - 1e-6           # never below the broker minimum
    assert rec["signal_bar_utc"] < rec["timestamp_utc"]   # signal precedes exit
    c.close()


def test_replay_matches_validated_backtest_in_sample():
    """On the real dense H1 set the replay in-sample PF/expR must be close to
    the validated backtest (PF ~1.41, expR ~0.22)."""
    src = ROOT / "data" / "btcusd_vx_H1_dense.csv"
    if not src.exists():
        pytest.skip("dense H1 dataset not present")
    import execution.paper_replay as R
    d = pd.read_csv(src)
    d["time"] = pd.to_datetime(d["time"], utc=True)
    trades = R._simulate(d, 5000.0, pd.Timestamp("2100-01-01T00:00:00Z"))  # all in-sample
    assert len(trades) > 800
    pnl = np.array([t["pnl_usd"] for t in trades], float)
    Rm = np.array([t["r_multiple"] for t in trades if t["r_multiple"] is not None], float)
    pf = pnl[pnl > 0].sum() / -pnl[pnl <= 0].sum()
    assert 1.25 < pf < 1.60, f"replay PF {pf:.3f} diverged from validated ~1.41"
    assert 0.12 < Rm.mean() < 0.32, f"replay expR {Rm.mean():.3f} diverged from validated ~0.22"


def test_replay_respects_kill_switch(export_csv, monkeypatch):
    import execution.paper_replay as R
    kill = ROOT / "state" / "KILL_SWITCH"
    kill.write_text("x", encoding="utf-8")
    try:
        st = R.run_from_csv(str(export_csv))
        assert "KILL_SWITCH" in st.get("error", "")
    finally:
        kill.unlink(missing_ok=True)
