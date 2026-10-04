from datetime import datetime, timezone

import pandas as pd

from execution import live_multi as lm


def test_size_volume_risk_based_and_steps():
    # $10k, 0.5 % -> $50 risk; stop 0.0040 on EURUSD ($100k per 1.0 move per lot) -> $400/lot -> 0.125 -> 0.12
    vol, risk, note = lm.size_volume(equity=10_000, risk_frac=0.005, max_risk_frac=0.01, sl_dist=0.0040,
                                     value_per_unit=100_000, vmin=0.01, vstep=0.01, vmax=50)
    assert vol == 0.12 and abs(risk - 48.0) < 1e-6 and note == "ok"


def test_size_volume_min_lot_allowed_only_under_ceiling():
    # $500 equity, BTC stop $1,000 x $1/unit -> $1,000 per lot; 0.01 lot = $10 = 2 % > 1 % ceiling -> skip
    vol, _, note = lm.size_volume(equity=500, risk_frac=0.005, max_risk_frac=0.01, sl_dist=1000,
                                  value_per_unit=1.0, vmin=0.01, vstep=0.01, vmax=50)
    assert vol == 0 and "min lot" in note
    vol, _, _ = lm.size_volume(equity=1500, risk_frac=0.005, max_risk_frac=0.01, sl_dist=1000,
                               value_per_unit=1.0, vmin=0.01, vstep=0.01, vmax=50)
    assert vol == 0.01                                       # $10 = 0.67 % <= 1 %


def test_stop_plan_geometry_and_broker_minimum():
    sl, tp, sl_d = lm.stop_plan("BUY", 100.0, 1.0, stops_level=0.1)
    assert (sl, tp, sl_d) == (98.0, 103.0, 2.0)
    sl, tp, sl_d = lm.stop_plan("SELL", 100.0, 0.01, stops_level=1.0)       # broker minimum dominates
    assert sl_d == 1.15 and sl == 101.15 and abs(tp - 98.85) < 1e-9


def test_completed_bars_drops_forming_bar():
    df = pd.DataFrame({"time": pd.date_range("2026-10-05 08:00", periods=4, freq="h", tz="UTC"), "close": [1, 2, 3, 4]})
    out = lm.completed_bars(df, datetime(2026, 10, 5, 11, 5, tzinfo=timezone.utc))
    assert list(out["close"]) == [1, 2, 3]


def test_magic_is_stable_and_distinct():
    assert lm.magic_for("EURUSD.vx") == lm.magic_for("EURUSD.vx")
    assert lm.magic_for("EURUSD.vx") != lm.magic_for("GBPUSD.vx")
    assert lm.magic_for("EURUSD.vx") != 26092601                       # never collides with live.py's BTC magic


def test_send_requires_every_switch(monkeypatch, tmp_path):
    demo = {"trade_mode": 0}
    real = {"trade_mode": 2}
    monkeypatch.setattr(lm.safety, "kill_switch_active", lambda: False)
    monkeypatch.delenv("LIVE_ACCOUNT_MODE", raising=False)
    monkeypatch.setenv("MULTI_LIVE_TRADING", "false"); monkeypatch.setenv("LIVE_TRADING", "true")
    assert lm.live_send_allowed(demo)[0] is False
    monkeypatch.setenv("MULTI_LIVE_TRADING", "true"); monkeypatch.setenv("LIVE_TRADING", "false")
    assert lm.live_send_allowed(demo)[0] is False
    monkeypatch.setenv("LIVE_TRADING", "true")
    assert lm.live_send_allowed(demo)[0] is True
    assert lm.live_send_allowed(real)[0] is False                      # real needs LIVE_ACCOUNT_MODE=real
    monkeypatch.setenv("LIVE_ACCOUNT_MODE", "real")
    assert lm.live_send_allowed(real)[0] is True
    monkeypatch.setattr(lm.safety, "kill_switch_active", lambda: True)
    assert lm.live_send_allowed(real)[0] is False


def test_btc_excluded_from_default_symbols(monkeypatch, tmp_path):
    (tmp_path / "multi_symbol_scan.json").write_text('{"passers": ["BTCUSD.vx", "NAS100.vx"]}')
    monkeypatch.setattr(lm, "REPORTS", tmp_path)
    monkeypatch.delenv("MULTI_SYMBOLS", raising=False)
    assert lm.symbols() == ["NAS100.vx"]
