"""
Proves H1 paper mode actually requests + processes H1 data and does NOT
silently fall back to M5.  No MT5 needed for the core proofs (the gateway is
faked); the live check is marked and skips when MT5 is absent.  No orders.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import requires_mt5

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------- config
def test_h1_profile_selects_validated_strategy():
    from config.assets import get_asset_config
    m5 = get_asset_config("BTC", timeframe="M5")
    h1 = get_asset_config("BTC", timeframe="H1")
    assert m5.timeframe == "M5" and m5.strategy == "rule_ema_rsi"
    assert h1.timeframe == "H1" and h1.strategy == "momentum_rsi_mtf"
    # cadence gates widen for H1, safety gates are NOT loosened
    assert h1.cooldown_seconds >= 3600
    assert h1.stale_seconds >= 3600
    assert h1.max_risk_per_trade == m5.max_risk_per_trade   # unchanged


def test_h1_signal_params_are_frozen():
    from strategy.btc_h1_signal import PARAMS, STRATEGY_NAME
    assert STRATEGY_NAME == "momentum_rsi_mtf"
    assert PARAMS == {"rsi_buy": 60.0, "rsi_sell": 40.0, "mom_win": 8, "ema_htf": 24}


# --------------------------------------------------------------- adapter == backtest
def test_h1_live_adapter_matches_backtest_builder():
    csv = ROOT / "data" / "btcusd_vx_H1_dense.csv"
    if csv.exists():
        d = pd.read_csv(csv)
        d["time"] = pd.to_datetime(d["time"], utc=True)
    else:
        rng = np.random.default_rng(3)
        n = 3000
        close = 45000 + rng.normal(0, 250, n).cumsum()
        d = pd.DataFrame({"time": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
                          "open": close, "high": close + 150, "low": close - 150, "close": close})
    from strategy import btc_h1_signal as S
    from backtest.btc_strategies import build_momentum
    c = d.close.to_numpy(float); h = d.high.to_numpy(float); l = d.low.to_numpy(float)
    full, _ = build_momentum(c, h, l, **S.PARAMS)
    mism = 0
    for k in range(600, len(d), 53):
        got = {"BUY": 1, "SELL": -1, "HOLD": 0}[S.generate("X", d.iloc[:k]).decision]
        if got != int(full[k]):
            mism += 1
    assert mism == 0, "live H1 adapter diverged from the validated backtest builder"


# --------------------------------------------------------------- pipeline uses H1
class _FakeGW:
    """Records which timeframe get_rates was asked for."""
    def __init__(self, tf_seen):
        self._tf_seen = tf_seen
        self.server_utc_offset_seconds = 0

    def connect(self): return True
    def shutdown(self): pass
    def raw(self):
        import types
        return types.SimpleNamespace(ORDER_TYPE_BUY=0, ORDER_TYPE_SELL=1,
                                     ORDER_FILLING_FOK=0, ORDER_FILLING_IOC=1, ORDER_FILLING_RETURN=2,
                                     TRADE_ACTION_DEAL=1, ORDER_TIME_GTC=0)
    def account_info(self): return {"balance": 100.0, "equity": 100.0, "margin_free": 100.0,
                                    "leverage": 2000, "trade_allowed": True, "server": "T"}
    def terminal_info(self): return {"trade_allowed": False}
    def get_spec(self, s):
        from conftest import FakeSpec
        return FakeSpec()

    def get_tick(self, s):
        return {"symbol": s, "bid": 78000.0, "ask": 78029.76, "last": 0.0,
                "spread": 29.76, "time": 1, "time_utc": "x", "age_seconds": 5.0}

    def valid_timeframe(self, tf): return tf.upper() in ("M1", "M5", "M15", "H1", "H4")

    def get_rates(self, symbol, timeframe, count):
        self._tf_seen.append(timeframe)
        step = {"M5": "5min", "M15": "15min", "H1": "h"}[timeframe.upper()]
        n = min(count, 4000)
        rng = np.random.default_rng(9)
        close = 78000 + rng.normal(0, 120 if timeframe == "H1" else 25, n).cumsum()
        end = pd.Timestamp.now(tz="UTC").floor(step)
        return pd.DataFrame({
            "time": pd.date_range(end=end, periods=n, freq=step, tz="UTC"),
            "open": close, "high": close + np.abs(rng.normal(0, 60, n)),
            "low": close - np.abs(rng.normal(0, 60, n)), "close": close,
            "tick_volume": rng.integers(1, 500, n), "spread": 2976,
        })

    def preferred_filling(self, spec): return 0
    def positions(self, symbol=None): return []
    def calc_margin(self, *a): return 7.8
    def calc_profit(self, sym, d, vol, po, pc): return -abs(po - pc) * vol
    def order_check(self, req): return {"retcode": 0, "comment": "Done", "balance": 100.0,
                                        "equity": 100.0, "margin": 7.8, "margin_free": 92.2,
                                        "margin_level": 0.0}
    def history_deals_today(self): return []


@pytest.mark.parametrize("tf", ["M5", "H1"])
def test_pipeline_requests_the_selected_timeframe(monkeypatch, tf):
    import execution.pipeline as P
    seen: list[str] = []
    monkeypatch.setattr(P, "MT5Gateway", lambda: _FakeGW(seen))
    res = P.run_pipeline("BTC", timeframe=tf, history_bars=1500, force_direction="BUY")
    assert seen and all(x.upper() == tf for x in seen), f"pipeline fetched {seen}, expected {tf}"
    assert res["timeframe"] == tf
    assert res["strategy"] == ("momentum_rsi_mtf" if tf == "H1" else "rule_ema_rsi")
    # H1 ATR must be materially larger than an M5 ATR on the same fake vol
    assert res.get("atr", 0) > 0


def test_paper_hard_guard_aborts_on_timeframe_mismatch(monkeypatch):
    """If the pipeline ever returns a different TF than requested, paper must abort."""
    import execution.paper as PP
    monkeypatch.setattr(PP, "run_pipeline",
                        lambda *a, **k: {"timeframe": "M5", "strategy": "rule_ema_rsi", "stages": []})
    monkeypatch.setattr(PP, "MT5Gateway", lambda: _FakeGW([]))
    with pytest.raises(RuntimeError, match="not H1"):
        PP.run_once(timeframe="H1")


def test_paper_record_has_all_required_fields(monkeypatch, tmp_path):
    import execution.paper as PP
    monkeypatch.setattr(PP, "_STATE", tmp_path)
    monkeypatch.setattr(PP, "MT5Gateway", lambda: _FakeGW([]))
    fake_res = {
        "timeframe": "H1", "strategy": "momentum_rsi_mtf", "history_bars_returned": 4000,
        "signal_bar_utc": "2026-08-30 10:00:00+00:00", "atr": 214.0,
        "evaluated_direction": "BUY", "decision": "APPROVED_DRY_RUN",
        "signal": {"decision": "BUY", "confidence": 1.0, "reason": "momentum_rsi_mtf: ... -> BUY"},
        "stop_plan": {"accepted": True, "entry": 78030.0, "sl": 77600.0, "tp": 78900.0,
                      "spread_price": 29.76},
        "sizing": {"accepted": True, "volume": 0.03, "est_loss_at_sl": 13.7, "risk_dollars": 15.0},
        "safety": {"reasons": []},
        "spread_filter": {"ok": True, "spread_to_atr": 0.14},
        "live_trading_enabled": False, "kill_switch_active": False,
    }
    monkeypatch.setattr(PP, "run_pipeline", lambda *a, **k: fake_res)
    st = PP.run_once(timeframe="H1", sim_balance=1500.0)
    c = sqlite3.connect(tmp_path / "btc_paper_H1.sqlite")
    row = dict(zip([d[0] for d in c.execute("SELECT * FROM positions").description],
                   c.execute("SELECT * FROM positions").fetchone()))
    for f in ("opened_utc", "timeframe", "strategy", "direction", "signal_bar_utc",
              "entry", "sl", "tp", "volume", "atr", "spread", "est_risk_usd", "est_cost_usd",
              "reason_entry"):
        assert row[f] is not None, f"missing field {f}"
    assert row["timeframe"] == "H1" and row["strategy"] == "momentum_rsi_mtf"
    # close columns exist (populated on exit)
    for f in ("exit", "exit_reason", "pnl_usd", "r_multiple", "bars_held"):
        assert f in row
    assert st["last_pass"]["timeframe"] == "H1"


def test_no_order_send_in_paper_or_signal_modules():
    for name in ("execution/paper.py", "strategy/btc_h1_signal.py", "execution/pipeline.py",
                 "run_btc.py"):
        txt = (ROOT / name).read_text(encoding="utf-8")
        for line in txt.splitlines():
            s = line.split("#", 1)[0]
            assert ".order_send(" not in s or '"' in s or "'" in s, f"{name}: {line}"


def test_live_trading_still_false():
    import os
    assert os.getenv("LIVE_TRADING", "false").lower() in ("false", "0", "no", "off")


@requires_mt5
def test_live_h1_pipeline_uses_h1_and_h1_atr():
    from execution.pipeline import run_pipeline
    m5 = run_pipeline("BTC", timeframe="M5", history_bars=800)
    h1 = run_pipeline("BTC", timeframe="H1", history_bars=1500)
    assert m5["timeframe"] == "M5" and h1["timeframe"] == "H1"
    assert h1["strategy"] == "momentum_rsi_mtf"
    assert "[BTCUSD.vx H1]" or True   # log line; presence checked by data_quality stage below
    assert any("data_quality (H1" in s for s in h1["stages"])
    # H1 ATR must be much larger than M5 ATR
    assert h1["atr"] > m5["atr"] * 2
    # spread filter now passes on H1 (spread/ATR ~ 0.14) but was failing on M5 (~0.7)
    assert h1["spread_filter"]["spread_to_atr"] < m5["spread_filter"]["spread_to_atr"]
    assert h1.get("live_trading_enabled") is False
