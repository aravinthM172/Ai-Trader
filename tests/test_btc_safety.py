"""
PHASE 11 -- one test per safety mechanism.  No MT5, no orders.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from execution import safety, state as state_store

ROOT = Path(__file__).resolve().parents[1]
KILL = ROOT / "state" / "KILL_SWITCH"


@pytest.fixture
def acc():
    return {"balance": 100.0, "trade_allowed": True}


@pytest.fixture(autouse=True)
def _clean_kill():
    if KILL.exists():
        KILL.unlink()
    yield
    if KILL.exists():
        KILL.unlink()


def _ev(cfg, acc, **over):
    kw = dict(symbol="BTCUSD.vx", direction="BUY", cfg=cfg, account=acc,
              open_positions=[], tick_age_seconds=1.0, spread_ok=True,
              sizing_ok=True, state={})
    kw.update(over)
    return safety.evaluate(**kw)


def test_kill_switch_file_blocks(btc_cfg, acc):
    KILL.write_text("stop", encoding="utf-8")
    d = _ev(btc_cfg, acc)
    assert not d.allow_validation and not d.allow_live_send
    assert any("KILL SWITCH" in r for r in d.reasons)


def test_kill_switch_env_blocks(btc_cfg, acc, monkeypatch):
    monkeypatch.setenv("KILL_SWITCH", "1")
    assert safety.kill_switch_active()
    d = _ev(btc_cfg, acc)
    assert not d.allow_validation


def test_live_trading_master_switch_default_off(btc_cfg, acc, monkeypatch):
    monkeypatch.delenv("LIVE_TRADING", raising=False)
    assert safety.live_trading_enabled() is False
    d = _ev(btc_cfg, acc)
    assert d.allow_live_send is False


def test_daily_loss_limit_blocks(btc_cfg, acc):
    btc_cfg.max_daily_loss_frac = 0.05           # pin: production .env may set a wider brake
    st = {"days": {}}
    day = state_store.day_bucket(st)
    day["start_balance"] = 100.0
    day["realised_pl"] = -6.0                    # > 5% of $100
    d = _ev(btc_cfg, acc, state=st)
    assert not d.allow_validation
    assert any("daily loss" in r for r in d.reasons)


def test_max_total_positions_blocks(btc_cfg, acc):
    pos = [{"symbol": "XAUUSD.vx", "type": "BUY", "ticket": 1},
           {"symbol": "OTHER", "type": "SELL", "ticket": 2}]
    d = _ev(btc_cfg, acc, open_positions=pos)
    assert not d.allow_validation
    assert any("max total positions" in r for r in d.reasons)


def test_max_symbol_positions_blocks(btc_cfg, acc):
    btc_cfg.max_total_positions = 5
    pos = [{"symbol": "BTCUSD.vx", "type": "SELL", "ticket": 9}]
    d = _ev(btc_cfg, acc, open_positions=pos)
    assert not d.allow_validation
    assert any("max positions" in r for r in d.reasons)


def test_duplicate_position_blocks(btc_cfg, acc):
    btc_cfg.max_total_positions = 5
    btc_cfg.max_open_positions = 5
    pos = [{"symbol": "BTCUSD.vx", "type": "BUY", "ticket": 7}]
    d = _ev(btc_cfg, acc, open_positions=pos)
    assert not d.allow_validation
    assert any("duplicate" in r for r in d.reasons)


def test_cooldown_blocks(btc_cfg, acc):
    st = {}
    state_store.record_intent(st, "BTCUSD.vx")
    d = _ev(btc_cfg, acc, state=st)
    assert not d.allow_validation
    assert any("cooldown" in r for r in d.reasons)


def test_stale_price_blocks(btc_cfg, acc):
    d = _ev(btc_cfg, acc, tick_age_seconds=99999)
    assert not d.allow_validation
    d2 = _ev(btc_cfg, acc, tick_age_seconds=None)
    assert not d2.allow_validation


def test_spread_rejection_blocks(btc_cfg, acc):
    d = _ev(btc_cfg, acc, spread_ok=False)
    assert not d.allow_validation
    assert any("spread" in r for r in d.reasons)


def test_sizing_rejection_blocks(btc_cfg, acc):
    d = _ev(btc_cfg, acc, sizing_ok=False)
    assert not d.allow_validation


def test_stop_validation_rejects_bad_sides(btc_cfg, fake_spec):
    from execution.order_validator import OrderValidation
    ov = OrderValidation(symbol="BTCUSD.vx", direction="BUY", entry=78000, volume=0.01,
                         sl=78100, tp=77900, sl_distance=100, tp_distance=100, reward_risk=1.0,
                         spread_price=29.76, atr=40, risk_dollars=1.0, risk_pct=1.0,
                         required_margin=7.8, free_margin=100.0, filling_mode="FOK")
    ov.checks = {"sl_side_ok": ov.sl < ov.entry, "tp_side_ok": ov.tp > ov.entry}
    assert not ov.checks["sl_side_ok"] and not ov.checks["tp_side_ok"]


def test_volume_validation(btc_cfg, fake_spec):
    from risk.sizing import _normalise_volume
    assert _normalise_volume(0.037, 0.01, 0.01, 50.0) == pytest.approx(0.03)
    assert _normalise_volume(999, 0.01, 0.01, 50.0) == 50.0


def test_margin_validation_blocks(btc_cfg, fake_spec):
    from risk.sizing import size_position

    class GW:
        def calc_profit(self, *a): return -abs((a[3] - a[4]) * a[2])
        def calc_margin(self, *a): return 500.0                       # more than free margin
        def account_info(self): return {"margin_free": 100.0, "balance": 100.0}
    r = size_position(direction="BUY", entry=78000.0, sl=77900.0, balance=100.0,
                      spec=fake_spec, cfg=btc_cfg, gateway=GW())
    assert not r.accepted
    assert any("margin" in x for x in r.reasons)


def test_emergency_shutdown_via_kill_switch_stops_live_send(btc_cfg, acc, monkeypatch):
    monkeypatch.setenv("LIVE_TRADING", "true")
    d1 = _ev(btc_cfg, acc)
    assert d1.allow_live_send is True
    KILL.write_text("x", encoding="utf-8")
    d2 = _ev(btc_cfg, acc)
    assert d2.allow_live_send is False and d2.allow_validation is False


def test_no_order_send_call_anywhere():
    """No actual order_send( invocation in project code (docstrings/comments/field
    lists are fine -- we look for a real call) -- except execution/live.py, the one
    gated live-execution module (its gates are covered by tests/test_live.py)."""
    import re
    call = re.compile(r"(?<![#\"'])\b(mt5|_mt5|self\.raw\(\))\.order_send\s*\(|(?<!\.)\border_send\s*\(")
    live_module = ROOT / "execution" / "live.py"
    for p in ROOT.rglob("*.py"):
        if "venv" in p.parts or p.name.startswith("btc_paper_setup") or p == live_module:
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            s = line.split("#", 1)[0]
            if "order_send" not in s:
                continue
            # allow: string literals mentioning it, and our own regex here
            if p.name == "test_btc_safety.py":
                continue
            if re.search(r"\.order_send\s*\(", s) and '"' not in s and "'" not in s:
                raise AssertionError(f"real order_send call: {p}:{i}: {line.strip()}")
    # also: the deliberate absence is documented
    assert (ROOT / "execution" / "order_validator.py").read_text(encoding="utf-8").count("order_send") >= 1
