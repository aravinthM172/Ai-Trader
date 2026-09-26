"""execution/live.py gates -- against a FAKE broker only (never the real terminal)."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from execution import live, safety, state as state_store


class FakeMT5:
    TRADE_ACTION_DEAL, TRADE_ACTION_SLTP = 1, 6
    ORDER_TYPE_BUY, ORDER_TYPE_SELL = 0, 1
    ORDER_TIME_GTC = 0
    ORDER_FILLING_FOK, ORDER_FILLING_IOC, ORDER_FILLING_RETURN = 0, 1, 2

    def __init__(self, broker):
        self.b = broker

    def order_send(self, req):
        self.b.sent.append(req)
        if req["action"] == self.TRADE_ACTION_DEAL and "position" not in req:
            self.b.positions.append(dict(
                ticket=1000 + len(self.b.sent), symbol=req["symbol"],
                type="BUY" if req["type"] == 0 else "SELL", volume=req["volume"],
                price_open=req["price"], sl=0.0 if self.b.drop_sl else req["sl"], tp=req["tp"],
                profit=0.0, time=int(datetime.now(timezone.utc).timestamp()), magic=req["magic"]))
        elif "position" in req and req["action"] == self.TRADE_ACTION_DEAL:
            self.b.positions = [p for p in self.b.positions if p["ticket"] != req["position"]]
        elif req["action"] == self.TRADE_ACTION_SLTP and not self.b.sltp_fails:
            for p in self.b.positions:
                if p["ticket"] == req["position"]:
                    p["sl"], p["tp"] = req["sl"], req["tp"]
        rc = 10006 if (req["action"] == self.TRADE_ACTION_SLTP and self.b.sltp_fails) else 10009
        return SimpleNamespace(retcode=rc, comment="fake", price=req.get("price", 0.0))

    def history_deals_get(self, *a):
        return []

    def last_error(self):
        return (0, "ok")


class FakeGW:
    broker = None

    def connect(self):
        return True

    def shutdown(self):
        pass

    def raw(self):
        return FakeMT5(self.broker)

    def get_spec(self, symbol):
        return SimpleNamespace(symbol=symbol, digits=2, value_per_price_unit_per_lot=1.0)

    def account_info(self):
        return {"balance": self.broker.balance, "equity": self.broker.balance, "trade_allowed": True}

    def terminal_info(self):
        return {"trade_allowed": True}

    def positions(self, symbol=None):
        return [dict(p) for p in self.broker.positions]

    def get_tick(self, symbol):
        return {"bid": 84000.0, "ask": 84029.76, "spread": 29.76}

    def to_utc(self, epoch):
        return datetime.fromtimestamp(epoch, tz=timezone.utc)

    def order_check(self, req):
        return {"retcode": 0, "comment": "Done"}

    def preferred_filling(self, spec):
        return 0


def _approved(direction="BUY", bar=None):
    bar = bar or live._signal_bar_due(datetime.now(timezone.utc))
    return {
        "signal_bar_utc": str(pd.Timestamp(bar)), "decision": "APPROVED_DRY_RUN",
        "evaluated_direction": direction, "signal": {"decision": direction}, "atr": 300.0,
        "stop_plan": {"entry": 84029.76, "sl": 83429.76, "tp": 84929.76},
        "sizing": {"volume": 0.01}, "safety": {"allow_live_send": True},
    }


@pytest.fixture
def env(tmp_path, monkeypatch):
    broker = SimpleNamespace(sent=[], positions=[], balance=300.0, drop_sl=False, sltp_fails=False)
    FakeGW.broker = broker
    kill = tmp_path / "KILL_SWITCH"
    monkeypatch.setattr(live, "MT5Gateway", FakeGW)
    monkeypatch.setattr(live, "_DB", tmp_path / "live.sqlite")
    monkeypatch.setattr(live, "_LIVE_STATE", tmp_path / "live_state.json")
    monkeypatch.setattr(live, "_KILL_FILE", kill)
    monkeypatch.setattr(live, "_REPORTS", tmp_path)
    monkeypatch.setattr(safety, "_KILL_FILE", kill)
    monkeypatch.setattr(state_store, "_PATH", tmp_path / "bot_state.json")
    monkeypatch.setattr(live, "entry_window_s", lambda: 3600.0)
    monkeypatch.setattr(live, "run_pipeline", lambda *a, **k: _approved())
    monkeypatch.setenv("LIVE_TRADING", "true")
    return SimpleNamespace(broker=broker, kill=kill, monkeypatch=monkeypatch)


def _sends(b):
    return [r for r in b.sent if r["action"] == FakeMT5.TRADE_ACTION_DEAL and "position" not in r]


def test_dry_run_never_sends(env):
    env.monkeypatch.setenv("LIVE_TRADING", "false")
    st = live.run_once()
    assert st["mode"] == "DRY_RUN" and st["last_entry_check"].get("dry_run") is True
    assert env.broker.sent == []


def test_live_sends_once_with_broker_stops_and_magic(env):
    st = live.run_once()
    assert st["last_entry_check"]["sent"]["ok"] is True
    (req,) = _sends(env.broker)
    assert req["magic"] == live.MAGIC and req["sl"] < req["price"] < req["tp"]
    # geometry kept: SL 600 / TP 900 away from the fill
    assert req["price"] - req["sl"] == pytest.approx(600.0) and req["tp"] - req["price"] == pytest.approx(900.0)
    live.run_once()                                   # position open -> no second order
    assert len(_sends(env.broker)) == 1


def test_same_signal_bar_never_sent_twice(env):
    live.run_once()
    env.broker.positions.clear()                      # stopped out within the same hour
    live.run_once()
    assert len(_sends(env.broker)) == 1


def test_kill_switch_blocks_send(env):
    env.kill.write_text("x", encoding="utf-8")
    st = live.run_once()
    assert st["last_entry_check"] == {"skipped": "kill switch active"}
    assert env.broker.sent == []


def test_blocked_by_safety_does_not_send(env):
    r = _approved(); r["safety"]["allow_live_send"] = False       # e.g. Algo Trading off
    env.monkeypatch.setattr(live, "run_pipeline", lambda *a, **k: r)
    live.run_once()
    assert env.broker.sent == []


def test_rejected_pipeline_does_not_send(env):
    r = _approved(); r["decision"] = "REJECTED"
    env.monkeypatch.setattr(live, "run_pipeline", lambda *a, **k: r)
    live.run_once()
    assert env.broker.sent == []


def test_unexpected_signal_bar_is_skipped(env):
    stale = pd.Timestamp(live._signal_bar_due(datetime.now(timezone.utc))) - pd.Timedelta(hours=3)
    env.monkeypatch.setattr(live, "run_pipeline", lambda *a, **k: _approved(bar=stale))
    st = live.run_once()
    assert "clock/feed" in st["last_entry_check"]["skipped"] and env.broker.sent == []


def test_late_start_skips_bar_without_running_pipeline(env):
    env.monkeypatch.setattr(live, "entry_window_s", lambda: -1.0)
    called = []
    env.monkeypatch.setattr(live, "run_pipeline", lambda *a, **k: called.append(1))
    live.run_once()
    assert called == [] and env.broker.sent == []


def test_missing_sl_is_repaired(env):
    env.broker.drop_sl = True
    live.run_once()
    assert env.broker.positions[0]["sl"] > 0


def test_unrepairable_sl_closes_position(env):
    env.broker.drop_sl = True
    env.broker.sltp_fails = True
    st = live.run_once()
    assert st["last_entry_check"]["sent"]["ok"] is False
    assert env.broker.positions == []


def test_time_exit_after_96h(env):
    old = int((datetime.now(timezone.utc) - timedelta(hours=97)).timestamp())
    env.broker.positions.append(dict(ticket=7, symbol="BTCUSD.vx", type="BUY", volume=0.01,
                                     price_open=80000.0, sl=79000.0, tp=81500.0, profit=0.0,
                                     time=old, magic=live.MAGIC))
    live.run_once()
    closes = [r for r in env.broker.sent if r.get("position") == 7]
    assert closes and closes[0]["type"] == FakeMT5.ORDER_TYPE_SELL


def test_foreign_positions_are_not_touched(env):
    old = int((datetime.now(timezone.utc) - timedelta(hours=200)).timestamp())
    env.broker.positions.append(dict(ticket=9, symbol="BTCUSD.vx", type="BUY", volume=0.01,
                                     price_open=80000.0, sl=0.0, tp=0.0, profit=0.0,
                                     time=old, magic=12345))
    live.run_once()
    assert not [r for r in env.broker.sent if r.get("position") == 9]


def test_drawdown_breaker_trips_kill_switch(env):
    live.run_once()                                   # peak = 300
    env.broker.positions.clear()
    env.broker.balance = 190.0                        # < 65% of 300
    st = live.run_once()
    assert env.kill.exists() and st["kill_switch_active"] is True
