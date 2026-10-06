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


# ---- prop controls wiring -------------------------------------------------------------
import json as _json
from datetime import timedelta as _td
from tools.challenge_tracker import trading_day as _tday0
_tday = lambda t: _tday0(t, 21)


def _prop_env(tmp_path, monkeypatch, challenge):
    from tools import challenge_tracker as ct
    out = tmp_path / "challenge.json"
    out.write_text(_json.dumps(challenge))
    monkeypatch.setattr(ct, "enabled", lambda: True)
    monkeypatch.setattr(ct, "OUT", out)
    monkeypatch.setattr(lm, "STATE", tmp_path)
    monkeypatch.setattr(lm.safety, "kill_switch_active", lambda: False)
    closed = []
    monkeypatch.setattr(lm, "_close", lambda gw, spec, p, reason, c: closed.append((p["ticket"], reason)) or True)
    c = lm._conn(tmp_path / "t.sqlite")
    return c, closed


CH1 = {"result": "in_progress", "phase": 1, "phase_start_balance": 5000.0}
NOW1 = datetime(2026, 10, 6, 10, 5, tzinfo=timezone.utc)
POS = [{"ticket": 1, "symbol": "XAUUSD.vx", "type": "BUY", "profit": -120.0},
       {"ticket": 2, "symbol": "DAX40.vx", "type": "SELL", "profit": -95.0}]
SPECS = {"XAUUSD.vx": object(), "DAX40.vx": object()}


def test_prop_daily_flatten_closes_all_when_live(tmp_path, monkeypatch):
    c, closed = _prop_env(tmp_path, monkeypatch, CH1)
    ls = {"prop_day": _tday(NOW1), "prop_day_ref": 5000.0}
    d = lm._prop_controls(None, c, ls, POS, SPECS, 4785.0, 5000.0, NOW1, True)      # -4.3 % today
    assert d.flatten_all and sorted(t for t, _ in closed) == [1, 2] and ls["prop_halt_day"]


def test_prop_dry_run_only_logs(tmp_path, monkeypatch):
    c, closed = _prop_env(tmp_path, monkeypatch, CH1)
    d = lm._prop_controls(None, c, {"prop_day": "x"}, POS, SPECS, 4785.0, 5000.0, NOW1, False)
    assert closed == [] and d.flatten_all is False or closed == []
    kinds = [r[0] for r in c.execute("SELECT kind FROM events")]
    assert closed == [] and (not d.flatten_all or "would_close" in kinds)


def test_prop_idea_loss_closes_only_that_trade(tmp_path, monkeypatch):
    c, closed = _prop_env(tmp_path, monkeypatch, CH1)
    pos = [{"ticket": 1, "symbol": "XAUUSD.vx", "type": "BUY", "profit": -46.0},
           {"ticket": 2, "symbol": "DAX40.vx", "type": "SELL", "profit": -10.0}]
    d = lm._prop_controls(None, c, {}, pos, SPECS, 4944.0, 5000.0, NOW1, True)
    assert closed == [(1, "idea_loss")] and not d.flatten_all


def test_prop_max_loss_writes_kill_switch(tmp_path, monkeypatch):
    c, closed = _prop_env(tmp_path, monkeypatch, CH1)
    d = lm._prop_controls(None, c, {"prop_day": _tday(NOW1),
                                    "prop_day_ref": 4600.0}, POS, SPECS, 4570.0, 4600.0, NOW1, True)
    assert d.kill and (tmp_path / "KILL_SWITCH").exists() and len(closed) == 2


def test_prop_cooldown_from_recent_losing_close(tmp_path, monkeypatch):
    c, _ = _prop_env(tmp_path, monkeypatch, CH1)
    c.execute("INSERT INTO trades(ticket, symbol, status, closed_utc, pnl_usd) VALUES(9,'DAX40.vx','CLOSED',?,-20)",
              ((NOW1 - _td(minutes=3)).isoformat(),))
    c.commit()
    d = lm._prop_controls(None, c, {}, [], SPECS, 5000.0, 5000.0, NOW1, True)
    assert not d.allow_entries and any("cooldown" in r for r in d.reasons)


def test_prop_idle_when_not_configured(monkeypatch, tmp_path):
    from tools import challenge_tracker as ct
    monkeypatch.setattr(ct, "enabled", lambda: False)
    assert lm._prop_controls(None, None, {}, [], {}, 5000.0, 5000.0, NOW1, True) is None


def test_emergency_flatten_closes_everything_when_allowed(tmp_path, monkeypatch):
    c, closed = _prop_env(tmp_path, monkeypatch, CH1)
    r = lm._emergency_flatten(None, c, POS, SPECS, True)
    assert sorted(t for t, why in closed) == [1, 2] and all(why == "emergency" for _, why in closed) and r["positions"] == 2


def test_emergency_flatten_dry_run_only_logs(tmp_path, monkeypatch):
    c, closed = _prop_env(tmp_path, monkeypatch, CH1)
    r = lm._emergency_flatten(None, c, POS, SPECS, False)
    assert closed == [] and all(x.get("dry_run") for x in r["results"])
    assert [k[0] for k in c.execute("SELECT kind FROM events")] == ["would_close", "would_close"]


def test_close_allowed_ignores_kill_switch_but_not_account_mode(monkeypatch):
    monkeypatch.setenv("MULTI_LIVE_TRADING", "true"); monkeypatch.setenv("LIVE_TRADING", "true")
    monkeypatch.delenv("LIVE_ACCOUNT_MODE", raising=False)
    monkeypatch.setattr(lm.safety, "kill_switch_active", lambda: True)
    assert lm.close_allowed({"trade_mode": 0}) is True            # demo, kill switch on -> may still close
    assert lm.close_allowed({"trade_mode": 2}) is False           # real account not allowed by LIVE_ACCOUNT_MODE
    monkeypatch.setenv("MULTI_LIVE_TRADING", "false")
    assert lm.close_allowed({"trade_mode": 0}) is False


# -- sizing value per point: never trust a too-small tick value (DAX40.vx 2026-10-05) --------------
class _Spec:
    def __init__(self, vpu, ccy):
        self.value_per_price_unit_per_lot, self.currency_profit = vpu, ccy


def test_value_per_unit_uses_larger_of_tick_and_calculator():
    v, note = lm.loss_value_per_unit(_Spec(1.0, "EUR"), 11.2, "USD")      # DAX: tick says $1, real ~$11.2
    assert v == 11.2 and "calculator" in note
    v, note = lm.loss_value_per_unit(_Spec(100.0, "USD"), 100.0, "USD")   # XAUUSD: both agree
    assert v == 100.0 and note == "ok"
    v, _ = lm.loss_value_per_unit(_Spec(12.0, "EUR"), 11.2, "USD")       # tick larger -> keep the larger
    assert v == 12.0


def test_value_per_unit_skips_unverifiable_foreign_currency():
    v, note = lm.loss_value_per_unit(_Spec(1.0, "EUR"), None, "USD")
    assert v is None and "cannot verify" in note
    v, note = lm.loss_value_per_unit(_Spec(1.0, "USD"), None, "USD")     # same currency: tick value is fine
    assert v == 1.0 and note == "ok"


def test_dax_trade_sized_to_planned_risk():
    vpu, _ = lm.loss_value_per_unit(_Spec(1.0, "EUR"), 11.2, "USD")
    vol, risk, _ = lm.size_volume(equity=5000, risk_frac=0.005, max_risk_frac=0.02, sl_dist=129.5,
                                  value_per_unit=vpu, vmin=0.01, vstep=0.01, vmax=50)
    assert vol == 0.01 and risk <= 25          # was 0.19 lots (~$276 real risk) before the fix


# -- per-symbol risk (MULTI_SYMBOL_RISK) ------------------------------------------------------
def test_parse_symbol_risk_ignores_bad_and_oversized_entries():
    r = lm.parse_symbol_risk(" XAUUSD=0.0025, GER40 = 0.0025 ,BAD,NAS100=abc,BTCUSD=0.5,=0.001")
    assert r == {"XAUUSD": 0.0025, "GER40": 0.0025}           # 0.5 > MAX_RISK_PER_TRADE -> dropped


def test_symbol_risk_falls_back_to_default(monkeypatch):
    monkeypatch.setattr(lm, "SYMBOL_RISK", {"XAUUSD": 0.0025})
    assert lm.symbol_risk("XAUUSD") == 0.0025
    assert lm.symbol_risk("BTCUSD") == lm.RISK_PER_TRADE
