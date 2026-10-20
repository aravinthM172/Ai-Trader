"""Dashboard bot settings inside execution/live_multi.py (added 2026-10-20): stop / target ATR, risk %, fixed lot and
the 1 % hard cap reach the order plan.  Skipped until the trader patch is applied (live_multi imports bot_settings)."""
import pytest

from execution import live_multi as lm
from tests import test_live_multi_fundednext as fn

pytestmark = pytest.mark.skipif(not hasattr(lm, "bot_settings"), reason="trader patch not applied yet")


def _plan(monkeypatch, tmp_path, mk, risk=0.0035):
    monkeypatch.setattr(lm, "NOTIONAL_CAP_SYMBOLS", set())
    monkeypatch.setattr(lm, "STATE", tmp_path)
    c = lm._conn(tmp_path / "t.sqlite")
    now = fn.pd.Timestamp("2026-10-12 12:05", tz="UTC").to_pydatetime()
    gst = lm.guard_state(c, [], {}, 15000.0, 15000.0, now)
    try:
        res = fn._call(monkeypatch, fn.Gw(), calm=frozenset(), news_blocks=False, c=c, spec=fn._spec(), gst=gst,
                       equity=15000.0, risk=risk, mk=mk)
    finally:
        c.close()
    assert res["decision"] == "dry_run", res
    return res["plan"]


def _mk(**kw):
    return {"enabled": True, "mode": "risk", "risk_pct": 0.35, "lots": 0.0, "sl_atr": 2.0, "tp_atr": 6.0, **kw}


def test_no_settings_keeps_today_plan(monkeypatch, tmp_path):
    p = _plan(monkeypatch, tmp_path, None)
    assert p["tp"] - p["entry"] == pytest.approx(3 * (p["entry"] - p["sl"]), rel=0.01)       # 2 ATR stop / 6 ATR target


def test_stop_and_target_atr_from_settings(monkeypatch, tmp_path):
    p = _plan(monkeypatch, tmp_path, _mk(sl_atr=1.5, tp_atr=7.5))
    assert p["tp"] - p["entry"] == pytest.approx(5 * (p["entry"] - p["sl"]), rel=0.01)
    assert p["entry"] - p["sl"] == pytest.approx(1.5 * p["atr"], rel=0.01)


def _prop_base_risk(monkeypatch, tmp_path, saved):
    """base_risk handed to prop_controls.decide by _prop_controls (the per-trade ceiling)."""
    from datetime import datetime, timezone
    from tools import challenge_tracker as ct
    seen = {}
    monkeypatch.setattr(ct, "enabled", lambda: True)
    monkeypatch.setattr(ct, "OUT", tmp_path / "missing.json")
    monkeypatch.setattr(lm.bot_settings, "load", lambda: saved)
    monkeypatch.setattr(lm.prop_controls, "decide", lambda **kw: seen.update(kw) or fn.NS(halt_today=False, flatten_all=False,
                        close_tickets=[], kill=False, allow_entries=True, risk_cap=kw["base_risk"], reasons=[], to_dict=lambda: {}))
    c = lm._conn(tmp_path / "t.sqlite")
    try:
        lm._prop_controls(None, c, {}, [], {}, 15000.0, 15000.0, datetime(2026, 10, 20, 7, tzinfo=timezone.utc), False)
    finally:
        c.close()
    return seen.get("base_risk")


@pytest.mark.skipif(not hasattr(lm, "bot_settings") or "base_cap" not in open(lm.__file__, encoding="utf-8").read(),
                    reason="allow-growth patch not applied yet")
def test_settings_raise_the_per_trade_ceiling_to_the_hard_cap(monkeypatch, tmp_path):
    assert _prop_base_risk(monkeypatch, tmp_path, {}) == lm.RISK_PER_TRADE                     # no settings: .env value
    saved = {"markets": {"BTCUSD": _mk()}, "auto_risk": {"enabled": True, "start_balance": 15000, "max_loss_pct": 10}}
    assert _prop_base_risk(monkeypatch, tmp_path, saved) == lm.bot_settings.HARD_MAX_RISK


def test_fixed_lot_and_hard_cap(monkeypatch, tmp_path):
    p = _plan(monkeypatch, tmp_path, _mk(mode="lots", lots=0.2))
    assert p["volume"] == pytest.approx(0.2)
    big = _plan(monkeypatch, tmp_path, _mk(mode="lots", lots=40))                            # would risk ~$8,000
    assert big["risk_usd"] <= 150 + 1e-6                                                     # 1 % of $15k


@pytest.mark.skipif(not hasattr(lm, "parse_symbol_count"), reason="per-symbol patch not applied yet")
def test_parse_symbol_count():
    assert lm.parse_symbol_count("BTCUSD=2") == {"BTCUSD": 2}
    assert lm.parse_symbol_count("BTCUSD=2, ETHUSD=x, NDX100=9, XAUUSD=0") == {"BTCUSD": 2}   # bad / out of range ignored
    assert lm.parse_symbol_count("") == {}
