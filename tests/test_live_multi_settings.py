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


def test_fixed_lot_and_hard_cap(monkeypatch, tmp_path):
    p = _plan(monkeypatch, tmp_path, _mk(mode="lots", lots=0.2))
    assert p["volume"] == pytest.approx(0.2)
    big = _plan(monkeypatch, tmp_path, _mk(mode="lots", lots=40))                            # would risk ~$8,000
    assert big["risk_usd"] <= 150 + 1e-6                                                     # 1 % of $15k
