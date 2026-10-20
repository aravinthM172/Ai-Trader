import json

import pytest

from strategy import bot_settings as bs


def _raw(**m):
    base = {"enabled": True, "mode": "risk", "risk_pct": 0.35, "lots": 0, "sl_atr": 2, "tp_atr": 6}
    return {"markets": {"BTCUSD": {**base, **m}}, "auto_risk": {"enabled": False}}


def test_validate_accepts_defaults_and_rejects_out_of_range():
    clean, err = bs.validate(_raw())
    assert not err and clean["markets"]["BTCUSD"]["sl_atr"] == 2.0
    for bad in ({"risk_pct": 5}, {"sl_atr": 0.1}, {"tp_atr": 1, "sl_atr": 2}, {"mode": "lots", "lots": 0}, {"mode": "yolo"}):
        assert bs.validate(_raw(**bad))[1], bad
    assert bs.validate({"markets": {"DOGE": {}}})[1]
    assert bs.validate({"auto_risk": {"enabled": True, "start_balance": 0}})[1]


def test_save_and_load_roundtrip_and_broken_file(tmp_path):
    p = tmp_path / "s.json"
    clean, _ = bs.validate(_raw(risk_pct=0.5))
    bs.save(clean, p)
    assert bs.load(p)["markets"]["BTCUSD"]["risk_pct"] == 0.5
    bs.save(clean, p)
    assert (tmp_path / "s.json.bak").exists()
    p.write_text(json.dumps({"markets": {"BTCUSD": {"risk_pct": 99}}}))
    assert bs.load(p) == {}                                   # never trade on a broken file
    assert bs.load(tmp_path / "missing.json") == {}


def test_cushion_multiplier():
    s = {"auto_risk": {"enabled": True, "start_balance": 15000, "max_loss_pct": 10}}
    assert bs.cushion_multiplier(s, 15000) == pytest.approx(1.0)
    assert bs.cushion_multiplier(s, 14250) == pytest.approx(0.5)      # half the room used -> half risk
    assert bs.cushion_multiplier(s, 13600) == pytest.approx(0.3)      # floor
    assert bs.cushion_multiplier(s, 16500) == pytest.approx(1.5)      # cap
    assert bs.cushion_multiplier({"auto_risk": {"enabled": False}}, 14000) == 1.0


def test_size_risk_mode_and_fixed_lot_with_hard_cap():
    kw = dict(equity=15000, sl_dist=1000, value_per_unit=1.0, vmin=0.01, vstep=0.01, vmax=100)
    vol, risk, _ = bs.size({"mode": "risk"}, risk_frac=0.0035, **kw)
    assert vol == pytest.approx(0.05) and risk == pytest.approx(50)
    vol, risk, note = bs.size({"mode": "lots", "lots": 0.1}, risk_frac=0.0035, **kw)
    assert vol == pytest.approx(0.1) and risk == pytest.approx(100)
    vol, risk, note = bs.size({"mode": "lots", "lots": 5}, risk_frac=0.0035, **kw)   # $5,000 risk asked
    assert vol == pytest.approx(0.15) and risk <= 150 + 1e-9 and "cut" in note
    vol, _, note = bs.size({"mode": "risk"}, risk_frac=0.0035, equity=15000, sl_dist=1000, value_per_unit=100,
                           vmin=0.01, vstep=0.01, vmax=100)                         # min lot = $1,000 risk
    assert vol == 0 and "limit" in note
