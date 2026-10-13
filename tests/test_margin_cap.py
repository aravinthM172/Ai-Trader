from risk import margin_cap as mc


def test_notional_and_used():
    assert mc.notional(0.03, 84000, 1) == 2520
    pos = [{"symbol": "BTCUSD", "volume": 0.03, "price_open": 84000}, {"symbol": "ETHUSD", "volume": 0.4, "price_open": 2500},
           {"symbol": "XAUUSD", "volume": 0.01, "price_open": 4100}]
    assert mc.used_notional(pos, {"BTCUSD": 1, "ETHUSD": 1, "XAUUSD": 100}, {"BTCUSD", "ETHUSD"}) == 2520 + 1000
    assert mc.used_notional(pos, {}, set()) == 0


def test_a_position_that_fits_is_untouched():
    vol, note = mc.cap_volume(0.05, price=84000, contract_size=1, equity=15000, used=0, max_frac=0.6, vstep=0.01, vmin=0.01)
    assert (vol, note) == (0.05, "ok")                          # $4,200 of $9,000


def test_tight_stop_position_is_cut_down_to_the_cap():
    # 0.35 % of $15k at a 0.53 % stop wants 0.12 BTC = $10,080; the cap leaves $9,000 -> 0.10
    vol, note = mc.cap_volume(0.12, price=84000, contract_size=1, equity=15000, used=0, max_frac=0.6, vstep=0.01, vmin=0.01)
    assert vol == 0.10 and "0.12 -> 0.1 lots" in note


def test_open_positions_use_up_the_room():
    kw = dict(price=2500, contract_size=1, equity=15000, max_frac=0.6, vstep=0.01, vmin=0.01)
    assert mc.cap_volume(1.0, used=8000, **kw)[0] == 0.40       # $1,000 left
    vol, note = mc.cap_volume(1.0, used=8990, **kw)
    assert vol == 0.0 and "min lot" in note                     # $10 left, the smallest lot is $25
    assert mc.cap_volume(1.0, used=9500, **kw)[0] == 0.0        # already over


def test_never_bigger_and_off_when_unset():
    kw = dict(price=84000, contract_size=1, equity=15000, used=0, vstep=0.01, vmin=0.01)
    for want in (0.01, 0.07, 0.3):
        assert mc.cap_volume(want, max_frac=0.6, **kw)[0] <= want
    assert mc.cap_volume(0.3, max_frac=0.0, **kw) == (0.3, "ok")
