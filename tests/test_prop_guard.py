from risk.prop_guard import PropRules, evaluate, max_risk_per_trade

R = PropRules(initial_balance=10_000)


def test_normal_day_allows():
    d = evaluate(R, equity=10_100, day_start=10_050)
    assert d.allow_entries and not d.flatten and not d.kill


def test_daily_block_then_flatten():
    d = evaluate(R, equity=9_640, day_start=10_000)              # -3.6 %
    assert not d.allow_entries and not d.flatten and d.day_halt
    d = evaluate(R, equity=9_570, day_start=10_000)              # -4.3 %
    assert d.flatten and d.day_halt and not d.kill


def test_static_max_loss_uses_initial_balance_not_peak():
    # equity well below a 12k peak but only 6 % under the 10k initial -> still allowed
    assert evaluate(R, equity=9_400, day_start=9_450).allow_entries
    d = evaluate(R, equity=9_280, day_start=9_300)               # -7.2 % from initial
    assert not d.allow_entries and not d.flatten
    d = evaluate(R, equity=9_140, day_start=9_150)               # -8.6 % from initial
    assert d.flatten and d.kill


def test_profit_target_stops_trading():
    d = evaluate(R, equity=11_010, day_start=10_900)
    assert not d.allow_entries and "target" in d.reason
    funded = PropRules(initial_balance=10_000, profit_target=None)
    assert evaluate(funded, equity=11_010, day_start=10_900).allow_entries


def test_risk_shrinks_with_daily_room():
    assert max_risk_per_trade(R, equity=10_000, day_start=10_000, n_open=0) == 0.005
    # 3 % lost today -> 0.5 % room left, 2 open -> 0.5/3 %
    assert abs(max_risk_per_trade(R, equity=9_700, day_start=10_000, n_open=2) - 0.005 / 3) < 1e-12
    assert max_risk_per_trade(R, equity=9_600, day_start=10_000, n_open=0) == 0.0
