from datetime import datetime, timedelta, timezone

from risk import prop_controls as pc

NOW = datetime(2026, 10, 6, 10, 5, tzinfo=timezone.utc)
CH = {"result": "in_progress", "phase": 1, "phase_start_balance": 5000.0, "day_ref": 5000.0}


def _d(**kw):
    base = dict(challenge=CH, equity=5000.0, positions=[], last_losing_close=None, now=NOW,
                trading_day="2026-10-06", halted_day=None)
    base.update(kw)
    return pc.decide(**base)


def test_normal_allows_entries_at_base_risk():
    d = _d()
    assert d.allow_entries and not d.flatten_all and d.risk_cap == 0.005 and d.close_tickets == []


def test_daily_flatten_halts_and_halt_persists_for_the_day():
    d = _d(equity=4785.0)                                    # -4.3 % today
    assert d.flatten_all and d.halt_today and not d.allow_entries
    d2 = _d(equity=4990.0, halted_day="2026-10-06")
    assert not d2.allow_entries and any("halted" in r for r in d2.reasons)
    assert _d(equity=4990.0, halted_day="2026-10-05").allow_entries


def test_max_loss_flatten_and_kill_from_phase_start():
    d = _d(equity=4570.0, challenge=dict(CH, day_ref=4600.0))   # -8.6 % from 5000, -0.65 % today
    assert d.flatten_all and d.kill


def test_target_reached_stops_entries():
    d = _d(equity=5401.0, challenge=dict(CH, day_ref=5390.0))
    assert not d.allow_entries and any("target" in r for r in d.reasons)
    funded = dict(CH, phase=3)
    assert _d(equity=5401.0, challenge=dict(funded, day_ref=5390.0)).allow_entries


def test_trade_idea_over_0_9_percent_is_closed():
    pos = [{"ticket": 1, "symbol": "XAUUSD.vx", "type": "BUY", "profit": -46.0},
           {"ticket": 2, "symbol": "DAX40.vx", "type": "SELL", "profit": -20.0}]
    d = _d(positions=pos, equity=4934.0)
    assert d.close_tickets == [1]


def test_cooldown_after_losing_close():
    assert not _d(last_losing_close=NOW - timedelta(minutes=4)).allow_entries
    assert _d(last_losing_close=NOW - timedelta(minutes=11)).allow_entries


def test_risk_shrinks_with_open_positions_and_daily_room():
    pos = [{"ticket": k, "symbol": f"S{k}", "type": "BUY", "profit": 0.0} for k in range(2)]
    d = _d(positions=pos, equity=4850.0)                     # -3 % today -> 0.5 % room / 3
    assert abs(d.risk_cap - 0.005 / 3) < 1e-9


def test_idle_without_challenge():
    d = _d(challenge=None)
    assert d.allow_entries and "idle" in d.reasons[0]
