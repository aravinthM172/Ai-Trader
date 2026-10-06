import json
from datetime import datetime, timedelta, timezone

from tools import challenge_tracker as ct

T0 = datetime(2026, 10, 6, 8, 0, tzinfo=timezone.utc)


def _days(n, start=T0):
    return {ct.trading_day(start + timedelta(days=k), 21) for k in range(n)}


def test_phase1_needs_target_and_min_days_then_phase2_rebases():
    st = ct.new_state(5000, T0)
    assert ct.update(st, now=T0, equity=5420, balance=5420, traded_days=_days(2)) == []      # only 2 days
    ev = ct.update(st, now=T0 + timedelta(days=2), equity=5420, balance=5420, traded_days=_days(3))
    assert any("PHASE 1 PASSED" in e for e in ev)
    assert st["phase"] == 2 and st["phase_start_balance"] == 5420 and st["trading_days"] == []
    later = T0 + timedelta(days=10)
    ev = ct.update(st, now=later, equity=5420 * 1.051, balance=5420 * 1.051, traded_days=_days(3, later))
    assert any("CHALLENGE PASSED" in e for e in ev) and st["result"] == "passed"


def test_daily_loss_uses_higher_of_balance_or_equity_at_day_start():
    st = ct.new_state(5000, T0)
    ct.update(st, now=T0, equity=5200, balance=5000, traded_days=set())        # day ref = 5200 (equity)
    ev = ct.update(st, now=T0 + timedelta(hours=2), equity=4935, balance=5000, traded_days=set())
    assert st["result"] == "failed" and any("daily loss" in e for e in ev)        # 265/5200 = 5.1 %


def test_max_loss_static_from_phase_start():
    st = ct.new_state(5000, T0)
    for k in range(5):                                                          # slow bleed, < 5 % per day
        ct.update(st, now=T0 + timedelta(days=k), equity=5000 - 100 * (k + 1) + 50, balance=5000 - 100 * k,
                  traded_days=set())
    ev = ct.update(st, now=T0 + timedelta(days=6), equity=4495, balance=4520, traded_days=set())
    assert st["result"] == "failed" and any("total loss" in e for e in ev)


def test_warning_at_70_percent_once_per_day():
    r5 = ct.ChallengeRules(daily_loss=0.05)                                   # a 5 % daily-limit account
    st = ct.new_state(5000, T0, r5)
    ct.update(st, now=T0, equity=5000, balance=5000, traded_days=set(), rules=r5)
    ev1 = ct.update(st, now=T0 + timedelta(hours=1), equity=4820, balance=5000, traded_days=set(), rules=r5)   # 3.6 %
    ev2 = ct.update(st, now=T0 + timedelta(hours=2), equity=4815, balance=5000, traded_days=set(), rules=r5)
    assert any("warning: daily" in e for e in ev1) and not ev2


def test_default_daily_limit_is_three_percent_and_reported(monkeypatch):
    monkeypatch.delenv("PROP_DAILY_LOSS_LIMIT", raising=False)
    r = ct.current_rules()
    assert r.daily_loss == 0.03
    st = ct.new_state(5000, T0, r)
    ct.update(st, now=T0, equity=5000, balance=5000, traded_days=set(), rules=r)
    ev = ct.update(st, now=T0 + timedelta(hours=1), equity=4845, balance=5000, traded_days=set(), rules=r)   # 3.1 %
    assert st["daily_loss_limit"] == 0.03 and st["result"] == "failed" and any("FAILED" in e for e in ev)
    monkeypatch.setenv("PROP_DAILY_LOSS_LIMIT", "0.05")
    assert ct.current_rules().daily_loss == 0.05


def test_run_from_files(tmp_path):
    status = tmp_path / "s.json"
    status.write_text(json.dumps({"generated_utc": T0.isoformat(), "equity": 5000.0}))
    ev = ct.run_from_files(state_path=tmp_path / "st.json", status_path=status, db=tmp_path / "none.sqlite",
                           out=tmp_path / "o.json")
    st = json.loads((tmp_path / "st.json").read_text())
    assert ev == [] and st["phase"] == 1 and st["target_usd"] == 5400.0


def test_inactivity_warns_then_fails():
    st = ct.new_state(5000, T0)
    ev = ct.update(st, now=T0 + timedelta(days=21), equity=5000, balance=5000, traded_days=set(), last_closed=None)
    assert any("no completed trade for 21 days" in e for e in ev) and st["result"] == "in_progress"
    ev = ct.update(st, now=T0 + timedelta(days=30, hours=1), equity=5000, balance=5000, traded_days=set(), last_closed=None)
    assert st["result"] == "failed" and any("inactivity" in e for e in ev)


def test_recent_trade_resets_inactivity_clock():
    st = ct.new_state(5000, T0)
    ct.update(st, now=T0 + timedelta(days=29), equity=5000, balance=5000, traded_days=set(),
              last_closed=T0 + timedelta(days=25))
    assert st["result"] == "in_progress" and st["days_since_last_closed_trade"] == 4.0
