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


def test_weekend_closes_are_not_trading_days_or_activity(tmp_path):
    import sqlite3
    db = tmp_path / "t.sqlite"
    c = sqlite3.connect(db)
    c.execute("CREATE TABLE trades (opened_utc TEXT, closed_utc TEXT, status TEXT)")
    c.executemany("INSERT INTO trades VALUES (?,?,?)", [
        ("2026-10-08T10:00:00+00:00", "2026-10-09T20:00:00+00:00", "CLOSED"),   # closed Friday before the roll: counts
        ("2026-10-09T12:00:00+00:00", "2026-10-10T12:00:00+00:00", "CLOSED"),   # weekday open, Saturday close: no
        ("2026-10-10T13:00:00+00:00", "2026-10-11T20:00:00+00:00", "CLOSED"),   # all weekend: no
        ("2026-10-11T10:00:00+00:00", "2026-10-11T22:00:00+00:00", "CLOSED"),   # weekend open, Monday (firm day) close: counts
        ("2026-10-12T10:00:00+00:00", None, "OPEN"),                            # still open: no
    ])
    c.commit()
    c.close()
    assert ct._traded_days(db, 21) == {"2026-10-09", "2026-10-12"}
    assert ct._last_closed(db, 21) == datetime.fromisoformat("2026-10-11T22:00:00+00:00")
    assert ct._last_closed(tmp_path / "none.sqlite") is None


def test_fundednext_preset_is_selected_by_prop_challenge(monkeypatch):
    monkeypatch.delenv("PROP_DAILY_LOSS_LIMIT", raising=False)
    monkeypatch.delenv("PROP_DAY_RESET_UTC_HOUR", raising=False)
    monkeypatch.setenv("PROP_CHALLENGE", "fundednext_2step")
    r = ct.current_rules()
    assert ct.enabled() and r.name == "FundedNext Stellar 2-Step"
    assert (r.targets, r.min_days, r.daily_loss, r.max_loss, r.inactivity_days) == ((0.08, 0.05), (5, 5), 0.05, 0.10, 60)
    assert r.weekend_counts and r.daily_base == "initial" and r.day_reset_utc_hour == 21
    monkeypatch.setenv("PROP_DAY_RESET_UTC_HOUR", "22")                       # their server in winter (UTC+2)
    assert ct.current_rules().day_reset_utc_hour == 22
    monkeypatch.setenv("PROP_CHALLENGE", "fundingpips_2step_standard")
    monkeypatch.delenv("PROP_DAY_RESET_UTC_HOUR")
    r = ct.current_rules()
    assert (r.name, r.min_days, r.daily_loss, r.inactivity_days, r.weekend_counts) ==         ("FundingPips 2-Step Standard", (3, 3), 0.03, 30, False)
    monkeypatch.setenv("PROP_CHALLENGE", "true")                              # the old on / off value
    assert ct.enabled() and ct.current_rules().name == "FundingPips 2-Step Standard"
    monkeypatch.setenv("PROP_CHALLENGE", "")
    assert not ct.enabled()


def test_fundednext_needs_five_days_and_measures_the_day_against_the_initial_balance():
    fn = ct.FUNDEDNEXT_2STEP
    st = ct.new_state(15000, T0, fn)
    assert ct.update(st, now=T0, equity=16250, balance=16250, traded_days=_days(4), rules=fn) == []     # 4 days: wait
    ev = ct.update(st, now=T0 + timedelta(days=4), equity=16250, balance=16250, traded_days=_days(5), rules=fn)
    assert any("PHASE 1 PASSED" in e for e in ev) and st["phase"] == 2
    # phase 2 from $16,250: $750 down on the day is 4.6 % of the day's start but 5 % of ... the phase's start balance
    st = ct.new_state(15000, T0, fn)
    ct.update(st, now=T0, equity=16000, balance=16000, traded_days=set(), rules=fn)
    ct.update(st, now=T0 + timedelta(hours=1), equity=15260, balance=16000, traded_days=set(), rules=fn)   # -$740
    assert st["result"] == "in_progress" and st["daily_headroom_usd"] == 10.0
    ev = ct.update(st, now=T0 + timedelta(hours=2), equity=15250, balance=16000, traded_days=set(), rules=fn)  # -$750
    assert st["result"] == "failed" and any("daily loss" in e for e in ev)


def test_weekend_closes_count_only_where_the_firm_counts_them(tmp_path):
    import sqlite3
    db = tmp_path / "t.sqlite"
    c = sqlite3.connect(db)
    c.execute("CREATE TABLE trades(status TEXT, closed_utc TEXT)")
    c.executemany("INSERT INTO trades VALUES('CLOSED', ?)", [("2026-10-10T12:00:00+00:00",),      # Saturday
                                                             ("2026-10-12T12:00:00+00:00",)])     # Monday
    c.commit()
    c.close()
    assert ct._traded_days(db, 21) == {"2026-10-12"}
    assert ct._traded_days(db, 21, weekend_counts=True) == {"2026-10-10", "2026-10-12"}

