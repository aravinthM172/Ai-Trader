from datetime import datetime, timezone

from strategy import weekend_flat as wf


def _t(day, hour, minute=0):
    return datetime(2026, 10, day, hour, minute, tzinfo=timezone.utc)      # 2026-10-16 is a Friday


def test_window_starts_friday_20_utc():
    assert not wf.in_flat_window(_t(16, 19, 59))
    assert wf.in_flat_window(_t(16, 20, 0))
    assert wf.in_flat_window(_t(16, 23, 30))


def test_whole_saturday_and_sunday_until_21_utc():
    assert wf.in_flat_window(_t(17, 0))
    assert wf.in_flat_window(_t(17, 12))
    assert wf.in_flat_window(_t(18, 20, 59))
    assert not wf.in_flat_window(_t(18, 21, 0))


def test_close_only_on_friday_evening():
    assert not wf.close_due(_t(16, 19, 59))
    assert wf.close_due(_t(16, 20, 0)) and wf.close_due(_t(16, 23, 59))
    assert not wf.close_due(_t(17, 12)) and not wf.close_due(_t(18, 20))   # shut: a close would be rejected
    assert not wf.close_due(_t(15, 21))                                    # Thursday


def test_weekdays_are_open():
    for day in (12, 13, 14, 15):                                           # Monday .. Thursday
        for hour in (0, 12, 20, 23):
            assert not wf.in_flat_window(_t(day, hour))
    assert not wf.in_flat_window(_t(19, 0))                                # Monday 00:00
