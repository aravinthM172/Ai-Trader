import json

from execution import live_multi as lm
from tools import loss_review as lr


def _db(tmp_path, trades, events=()):
    p = tmp_path / "t.sqlite"
    c = lm._conn(p)
    for t in trades:
        c.execute("""INSERT INTO trades(ticket, symbol, direction, opened_utc, entry, sl, status, closed_utc, exit,
                     exit_reason, pnl_usd, r_multiple) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""", t)
    for kind, payload in events:
        c.execute("INSERT INTO events(ts_utc, kind, payload) VALUES('t', ?, ?)", (kind, json.dumps(payload)))
    c.commit()
    c.close()
    return p


def test_labels():
    assert lr.label("target", 2.9, None) == "win"
    assert lr.label("stop", -1.02, None) == "normal stop"
    assert lr.label("stop", -1.6, None) == "slipped stop"
    assert lr.label("manual", 2.7, None) == "manual"
    assert lr.label("manual/time", -0.3, "time") == "time exit"
    assert lr.label("manual/time", 0.8, "reversal") == "bot exit"
    assert lr.label("manual/time", 0.1, None) == "closed by bot"


def test_band_narrows_with_more_trades():
    lo10, hi10 = lr.band("XAUUSD", 10)
    lo100, hi100 = lr.band("XAUUSD", 100)
    assert lo10 < lo100 < 0.117 < hi100 < hi10
    assert lr.band("GER40", 10) is None


def test_review_today_gold_case(tmp_path):
    """The 2026-10-14 gold trade: sent at 4140.91, filled 4140.85, closed by hand at +2.71R."""
    p = _db(tmp_path, [
        (1, "XAUUSD", "SELL", "2026-10-14T11:00:22", 4140.85, 4163.35, "CLOSED", "2026-10-14T16:08:58", 4079.85,
         "manual", 61.0, 2.711),
        (2, "BTCUSD", "SELL", "2026-10-14T13:00:23", 82765.5, 83674.2, "OPEN", None, None, None, None, None)],
        [("order_send", {"symbol": "XAUUSD", "retcode": 10009, "request": {"price": 4140.91}}),
         ("order_send", {"symbol": "BTCUSD", "retcode": 10009, "request": {"price": 82763.82}})])
    rep = lr.review(p)
    assert rep["closed"] == 1 and rep["open"] == 1 and rep["kinds"] == {"manual": 1}
    t = rep["trades"][0]
    assert t["kind"] == "manual" and t["entry_slip_r"] == round(-(4140.91 - 4140.85) / 22.5, 3)
    assert rep["by_symbol"]["XAUUSD"]["verdict"].startswith("too early")
    assert any("by hand" in f for f in rep["findings"])
    assert "Loss review" in lr.render(rep)


def test_failed_sends_do_not_shift_the_slippage_match(tmp_path):
    p = _db(tmp_path, [(1, "ETHUSD", "BUY", "a", 100.0, 98.0, "CLOSED", "b", 106.0, "target", 30.0, 3.0)],
            [("order_send", {"symbol": "ETHUSD", "retcode": 10031, "request": {"price": 90.0}}),
             ("order_send", {"symbol": "ETHUSD", "retcode": 10009, "request": {"price": 99.9}})])
    assert lr.review(p)["trades"][0]["entry_slip_r"] == -0.05


def test_market_below_backtest_is_a_finding(tmp_path):
    rows = [(i, "XAUUSD", "BUY", f"2026-10-{i:02d}", 100.0, 98.0, "CLOSED", "x", 98.0, "stop", -10.0, -1.0)
            for i in range(1, 21)]
    rep = lr.review(_db(tmp_path, rows))
    assert rep["by_symbol"]["XAUUSD"]["verdict"].startswith("BELOW")
    assert any(f.startswith("XAUUSD: BELOW") for f in rep["findings"])


def test_bot_time_exit_is_labelled_from_the_close_event(tmp_path):
    p = _db(tmp_path, [(7, "USDJPY", "BUY", "a", 158.0, 157.7, "CLOSED", "b", 158.1, "manual/time", 7.0, 0.33)],
            [("close_send", {"ticket": 7, "reason": "time", "retcode": 10009})])
    assert lr.review(p)["kinds"] == {"time exit": 1}
