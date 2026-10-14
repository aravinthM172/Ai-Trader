"""A fill the bot could not find straight after the send must still end up in the trade DB, so its close is
counted (FundedNext 2026-10-14: 4 fills, none recorded, a hand close showed "0 closed")."""
from datetime import datetime, timezone
from types import SimpleNamespace as NS

from execution import live_multi as lm

MAGIC = 26128981


class FakeMT5:
    ORDER_TYPE_BUY, ORDER_TYPE_SELL = 0, 1

    def __init__(self, deals=(), orders=(), loss=-22.5):
        self.deals, self.orders, self.loss = list(deals), list(orders), loss

    def history_deals_get(self, *a, position=None):
        return [d for d in self.deals if position is None or d.position_id == position]

    def history_orders_get(self, position=None):
        return [o for o in self.orders if o.position_id == position]

    def order_calc_profit(self, kind, symbol, volume, price_open, price_close):
        return self.loss


class FakeGW:
    def __init__(self, mt5, positions=(), appear_after=0):
        self.mt5, self.pos, self.calls, self.appear_after = mt5, list(positions), 0, appear_after

    def raw(self):
        return self.mt5

    def positions(self, symbol=None):
        self.calls += 1
        if self.calls <= self.appear_after:                  # not visible yet just after the fill
            return []
        return [p for p in self.pos if symbol is None or p["symbol"] == symbol]

    def to_utc(self, t):
        return datetime.fromtimestamp(t, timezone.utc)


def _pos(ticket=282346588, magic=MAGIC, time=1_791_986_422):
    return dict(ticket=ticket, symbol="XAUUSD", type="SELL", volume=0.01, price_open=4140.85, sl=4163.35,
                tp=4073.6, profit=0.0, swap=0.0, time=time, magic=magic)


def _deal(pid, entry, magic, price, profit=0.0, reason=0, type_=1):
    return NS(position_id=pid, entry=entry, magic=magic, price=price, profit=profit, commission=-0.07, swap=0.0,
              fee=0.0, reason=reason, time=1_792_004_938 if entry else 1_791_986_422, symbol="XAUUSD",
              volume=0.01, type=type_)


def test_find_position_waits_for_a_late_position():
    gw = FakeGW(FakeMT5(), [_pos()], appear_after=3)
    p = lm._find_position(gw, "XAUUSD", MAGIC, 282346588, wait_s=5, step_s=0)
    assert p["ticket"] == 282346588 and gw.calls == 4


def test_find_position_gives_up_after_the_wait():
    assert lm._find_position(FakeGW(FakeMT5(), []), "XAUUSD", MAGIC, wait_s=0, step_s=0) is None


def test_find_position_prefers_the_order_ticket():
    gw = FakeGW(FakeMT5(), [_pos(1, time=200), _pos(2, time=100)])
    assert lm._find_position(gw, "XAUUSD", MAGIC, 2, wait_s=0)["ticket"] == 2
    assert lm._find_position(gw, "XAUUSD", MAGIC, 0, wait_s=0)["ticket"] == 1        # else the newest


def test_open_bot_position_missing_from_db_is_adopted(tmp_path):
    c = lm._conn(tmp_path / "t.sqlite")
    gw = FakeGW(FakeMT5(), [_pos()])
    assert lm._adopt_open(gw, c, gw.positions(), {"XAUUSD": "Metals"}) == 1
    row = c.execute("SELECT symbol, grp, direction, entry, sl, tp, atr, risk_usd, status FROM trades").fetchone()
    assert row == ("XAUUSD", "Metals", "SELL", 4140.85, 4163.35, 4073.6, 11.25, 22.5, "OPEN")
    assert lm._adopt_open(gw, c, gw.positions(), {}) == 0                     # only once


def test_hand_close_of_an_unrecorded_bot_trade_is_counted(tmp_path):
    """The 2026-10-14 case: gold filled, never recorded, then closed by hand (+$61)."""
    lm._NOT_MINE.clear()
    c = lm._conn(tmp_path / "t.sqlite")
    pid = 282346588
    mt5 = FakeMT5(deals=[_deal(pid, 0, MAGIC, 4140.85), _deal(pid, 1, 0, 4079.85, profit=61.0)],
                  orders=[NS(position_id=pid, sl=4163.35, tp=4073.6)])
    lm._sync_closed(FakeGW(mt5), c, {MAGIC})
    row = c.execute("SELECT status, exit_reason, entry, exit, pnl_usd, r_multiple FROM trades WHERE ticket=?",
                    (pid,)).fetchone()
    assert row == ("CLOSED", "manual", 4140.85, 4079.85, 60.93, round(60.93 / 22.5, 3))


def test_hand_trade_the_bot_never_opened_is_left_alone(tmp_path):
    lm._NOT_MINE.clear()
    c = lm._conn(tmp_path / "t.sqlite")
    mt5 = FakeMT5(deals=[_deal(5, 0, 0, 4140.0), _deal(5, 1, 0, 4150.0, profit=10.0)])
    lm._sync_closed(FakeGW(mt5), c, {MAGIC})
    lm._sync_closed(FakeGW(mt5), c, {MAGIC})
    assert c.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 0 and 5 in lm._NOT_MINE
