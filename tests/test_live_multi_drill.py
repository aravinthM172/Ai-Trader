"""Safety drill: the real execution.live_multi.run_once() against a simulated broker, pass after pass.

The pieces (prop_guard, prop_controls, portfolio_guard, _emergency_flatten) have their own unit tests; this file
checks that they are wired together: a loss really closes the trades, a stop really stops the next entry."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

import json
import pandas as pd
import pytest

from execution import live_multi as lm

SYMS = ["BTCUSD", "XAUUSD", "USDJPY"]
T0 = datetime(2026, 10, 6, 10, 5, tzinfo=timezone.utc)          # Tuesday, 5 minutes into the hour


class SimBroker:
    """Just enough of mt5.gateway.MT5Gateway + the MetaTrader5 module for run_once()."""
    TRADE_ACTION_DEAL, TRADE_ACTION_SLTP = 1, 6
    ORDER_TYPE_BUY, ORDER_TYPE_SELL, ORDER_TIME_GTC = 0, 1, 0

    def __init__(self, balance=5000.0):
        self.balance, self.now = balance, T0
        self.pos, self.deals, self.signal = [], [], {}
        self.connected, self.send_retcode, self.close_retcode, self.sltp_retcode = True, 10009, 10009, 10009
        self.strip_sl, self._ticket = False, 100

    # -- gateway ------------------------------------------------------------------
    def __call__(self):                      # run_once() does MT5Gateway()
        return self

    def connect(self): return self.connected
    def shutdown(self): pass
    def raw(self): return self
    def to_utc(self, t): return datetime.fromtimestamp(t, timezone.utc)
    def preferred_filling(self, spec): return 0
    def order_check(self, req): return {"retcode": 0}
    def calc_value_per_unit(self, s, direction, px): return 1.0
    def get_tick(self, s): return {"bid": 999.9, "ask": 1000.1, "spread": 0.2, "age_seconds": 1.0}

    def account_info(self):
        return {"balance": self.balance, "equity": self.balance + sum(p["profit"] for p in self.pos),
                "trade_mode": 0, "currency": "USD"}

    def get_spec(self, s):
        return NS(symbol=s, digits=2, stops_level_price=0.0, value_per_price_unit_per_lot=1.0, volume_min=0.01,
                  volume_step=0.01, volume_max=100.0, currency_profit="USD")

    def positions(self, symbol=None):
        return [dict(p) for p in self.pos if symbol is None or p["symbol"] == symbol]

    def get_rates(self, s, tf, n):           # flat 1000 with a 20-point range -> ATR 20; the last bar is still forming
        t = pd.date_range(end=pd.Timestamp(self.now).floor("1h"), periods=400, freq="h")
        return pd.DataFrame({"time": t, "open": 1000.0, "high": 1010.0, "low": 990.0, "close": 1000.0})

    # -- MetaTrader5 module -------------------------------------------------------
    def symbol_info(self, s): return NS(path=f"group-{s}\\{s}")
    def history_deals_get(self, a, b): return list(self.deals)

    def order_send(self, req):
        if req["action"] == self.TRADE_ACTION_SLTP:
            if self.sltp_retcode == 10009:
                for p in self.pos:
                    if p["ticket"] == req["position"]:
                        p["sl"], p["tp"] = req["sl"], req["tp"]
            return NS(retcode=self.sltp_retcode)
        if "position" in req:                                              # close
            if self.close_retcode != 10009:
                return NS(retcode=self.close_retcode)
            p = next(p for p in self.pos if p["ticket"] == req["position"])
            self.pos.remove(p)
            self.balance += p["profit"]
            self.deals.append(NS(position_id=p["ticket"], entry=1, magic=req["magic"], reason=3, profit=p["profit"],
                                 commission=0.0, swap=0.0, fee=0.0, price=req["price"], time=self.now.timestamp(),
                                 symbol=p["symbol"]))
            return NS(retcode=10009)
        if self.send_retcode != 10009:
            return NS(retcode=self.send_retcode)
        self.open(req["symbol"], "BUY" if req["type"] == self.ORDER_TYPE_BUY else "SELL", volume=req["volume"],
                  sl=0.0 if self.strip_sl else req["sl"], tp=0.0 if self.strip_sl else req["tp"], price=req["price"])
        return NS(retcode=10009)

    # -- test helpers -------------------------------------------------------------
    def open(self, symbol, side="BUY", *, profit=0.0, volume=0.31, sl=960.0, tp=1120.0, price=1000.0, at=None):
        self._ticket += 1
        self.pos.append(dict(ticket=self._ticket, symbol=symbol, type=side, volume=volume, price_open=price, sl=sl, tp=tp,
                             magic=lm.magic_for(symbol), time=(at or self.now).timestamp(), profit=profit))
        return self._ticket

    def float_loss(self, each):
        for p in self.pos:
            p["profit"] = each


@pytest.fixture
def drill(tmp_path, monkeypatch):
    from tools import challenge_tracker as ct
    import mt5.gateway as gateway

    b = SimBroker()
    monkeypatch.setattr(gateway, "MT5Gateway", b)
    for k, v in dict(MULTI_LIVE_TRADING="true", LIVE_TRADING="true", LIVE_ACCOUNT_MODE="demo", KILL_SWITCH="0",
                     MULTI_SYMBOLS=",".join(SYMS), PROP_CHALLENGE="fundingpips_2step_standard").items():
        monkeypatch.setenv(k, v)
    conn = lm._conn
    monkeypatch.setattr(lm, "_conn", lambda path=None: conn(tmp_path / "multi.sqlite"))
    for name, val in dict(STATE=tmp_path, REPORTS=tmp_path, LS=tmp_path / "ls.json", FLATTEN=tmp_path / "EMERGENCY_FLATTEN",
                          RISK_PER_TRADE=0.0025, MAX_RISK_PER_TRADE=0.01, SYMBOL_RISK={}, SYMBOL_MAX_RISK={},
                          REGIME_SYMBOLS=set(), LONG_ONLY_SYMBOLS=set(), STRONG_ONLY_SYMBOLS=set(), ENTRY_WINDOW_S=900,
                          MAX_SPREAD_ATR=0.35, MAX_TICK_AGE_S=120, MAX_HOLD_H=96,
                          GUARD=lm.pg.GuardConfig(daily_loss_limit=0.04, max_drawdown_pause=0.08)).items():
        monkeypatch.setattr(lm, name, val)
    monkeypatch.setattr(lm.safety, "_KILL_FILE", tmp_path / "KILL_SWITCH")
    monkeypatch.setattr(lm, "NewsFilter", lambda **kw: NS(blocking_event=lambda s, now: None, is_blocked=lambda s, now: False))
    monkeypatch.setattr(lm.btc_h1_signal, "generate", lambda s, done: NS(
        decision=b.signal.get(s, "HOLD"), signal_bar_utc=str(done["time"].iloc[-1]), features={}))
    monkeypatch.setattr(ct, "OUT", tmp_path / "challenge.json")
    ct.OUT.write_text(json.dumps({"result": "in_progress", "phase": 1, "phase_start_balance": 5000.0,
                                  "daily_loss_limit": 0.03}))

    def run(now):
        b.now = now
        return lm.run_once(now=now)

    b.run, b.dir = run, tmp_path
    return b


def _all_buy(b):
    b.signal = {s: "BUY" for s in SYMS}


# ---- normal operation ---------------------------------------------------------------------
def test_opens_one_sized_trade_per_signal_with_stop_and_target(drill):
    _all_buy(drill)
    st = drill.run(T0)
    assert st["mode"] == "LIVE" and [p["symbol"] for p in drill.pos] == SYMS
    for p in drill.pos:
        assert p["volume"] == 0.31                                   # $12.50 planned / (2 x ATR 20 x $1) -> 0.3125 -> 0.31
        assert p["sl"] == pytest.approx(p["price_open"] - 40) and p["tp"] == pytest.approx(p["price_open"] + 120)


def test_no_second_trade_on_a_later_pass_or_after_a_restart_that_lost_its_state(drill):
    _all_buy(drill)
    drill.run(T0)
    drill.run(T0 + timedelta(minutes=2))                              # same bar, next pass
    (drill.dir / "ls.json").unlink()                                  # restart with the state file gone
    drill.run(T0 + timedelta(minutes=4))
    drill.run(T0 + timedelta(hours=1))                                # next bar, signals still on
    assert len(drill.pos) == 3


def test_late_start_does_not_chase_the_bar(drill):
    _all_buy(drill)
    st = drill.run(T0 + timedelta(minutes=20))                        # 25 minutes into the hour
    assert drill.pos == [] and "window" in st["entries"]["*"]


# ---- daily loss limit -----------------------------------------------------------------------
def test_daily_loss_closes_everything_blocks_the_day_and_resumes_next_day(drill):
    _all_buy(drill)
    drill.run(T0)
    drill.float_loss(-44.0)                                           # 3 x $44 = 2.64 % of the day's start (limit 3 %)
    st = drill.run(T0 + timedelta(minutes=25))
    assert drill.pos == [] and st["prop"]["flatten_all"] and drill.balance == pytest.approx(5000 - 132)

    st = drill.run(T0 + timedelta(hours=1))                           # next bar, same trading day
    assert drill.pos == [] and st["entries"]["*"].startswith("prop controls")

    st = drill.run(datetime(2026, 10, 6, 21, 5, tzinfo=timezone.utc))  # trading day rolls at 21:00 UTC
    assert len(drill.pos) == 3 and st["prop"]["allow_entries"]


def test_daily_loss_also_closes_while_the_bot_is_paused(drill):
    """Website Pause = kill switch.  It must stop new trades, not the loss protection."""
    _all_buy(drill)
    drill.run(T0)
    (drill.dir / "KILL_SWITCH").write_text("website pause")
    drill.float_loss(-44.0)
    drill.run(T0 + timedelta(minutes=25))
    assert drill.pos == []


def test_one_trade_losing_too_much_is_closed_alone(drill):
    _all_buy(drill)
    drill.run(T0)
    drill.pos[0]["profit"] = -46.0                                    # 0.92 % of the account on one idea
    drill.run(T0 + timedelta(minutes=25))
    assert [p["symbol"] for p in drill.pos] == SYMS[1:]


# ---- total loss limit -----------------------------------------------------------------------
def test_total_loss_closes_everything_and_stops_the_bot_for_good(drill):
    drill.balance = 4600.0
    drill.open("BTCUSD", profit=-15.0)
    drill.open("XAUUSD", profit=-15.0)                                # equity 4570 = 8.6 % below the start
    st = drill.run(T0)
    assert drill.pos == [] and st["prop"]["kill"] and (drill.dir / "KILL_SWITCH").exists()

    _all_buy(drill)
    for now in (T0 + timedelta(hours=1), datetime(2026, 10, 7, 21, 5, tzinfo=timezone.utc)):
        st = drill.run(now)
        assert drill.pos == [] and st["mode"] == "DRY_RUN" and st["entries"]["*"] == "kill switch active"


def test_total_loss_close_that_fails_is_tried_again(drill):
    drill.balance = 4600.0
    drill.open("BTCUSD", profit=-15.0)
    drill.open("XAUUSD", profit=-15.0)
    drill.close_retcode = 10018                                       # market closed
    drill.run(T0)
    assert len(drill.pos) == 2 and (drill.dir / "KILL_SWITCH").exists()
    drill.close_retcode = 10009
    drill.run(T0 + timedelta(minutes=1))
    assert drill.pos == []


def test_no_new_trade_once_seven_percent_down(drill):
    drill.balance = 4640.0                                            # 7.2 % below the start
    _all_buy(drill)
    st = drill.run(T0)
    assert drill.pos == [] and st["entries"]["*"].startswith("prop controls")


# ---- emergency stop -------------------------------------------------------------------------
def test_emergency_stop_closes_everything_even_when_paused(drill):
    _all_buy(drill)
    drill.run(T0)
    (drill.dir / "KILL_SWITCH").write_text("website emergency stop")
    (drill.dir / "EMERGENCY_FLATTEN").write_text("x")
    st = drill.run(T0 + timedelta(minutes=30))
    assert drill.pos == [] and st["emergency_flatten"]["positions"] == 3 and not (drill.dir / "EMERGENCY_FLATTEN").exists()


def test_emergency_stop_keeps_trying_until_the_trades_are_closed(drill):
    _all_buy(drill)
    drill.run(T0)
    (drill.dir / "KILL_SWITCH").write_text("website emergency stop")
    (drill.dir / "EMERGENCY_FLATTEN").write_text("x")
    drill.close_retcode = 10018
    drill.run(T0 + timedelta(minutes=30))
    assert len(drill.pos) == 3 and (drill.dir / "EMERGENCY_FLATTEN").exists()
    drill.close_retcode = 10009
    drill.run(T0 + timedelta(minutes=31))
    assert drill.pos == [] and not (drill.dir / "EMERGENCY_FLATTEN").exists()


# ---- things going wrong ---------------------------------------------------------------------
def test_mt5_down_is_reported_not_a_crash(drill):
    drill.connected = False
    assert drill.run(T0)["error"] == "MT5 connection failed"


def test_order_refused_for_no_connection_is_sent_once_on_the_next_pass(drill):
    drill.signal = {"BTCUSD": "BUY"}
    drill.send_retcode = 10031
    assert drill.run(T0)["entries"]["BTCUSD"]["decision"] == "retry" and drill.pos == []
    drill.send_retcode = 10009
    drill.run(T0 + timedelta(minutes=1))
    drill.run(T0 + timedelta(minutes=2))
    assert len(drill.pos) == 1


def test_trade_filled_without_a_stop_gets_one_or_is_closed(drill):
    drill.signal = {"BTCUSD": "BUY"}
    drill.strip_sl = True                                             # broker drops SL / TP on the fill
    drill.run(T0)
    assert len(drill.pos) == 1 and drill.pos[0]["sl"] > 0 and drill.pos[0]["tp"] > 0      # repaired straight away

    drill.pos.clear()
    drill.sltp_retcode = 10013                                        # ... and refuses to set them afterwards
    st = drill.run(T0 + timedelta(hours=1))
    assert drill.pos == [] and st["entries"]["BTCUSD"]["decision"] == "send_failed"


def test_stop_removed_later_is_put_back(drill):
    drill.signal = {"BTCUSD": "BUY"}
    drill.run(T0)
    drill.signal = {}
    drill.pos[0]["sl"] = 0.0                                          # e.g. removed by hand in MT5
    drill.run(T0 + timedelta(minutes=30))
    assert drill.pos[0]["sl"] == pytest.approx(drill.pos[0]["price_open"] - 40)


def test_trade_is_closed_after_96_hours(drill):
    drill.open("BTCUSD", at=T0 - timedelta(hours=97))
    drill.open("XAUUSD", at=T0 - timedelta(hours=95))
    drill.run(T0)
    assert [p["symbol"] for p in drill.pos] == ["XAUUSD"]
