import base64
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from dashboard import cloud_app as ca
from dashboard import data as dd


def _df(trend, n=400):
    c = 100 + np.cumsum(np.full(n, trend) + np.random.default_rng(0).normal(0, 0.05, n))
    return pd.DataFrame({"time": pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC"),
                         "open": c, "high": c + 0.3, "low": c - 0.3, "close": c})


def test_readiness_buy_in_uptrend_sell_in_downtrend():
    up, dn = dd.signal_readiness(_df(+0.2)), dd.signal_readiness(_df(-0.2))
    assert up["decision"] == "BUY" and up["missing"] == [] and up["plan"]["sl"] < up["plan"]["entry"] < up["plan"]["tp"]
    assert dn["decision"] == "SELL" and dn["plan"]["tp"] < dn["plan"]["entry"] < dn["plan"]["sl"]


def test_readiness_matches_live_signal_module():
    from strategy import btc_h1_signal
    for trend in (+0.2, -0.2, 0.0):
        df = _df(trend)
        assert dd.signal_readiness(df)["decision"] == btc_h1_signal.generate("X", df).decision


def test_entry_window():
    assert dd.next_entry_window(datetime(2026, 10, 5, 10, 5, tzinfo=timezone.utc))["open_now"]
    w = dd.next_entry_window(datetime(2026, 10, 5, 10, 30, tzinfo=timezone.utc))
    assert not w["open_now"] and w["in_min"] == 30.0


def test_equity_curve():
    closed = [{"closed_utc": "2026-10-06T10:00:00+00:00", "pnl_usd": 50},
              {"closed_utc": "2026-10-05T10:00:00+00:00", "pnl_usd": -20}]
    assert [p["value"] for p in dd.equity_curve(closed, start_balance=5000)] == [4980, 5030]


def test_cloud_auth_helpers():
    assert ca.bearer_ok("Bearer s3cret", "s3cret") and not ca.bearer_ok("Bearer nope", "s3cret")
    assert not ca.bearer_ok("Bearer x", "")                       # no token configured -> always reject
    good = "Basic " + base64.b64encode(b"me:pw").decode()
    assert ca.basic_ok(good, "me", "pw") and not ca.basic_ok(good, "me", "other")
    assert not ca.basic_ok(good, "", "")                          # no credentials configured -> reject


def test_store_and_state_keep_equity_per_account(tmp_path):
    ca.store({"market": {"account": {"login": 1, "equity": 5000, "balance": 5000}}}, data_dir=tmp_path)
    st = ca.state(data_dir=tmp_path)
    assert st["received_utc"] and len(st["equity_history"]) == 1
    ca.store({"market": {"account": {"login": 2, "equity": 99, "balance": 99}}}, data_dir=tmp_path)
    assert [h["login"] for h in ca.state(data_dir=tmp_path)["equity_history"]] == [2]
    assert json.loads((tmp_path / "latest.json").read_text())["market"]["account"]["login"] == 2


def _rules(**over):
    base = dict(challenge={"result": "in_progress", "phase": 1, "target_usd": 5400.0, "phase_start_balance": 5000.0,
                           "profit_pct": 1.0, "trading_days": ["a", "b", "c"], "daily_loss_pct": 0.5,
                           "daily_headroom_usd": 225.0, "total_headroom_usd": 550.0},
                account={"equity": 5050.0, "type": "demo", "algo_terminal": True, "algo_account": True},
                multi_status={"risk_per_trade": 0.005, "symbols": ["BTCUSD.vx", "XAUUSD.vx", "DAX40.vx"]},
                trades=[{"ticket": 1, "status": "OPEN", "risk_usd": 25.0, "grp": "Crypto", "opened_utc": "2026-10-06T10:00:00+00:00"}],
                processes={"live_multi": True, "watchdog": True, "mt5": True, "btc_live": False},
                kill_switch=None, edge=None, news_events=[], now=datetime(2026, 10, 6, 12, tzinfo=timezone.utc))
    base.update(over)
    return {r["rule"]: r["status"] for r in dd.rules_check(**base)}


def test_rules_healthy_state_has_no_failures():
    st = _rules()
    assert "fail" not in st.values()
    assert st["Daily loss (day starts 21:00 UTC)"] == "pass" and st["Minimum trading days"] == "pass" and st["Max open positions"] == "pass"


def test_rules_flag_breaches():
    ch = {"result": "in_progress", "phase": 1, "target_usd": 5400.0, "phase_start_balance": 5000.0, "profit_pct": -5.2,
          "trading_days": [], "daily_loss_pct": 5.2, "daily_headroom_usd": -10, "total_headroom_usd": 240}
    st = _rules(challenge=ch, account={"equity": 4740.0, "type": "demo", "algo_terminal": False, "algo_account": True},
                processes={"live_multi": True, "watchdog": False, "mt5": True, "btc_live": True})
    assert st["Daily loss (day starts 21:00 UTC)"] == "fail" and st["Algo trading enabled"] == "fail"
    assert st["BTC-only bot NOT on the same account"] == "fail" and st["Watchdog running"] == "fail"
    assert st["Max loss (static from start)"] == "pass"          # 5.2 % < 7 % warn line


def test_rules_news_window_and_position_caps():
    trades = [{"ticket": k, "status": "OPEN", "risk_usd": 25.0, "grp": "Metals", "opened_utc": "2026-10-07T18:02:00+00:00"}
              for k in range(7)]
    st = _rules(trades=trades, news_events=[{"time_utc": "2026-10-07T18:00:00+00:00", "event": "FOMC"}])
    assert st["Max open positions"] == "fail" and st["Positions per market group"] == "fail"
    assert st["News window (funded only: ±5 min news / ±10 min speeches; trades opened 5 h+ before are exempt)"] == "info"   # evaluation phase


def test_tick_endpoint_roundtrip(tmp_path, monkeypatch):
    import gzip, threading, urllib.request
    from http.server import ThreadingHTTPServer
    monkeypatch.setenv("DASHBOARD_PUSH_TOKEN", "tok"); monkeypatch.setenv("DASHBOARD_USER", "u"); monkeypatch.setenv("DASHBOARD_PASSWORD", "p")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), ca.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}"
    body = gzip.compress(json.dumps({"symbols": {"BTCUSD.vx": {"bid": 85000.5, "ask": 85030, "time_utc": "2026-10-05T10:00:01+00:00"}}}).encode())
    req = urllib.request.Request(url + "/api/tick", data=body, method="POST",
                                 headers={"Authorization": "Bearer tok", "Content-Encoding": "gzip"})
    assert urllib.request.urlopen(req).status == 200
    got = json.load(urllib.request.urlopen(urllib.request.Request(
        url + "/api/tick", headers={"Authorization": "Basic " + base64.b64encode(b"u:p").decode()})))
    assert got["symbols"]["BTCUSD.vx"]["bid"] == 85000.5 and got["received"] > 0
    srv.shutdown()


def _server(monkeypatch):
    import threading
    from http.server import ThreadingHTTPServer
    monkeypatch.setenv("DASHBOARD_PUSH_TOKEN", "tok"); monkeypatch.setenv("DASHBOARD_USER", "u"); monkeypatch.setenv("DASHBOARD_PASSWORD", "p")
    ca._CMD.clear()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), ca.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}"


def _req(url, *, data=None, basic=False, bearer=False):
    import urllib.request, urllib.error
    h = {"Content-Type": "application/json"}
    if basic:
        h["Authorization"] = "Basic " + base64.b64encode(b"u:p").decode()
    if bearer:
        h["Authorization"] = "Bearer tok"
    r = urllib.request.Request(url, data=None if data is None else json.dumps(data).encode(), headers=h,
                               method="GET" if data is None else "POST")
    try:
        with urllib.request.urlopen(r) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, {}


def test_command_flow_page_to_pc_and_back(monkeypatch):
    srv, url = _server(monkeypatch)
    assert _req(url + "/api/command", data={"action": "stop", "confirm": "NOPE"}, basic=True)[0] == 400
    assert _req(url + "/api/command", data={"action": "stop", "confirm": "STOP"})[0] == 401          # no login
    code, cmd = _req(url + "/api/command", data={"action": "stop", "confirm": "STOP"}, basic=True)
    assert code == 200 and cmd["status"] == "pending"
    assert _req(url + "/api/command/next")[0] == 401                                                   # PC needs token
    _, nxt = _req(url + "/api/command/next", bearer=True)
    assert nxt["id"] == cmd["id"] and nxt["action"] == "stop"
    _req(url + "/api/command/ack", data={"id": cmd["id"], "status": "done", "message": "ok"}, bearer=True)
    _, st = _req(url + "/api/command", basic=True)
    assert st["status"] == "done" and _req(url + "/api/command/next", bearer=True)[1] == {}
    srv.shutdown()


def test_publisher_execute_stop_and_start(tmp_path):
    from dashboard import publisher as pb
    kill, flat = tmp_path / "K", tmp_path / "F"
    st, msg = pb.execute_command({"action": "stop"}, kill=kill, flatten=flat, running=lambda: True)
    assert st == "done" and kill.exists() and "website emergency stop" in kill.read_text() and flat.exists()
    started = []
    st, msg = pb.execute_command({"action": "start"}, kill=kill, flatten=flat, running=lambda: False,
                                 start_task=lambda: started.append(1))
    assert st == "done" and not kill.exists() and started == [1]


def test_publisher_start_refuses_to_override_safety_stop_without_force(tmp_path):
    from dashboard import publisher as pb
    kill = tmp_path / "K"
    kill.write_text("2026-10-06  prop controls: equity 8.6% below initial")
    st, _ = pb.execute_command({"action": "start"}, kill=kill, flatten=tmp_path / "F", running=lambda: True)
    assert st == "refused" and kill.exists()
    st, _ = pb.execute_command({"action": "start", "force": True}, kill=kill, flatten=tmp_path / "F", running=lambda: True)
    assert st == "done" and not kill.exists()


def test_order_blocks_pair_group_and_trend_filter():
    pos = [{"symbol": "BTCUSD", "type": "SELL"}]
    groups = {"BTCUSD": "Crypto", "ETHUSD": "Crypto", "NDX100": "Indices"}
    kw = dict(open_positions=pos, groups=groups)
    assert dd.order_blocks("BTCUSD", "SELL", max_per_group=2, regime=None, **kw) == ["SELL already open (one per pair)"]
    assert dd.order_blocks("ETHUSD", "SELL", max_per_group=2, regime=None, **kw) == []
    assert dd.order_blocks("ETHUSD", "SELL", max_per_group=1, regime=None, **kw) == \
        ["group limit: BTCUSD already open in Crypto (max 1)"]
    up = {"only": "BUY", "why": "daily close 31000 above EMA200 28000"}
    assert dd.order_blocks("NDX100", "SELL", max_per_group=1, regime=up, **kw) == \
        ["trend filter: buys only (daily close 31000 above EMA200 28000)"]
    assert dd.order_blocks("NDX100", "BUY", max_per_group=1, regime=up, **kw) == []
    assert dd.order_blocks("NDX100", "HOLD", max_per_group=1, regime=up, **kw) == []
    assert dd.order_blocks("NDX100", "BUY", max_per_group=1, regime={"only": None, "why": "no daily bars"}, **kw) == \
        ["trend filter: no daily bars"]


def test_add_blocks_uses_group_cap_from_env(monkeypatch):
    monkeypatch.setenv("MULTI_MAX_PER_GROUP", "1")
    mkt = {"symbols": {"BTCUSD": {"group": "Crypto", "readiness": {"decision": "SELL"}},
                       "ETHUSD": {"group": "Crypto", "readiness": {"decision": "SELL"}, "forming": {"decision": "SELL"}}}}
    dd.add_blocks(mkt, [{"symbol": "BTCUSD", "type": "SELL"}])
    assert mkt["symbols"]["ETHUSD"]["blocks"] == ["group limit: BTCUSD already open in Crypto (max 1)"]
    assert mkt["symbols"]["ETHUSD"]["blocks_forming"] == mkt["symbols"]["ETHUSD"]["blocks"]
    assert mkt["symbols"]["BTCUSD"]["blocks"] == ["SELL already open (one per pair)"]

def test_switch_account_password_reaches_pc_once_and_never_the_page(monkeypatch):
    srv, url = _server(monkeypatch)
    body = {"action": "switch_account", "confirm": "SWITCH", "login": "12345678", "server": "FundingPips-Trial",
            "password": "s3cret!", "close_first": True, "account_type": "demo", "account_size": "10000", "phase": "1"}
    assert _req(url + "/api/command", data={**body, "confirm": "STOP"}, basic=True)[0] == 400
    assert _req(url + "/api/command", data={**body, "login": "12ab"}, basic=True)[0] == 400
    assert _req(url + "/api/command", data={**body, "password": ""}, basic=True)[0] == 400
    code, cmd = _req(url + "/api/command", data=body, basic=True)
    assert code == 200 and "password" not in cmd and cmd["login"] == "12345678"
    assert "password" not in _req(url + "/api/command", basic=True)[1]                    # page never sees it
    _, nxt = _req(url + "/api/command/next", bearer=True)
    assert nxt["password"] == "s3cret!" and nxt["close_first"] is True
    assert "password" not in ca._CMD and ca._CMD["status"] == "running"                  # wiped after pickup
    assert _req(url + "/api/command/next", bearer=True)[1] == {}
    srv.shutdown()


def test_unclaimed_switch_password_is_wiped_on_expiry(monkeypatch):
    srv, url = _server(monkeypatch)
    _req(url + "/api/command", data={"action": "switch_account", "confirm": "SWITCH", "login": "123456",
                                     "server": "S", "password": "pw", "account_type": "demo",
                                     "account_size": "5000", "phase": "1"}, basic=True)
    ca._CMD["created"] -= ca.CMD_TTL + 1
    assert _req(url + "/api/command/next", bearer=True)[1] == {}
    assert ca._CMD["status"] == "expired" and "password" not in ca._CMD
    srv.shutdown()


def test_publisher_pause_keeps_trades(tmp_path):
    from dashboard import publisher as pb
    kill, flat = tmp_path / "K", tmp_path / "F"
    st, _ = pb.execute_command({"action": "pause"}, kill=kill, flatten=flat, running=lambda: True)
    assert st == "done" and "website pause" in kill.read_text() and not flat.exists()
    st, _ = pb.execute_command({"action": "start"}, kill=kill, flatten=flat, running=lambda: True)
    assert st == "done" and not kill.exists()                                              # pause is not protective


class _Acct:
    def __init__(self, login, server, balance=5000.0, trade_mode=0):
        self.login, self.server, self.balance, self.currency, self.trade_mode = login, server, balance, "USD", trade_mode


class _Pos:
    def __init__(self, magic):
        self.magic = magic


class FakeMT5:
    def __init__(self, login=111, server="Old-Srv", positions=(), good_password="pw", close_after=None, new_mode=0):
        self.acct, self.positions, self.good, self.logins = _Acct(login, server), list(positions), good_password, []
        self.new_mode = new_mode                            # trade_mode of accounts logged in to with a password
        self.close_after = close_after                      # positions_get calls until the bot's trades are closed

    def initialize(self):
        return True

    def last_error(self):
        return (-6, "Authorization failed")

    def account_info(self):
        return self.acct

    def positions_get(self):
        if self.close_after is not None:
            self.close_after -= 1
            if self.close_after < 0:
                self.positions = []
        return tuple(self.positions)

    def login(self, login, password="", server="", timeout=0):
        self.logins.append((login, password, server))
        if password and password != self.good:
            return False
        self.acct = _Acct(login, server, balance=10000.0, trade_mode=self.new_mode if password else 0)
        return True


def _switch(fake, tmp_path, **cmd):
    from dashboard import account_switch as sw
    calls = []
    st, msg = sw.switch_account({"login": "222", "server": "New-Srv", "password": "pw", "account_type": "demo",
                                 "account_size": "10000", "phase": "1", **cmd},
                                mt5=fake, magics={7}, state=tmp_path, env=tmp_path / ".env", sleep=lambda s: None,
                                stop=lambda: calls.append("stop"), start=lambda: calls.append("start"))
    return st, msg, calls


def test_account_switch_logs_in_archives_state_and_stays_paused(tmp_path):
    for f in ("multi_live.sqlite", "multi_live_state.json", "challenge_state.json", "watchdog_state.json"):
        (tmp_path / f).write_text("old")
    fake = FakeMT5(positions=[_Pos(99)])                     # a manual trade (other magic) does not block
    st, msg, calls = _switch(fake, tmp_path)
    assert st == "done" and "Now on 222" in msg and "press Start" in msg
    assert fake.logins == [(222, "pw", "New-Srv")] and calls == ["stop", "start"]
    arch = [d for d in tmp_path.iterdir() if d.name.startswith("archive_111_")]
    assert len(arch) == 1 and (arch[0] / "multi_live.sqlite").read_text() == "old"
    assert not (tmp_path / "multi_live.sqlite").exists()
    assert "account switch" in (tmp_path / "KILL_SWITCH").read_text()
    saved = json.loads((tmp_path / "account.json").read_text())
    assert saved == {"login": 222, "server": "New-Srv", "switched_utc": saved["switched_utc"], "account_size": 10000, "phase": 1}
    assert "pw" not in (tmp_path / "account.json").read_text()


def test_account_switch_refuses_while_bot_has_trades(tmp_path):
    fake = FakeMT5(positions=[_Pos(7)])
    st, msg, calls = _switch(fake, tmp_path)
    assert st == "refused" and fake.logins == [] and calls == [] and not (tmp_path / "KILL_SWITCH").exists()


def test_account_switch_close_first_flattens_then_switches(tmp_path):
    fake = FakeMT5(positions=[_Pos(7)], close_after=2)
    st, msg, calls = _switch(fake, tmp_path, close_first=True)
    assert st == "done" and calls == ["stop", "start"] and not (tmp_path / "EMERGENCY_FLATTEN").exists()


def test_account_switch_bad_password_goes_back_to_old_account(tmp_path):
    (tmp_path / "multi_live.sqlite").write_text("old")
    fake = FakeMT5(good_password="right")
    st, msg, calls = _switch(fake, tmp_path, password="wrong")
    assert st == "failed" and "Still on 111" in msg and "wrong" not in msg
    assert fake.logins[-1] == (111, "", "Old-Srv") and fake.acct.login == 111
    assert (tmp_path / "multi_live.sqlite").read_text() == "old" and calls == ["stop", "start"]
    assert (tmp_path / "KILL_SWITCH").exists()


def test_account_switch_validates_and_skips_same_account(tmp_path):
    from dashboard import account_switch as sw
    assert sw.validate({"login": "abc", "server": "S", "password": "p"})[3]
    assert sw.validate({"login": "123", "server": "", "password": "p"})[3]
    st, msg, calls = _switch(FakeMT5(login=222, server="New-Srv"), tmp_path)
    assert st == "done" and "Already" in msg and calls == []


def test_switch_to_real_needs_switch_real_confirmation(monkeypatch):
    srv, url = _server(monkeypatch)
    body = {"action": "switch_account", "login": "123456", "server": "S", "password": "pw", "account_type": "real",
            "account_size": "10000", "phase": "1"}
    assert _req(url + "/api/command", data={**body, "confirm": "SWITCH"}, basic=True)[0] == 400
    code, cmd = _req(url + "/api/command", data={**body, "confirm": "SWITCH REAL"}, basic=True)
    assert code == 200 and cmd["account_type"] == "real"
    assert _req(url + "/api/command", data={**body, "account_type": "", "confirm": "SWITCH"}, basic=True)[0] == 400
    srv.shutdown()


def test_account_switch_sets_live_account_mode_and_keeps_other_env_lines(tmp_path):
    env = tmp_path / ".env"
    env.write_text("LIVE_TRADING=true\n# LIVE_ACCOUNT_MODE=demo  (comment stays)\nLIVE_ACCOUNT_MODE=demo\nX=1\n")
    st, msg, _ = _switch(FakeMT5(new_mode=2), tmp_path, account_type="real")
    assert st == "done" and "REAL" in msg
    assert env.read_text().splitlines() == ["LIVE_TRADING=true", "# LIVE_ACCOUNT_MODE=demo  (comment stays)",
                                            "LIVE_ACCOUNT_MODE=real", "X=1"]


def test_account_switch_type_mismatch_is_undone(tmp_path):
    env = tmp_path / ".env"
    env.write_text("LIVE_ACCOUNT_MODE=demo\n")
    (tmp_path / "multi_live.sqlite").write_text("old")
    fake = FakeMT5(new_mode=0)                                # MT5 says DEMO
    st, msg, calls = _switch(fake, tmp_path, account_type="real")
    assert st == "failed" and "MT5 says account 222 is DEMO" in msg and "Still on 111" in msg
    assert fake.acct.login == 111 and env.read_text() == "LIVE_ACCOUNT_MODE=demo\n"
    assert (tmp_path / "multi_live.sqlite").read_text() == "old" and calls == ["stop", "start"]


def test_account_switch_requires_account_type(tmp_path):
    from dashboard import account_switch as sw
    assert "demo or real" in sw.validate({"login": "123", "server": "S", "password": "p"})[3]
    assert sw.validate({"login": "123", "server": "S", "password": "p", "account_type": "demo",
                        "account_size": "10000", "phase": "2"})[3] == ""


# ---- account size + phase on switch (loss limits measured from the bought size) ---------
def test_account_switch_requires_size_and_phase(tmp_path):
    from dashboard import account_switch as sw
    base = {"login": "123", "server": "S", "password": "p", "account_type": "demo"}
    assert "account size" in sw.validate({**base, "phase": "1"})[3]
    assert "account size" in sw.validate({**base, "account_size": "7000", "phase": "1"})[3]
    assert "phase" in sw.validate({**base, "account_size": "5000"})[3]
    assert "phase" in sw.validate({**base, "account_size": "5000", "phase": "3"})[3]


def test_cloud_rejects_switch_without_valid_size(monkeypatch):
    srv, url = _server(monkeypatch)
    body = {"action": "switch_account", "confirm": "SWITCH", "login": "123456", "server": "S", "password": "pw",
            "account_type": "demo", "phase": "1"}
    assert _req(url + "/api/command", data=body, basic=True)[0] == 400
    assert _req(url + "/api/command", data={**body, "account_size": "7000"}, basic=True)[0] == 400
    assert _req(url + "/api/command", data={**body, "account_size": "10000", "phase": "x"}, basic=True)[0] == 400
    code, cmd = _req(url + "/api/command", data={**body, "account_size": "10000"}, basic=True)
    assert code == 200 and cmd["account_size"] == 10000 and cmd["phase"] == "1"
    srv.shutdown()


def test_account_switch_seeds_challenge_from_chosen_size(tmp_path):
    (tmp_path / "challenge_state.json").write_text('{"phase_start_balance": 5003.0}')     # old account's state
    st, msg, _ = _switch(FakeMT5(), tmp_path, phase="2")                                   # MT5 balance 10,000
    assert st == "done" and "$10,000 (phase 2)" in msg
    ch = json.loads((tmp_path / "challenge_state.json").read_text())
    assert ch["phase_start_balance"] == 10000.0 and ch["phase"] == 2 and ch["result"] == "in_progress"
    acct = json.loads((tmp_path / "account.json").read_text())
    assert acct["account_size"] == 10000 and acct["phase"] == 2
    arch = [d for d in tmp_path.iterdir() if d.name.startswith("archive_111_")]
    assert json.loads((arch[0] / "challenge_state.json").read_text())["phase_start_balance"] == 5003.0


def test_account_switch_wrong_size_is_undone(tmp_path):
    env = tmp_path / ".env"
    env.write_text("LIVE_ACCOUNT_MODE=demo\n")
    (tmp_path / "multi_live.sqlite").write_text("old")
    fake = FakeMT5()                                                                       # MT5 balance 10,000
    st, msg, calls = _switch(fake, tmp_path, account_size="5000")                          # 200 % of a $5k size
    assert st == "failed" and "$5,000 account" in msg and "10,000.00" in msg and "Still on 111" in msg
    assert fake.acct.login == 111 and env.read_text() == "LIVE_ACCOUNT_MODE=demo\n"
    assert (tmp_path / "multi_live.sqlite").read_text() == "old" and not (tmp_path / "challenge_state.json").exists()
    assert calls == ["stop", "start"]
