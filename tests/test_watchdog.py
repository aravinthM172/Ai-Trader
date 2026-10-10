import json
import sqlite3
from datetime import datetime, timedelta, timezone

from tools import watchdog as wd

NOW = datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)
BOT = ("python.exe", "venv\\scripts\\python.exe -u run_live.py --loop 30")
MT5 = ("terminal64.exe", "terminal64.exe")


def _status(path, age_min=0.5, **acct):
    a = dict(balance=500.0, peak_balance=500.0, drawdown_floor=325.0,
             terminal_trade_allowed=True, account_trade_allowed=True, open_positions=[])
    a.update(acct)
    path.write_text(json.dumps({"generated_utc": (NOW - timedelta(minutes=age_min)).isoformat(),
                                "mode": "LIVE", "kill_switch_active": False, "account": a,
                                "closed_trades": 0, "net_pl_usd": 0, "expectancy_R": None}))


def _db(path):
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE trades(ticket INTEGER PRIMARY KEY, direction TEXT, volume REAL, entry REAL, "
              "sl REAL, tp REAL, risk_usd REAL)")
    c.execute("CREATE TABLE events(id INTEGER PRIMARY KEY AUTOINCREMENT, ts_utc TEXT, kind TEXT, payload TEXT)")
    c.commit()
    return c


def _run(tmp, ws, procs=(BOT, MT5), **kw):
    calls = {"restart": 0, "mt5": 0}
    out = wd.check(NOW, ws, procs=list(procs), status_path=tmp / "s.json", db=tmp / "l.sqlite",
                   kill=tmp / "KILL", restart=lambda: calls.__setitem__("restart", calls["restart"] + 1),
                   start_mt5=lambda: calls.__setitem__("mt5", calls["mt5"] + 1) or False,
                   run_edge=lambda enforce: {"enforced": False}, **kw)
    return out, calls


def test_healthy_pass_only_heartbeat(tmp_path):
    _status(tmp_path / "s.json")
    alerts, calls = _run(tmp_path, {})
    assert [a for a in alerts if not a.startswith("daily heartbeat")] == []
    assert calls == {"restart": 0, "mt5": 0}


def test_stale_and_dead_bot_restarts_once(tmp_path, monkeypatch):
    monkeypatch.setenv("WATCHDOG_WATCH_BTC", "true")
    _status(tmp_path / "s.json", age_min=30)
    ws = {}
    alerts, calls = _run(tmp_path, ws, procs=[MT5])
    assert any("NOT UPDATING" in a for a in alerts) and calls["restart"] == 1
    alerts, calls = _run(tmp_path, ws, procs=[MT5])            # same pass again: no duplicate alert/restart
    assert not any("NOT UPDATING" in a for a in alerts) and calls["restart"] == 0


def test_stale_but_process_alive_does_not_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("WATCHDOG_WATCH_BTC", "true")
    _status(tmp_path / "s.json", age_min=30)
    _, calls = _run(tmp_path, {})
    assert calls["restart"] == 0


def test_mt5_down_and_algo_off(tmp_path):
    _status(tmp_path / "s.json", terminal_trade_allowed=False)
    alerts, calls = _run(tmp_path, {}, procs=[BOT])
    assert any("MT5 TERMINAL NOT RUNNING" in a for a in alerts) and calls["mt5"] == 1
    assert any("ALGO TRADING IS OFF" in a for a in alerts)


def test_kill_switch_and_drawdown(tmp_path):
    _status(tmp_path / "s.json", balance=340.0)
    (tmp_path / "KILL").write_text("2026  edge_monitor: test")
    alerts, _ = _run(tmp_path, {})
    assert any("KILL SWITCH ACTIVE" in a and "edge_monitor" in a for a in alerts)
    assert any("DRAWDOWN WARNING" in a for a in alerts)


def test_new_events_and_trades_after_first_run(tmp_path):
    _status(tmp_path / "s.json")
    c = _db(tmp_path / "l.sqlite")
    c.execute("INSERT INTO events(ts_utc,kind,payload) VALUES('t','closed','old')")
    c.commit()
    ws = {}
    alerts, _ = _run(tmp_path, ws)                              # first run: history not replayed
    assert not any("TRADE CLOSED" in a for a in alerts)
    c.execute("INSERT INTO trades VALUES(1,'BUY',0.01,60000,59000,61500,10.0)")
    c.execute("INSERT INTO events(ts_utc,kind,payload) VALUES('t','closed','{\"pnl\": 5}')")
    c.commit()
    c.close()
    alerts, _ = _run(tmp_path, ws)
    assert any("LIVE OPEN #1" in a for a in alerts)
    assert any("TRADE CLOSED" in a for a in alerts)


def test_telegram_kill_command_writes_file(tmp_path, monkeypatch):
    monkeypatch.setattr(wd, "multi_enabled", lambda: False)      # isolate from real status files
    monkeypatch.setattr(wd.challenge_tracker, "enabled", lambda: False)
    _status(tmp_path / "s.json")
    replies = wd.handle_commands(["/kill", "/status", "/resume"], kill=tmp_path / "KILL",
                                 status_path=tmp_path / "s.json")
    assert (tmp_path / "KILL").exists()
    assert len(replies) == 2                                     # /resume deliberately unsupported


def test_report_only_pass_never_acts(tmp_path, monkeypatch):
    monkeypatch.setenv("WATCHDOG_WATCH_BTC", "true")
    _status(tmp_path / "s.json", age_min=30)
    calls = {"restart": 0, "mt5": 0, "enforce": None}

    def edge(enforce):
        calls["enforce"] = enforce
        return {"enforced": False}

    wd.check(NOW, {}, procs=[], status_path=tmp_path / "s.json", db=tmp_path / "l.sqlite",
             kill=tmp_path / "KILL", restart=lambda: calls.__setitem__("restart", 1),
             start_mt5=lambda: calls.__setitem__("mt5", 1) or True, run_edge=edge, allow_actions=False)
    assert calls == {"restart": 0, "mt5": 0, "enforce": False}


def test_multi_checks_stale_restart_and_new_trade(tmp_path):
    st = tmp_path / "m.json"
    st.write_text(json.dumps({"generated_utc": (NOW - timedelta(minutes=30)).isoformat(),
                              "equity": 9300.0, "peak_equity": 10000.0}))
    c = sqlite3.connect(tmp_path / "m.sqlite")
    c.execute("CREATE TABLE trades(ticket INTEGER PRIMARY KEY, symbol TEXT, direction TEXT, volume REAL, entry REAL, "
              "sl REAL, tp REAL, risk_usd REAL)")
    c.execute("CREATE TABLE events(id INTEGER PRIMARY KEY AUTOINCREMENT, ts_utc TEXT, kind TEXT, payload TEXT)")
    c.commit()
    calls = {"restart": 0}
    kw = dict(procs=[MT5], status_path=st, db=tmp_path / "m.sqlite", ref=tmp_path / "none.csv",
              kill=tmp_path / "KILL", restart=lambda: calls.__setitem__("restart", calls["restart"] + 1),
              run_edge=lambda **k: {"enforced": False})
    ws = {}
    alerts = wd.check_multi(NOW, ws, **kw)
    assert any("NOT UPDATING" in a for a in alerts) and calls["restart"] == 1
    assert any("DRAWDOWN 7.0%" in a for a in alerts)
    c.execute("INSERT INTO trades VALUES(5,'NAS100.vx','BUY',0.1,20000,19900,20150,50)")
    c.commit(); c.close()
    alerts = wd.check_multi(NOW, ws, **kw)
    assert any("LIVE OPEN #5 NAS100.vx" in a for a in alerts)
    assert calls["restart"] == 1                                  # no second restart inside the back-off


def test_btc_bot_never_restarted_when_multi_trades_btc(tmp_path, monkeypatch):
    monkeypatch.setenv("MULTI_SYMBOLS", "BTCUSD.vx,XAUUSD.vx,DAX40.vx")
    monkeypatch.delenv("WATCHDOG_WATCH_BTC", raising=False)
    _status(tmp_path / "s.json", age_min=600)                  # BTC-only bot long dead
    alerts, calls = _run(tmp_path, {}, procs=[MT5])
    assert calls["restart"] == 0 and not any("NOT UPDATING" in a for a in alerts)
    monkeypatch.setenv("MULTI_SYMBOLS", "BTCUSD,XAUUSD,GER40")   # FundingPips names, no suffix
    assert not wd.btc_bot_watched()


def test_btc_bot_watched_when_it_is_the_active_bot(tmp_path, monkeypatch):
    monkeypatch.setenv("MULTI_SYMBOLS", "XAUUSD.vx,DAX40.vx")
    _status(tmp_path / "s.json", age_min=600)
    _, calls = _run(tmp_path, {}, procs=[MT5])
    assert calls["restart"] == 1


def test_startup_line_reports_the_multi_trader_when_it_is_the_active_bot(tmp_path, monkeypatch):
    ms = tmp_path / "multi.json"
    ms.write_text('{"mode": "LIVE", "equity": 4985.11, "peak_equity": 5000.0, "open_positions": [1, 2], '
                  '"closed_trades": 14, "net_pl_usd": -9.7, "expectancy_R": -0.05}', encoding="utf-8")
    monkeypatch.setattr(wd, "MULTI_STATUS", ms)
    monkeypatch.setattr(wd, "multi_enabled", lambda: True)
    monkeypatch.setattr(wd, "btc_bot_watched", lambda: False)
    line = wd.startup_line()
    assert line.startswith("[multi] LIVE | equity $4985.11") and "open 2 | closed 14" in line
    monkeypatch.setattr(wd, "multi_enabled", lambda: False)      # BTC-only set-up keeps the old line
    assert "[multi]" not in wd.startup_line()
