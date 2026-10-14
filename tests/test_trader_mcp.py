import json

from mcp_server import trader_mcp as tm


def test_tools_are_read_only_and_listed():
    names = {"health", "bot_status", "recent_trades", "live_vs_backtest", "recent_log", "upcoming_news", "read_report"}
    src = open(tm.__file__, encoding="utf-8").read()
    for bad in ("order_send", "MT5Gateway", "write_text", "KILL.write", "subprocess", "eval(", "exec("):
        assert bad not in src, bad
    assert names <= {n for n in dir(tm)}


def test_health_and_unknown_inputs(tmp_path, monkeypatch):
    st = tmp_path / "s.json"
    st.write_text(json.dumps({"generated_utc": "2026-10-04T10:00:00+00:00", "mode": "DRY_RUN",
                              "account": {"balance": 99.7, "open_positions": []}, "closed_trades": 0}))
    monkeypatch.setitem(tm.BOTS, "btc", dict(tm.BOTS["btc"], status=st))
    h = tm.health()
    assert h["btc"]["mode"] == "DRY_RUN" and h["btc"]["stale"] is True
    assert "error" in tm.bot_status("nope")
    assert "error" in tm.read_report("../../.env")              # not whitelisted


SNAP = {
    "now_utc": "2026-10-14T05:30:00+00:00", "vps_boot_utc": "2026-10-12T07:36:41+00:00",
    "status": {"generated_utc": "2026-10-14T05:29:45+00:00", "mode": "LIVE", "kill_switch_active": False,
               "equity": 15000.0, "balance": 15000.0, "open_positions": [], "closed_trades": 0,
               "entries": {"BTCUSD": "hold"}},
    "challenge": {"rules": "FundedNext Stellar 2-Step", "phase": 1, "daily_headroom_usd": 750.0},
    "kill_switch_file": None,
    "decisions": {"BTCUSD|2026-10-13T20:00:00+00:00": "hold", "XAUUSD|2026-10-14T00:00:00+00:00": "blocked",
                  "BTCUSD|2026-10-14T04:00:00+00:00": "hold"},
    "why": {"XAUUSD|2026-10-14T00:00:00+00:00": "BUY regime: daily close 4136 below EMA200 4307"},
    "settings": {"MULTI_LIVE_TRADING": "true", "MT5_PASSWORD": "<set>"},
    "status_lines": ["[a]", "[b]"], "problem_lines": ["x ERROR y"], "log_tail": ["t"], "watchdog_tail": ["w"],
    "trades": [{"ticket": i} for i in range(5)],
    "processes": [{"pid": 1, "name": "python.exe", "started_utc": "T1", "cmd": "-u -m execution.live_multi --loop 30"},
                  {"pid": 2, "name": "python.exe", "started_utc": "T2", "cmd": "-u -m tools.watchdog --interval 60"},
                  {"pid": 3, "name": "terminal64.exe", "started_utc": "T3", "cmd": "C:/MT5/terminal64.exe"}],
}


def test_live_tools_read_the_vps_snapshot(monkeypatch):
    monkeypatch.setattr(tm.vps, "snapshot", lambda force=False: SNAP)
    h = tm.live_health()
    assert h["mode"] == "LIVE" and h["kill_switch"] is False and h["stale"] is False
    assert h["running"] == {"trader": "T1", "watchdog": "T2", "publisher": None, "mt5": "T3"}
    assert h["challenge"]["rules"] == "FundedNext Stellar 2-Step"
    d = tm.live_decisions(hours=6)                                 # 23:30 cut: drops the 20:00 bar
    assert list(d["bars"]) == ["2026-10-14T00:00:00+00:00", "2026-10-14T04:00:00+00:00"]
    assert d["bars"]["2026-10-14T00:00:00+00:00"] == {"XAUUSD": "blocked: BUY regime: daily close 4136 below EMA200 4307"}
    assert d["bars"]["2026-10-14T04:00:00+00:00"] == {"BTCUSD": "hold"}
    assert [t["ticket"] for t in tm.live_trades(n=2)["trades"]] == [0, 1]
    assert tm.live_settings()["settings"]["MT5_PASSWORD"] == "<set>"
    assert tm.live_log("problems")["lines"] == ["x ERROR y"] and "error" in tm.live_log("nope")
    assert tm.live_challenge()["daily_headroom_usd"] == 750.0


def test_live_tools_pass_errors_through(monkeypatch):
    monkeypatch.setattr(tm.vps, "snapshot", lambda force=False: {"error": "VPS password not set"})
    for f in (tm.live_health, tm.live_decisions, tm.live_trades, tm.live_challenge, tm.live_settings, tm.live_log):
        assert f() == {"error": "VPS password not set"}


def test_vps_password_and_fixed_snapshot_script(tmp_path, monkeypatch):
    import zlib, base64
    from mcp_server import vps
    monkeypatch.delenv("VPS_PW", raising=False)
    monkeypatch.setattr(vps, "ENV_FILE", tmp_path / ".env.vps")
    assert vps.password() is None and "password not set" in vps.snapshot(force=True)["error"]
    vps._cache.update(t=0.0, data=None)
    (tmp_path / ".env.vps").write_text("# comment\nVPS_PW='abc'\n")
    assert vps.password() == "abc"
    monkeypatch.setenv("VPS_PW", "env-wins")
    assert vps.password() == "env-wins"
    code = vps.bootstrap_command()[1]
    assert code.startswith('"') and code.endswith('"')                # one argument for cmd.exe
    blob = code.split("'")[1]
    src = zlib.decompress(base64.b64decode(blob)).decode()
    assert src == vps.SNAPSHOT_SRC.read_text(encoding="utf-8").replace("\r\n", "\n") or "mode=ro" in src
    for bad in ("order_send", "write_text", "MetaTrader5", "unlink", "Stop-Process"):
        assert bad not in src, bad
    assert len(vps.bootstrap_command()[1]) < 7000                 # Windows command-line limit 8191
