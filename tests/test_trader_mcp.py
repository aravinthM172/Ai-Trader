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
