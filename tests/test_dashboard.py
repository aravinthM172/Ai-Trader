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
    assert st["Daily loss"] == "pass" and st["Minimum trading days"] == "pass" and st["Max open positions"] == "pass"


def test_rules_flag_breaches():
    ch = {"result": "in_progress", "phase": 1, "target_usd": 5400.0, "phase_start_balance": 5000.0, "profit_pct": -5.2,
          "trading_days": [], "daily_loss_pct": 5.2, "daily_headroom_usd": -10, "total_headroom_usd": 240}
    st = _rules(challenge=ch, account={"equity": 4740.0, "type": "demo", "algo_terminal": False, "algo_account": True},
                processes={"live_multi": True, "watchdog": False, "mt5": True, "btc_live": True})
    assert st["Daily loss"] == "fail" and st["Algo trading enabled"] == "fail"
    assert st["BTC-only bot NOT on the same account"] == "fail" and st["Watchdog running"] == "fail"
    assert st["Max loss (static from start)"] == "pass"          # 5.2 % < 7 % warn line


def test_rules_news_window_and_position_caps():
    trades = [{"ticket": k, "status": "OPEN", "risk_usd": 25.0, "grp": "Metals", "opened_utc": "2026-10-07T18:02:00+00:00"}
              for k in range(7)]
    st = _rules(trades=trades, news_events=[{"time_utc": "2026-10-07T18:00:00+00:00", "event": "FOMC"}])
    assert st["Max open positions"] == "fail" and st["Positions per market group"] == "fail"
    assert st["No trades ±5 min of high-impact news (funded: profit not counted)"] == "warn"


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
