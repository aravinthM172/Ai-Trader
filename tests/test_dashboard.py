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
