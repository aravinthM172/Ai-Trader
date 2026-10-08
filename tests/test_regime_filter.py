from datetime import datetime, timezone

import numpy as np
import pandas as pd

from strategy import regime_filter as rf


def _d1(closes, start="2024-01-01 21:00"):
    t = pd.date_range(start, periods=len(closes), freq="D", tz="UTC")
    c = np.asarray(closes, float)
    return pd.DataFrame({"time": t, "open": c, "high": c, "low": c, "close": c})


def _now_after(d1, hours=1):
    return (d1["time"].iloc[-1] + pd.Timedelta(days=1, hours=hours)).to_pydatetime()


def test_parse_symbols():
    assert rf.parse_symbols(" XAUUSD, BTCUSD ,") == {"XAUUSD", "BTCUSD"}
    assert rf.parse_symbols("") == set()
    assert rf.parse_symbols(None) == set()


def test_uptrend_allows_buy_blocks_sell():
    d1 = _d1(np.linspace(100, 200, 600))
    now = _now_after(d1)
    assert rf.side200("BUY", d1, now)[0] is True
    ok, why = rf.side200("SELL", d1, now)
    assert ok is False and "above" in why


def test_downtrend_allows_sell_blocks_buy():
    d1 = _d1(np.linspace(200, 100, 600))
    now = _now_after(d1)
    assert rf.side200("SELL", d1, now)[0] is True
    assert rf.side200("BUY", d1, now)[0] is False


def test_forming_daily_bar_is_ignored():
    # 599 rising days then a forming bar that crashes far below the EMA: the decision must not see it
    closes = list(np.linspace(100, 200, 599)) + [10.0]
    d1 = _d1(closes)
    now = (d1["time"].iloc[-1] + pd.Timedelta(hours=5)).to_pydatetime()     # last bar still forming
    assert rf.side200("BUY", d1, now)[0] is True
    later = _now_after(d1)                                                  # now it is complete
    assert rf.side200("BUY", d1, later)[0] is False


def test_not_enough_data_is_undecided():
    d1 = _d1(np.linspace(100, 200, 100))
    ok, why = rf.side200("BUY", d1, _now_after(d1))
    assert ok is None and "completed daily bars" in why
    assert rf.side200("BUY", None, datetime.now(timezone.utc))[0] is None
    assert rf.side200("BUY", d1.iloc[:0], datetime.now(timezone.utc))[0] is None
