import pandas as pd

from tools import twin_check as tc


def _bars(rows):
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"])


def test_simulate_exit_target_stop_and_stop_first():
    # BUY at 100, ATR 10 -> stop 80, target 130
    assert tc.simulate_exit("BUY", 100, 10, _bars([[100, 105, 95, 101], [101, 131, 99, 130]]))[:2] == ("target", 1.5)
    assert tc.simulate_exit("BUY", 100, 10, _bars([[100, 101, 79, 80]]))[:2] == ("stop", -1.0)
    # both touched inside one bar -> stop first (conservative)
    assert tc.simulate_exit("BUY", 100, 10, _bars([[100, 135, 75, 100]]))[:2] == ("stop", -1.0)
    # SELL mirror
    assert tc.simulate_exit("SELL", 100, 10, _bars([[100, 102, 69, 70]]))[:2] == ("target", 1.5)
    reason, r, held = tc.simulate_exit("SELL", 100, 10, _bars([[100, 101, 99, 100]] * 3))
    assert reason == "open" and held == 3


def test_simulate_exit_time_exit_after_max_hold():
    bars = _bars([[100, 101, 99, 100]] * (tc.MAX_HOLD_H + 2))
    reason, r, held = tc.simulate_exit("BUY", 100, 10, bars)
    assert reason == "time" and held == tc.MAX_HOLD_H and r == 0.0


def test_pair_verdicts():
    bar = "2026-10-05T04:00:00+00:00"
    bt = [{"feed": "BTCUSD", "signal_bar_utc": bar, "direction": "BUY", "exit_reason": "stop", "r": -1.0},
          {"feed": "GER40", "signal_bar_utc": bar, "direction": "BUY", "exit_reason": "target", "r": 1.5}]
    live = [{"feed": "BTCUSD", "symbol": "BTCUSD.vx", "signal_bar_utc": "2026-10-05 04:00:00+00:00",
             "direction": "BUY", "exit_reason": "stop", "r_multiple": -1.01, "account": "valetax"},
            {"feed": "XAUUSD", "symbol": "XAUUSD", "signal_bar_utc": bar, "direction": "SELL",
             "exit_reason": "stop", "r_multiple": -1.0, "account": "fundingpips"},
            {"feed": "XAUUSD", "symbol": "XAUUSD", "signal_bar_utc": "2026-10-05T06:00:00+00:00",
             "direction": "BUY", "exit_reason": "target", "r_multiple": 1.5, "account": "fundingpips"}]
    raw = {"XAUUSD": {"2026-10-05T06:00:00+00:00": "BUY"}}
    v = {(r["symbol"], r["signal_bar_utc"]): r["verdict"] for r in tc.pair(bt, live, raw)}
    assert v[("BTCUSD", bar)] == "match"
    assert v[("GER40", bar)] == "missed"
    assert v[("XAUUSD", bar)] == "extra (no signal)"
    assert v[("XAUUSD", "2026-10-05T06:00:00+00:00")] == "extra (backtest in a trade)"
