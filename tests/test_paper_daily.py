import numpy as np
import pandas as pd

from backtest import strategy_round3_daily as r3
from execution import paper_daily as pdly


def _bars(n=900, seed=7):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
    o = np.r_[c[0], c[:-1]] * (1 + rng.normal(0, 0.002, n))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.006, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.006, n)))
    return o, h, l, c, pd.bdate_range("2020-01-01", periods=n)


def test_state_hook_does_not_change_trades():
    o, h, l, c, d = _bars()
    for name in ("turtle55", "clenow_trend", "connors_rsi2"):
        fn = r3.STRATEGIES[name]
        s = {}
        assert fn(o, h, l, c, d) == fn(o, h, l, c, d, state=s)
        assert "pos" in s


def test_next_action_text():
    assert "ENTER LONG" in pdly.next_action("connors_rsi2", dict(pos=0, enter_next_open=1, risk_if_entered=5.0), 100)
    assert "EXIT" in pdly.next_action("clenow_trend", dict(pos=1, exit_next_open=True, trail_close=90.0), 100)
    assert "buy stop" in pdly.next_action("turtle55", dict(pos=0, buy_stop=110.0, sell_stop=90.0, N=2.0), 100)


def test_sleeves_only_use_round3_strategies():
    for members in pdly.SLEEVES.values():
        for st, sym in members:
            assert st in r3.STRATEGIES and not sym.endswith(".vx")
