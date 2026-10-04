import numpy as np
import pandas as pd

from backtest import research_gate as rg
from backtest import btc_strategies as strat
from backtest.research_strategies import LIVE_MOMENTUM


def _df(n=3000, seed=0):
    c = 100 + np.cumsum(np.random.default_rng(seed).normal(0, 1, n))
    return pd.DataFrame({"close": c, "high": c + 0.5, "low": c - 0.5})


def leaky(c, h, l):
    """Cheats: trades in the direction of the NEXT bar's move."""
    nxt = np.sign(np.diff(c, append=c[-1]))
    return nxt.astype(np.int8), 10


def test_lookahead_check_rejects_peeking_strategy():
    assert rg.lookahead_ok(leaky, {}, _df()) is False


def test_lookahead_check_accepts_live_strategy():
    assert rg.lookahead_ok(strat.build_momentum, LIVE_MOMENTUM, _df()) is True
