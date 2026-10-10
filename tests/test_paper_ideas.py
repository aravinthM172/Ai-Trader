"""execution/paper_ideas.py must trade exactly like the research backtests it forward-tests."""
import numpy as np
import pandas as pd
import pytest

from execution import paper_ideas as pi


def _bars(n=6000, seed=7, start="2025-01-01", price=100.0):
    rng = np.random.default_rng(seed)
    c = price * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    o = np.concatenate([[price], c[:-1]]) * (1 + rng.normal(0, 0.0005, n))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.002, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.002, n)))
    return pd.DataFrame(dict(time=pd.date_range(start, periods=n, freq="h", tz="UTC"), open=o, high=h, low=l, close=c))


def _closed(t):
    return t[t.status == "CLOSED"].reset_index(drop=True)


@pytest.mark.parametrize("trail", [False, True])
def test_breakout_matches_the_backtest(trail):
    r6 = pytest.importorskip("backtest.strategy_round6")          # needs the research data files
    df = _bars()
    mine = _closed(pi.breakout_trades(df, trail))
    ref = r6.breakout_piece(df, dict(cost_frac=0.0, swap_long=0.0, swap_short=0.0), trail)
    ref = ref.iloc[:len(mine)]                                     # the backtest also closes the last, still running trade
    assert len(mine) > 50 and len(ref) == len(mine)
    assert (pd.to_datetime(mine.entry, utc=True).to_numpy() == ref.entry.to_numpy()).all()
    assert np.allclose(mine.R_gross.to_numpy(), ref.R.to_numpy(), atol=1e-9)


def test_gold_asia_matches_the_backtest():
    r7 = pytest.importorskip("backtest.strategy_round7")
    df = _bars(seed=11)
    mine = _closed(pi.gold_asia_trades(df))
    cost = dict(cost_frac=0.0, swap_long=0.0, swap_short=0.0)
    ref = r7.window(df, cost, df.time.dt.tz_convert("America/New_York").dt.hour.to_numpy() == 19, np.ones(len(df), np.int8),
                    df.time.dt.hour.to_numpy() == 7).iloc[:len(mine)]
    assert len(mine) > 100 and len(ref) == len(mine)
    assert (pd.to_datetime(mine.entry, utc=True).to_numpy() == ref.entry.to_numpy()).all()
    assert np.allclose(mine.R_gross.to_numpy(), ref.R.to_numpy(), atol=1e-9)


def test_a_running_trade_is_reported_open_and_never_duplicated(tmp_path, monkeypatch):
    monkeypatch.setattr(pi, "DB", tmp_path / "p.sqlite")
    monkeypatch.setattr(pi, "STATUS", tmp_path / "s.json")
    df = _bars(n=3000, start="2026-09-01")
    monkeypatch.setattr(pi, "PAPER_START", df.time.iloc[1000])

    class Gw:
        def __init__(self, upto): self.upto = upto
        def get_rates(self, s, tf, n): return df.iloc[:self.upto].copy()
        def get_spec(self, s): return type("S", (), dict(spread_points=10.0, point=0.001))()

    a = pi.run_once(Gw(2000), now=(df.time.iloc[1999] + pd.Timedelta(hours=1)).to_pydatetime())
    b = pi.run_once(Gw(3000), now=(df.time.iloc[2999] + pd.Timedelta(hours=1)).to_pydatetime())
    k = "breakout_nextday|BTCUSD"
    assert b["ideas"][k]["closed"] > a["ideas"][k]["closed"] > 0
    # the second pass must contain the first pass's closed trades unchanged
    import sqlite3
    rows = pd.read_sql("SELECT * FROM trades WHERE idea='breakout_nextday' AND symbol='BTCUSD'", sqlite3.connect(pi.DB))
    assert rows.entry.is_unique and (rows.status == "OPEN").sum() <= 1
    assert all(pd.to_datetime(rows.entry, utc=True) >= df.time.iloc[1000])
