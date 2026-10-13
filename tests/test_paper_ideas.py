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


ZERO = dict(cost_frac=0.0, swap_long=0.0, swap_short=0.0)


def _same(mine, ref, least):
    ref = ref.iloc[:len(mine)]
    assert len(mine) > least and len(ref) == len(mine)
    assert (pd.to_datetime(mine.entry, utc=True).to_numpy() == ref.entry.to_numpy()).all()
    assert np.allclose(mine.R_gross.to_numpy(), ref.R.to_numpy(), atol=1e-9)


def test_gold_pmfix_matches_the_backtest():
    g8 = pytest.importorskip("backtest.gold_round8")
    df = _bars(seed=13)
    _same(_closed(pi.gold_pmfix_trades(df)), g8.pmfix_long(df, ZERO), 100)


def test_gold_ema_matches_the_backtest():
    src = pytest.importorskip("backtest.gold_sources_retest")
    df = _bars(seed=17)
    _same(_closed(pi.gold_ema_trades(df)), src.ema_cross(df, ZERO, 9, 21, True, 2.0), 50)


def test_gold_fomc_matches_the_backtest(monkeypatch):
    g8 = pytest.importorskip("backtest.gold_round8")
    df = _bars(seed=19)
    days = tuple(str(d.date()) for d in pd.date_range("2025-01-08", "2025-08-30", freq="5D"))
    monkeypatch.setattr(pi, "FOMC_DAYS", days)
    when = pd.DatetimeIndex([pd.Timestamp(d + " 14:00", tz="America/New_York").tz_convert("UTC") for d in days])
    _same(_closed(pi.gold_fomc_trades(df)), g8.news_follow(df, ZERO, when, False), 10)


def test_new_ideas_count_from_their_own_start(tmp_path, monkeypatch):
    monkeypatch.setattr(pi, "DB", tmp_path / "p.sqlite")
    monkeypatch.setattr(pi, "STATUS", tmp_path / "s.json")
    df = _bars(n=3000, start="2026-09-01")
    monkeypatch.setattr(pi, "PAPER_START", df.time.iloc[1000])
    monkeypatch.setattr(pi, "START", {"gold_ema921": df.time.iloc[2000]})
    gw = type("G", (), dict(get_rates=lambda self, s, tf, n: df.copy(),
                            get_spec=lambda self, s: type("S", (), dict(spread_points=10.0, point=0.001))()))()
    pi.run_once(gw, now=(df.time.iloc[-1] + pd.Timedelta(hours=1)).to_pydatetime())
    import sqlite3
    rows = pd.read_sql("SELECT idea, entry FROM trades", sqlite3.connect(pi.DB))
    first = rows.groupby("idea").entry.min().map(lambda x: pd.Timestamp(x, tz="UTC"))
    assert first["gold_ema921"] >= df.time.iloc[2000] and first["gold_asia"] < df.time.iloc[2000]


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


def _daily(n=600, end="2025-01-01", rising=True):
    t = pd.date_range(end=pd.Timestamp(end, tz="UTC") - pd.Timedelta(days=1), periods=n, freq="D")
    c = np.linspace(100, 200, n) if rising else np.linspace(200, 100, n)
    return pd.DataFrame(dict(time=t, close=c))


def test_daily_up_reads_only_completed_days():
    d1 = _daily(600, rising=False)
    d1.loc[len(d1) - 1, "close"] = 500.0                           # the last day closes far above the average
    last = d1.time.iloc[-1]
    when = pd.Series([last + pd.Timedelta(hours=23), last + pd.Timedelta(days=1)])
    assert pi.daily_up(when, d1).tolist() == [False, True]         # known only once that day has ended
    assert not pi.daily_up(when, d1.iloc[-300:]).any()             # too little daily history: no trade
    assert not pi.daily_up(when, None).any()


def test_ndx_tight_follows_its_rules():
    df = _bars(seed=23)
    assert len(pi.ndx_tight_trades(df, _daily(rising=False))) == 0  # daily trend down: no buys
    tr = _closed(pi.ndx_tight_trades(df, _daily()))
    assert len(tr) > 50 and (tr.dir == 1).all()
    c = df.close.to_numpy()
    j = pd.Index(df.time).get_indexer(pd.to_datetime(tr.entry, utc=True)) - 1          # signal bar
    atr = pi._arrays(df)[5]
    assert (pi._rsi(c, 14)[j] >= 65).all() and (c[j] > c[j - 8]).all() and (c[j] > pi._ema(c, 96)[j]).all()
    assert np.allclose(tr.risk_px, atr[j]) and np.allclose(tr.entry_px - tr.stop_px, atr[j])
    assert set(tr.reason) <= {"stop", "target", "time"} and (tr.reason == "target").sum() > 5
    at_open = np.isclose(tr.exit_px, df.open.to_numpy()[pd.Index(df.time).get_indexer(pd.to_datetime(tr.exit, utc=True))])
    hit = tr[(tr.reason == "target") & ~at_open]
    assert len(hit) and np.allclose(hit.R_gross, 3.0)              # target = 3 x the first risk
    assert (tr[tr.reason == "target"].R_gross >= 3.0 - 1e-9).all() and (tr[tr.reason == "stop"].R_gross <= -1.0 + 1e-9).all()
    assert (pd.to_datetime(tr.exit, utc=True) - pd.to_datetime(tr.entry, utc=True) <= pd.Timedelta(hours=96)).all()
