import numpy as np
import pandas as pd

from backtest import strategy_round4_intermarket as r4


def _walk(n=3000, seed=1):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    o = np.r_[c[0], c[:-1]] * (1 + rng.normal(0, 0.002, n))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.004, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.004, n)))
    d = pd.bdate_range("2005-01-03", periods=n)
    return pd.DataFrame(dict(open=o, high=h, low=l, close=c), index=d)


def test_zigzag_alternates_and_confirms_after_extreme():
    d = _walk()
    A = r4.r3.atr(d.high.to_numpy(), d.low.to_numpy(), d.close.to_numpy(), 20)
    sw = r4.zigzag(d.high.to_numpy(), d.low.to_numpy(), A)
    assert len(sw) > 50
    assert all(a[2] != b[2] for a, b in zip(sw, sw[1:]))          # high, low, high, ...
    assert all(cf > ix for ix, _, _, cf in sw)                     # known only after the extreme


def test_no_lookahead_patterns_and_pairs():
    d, e = _walk(seed=2), _walk(seed=3)
    cut = 2200
    for fn in r4.PATTERNS.values():
        full = fn(*(d[k].to_numpy() for k in ("open", "high", "low", "close")), d.index)
        part = fn(*(d.iloc[:cut][k].to_numpy() for k in ("open", "high", "low", "close")), d.index[:cut])
        closed_full = [t for t in full if t[1] < cut - 1]
        closed_part = [t for t in part if t[1] < cut - 1]
        assert closed_full == closed_part and len(closed_full) > 5
    sw = ((0.0, 0.0), (0.0, 0.0))
    f = r4.pair_trades(d, e, 1e-4, 1e-4, sw)
    p = r4.pair_trades(d.iloc[:cut], e.iloc[:cut], 1e-4, 1e-4, sw)
    lim = d.index[cut - 2]
    pd.testing.assert_frame_equal(f[f.exit < lim].reset_index(drop=True), p[p.exit < lim].reset_index(drop=True))
    assert len(f) > 20


def test_pair_mean_reverting_spread_wins():
    rng = np.random.default_rng(5)
    n = 2500
    b = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.8 * noise[i - 1] + rng.normal(0, 0.01)        # strongly mean-reverting
    a = b * np.exp(noise)
    idx = pd.bdate_range("2005-01-03", periods=n)
    mk = lambda c: pd.DataFrame(dict(open=c, high=c, low=c, close=c), index=idx)
    t = r4.pair_trades(mk(a), mk(b), 0.0, 0.0, ((0, 0), (0, 0)))
    assert len(t) > 30 and t.R.mean() > 0.1, (len(t), t.R.mean())


def test_dxy_regime_known_next_day_and_direction():
    d = pd.bdate_range("2020-01-01", periods=120)
    dxy = pd.DataFrame(dict(close=np.r_[np.full(60, 100.0), np.linspace(100, 110, 60)]), index=d)
    reg = r4.dxy_regime(dxy)
    assert reg.index[0] == d[49] + pd.Timedelta(days=1)
    assert r4.regime_at(reg, [d[-1]])[0] == 1                      # dollar strong at the end
    assert r4.usd_sign("EURUSD.vx") == 1 and r4.usd_sign("USDJPY.vx") == -1
    trades = [(100, 110, 1, 1.0, 1.0, 1.0), (101, 111, -1, 1.0, 1.0, 1.0)]
    kept = r4.dxy_filtered(lambda *a: trades, reg, 1)(None, None, None, None, d)
    assert kept == [trades[1]]                                     # strong dollar: EURUSD shorts only


def test_filter_test_rule():
    e = pd.date_range("2010-01-01", periods=400, freq="7D", tz="UTC")
    R = np.where(np.arange(400) % 2 == 0, 1.0, -0.5) + np.random.default_rng(0).normal(0, 0.1, 400)
    keep = np.arange(400) % 2 == 0
    r = r4.filter_test(pd.DataFrame(dict(entry=e, R=R)), keep)
    assert r["passes"] and r["kept"] == 200
    assert not r4.filter_test(pd.DataFrame(dict(entry=e, R=R)), ~keep)["passes"]
