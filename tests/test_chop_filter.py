import numpy as np
import pandas as pd

from backtest import chop_filter_study as cf


def _walk(n=1500, seed=1, drift=0.0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(drift, 0.01, n)))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * 1.002
    l = np.minimum(o, c) * 0.998
    return h, l, c


def test_indicators_have_no_lookahead():
    h, l, c = _walk()
    cut = 1000
    for f in (lambda h, l, c: cf.adx(h, l, c), cf.chop, lambda h, l, c: cf.efficiency_ratio(c)):
        full, part = f(h, l, c), f(h[:cut], l[:cut], c[:cut])
        np.testing.assert_allclose(full[:cut], part, equal_nan=True)


def test_trend_reads_trending_and_noise_reads_choppy():
    ht, lt, ct = _walk(seed=2, drift=0.004)          # strong steady trend
    hn, ln, cn = _walk(seed=2, drift=0.0)
    assert np.nanmean(cf.adx(ht, lt, ct)[100:]) > np.nanmean(cf.adx(hn, ln, cn)[100:])
    assert np.nanmean(cf.chop(ht, lt, ct)[100:]) < np.nanmean(cf.chop(hn, ln, cn)[100:])
    assert np.nanmean(cf.efficiency_ratio(ct)[100:]) > np.nanmean(cf.efficiency_ratio(cn)[100:])
    er = cf.efficiency_ratio(np.arange(50, dtype=float))
    assert np.allclose(er[20:], 1.0)                 # a straight line is perfectly efficient


def test_filter_uses_signal_bar_not_entry_bar():
    h, l, c = _walk()
    df = pd.DataFrame(dict(time=pd.date_range("2020-01-01", periods=len(c), freq="h", tz="UTC"),
                           open=np.r_[c[0], c[:-1]], high=h, low=l, close=c))
    t = cf.momentum_trades(df, 0.0, 3.0)
    assert len(t) > 10
    k = cf.keep_mask(df, t, "er030")
    er = cf.efficiency_ratio(c) >= 0.30
    assert (k == er[t["sig_i"].to_numpy()]).all()
    assert (t["sig_i"] + 1 == df.index[df["time"].isin(t["entry"])]).all()
