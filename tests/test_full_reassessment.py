import numpy as np
import pandas as pd

from backtest import full_reassessment as fr
from strategy import btc_h1_signal

COST0 = dict(cost_frac=0.0, swap_long=0.0, swap_short=0.0)


def _walk(n=3000, seed=4):
    rng = np.random.default_rng(seed)
    c = 2000 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    o = np.r_[c[0], c[:-1]] * (1 + rng.normal(0, 0.0005, n))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.002, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.002, n)))
    t = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(dict(time=t, open=o, high=h, low=l, close=c))


def test_entries_match_the_live_signal_builder():
    """Full-series signals == what live_multi gets from btc_h1_signal.generate on its 400-bar window."""
    d = _walk()
    t = fr.simulate(d, 3.0, COST0)
    idx = d.index[d.time.isin(t.entry)]
    idx = [i for i in idx if i >= 600]
    assert len(idx) > 25
    for i in idx[:60]:
        sig = btc_h1_signal.generate("X", d.iloc[i - 400:i].reset_index(drop=True))
        assert sig.decision == ("BUY" if t.set_index("entry").loc[d.time[i], "dir"] > 0 else "SELL")


def test_no_lookahead_and_one_position():
    d = _walk()
    full, part = fr.simulate(d, 3.0, COST0), fr.simulate(d.iloc[:2000], 3.0, COST0)
    cut = d.time[1990]
    pd.testing.assert_frame_equal(full[full.exit < cut].reset_index(drop=True), part[part.exit < cut].reset_index(drop=True))
    assert (full.entry.iloc[1:].to_numpy() >= full.exit.iloc[:-1].to_numpy()).all()


def test_stop_target_R_and_costs():
    d = _walk()
    t = fr.simulate(d, 3.0, COST0)
    assert np.allclose(t.R[t.reason == "stop"].clip(upper=-1.0), t.R[t.reason == "stop"].clip(upper=-1.0))
    assert np.allclose(t.R[t.reason == "target"][t.R[t.reason == "target"] < 2], 1.5)
    assert (t.R[t.reason == "stop"] <= -1.0 + 1e-9).all()
    tc = fr.simulate(d, 3.0, dict(cost_frac=0.001, swap_long=-0.5, swap_short=-0.5))
    assert tc.R.mean() < t.R.mean()


def test_nights_counts_2100_utc_rollovers():
    ts = pd.Timestamp
    assert fr.nights(ts("2026-01-05 10:00", tz="UTC"), ts("2026-01-05 20:00", tz="UTC")) == 0
    assert fr.nights(ts("2026-01-05 20:00", tz="UTC"), ts("2026-01-05 22:00", tz="UTC")) == 1
    assert fr.nights(ts("2026-01-05 10:00", tz="UTC"), ts("2026-01-08 10:00", tz="UTC")) == 3


def test_prop_sim_outcomes():
    days = pd.date_range("2020-01-01", periods=400, freq="D", tz="UTC")
    win = pd.DataFrame(dict(exit=days, R=1.0))
    assert fr.prop_sim(win, 0.01)["outcomes"].get("pass") == 1.0
    lose = pd.DataFrame(dict(exit=days, R=-1.0))
    assert fr.prop_sim(lose, 0.01)["outcomes"].get("max_loss") == 1.0
    crash = pd.DataFrame(dict(exit=days, R=-4.0))
    assert fr.prop_sim(crash, 0.01)["outcomes"].get("daily") == 1.0


def test_no_trade_across_a_data_hole():
    d = _walk(4000)
    d.loc[2000:, "time"] = d.loc[2000:, "time"] + pd.Timedelta(days=400)
    for k in ("open", "high", "low", "close"):
        d.loc[2000:, k] *= 1.5                       # price jump across the hole
    t = fr.simulate(d, 3.0, COST0)
    hole0, hole1 = d.time[1999], d.time[2000]
    assert not ((t.entry <= hole0) & (t.exit >= hole1)).any()
    assert t.R.min() >= -1.5 and len(fr.segments(d)) == 2
