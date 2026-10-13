import numpy as np
import pandas as pd

from strategy import reversal_exit as rx


def _bars(last, n=120, price=100.0):
    """Quiet bars 1.0 high-to-low around `price` (ATR about 1), then one last bar (open, high, low, close)."""
    t = pd.date_range("2026-10-01", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame({"time": t, "open": price, "high": price + 0.5, "low": price - 0.5, "close": price})
    df.loc[n - 1, ["open", "high", "low", "close"]] = last
    return df


OPENED = pd.Timestamp("2026-10-03 00:00", tz="UTC").to_pydatetime()


def _buy(last, **kw):
    return rx.check(_bars(last), direction="BUY", entry=96.0, atr_entry=1.0, opened=OPENED, **kw)


def test_big_candle_against_a_winning_buy_closes_it():
    hit, why = _buy((101.0, 101.1, 99.0, 99.2))            # 2.1 tall, down, still 3.2 ATR above the entry
    assert hit and why.startswith("reversal") and "3.2 ATR in profit" in why


def test_not_enough_profit_or_a_small_or_friendly_candle_does_nothing():
    assert _buy((99.0, 99.1, 97.0, 97.2)) == (False, "1.2 ATR in profit (needs 2)")      # big and against, too little profit
    assert _buy((100.4, 100.5, 99.6, 99.7))[1].startswith("candle against is")           # against but small
    assert _buy((99.0, 101.2, 98.9, 101.0)) == (False, "last candle went with the trade")


def test_sell_is_the_mirror_image():
    kw = dict(direction="SELL", entry=104.0, atr_entry=1.0, opened=OPENED)
    assert rx.check(_bars((99.0, 101.0, 98.9, 100.8)), **kw)[0] is True                  # up candle, 3.2 ATR in profit
    assert rx.check(_bars((101.0, 101.1, 99.0, 99.2)), **kw)[0] is False                 # down candle helps a SELL


def test_profit_is_measured_in_the_trades_own_atr():
    last = (101.0, 101.1, 99.0, 99.2)
    assert rx.check(_bars(last), direction="BUY", entry=96.0, atr_entry=2.0, opened=OPENED)[0] is False   # 1.6 ATR up
    assert rx.check(_bars(last), direction="BUY", entry=95.0, atr_entry=2.0, opened=OPENED)[0] is True    # 2.1 ATR up


def test_a_bar_from_before_the_entry_is_ignored():
    late = pd.Timestamp("2026-10-06 00:00", tz="UTC").to_pydatetime()                    # the last bar is 10-05 23:00
    assert rx.check(_bars((101.0, 101.1, 99.0, 99.2)), direction="BUY", entry=96.0, atr_entry=1.0, opened=late) == \
        (False, "no completed bar since the entry")
    same_hour = pd.Timestamp("2026-10-05 23:07", tz="UTC").to_pydatetime()               # opened inside that bar: counts
    assert rx.check(_bars((101.0, 101.1, 99.0, 99.2)), direction="BUY", entry=96.0, atr_entry=1.0, opened=same_hour)[0]


def test_thresholds_are_adjustable_and_bad_input_is_safe():
    last = (101.0, 101.1, 99.0, 99.2)
    assert _buy(last, min_profit_atr=4.0)[0] is False
    assert _buy(last, candle_atr=3.0)[0] is False
    assert rx.check(None, direction="BUY", entry=96.0, atr_entry=1.0, opened=OPENED) == (False, "not enough data")
    assert rx.check(_bars(last, n=20), direction="BUY", entry=96.0, atr_entry=1.0, opened=OPENED)[0] is False
    assert rx.check(_bars(last), direction="BUY", entry=96.0, atr_entry=0.0, opened=OPENED)[0] is False


def test_matches_the_backtest_rule_bar_by_bar():
    """Same decision as scratchpad reversal.py: profit at the close >= 2 x entry ATR, a candle against, range >= 1.5 ATR."""
    from backtest.btc_strategies import base_atr
    rng = np.random.default_rng(5)
    c = 100 + np.cumsum(rng.normal(0.05, 0.6, 600))
    o = np.r_[c[0], c[:-1]]
    h, l = np.maximum(o, c) + np.abs(rng.normal(0, 0.3, 600)), np.minimum(o, c) - np.abs(rng.normal(0, 0.3, 600))
    df = pd.DataFrame({"time": pd.date_range("2026-01-01", periods=600, freq="h", tz="UTC"), "open": o, "high": h, "low": l, "close": c})
    A = base_atr(h, l, c, 14)
    entry, a0, hits = c[200], A[199], 0
    for k in range(300, 600):
        want = (c[k] - entry >= 2 * a0) and (c[k] < o[k]) and (h[k] - l[k] >= 1.5 * A[k])
        got = rx.check(df.iloc[k + 1 - rx.H1_BARS:k + 1].reset_index(drop=True), direction="BUY", entry=entry, atr_entry=a0,
                       opened=df.time[200].to_pydatetime())[0]
        hits += want
        if abs((h[k] - l[k]) / A[k] - 1.5) > 0.03:              # away from the threshold the short window agrees exactly
            assert got == want, k
    assert hits > 0
