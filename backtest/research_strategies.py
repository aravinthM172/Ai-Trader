"""
Round-1 research strategies built from the 2026-10 GitHub review
(research/github_algo_trading/ADD_LIST.md).  Kept OUT of btc_strategies.py so the
frozen live strategy's module is untouched.

Same contract as btc_strategies: each builder returns (entries, warmup) where
entries[i] in {-1, 0, +1} is the direction to OPEN at bar i, decided only from
bars <= i-1.  Exits are the engine's (2 ATR stop, 3 ATR target, time exit).

Parameters are fixed from the source idea -- NOT tuned on our data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest import btc_strategies as base
from strategy.btc_features import _atr, _ema, _rsi

_shift1 = base._shift1
warmup_for = base.warmup_for

# frozen params of the live strategy and the two other positive H1 candidates
LIVE_MOMENTUM = dict(rsi_buy=60.0, rsi_sell=40.0, mom_win=8, ema_htf=24)
DONCHIAN = dict(lookback=20, expansion=1.3, atr_rank_min=0.20)
EMA_TREND = dict(fast=20, slow=50, trend=200, adx_min=0.0, atr_rank_max=1.0)


def _to_entries(up, dn, n):
    out = np.zeros(n, np.int8)
    out[np.nan_to_num(np.asarray(up, float)) > 0] = 1
    out[np.nan_to_num(np.asarray(dn, float)) > 0] = -1
    return out


# 1. Multi-horizon time-series momentum (Moskowitz-Ooi-Pedersen 2012; rkohli3/TSMOM).
#    Trade only when 1-day, 3-day and 1-week returns all point the same way.
def build_tsmom_multi(c, h, l, *, horizons=(24, 72, 168)):
    s = pd.Series(c)
    signs = np.vstack([np.sign(_shift1(s.diff(k).to_numpy())) for k in horizons])
    up = np.all(signs > 0, axis=0)
    dn = np.all(signs < 0, axis=0)
    return _to_entries(up, dn, len(c)), warmup_for(max(horizons))


# 2. Consensus of the three positive H1 strategies (ensemble idea): open when at
#    least 2 of {momentum_rsi_mtf, donchian_breakout, ema_trend} agree.
def build_consensus(c, h, l, *, min_votes=2):
    m, wm = base.build_momentum(c, h, l, **LIVE_MOMENTUM)
    d, wd = base.build_breakout(c, h, l, **DONCHIAN)
    e, we = base.build_trend(c, h, l, **EMA_TREND)
    votes = m.astype(int) + d.astype(int) + e.astype(int)
    return _to_entries(votes >= min_votes, votes <= -min_votes, len(c)), max(wm, wd, we)


# 3. Live momentum, skipping the extreme-volatility regime (ATR above its rolling
#    90th percentile) -- prop-firm-monte-carlo regime buckets; DXRG "volatility-blind" finding.
def build_momentum_vol_regime(c, h, l, *, atr_rank_win=500, atr_rank_max=0.90):
    m, wm = base.build_momentum(c, h, l, **LIVE_MOMENTUM)
    atr, _ = _atr(h, l, c, 14)
    rank = _shift1(pd.Series(atr).rolling(atr_rank_win, min_periods=100).rank(pct=True).to_numpy())
    ok = np.nan_to_num(rank, nan=1.0) <= atr_rank_max
    out = np.where(ok, m, 0).astype(np.int8)
    return out, max(wm, warmup_for(atr_rank_win))


# 4. Live momentum with a change-point ("shock") filter -- slow momentum / fast
#    reversion idea (kieranjwood): skip entries right after a 3-bar move larger than
#    3 ATR, when momentum is most likely to reverse.
def build_momentum_no_shock(c, h, l, *, shock_bars=3, shock_atr=3.0):
    m, wm = base.build_momentum(c, h, l, **LIVE_MOMENTUM)
    atr, _ = _atr(h, l, c, 14)
    move = np.abs(pd.Series(c).diff(shock_bars).to_numpy())
    shock = _shift1(move / np.where(atr > 0, atr, np.nan))
    ok = np.nan_to_num(shock, nan=0.0) < shock_atr
    out = np.where(ok, m, 0).astype(np.int8)
    return out, wm


ROUND1 = {
    "tsmom_multi":          (build_tsmom_multi, {}),
    "consensus_2of3":       (build_consensus, {}),
    "momentum_vol_regime":  (build_momentum_vol_regime, {}),
    "momentum_no_shock":    (build_momentum_no_shock, {}),
}


# ---------------------------------------------------------------------------
# Round 2 (2026-10-05): the classic open-source families -- the strategies most often
# shipped in freqtrade / TradingView / Jesse examples and in trend-following papers.
# Parameters are the textbook defaults, NOT tuned on our data.  Exits stay the engine's
# fixed 2 ATR / 3 ATR bracket (DXRG: fixed brackets beat discretionary exits).
# ---------------------------------------------------------------------------
def _supertrend_dir(h, l, c, period=10, mult=3.0):
    """+1 / -1 Supertrend direction at each bar close (standard recursive final bands)."""
    atr, _ = _atr(h, l, c, period)
    hl2 = (h + l) / 2.0
    n = len(c)
    up_b, dn_b = hl2 - mult * atr, hl2 + mult * atr
    fu, fd = np.full(n, np.nan), np.full(n, np.nan)
    d = np.zeros(n, np.int8)
    start = period
    fu[start], fd[start], d[start] = up_b[start], dn_b[start], 1
    for i in range(start + 1, n):
        fu[i] = max(up_b[i], fu[i - 1]) if c[i - 1] > fu[i - 1] else up_b[i]
        fd[i] = min(dn_b[i], fd[i - 1]) if c[i - 1] < fd[i - 1] else dn_b[i]
        d[i] = 1 if c[i] > fd[i - 1] else (-1 if c[i] < fu[i - 1] else d[i - 1])
    return d


# 5. Supertrend(10, 3) flip in the direction of the 200 EMA (TradingView / freqtrade classic).
def build_supertrend_flip(c, h, l, *, period=10, mult=3.0, trend=200):
    d = _supertrend_dir(h, l, c, period, mult).astype(float)
    flip = np.concatenate([[0.0], np.diff(d)])
    e = _ema(c, trend)
    up = _shift1((flip > 0) & (c > e))
    dn = _shift1((flip < 0) & (c < e))
    return _to_entries(up, dn, len(c)), warmup_for(trend)


# 6. Turtle-style 55-bar Donchian breakout confirmed by Supertrend and the 200 EMA
#    (Turtle System 2; Zarattini-Pagani-Barbon 2025 crypto trend recipe, on the H1 grid).
def build_donchian_supertrend(c, h, l, *, lookback=55, period=10, mult=3.0, trend=200):
    hi = pd.Series(h).rolling(lookback).max().shift(1).to_numpy()     # channel up to the previous bar
    lo = pd.Series(l).rolling(lookback).min().shift(1).to_numpy()
    d = _supertrend_dir(h, l, c, period, mult)
    e = _ema(c, trend)
    up = _shift1((c > hi) & (d > 0) & (c > e))
    dn = _shift1((c < lo) & (d < 0) & (c < e))
    return _to_entries(up, dn, len(c)), warmup_for(trend, lookback)


# 7. EMA 20/50 crossover with the 200 EMA filter (freqtrade sample-strategy family).
def build_ema_cross(c, h, l, *, fast=20, slow=50, trend=200):
    ef, es, et = _ema(c, fast), _ema(c, slow), _ema(c, trend)
    diff = np.sign(ef - es)
    cross = np.concatenate([[0.0], np.diff(np.nan_to_num(diff))])
    up = _shift1((cross > 0) & (c > et))
    dn = _shift1((cross < 0) & (c < et))
    return _to_entries(up, dn, len(c)), warmup_for(trend)


# 8. MACD(12, 26, 9) signal-line cross in the direction of the 200 EMA (most-copied TradingView strategy).
def build_macd_trend(c, h, l, *, fast=12, slow=26, signal=9, trend=200):
    macd = _ema(c, fast) - _ema(c, slow)
    sig = pd.Series(macd).ewm(span=signal, adjust=False).mean().to_numpy()
    diff = np.sign(np.nan_to_num(macd - sig))
    cross = np.concatenate([[0.0], np.diff(diff)])
    et = _ema(c, trend)
    up = _shift1((cross > 0) & (macd < 0) & (c > et))     # bullish cross below zero, uptrend
    dn = _shift1((cross < 0) & (macd > 0) & (c < et))
    return _to_entries(up, dn, len(c)), warmup_for(trend, slow + signal)


# 9. TTM-style squeeze breakout: Bollinger(20, 2) inside Keltner(20, 1.5 ATR), then the first
#    close outside the Bollinger band once the squeeze releases (John Carter; LazyBear's script).
def build_squeeze_breakout(c, h, l, *, win=20, bb_k=2.0, kc_k=1.5):
    s = pd.Series(c)
    m = s.rolling(win).mean().to_numpy()
    sd = s.rolling(win).std(ddof=0).to_numpy()
    atr, _ = _atr(h, l, c, win)
    squeeze = (m + bb_k * sd < m + kc_k * atr) & (m - bb_k * sd > m - kc_k * atr)
    was_sq = np.concatenate([[False], squeeze[:-1]])
    release = was_sq & ~squeeze
    up = _shift1(release & (c > m + bb_k * sd))
    dn = _shift1(release & (c < m - bb_k * sd))
    return _to_entries(up, dn, len(c)), warmup_for(win, 30)


# 10. Connors RSI(2) pullback inside the 200 EMA trend (buy dips in uptrends, sell rips in downtrends).
def build_rsi2_pullback(c, h, l, *, rsi_period=2, lo=10.0, hi=90.0, trend=200):
    r = _rsi(c, rsi_period)
    et = _ema(c, trend)
    up = _shift1((r < lo) & (c > et))
    dn = _shift1((r > hi) & (c < et))
    return _to_entries(up, dn, len(c)), warmup_for(trend)


ROUND2 = {
    "supertrend_flip":      (build_supertrend_flip, {}),
    "donchian_supertrend":  (build_donchian_supertrend, {}),
    "ema_cross_200":        (build_ema_cross, {}),
    "macd_trend":           (build_macd_trend, {}),
    "squeeze_breakout":     (build_squeeze_breakout, {}),
    "rsi2_pullback":        (build_rsi2_pullback, {}),
}
