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
from strategy.btc_features import _atr

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
