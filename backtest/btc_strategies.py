"""
PHASE 5 -- a SMALL set of economically sensible BTC strategies.

Each ``build_*`` returns an int8 array ``entries`` where entries[i] in {-1,0,+1}
is the direction to OPEN at bar i, decided ONLY from information at bars <= i-1.
The lab fills at open[i].  No parameter brute force -- a handful of sensible
defaults, plus small neighbourhoods for the robustness phase.

Also: regime_labels() -- trending / ranging / high-vol / low-vol per bar (causal).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.btc_features import _atr, _ema, _rsi


def _sma(x, w):
    return pd.Series(x).rolling(w, min_periods=w).mean().to_numpy()


def _rollmax(x, w):
    return pd.Series(x).shift(1).rolling(w, min_periods=w).max().to_numpy()


def _rollmin(x, w):
    return pd.Series(x).shift(1).rolling(w, min_periods=w).min().to_numpy()


def _adx(h, l, c, period=14):
    n = len(c)
    up = np.diff(h, prepend=h[0])
    dn = -np.diff(l, prepend=l[0])
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    _, tr = _atr(h, l, c, period)
    atr = pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean().to_numpy()
    pdi = 100 * pd.Series(plus_dm).ewm(alpha=1 / period, adjust=False).mean().to_numpy() / np.where(atr > 0, atr, np.nan)
    mdi = 100 * pd.Series(minus_dm).ewm(alpha=1 / period, adjust=False).mean().to_numpy() / np.where(atr > 0, atr, np.nan)
    dx = 100 * np.abs(pdi - mdi) / np.where((pdi + mdi) > 0, pdi + mdi, np.nan)
    adx = pd.Series(dx).ewm(alpha=1 / period, adjust=False).mean().to_numpy()
    return adx, pdi, mdi


def _shift1(a):
    return np.concatenate([[np.nan], a[:-1]])


def warmup_for(*periods) -> int:
    return int(max(periods)) + 10


# ---------------------------------------------------------------------------
# A. TREND FOLLOWING  -- EMA structure + ADX + ATR regime
# ---------------------------------------------------------------------------
def build_trend(c, h, l, *, fast=20, slow=50, trend=200, adx_min=20.0,
                atr_period=14, atr_rank_win=200, atr_rank_max=0.85):
    ef, es, et = _ema(c, fast), _ema(c, slow), _ema(c, trend)
    adx, _, _ = _adx(h, l, c, 14)
    atr, _ = _atr(h, l, c, atr_period)
    atr_rank = pd.Series(atr).rolling(atr_rank_win, min_periods=50).rank(pct=True).to_numpy()
    ef1, es1, et1, adx1, ar1 = _shift1(ef), _shift1(es), _shift1(et), _shift1(adx), _shift1(atr_rank)
    up = (ef1 > es1) & (es1 > et1) & (adx1 >= adx_min) & (ar1 <= atr_rank_max)
    dn = (ef1 < es1) & (es1 < et1) & (adx1 >= adx_min) & (ar1 <= atr_rank_max)
    out = np.zeros(len(c), np.int8)
    out[np.nan_to_num(up.astype(float)) > 0] = 1
    out[np.nan_to_num(dn.astype(float)) > 0] = -1
    return out, warmup_for(trend, atr_rank_win)


# ---------------------------------------------------------------------------
# B. BREAKOUT  -- Donchian channel + volatility expansion + ATR filter
# ---------------------------------------------------------------------------
def build_breakout(c, h, l, *, lookback=48, atr_period=14, expansion=1.3,
                   atr_rank_win=200, atr_rank_min=0.20):
    rhi = _rollmax(h, lookback)
    rlo = _rollmin(l, lookback)
    atr, tr = _atr(h, l, c, atr_period)
    tr_mean = _sma(tr, 20)
    expand = tr / np.where(tr_mean > 0, tr_mean, np.nan)
    atr_rank = pd.Series(atr).rolling(atr_rank_win, min_periods=50).rank(pct=True).to_numpy()
    # decided at i-1: close[i-1] broke the channel measured up to i-2, with expansion
    c1, rhi1, rlo1, ex1, ar1 = _shift1(c), _shift1(rhi), _shift1(rlo), _shift1(expand), _shift1(atr_rank)
    up = (c1 > rhi1) & (ex1 >= expansion) & (ar1 >= atr_rank_min)
    dn = (c1 < rlo1) & (ex1 >= expansion) & (ar1 >= atr_rank_min)
    out = np.zeros(len(c), np.int8)
    out[np.nan_to_num(up.astype(float)) > 0] = 1
    out[np.nan_to_num(dn.astype(float)) > 0] = -1
    return out, warmup_for(lookback, atr_rank_win)


# ---------------------------------------------------------------------------
# C. MOMENTUM  -- RSI + higher-timeframe trend confirmation
# ---------------------------------------------------------------------------
def build_momentum(c, h, l, htf_close_on_grid=None, *, rsi_period=14,
                   rsi_buy=60.0, rsi_sell=40.0, mom_win=12, ema_htf=50):
    rsi = _rsi(c, rsi_period)
    mom = pd.Series(c).diff(mom_win).to_numpy()
    if htf_close_on_grid is not None:
        eh = _ema(htf_close_on_grid, ema_htf)
        htf_up = (htf_close_on_grid > eh)
    else:
        eh = _ema(c, ema_htf * 4)                    # crude HTF proxy on same series
        htf_up = (c > eh)
    r1, m1, hu1 = _shift1(rsi), _shift1(mom), _shift1(htf_up.astype(float))
    up = (r1 >= rsi_buy) & (m1 > 0) & (hu1 > 0)
    dn = (r1 <= rsi_sell) & (m1 < 0) & (hu1 < 0.5)
    out = np.zeros(len(c), np.int8)
    out[np.nan_to_num(up.astype(float)) > 0] = 1
    out[np.nan_to_num(dn.astype(float)) > 0] = -1
    return out, warmup_for(rsi_period, ema_htf * 4, mom_win)


# ---------------------------------------------------------------------------
# D. MEAN REVERSION  -- only inside an explicitly detected ranging regime
# ---------------------------------------------------------------------------
def build_mean_reversion(c, h, l, *, z_win=20, z_entry=2.0, adx_max=18.0,
                         atr_period=14):
    m = _sma(c, z_win)
    sd = pd.Series(c).rolling(z_win, min_periods=z_win).std(ddof=0).to_numpy()
    z = (c - m) / np.where(sd > 0, sd, np.nan)
    adx, _, _ = _adx(h, l, c, 14)
    z1, adx1 = _shift1(z), _shift1(adx)
    ranging = adx1 <= adx_max
    up = ranging & (z1 <= -z_entry)          # stretched down in a range -> buy
    dn = ranging & (z1 >= z_entry)
    out = np.zeros(len(c), np.int8)
    out[np.nan_to_num(up.astype(float)) > 0] = 1
    out[np.nan_to_num(dn.astype(float)) > 0] = -1
    return out, warmup_for(z_win, 30)


# ---------------------------------------------------------------------------
# E. REGIME CLASSIFIER (causal)
# ---------------------------------------------------------------------------
def regime_labels(c, h, l, *, adx_period=14, adx_trend=22.0, atr_period=14,
                  atr_rank_win=250):
    adx, _, _ = _adx(h, l, c, adx_period)
    atr, _ = _atr(h, l, c, atr_period)
    atr_rank = pd.Series(atr).rolling(atr_rank_win, min_periods=60).rank(pct=True).to_numpy()
    adx1, ar1 = _shift1(adx), _shift1(atr_rank)
    trend = np.where(adx1 >= adx_trend, "trending", "ranging")
    vol = np.where(ar1 >= 0.66, "high_vol", np.where(ar1 <= 0.33, "low_vol", "mid_vol"))
    return trend, vol


STRATEGIES = {
    "trend_ema": build_trend,
    "breakout_donchian": build_breakout,
    "momentum_rsi_mtf": build_momentum,
    "mean_reversion_range": build_mean_reversion,
}


def base_atr(h, l, c, period=14):
    a, _ = _atr(h, l, c, period)
    return a
