"""
Strategy round 7 -- the full sweep: every standard indicator rule and every documented session / calendar
idea, on the seven requested markets, hourly and daily, on the longest history available.  (2026-10-10)

    python -m backtest.strategy_round7             -> STRATEGY_ROUND7.md, reports/strategy_round7.json
    python -m backtest.strategy_round7 --ledger    same, and record every test in research/ledger.csv

Markets   BTCUSD, ETHUSD, NDX100, XAUUSD, GER40, EURUSD, GBPUSD.
History   H1: Bitstamp (BTC 2014-), FundingPips (ETH 2018-), Dukascopy (gold / EURUSD / GBPUSD 2005-,
          NDX100 2013-, GER40 2014-).  D1: Yahoo, capped at 30 years (NDX100 and GER40 1996-, gold futures
          2000-, EURUSD / GBPUSD 2003-, BTC 2014-, ETH 2017-).  Free hourly data does not go back further.
Costs     FundingPips 1-year median spread + 0.25 x spread slippage per side + FundingPips swap.

PRE-REGISTERED (written before the first run; textbook parameters, nothing tuned):

PART 1  indicator rules (RULES below), each on H1 and on D1.  Signal on a closed bar, only when it is new
        (not true on the bar before), entry at the next open, one position per (rule, market).
        Trend rules          stop 2 ATR, target 6 ATR, time exit 96 bars (H1) / 20 bars (D1).
        Mean-reversion rules stop 2 ATR, target 2 ATR, time exit 24 bars (H1) /  5 bars (D1).
PART 2  session and calendar ideas from the literature, H1, held between fixed clock times with a
        protective stop of 3 ATR (= 1 R), no target:
        fx_local_hours   Breedon & Ranaldo: a currency falls in its own trading hours.  Sell 07:00 UTC,
                         close 12:00 UTC; buy 15:00 UTC, close 21:00 UTC.  (EURUSD, GBPUSD)
        gold_asia_long   gold rises in eastern hours: buy 23:00 UTC, close 07:00 UTC.
        gold_london_short and falls afterwards: sell 08:00 UTC, close 15:00 UTC.
        gold_asia_after_reopen  POST-HOC check of gold_asia_long: buy 19:00 New York (one hour after the daily
                         reopening, when the spread is back to normal), close 07:00 UTC.
        idx_overnight    equity premium is earned overnight: buy at the cash close (16:00 New York /
                         17:30 Frankfurt, next full hour), close at the next cash open (09:00 local hour).
        idx_intraday     the mirror: buy at the first full hour after the open, close at the cash close.
        idx_first_last   Gao et al.: direction of the day up to the end of the first full hour is traded
                         in the last hour of the cash session.
        idx_orb          opening range breakout: range of the first two local cash hours; first close
                         outside it is traded to the cash close, stop at the other side of the range.
        pre_fomc         Lucca & Moench: buy 24 hours before each FOMC decision, close 1 hour before it.
                         (NDX100, GER40, XAUUSD)
        crypto_weekend   buy Friday 21:00 UTC, close Monday 00:00 UTC.  (BTC, ETH)
        crypto_22utc     Quantpedia: buy 22:00 UTC, close 00:00 UTC.  (BTC, ETH)
        turn_of_month    buy at the open of the last trading day of the month, close at the open of the
                         4th trading day of the next month (daily bars; all markets).

PASS RULE (r3.evaluate, unchanged): >= 10 years, >= 100 trades, mean R > 0 on full / first 70 % / last 30 %,
  >= 6 of 8 folds, > 0 at 2x costs, deflated Sharpe >= 0.95 with N = ledger + every test here.
"works" (reported, weaker than a pass): all of that except years_10 and dsr_95, and t >= 2.75.
Read-only; no orders.
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from backtest import fp_strategy_build as fb
from backtest import full_reassessment as fr
from backtest import long_history_backtest as lh
from backtest import strategy_round3_daily as r3
from backtest import strategy_round5 as r5
from backtest import strategy_round6 as r6
from backtest.btc_strategies import base_atr

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MARKETS = ["BTCUSD", "ETHUSD", "NDX100", "XAUUSD", "GER40", "EURUSD", "GBPUSD"]
D1_FILES = {**lh.YAHOO, "GBPUSD": "yahoo_D1/GBPUSD_X.csv", "DJI30": "yahoo_D1/_DJI.csv", "SPX500": "yahoo_D1/_GSPC.csv"}
EXITS = {"H1": dict(T=(2.0, 6.0, 96), M=(2.0, 2.0, 24)), "D1": dict(T=(2.0, 6.0, 20), M=(2.0, 2.0, 5))}
LOCAL = {"NDX100": ("America/New_York", 9, 16), "GER40": ("Europe/Berlin", 9, 17)}      # tz, first cash hour, close hour


# -- data ----------------------------------------------------------------------------------------
def h1(sym: str) -> tuple[pd.DataFrame, str]:
    if sym == "GBPUSD":
        p = DATA / "dukascopy_H1" / "GBPUSD.vx.csv"
        if p.exists():
            return fr._dense_from(fr._read(p)), "Dukascopy"
        return fb.load_h1(sym), "FundingPips"
    df, src = lh.h1(sym)
    if sym == "GER40":                                   # Dukascopy DAX: 1 January 2014 is a broken holiday session
        df = df[df.time >= "2014-01-06"].reset_index(drop=True)
    return df, src


def d1(sym: str) -> pd.DataFrame:
    d = pd.read_csv(DATA / D1_FILES[sym])
    d["time"] = pd.to_datetime(d["date"], utc=True) + pd.Timedelta(hours=21)
    d = d.dropna(subset=["open", "high", "low", "close"])
    d = d[(d.high > d.low) & (d.close > 0)]
    d = d.assign(high=d[["open", "high", "close"]].max(axis=1), low=d[["open", "low", "close"]].min(axis=1))
    rng = d.high - d.low
    d = d[~(rng > 8 * rng.rolling(20, min_periods=5).median().shift(1))]                 # bad ticks
    d = d[d.time >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=30 * 365.25)]
    return d[["time", "open", "high", "low", "close"]].sort_values("time").drop_duplicates("time").reset_index(drop=True)


# -- indicators ----------------------------------------------------------------------------------
def sma(x, n): return pd.Series(x).rolling(n).mean().to_numpy()
def ema(x, n): return pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()
def wilder(x, n): return pd.Series(x).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
def hi(x, n): return pd.Series(x).rolling(n).max().to_numpy()
def lo(x, n): return pd.Series(x).rolling(n).min().to_numpy()
def prev(x, k=1): return np.concatenate([np.full(k, np.nan), x[:-k]])


def rsi(c, n):
    d = np.diff(c, prepend=c[0])
    up, dn = wilder(np.maximum(d, 0), n), wilder(np.maximum(-d, 0), n)
    return 100 - 100 / (1 + up / np.where(dn > 0, dn, np.nan))


def dmi(h, l, c, n=14):
    up, dn = np.diff(h, prepend=h[0]), -np.diff(l, prepend=l[0])
    plus, minus = np.where((up > dn) & (up > 0), up, 0.0), np.where((dn > up) & (dn > 0), dn, 0.0)
    pc = np.concatenate([[c[0]], c[:-1]])
    a = wilder(np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc))), n)
    a = np.where(a > 0, a, np.nan)
    pdi, mdi = 100 * wilder(plus, n) / a, 100 * wilder(minus, n) / a
    return pdi, mdi, wilder(np.nan_to_num(100 * np.abs(pdi - mdi) / (pdi + mdi)), n)


@njit(cache=True)
def _wma(x, n):
    out = np.full(len(x), np.nan)
    den = n * (n + 1) / 2
    for i in range(n - 1, len(x)):
        s = 0.0
        for k in range(n):
            s += x[i - n + 1 + k] * (k + 1)
        out[i] = s / den
    return out


@njit(cache=True)
def _mad(x, n):
    out = np.full(len(x), np.nan)
    for i in range(n - 1, len(x)):
        m = 0.0
        for k in range(i - n + 1, i + 1):
            m += x[k]
        m /= n
        s = 0.0
        for k in range(i - n + 1, i + 1):
            s += abs(x[k] - m)
        out[i] = s / n
    return out


@njit(cache=True)
def _age(x, n, want_max):
    """Bars since the highest (or lowest) value of the last n + 1 bars."""
    out = np.full(len(x), np.nan)
    for i in range(n, len(x)):
        b = i - n
        for k in range(i - n, i + 1):
            if (want_max and x[k] >= x[b]) or (not want_max and x[k] <= x[b]):
                b = k
        out[i] = i - b
    return out


@njit(cache=True)
def _psar(h, l, step=0.02, cap=0.2):
    n = len(h)
    up = np.ones(n, np.bool_)
    sar, ep, af = l[0], h[0], step
    for i in range(1, n):
        sar = sar + af * (ep - sar)
        if up[i - 1]:
            sar = min(sar, l[i - 1], l[i - 2] if i > 1 else l[i - 1])
            if l[i] < sar:
                up[i], sar, ep, af = False, ep, l[i], step
            else:
                up[i] = True
                if h[i] > ep:
                    ep, af = h[i], min(af + step, cap)
        else:
            sar = max(sar, h[i - 1], h[i - 2] if i > 1 else h[i - 1])
            if h[i] > sar:
                up[i], sar, ep, af = True, ep, h[i], step
            else:
                up[i] = False
                if l[i] < ep:
                    ep, af = l[i], min(af + step, cap)
    return up


@njit(cache=True)
def _supertrend(h, l, c, atr, mult=3.0):
    n = len(c)
    up = np.ones(n, np.bool_)
    fu, fl = 0.0, 0.0
    for i in range(n):
        if not (atr[i] > 0):
            continue
        mid = (h[i] + l[i]) / 2
        bu, bl = mid + mult * atr[i], mid - mult * atr[i]
        if i == 0 or fu == 0.0:
            fu, fl = bu, bl
            continue
        fu = bu if (bu < fu or c[i - 1] > fu) else fu
        fl = bl if (bl > fl or c[i - 1] < fl) else fl
        up[i] = True if c[i] > fu else (False if c[i] < fl else up[i - 1])
    return up


@njit(cache=True)
def _kama(c, n=10, fast=2, slow=30):
    out = np.full(len(c), np.nan)
    if len(c) <= n:
        return out
    out[n] = c[n]
    f, s = 2.0 / (fast + 1), 2.0 / (slow + 1)
    for i in range(n + 1, len(c)):
        vol = 0.0
        for k in range(i - n + 1, i + 1):
            vol += abs(c[k] - c[k - 1])
        er = abs(c[i] - c[i - n]) / vol if vol > 0 else 0.0
        sc = (er * (f - s) + s) ** 2
        out[i] = out[i - 1] + sc * (c[i] - out[i - 1])
    return out


@njit(cache=True)
def _heikin(o, h, l, c):
    n = len(c)
    up = np.zeros(n, np.bool_)
    ho = (o[0] + c[0]) / 2
    hc_prev = ho
    for i in range(n):
        hc = (o[i] + h[i] + l[i] + c[i]) / 4
        if i > 0:
            ho = (ho + hc_prev) / 2
        up[i] = hc > ho
        hc_prev = hc
    return up


def cross_up(a, b):
    return (a > b) & (prev(a) <= prev(b))


def rules(o, h, l, c) -> dict:
    """name -> (buy, sell, 'T' trend | 'M' mean reversion).  Every array at bar j uses bars <= j only."""
    A = base_atr(h, l, c, 14)
    e9, e20, e21, e50, e200 = ema(c, 9), ema(c, 20), ema(c, 21), ema(c, 50), ema(c, 200)
    s20, s50, s200 = sma(c, 20), sma(c, 50), sma(c, 200)
    sd20 = pd.Series(c).rolling(20).std(ddof=0).to_numpy()
    r14, r2 = rsi(c, 14), rsi(c, 2)
    pdi, mdi, adx = dmi(h, l, c)
    macd = ema(c, 12) - ema(c, 26)
    sig = ema(macd, 9)
    hh14, ll14 = hi(h, 14), lo(l, 14)
    raw = 100 * (c - ll14) / np.where(hh14 > ll14, hh14 - ll14, np.nan)
    K = sma(raw, 3)
    Dk = sma(K, 3)
    wr = -100 * (hh14 - c) / np.where(hh14 > ll14, hh14 - ll14, np.nan)
    tp = (h + l + c) / 3
    cci = (tp - sma(tp, 20)) / (0.015 * _mad(tp, 20))
    c1, c2, c3, o1, h1_, l1 = prev(c), prev(c, 2), prev(c, 3), prev(o), prev(h), prev(l)
    up_trend, dn_trend = c > e200, c < e200
    R = {}
    X = lambda a, b: (cross_up(a, b), cross_up(b, a))

    # --- trend / breakout
    R["sma_50_200_cross"] = (*X(s50, s200), "T")
    R["ema_20_50_cross"] = (*X(e20, e50), "T")
    R["ema_9_21_cross"] = (*X(e9, e21), "T")
    R["price_sma200_cross"] = (*X(c, s200), "T")
    R["macd_signal_cross"] = (*X(macd, sig), "T")
    R["macd_zero_trend"] = (cross_up(macd, 0 * c) & up_trend, cross_up(0 * c, macd) & dn_trend, "T")
    R["adx_di_cross"] = (cross_up(pdi, mdi) & (adx > 25), cross_up(mdi, pdi) & (adx > 25), "T")
    ps = _psar(h, l)
    R["parabolic_sar_flip"] = (ps, ~ps, "T")
    st = _supertrend(h, l, c, base_atr(h, l, c, 10))
    R["supertrend_flip"] = (st, ~st, "T")
    R["donchian20_breakout"] = (c > prev(hi(h, 20)), c < prev(lo(l, 20)), "T")
    R["donchian55_breakout"] = (c > prev(hi(h, 55)), c < prev(lo(l, 55)), "T")
    R["keltner_breakout"] = (c > e20 + 2 * A, c < e20 - 2 * A, "T")
    bw = sd20 / np.where(s20 > 0, s20, np.nan)
    wide = bw > prev(bw)
    R["bollinger_breakout"] = ((c > s20 + 2 * sd20) & wide, (c < s20 - 2 * sd20) & wide, "T")
    ha = _heikin(o, h, l, c)
    haf = np.nan_to_num(ha.astype(float))
    R["heikin_ashi_3"] = ((haf == 1) & (prev(haf) == 1) & (prev(haf, 2) == 1) & (prev(haf, 3) == 0),
                          (haf == 0) & (prev(haf) == 0) & (prev(haf, 2) == 0) & (prev(haf, 3) == 1), "T")
    ten, kij = (hi(h, 9) + lo(l, 9)) / 2, (hi(h, 26) + lo(l, 26)) / 2
    R["ichimoku_tk_cross"] = (cross_up(ten, kij) & (c > kij), cross_up(kij, ten) & (c < kij), "T")
    au, ad = 100 * (25 - _age(h, 25, True)) / 25, 100 * (25 - _age(l, 25, False)) / 25
    R["aroon_cross"] = (cross_up(au, ad) & (au > 70), cross_up(ad, au) & (ad > 70), "T")
    roc = c / prev(c, 20) - 1
    R["roc20_zero_cross"] = (*X(roc, 0 * c), "T")
    t3 = ema(ema(ema(np.log(c), 15), 15), 15)
    trix = t3 - prev(t3)
    R["trix_zero_cross"] = (*X(trix, 0 * c), "T")
    hma = _wma(2 * np.nan_to_num(_wma(c, 27)) - np.nan_to_num(_wma(c, 55)), 7)
    hma[:70] = np.nan
    hs = hma - prev(hma)
    R["hull_ma_turn"] = (*X(hs, 0 * c), "T")
    kama = _kama(c)
    R["kama_cross"] = (*X(c, kama), "T")
    idx = pd.Series(np.arange(len(c), dtype=float))
    slope = (pd.Series(c).rolling(50).cov(idx) / idx.rolling(50).var()).to_numpy()
    R["linreg_slope_turn"] = (*X(slope, 0 * c), "T")
    tr = np.maximum(h - l, np.maximum(np.abs(h - c1), np.abs(l - c1)))
    trs = pd.Series(tr).rolling(14).sum().to_numpy()
    vp = pd.Series(np.abs(h - l1)).rolling(14).sum().to_numpy() / trs
    vm = pd.Series(np.abs(l - h1_)).rolling(14).sum().to_numpy() / trs
    R["vortex_cross"] = (*X(vp, vm), "T")
    R["cci_100_breakout"] = (cross_up(cci, 0 * c + 100), cross_up(0 * c - 100, cci), "T")
    R["rsi_50_cross"] = (*X(r14, 0 * c + 50), "T")
    mom = c - prev(c, 8)
    R["live_momentum_rsi60"] = ((r14 >= 60) & (mom > 0) & (c > ema(c, 96)), (r14 <= 40) & (mom < 0) & (c < ema(c, 96)), "T")
    inside = (h1_ < prev(h, 2)) & (l1 > prev(l, 2))
    R["inside_bar_breakout"] = (inside & (c > h1_), inside & (c < l1), "T")
    R["engulfing_trend"] = ((c > o) & (c1 < o1) & (c >= o1) & (o <= c1) & up_trend,
                            (c < o) & (c1 > o1) & (c <= o1) & (o >= c1) & dn_trend, "T")
    gap = o - c1
    R["gap_continuation"] = ((gap > 0.5 * A) & (c > o), (gap < -0.5 * A) & (c < o), "T")
    hh50, ll50 = hi(h, 50), lo(l, 50)
    R["fib_618_pullback"] = (up_trend & (l <= hh50 - 0.618 * (hh50 - ll50)) & (c > hh50 - 0.618 * (hh50 - ll50)) & (c > o),
                             dn_trend & (h >= ll50 + 0.618 * (hh50 - ll50)) & (c < ll50 + 0.618 * (hh50 - ll50)) & (c < o), "T")
    R["ema20_pullback"] = ((c > e50) & (e50 > e200) & (l <= e20) & (c > e20) & (c > o),
                           (c < e50) & (e50 < e200) & (h >= e20) & (c < e20) & (c < o), "T")
    # --- mean reversion
    R["rsi14_30_70"] = (r14 < 30, r14 > 70, "M")
    R["rsi2_trend_dip"] = ((r2 < 10) & (c > s200), (r2 > 90) & (c < s200), "M")
    R["stochastic_cross"] = (cross_up(K, Dk) & (K < 20), cross_up(Dk, K) & (K > 80), "M")
    R["williams_r"] = (wr < -90, wr > -10, "M")
    R["cci_100_reversal"] = (cross_up(cci, 0 * c - 100), cross_up(0 * c + 100, cci), "M")
    R["bollinger_reentry"] = ((c1 < prev(s20 - 2 * sd20)) & (c > s20 - 2 * sd20), (c1 > prev(s20 + 2 * sd20)) & (c < s20 + 2 * sd20), "M")
    z = (c - s20) / np.where(sd20 > 0, sd20, np.nan)
    R["zscore_2"] = (z < -2, z > 2, "M")
    R["keltner_fade"] = (c < e20 - 2 * A, c > e20 + 2 * A, "M")
    R["three_in_a_row"] = ((c < c1) & (c1 < c2) & (c2 < c3), (c > c1) & (c1 > c2) & (c2 > c3), "M")
    rng = h - l
    ibs = np.where(rng > 0, (c - l) / np.where(rng > 0, rng, 1), 0.5)
    R["ibs_extreme"] = (ibs < 0.2, ibs > 0.8, "M")
    dist = (c - s50) / np.where(A > 0, A, np.nan)
    R["stretch_from_sma50"] = (dist < -3, dist > 3, "M")
    body = np.abs(c - o)
    lw, uw = np.minimum(o, c) - l, h - np.maximum(o, c)
    R["pin_bar_extreme"] = ((lw >= 2 * body) & (uw <= body) & (l <= lo(l, 20)) & (rng > 0),
                            (uw >= 2 * body) & (lw <= body) & (h >= hi(h, 20)) & (rng > 0), "M")
    R["rsi_divergence"] = ((c <= lo(c, 20)) & (r14 > lo(r14, 20) + 5) & (r14 < 40), (c >= hi(c, 20)) & (r14 < hi(r14, 20) - 5) & (r14 > 60), "M")
    return R


def signals(p: pd.DataFrame, tf: str) -> dict:
    o, h, l, c = (p[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    A = base_atr(h, l, c, 14)
    out = {}
    for name, (buy, sell, kind) in rules(o, h, l, c).items():
        buy, sell = np.nan_to_num(buy.astype(float)) > 0, np.nan_to_num(sell.astype(float)) > 0
        buy, sell = buy & ~np.concatenate([[False], buy[:-1]]), sell & ~np.concatenate([[False], sell[:-1]])
        sl, tp, hold = EXITS[tf][kind]
        out[name] = dict(r5._put(r5._sig(len(c)), buy, sell, c - sl * A, c + sl * A, c + tp * A, c - tp * A), hold=hold)
    return out


RULE_KIND = {k: v[2] for k, v in rules(*(np.linspace(1, 2, 400) + x for x in (0.0, 0.01, -0.01, 0.0))).items()}


def _cat(parts: list) -> pd.DataFrame:
    """Join trade tables, ignoring empty ones (an empty table would turn the time columns into objects)."""
    full = [x for x in parts if len(x)]
    return pd.concat(full, ignore_index=True) if full else parts[0]


def sweep(df: pd.DataFrame, cost: dict, tf: str) -> dict:
    parts = {k: [] for k in RULE_KIND}
    pieces = fr.segments(df) if tf == "H1" else [df]
    for p in pieces:
        s = signals(p, tf)
        for k in RULE_KIND:
            parts[k].append(r5.simulate_piece(p, s[k], cost))
    return {k: _cat(v) for k, v in parts.items()}


def lookahead_ok(p: pd.DataFrame, cut: int = 12000) -> list:
    a, b = signals(p.iloc[:cut].reset_index(drop=True), "H1"), signals(p.iloc[:cut + 2000].reset_index(drop=True), "H1")
    bad = []
    for k in RULE_KIND:
        m = slice(600, cut - 5)
        if not (np.array_equal(a[k]["d"][m], b[k]["d"][m]) and np.allclose(a[k]["stop"][m], b[k]["stop"][m], equal_nan=True)):
            bad.append(k)
    return bad


# -- PART 2: session and calendar ideas ----------------------------------------------------------
def window(p: pd.DataFrame, cost: dict, enter: np.ndarray, d: np.ndarray, leave: np.ndarray, hold: int = 120) -> pd.DataFrame:
    """enter[i]: open a trade at the open of bar i in direction d[i]; leave[k]: close at the open of bar k."""
    h, l, c = (p[k].to_numpy(float) for k in ("high", "low", "close"))
    A, n = base_atr(h, l, c, 14), len(c)
    sig_d = np.zeros(n, np.int8)
    sig_d[:-1] = np.where(enter[1:], d[1:], 0)                       # the order is placed on the bar before
    s = r5._sig(n)
    s["d"] = sig_d
    s["stop"] = np.where(sig_d > 0, c - 3 * A, np.where(sig_d < 0, c + 3 * A, np.nan))
    s["tgt"] = np.where(sig_d > 0, np.inf, np.where(sig_d < 0, -np.inf, np.nan))
    return r5.simulate_piece(p, dict(s, hold=hold, flat=leave), cost)


def session_ideas(sym: str, df: pd.DataFrame, cost: dict, fomc: pd.DatetimeIndex) -> dict:
    out = {}

    def run(name, f):
        parts = [f(p) for p in fr.segments(df)]
        out[name] = _cat(parts)

    def clock(p, tz=None):
        t = p["time"] if tz is None else p["time"].dt.tz_convert(tz)
        return t.dt.hour.to_numpy(), t.dt.dayofweek.to_numpy()

    one = lambda n, v=1: np.full(n, v, np.int8)
    if sym in ("EURUSD", "GBPUSD"):
        def f(p):
            hr, _ = clock(p)
            return window(p, cost, (hr == 7) | (hr == 15), np.where(hr == 7, -1, 1).astype(np.int8), (hr == 12) | (hr == 21))
        run("fx_local_hours", f)
    if sym == "XAUUSD":
        run("gold_asia_long", lambda p: window(p, cost, clock(p)[0] == 23, one(len(p)), clock(p)[0] == 7))
        run("gold_london_short", lambda p: window(p, cost, clock(p)[0] == 8, one(len(p), -1), clock(p)[0] == 15))
        # POST-HOC (added after the first run, counted in N): gold reopens at 18:00 New York with a wide spread, and
        # bid candles drift up as it narrows -- 23:00 UTC is that reopening hour in winter.  Enter one hour later.
        run("gold_asia_after_reopen", lambda p: window(p, cost, clock(p, "America/New_York")[0] == 19, one(len(p)), clock(p)[0] == 7))
    if sym in LOCAL:
        tz, op, cl = LOCAL[sym]
        run("idx_overnight", lambda p: window(p, cost, clock(p, tz)[0] == cl, one(len(p)), clock(p, tz)[0] == op))
        run("idx_intraday", lambda p: window(p, cost, clock(p, tz)[0] == op + 1, one(len(p)), clock(p, tz)[0] == cl))

        def first_last(p):
            hr, _ = clock(p, tz)
            c = p["close"].to_numpy(float)
            day = p["time"].dt.tz_convert(tz).dt.normalize()
            prev_close = pd.Series(np.where(hr == cl - 1, c, np.nan)).ffill().shift(1).to_numpy()       # yesterday's cash close
            first = pd.Series(np.where(hr == op + 1, np.sign(c - prev_close), np.nan)).groupby(day.to_numpy()).ffill().to_numpy()
            return window(p, cost, (hr == cl - 1) & np.isfinite(first) & (first != 0), np.nan_to_num(first).astype(np.int8), hr == cl)
        run("idx_first_last", first_last)

        def orb(p):
            hr, _ = clock(p, tz)
            hi_, lo_, c = p["high"].to_numpy(float), p["low"].to_numpy(float), p["close"].to_numpy(float)
            day = p["time"].dt.tz_convert(tz).dt.normalize().to_numpy()
            inr = (hr == op) | (hr == op + 1)
            g = pd.DataFrame(dict(h=np.where(inr, hi_, np.nan), l=np.where(inr, lo_, np.nan))).groupby(day, sort=False)
            rh, rl = g.h.transform("max").to_numpy(), g.l.transform("min").to_numpy()     # read after the range hours only
            win = (hr >= op + 2) & (hr < cl - 1)
            brk = r5._first_of_group(win & ((c > rh) | (c < rl)), day)
            s = r5._put(r5._sig(len(c)), brk & (c > rh), brk & (c < rl), rl, rh, np.full(len(c), np.inf), np.full(len(c), -np.inf))
            return r5.simulate_piece(p, dict(s, hold=12, flat=hr == cl), cost)
        run("idx_orb", orb)
    if sym in ("NDX100", "GER40", "XAUUSD"):
        def f(p):
            t = pd.DatetimeIndex(p["time"])
            return window(p, cost, t.isin(fomc - pd.Timedelta(hours=24)), one(len(p)), t.isin(fomc - pd.Timedelta(hours=1)), hold=40)
        run("pre_fomc", f)
    if sym in ("BTCUSD", "ETHUSD"):
        def f(p):
            hr, dow = clock(p)
            return window(p, cost, (dow == 4) & (hr == 21), one(len(p)), (dow == 0) & (hr == 0))
        run("crypto_weekend", f)
        run("crypto_22utc", lambda p: window(p, cost, clock(p)[0] == 22, one(len(p)), clock(p)[0] == 0))
    return out


def turn_of_month(d: pd.DataFrame, cost: dict) -> pd.DataFrame:
    t = d["time"]
    ym = (t.dt.year * 12 + t.dt.month).to_numpy()
    last = np.concatenate([ym[1:] != ym[:-1], [False]])
    nth = pd.Series(np.ones(len(d))).groupby(ym).cumsum().to_numpy()
    return window(d, cost, last, np.ones(len(d), np.int8), nth == 4, hold=10)


# -- run ------------------------------------------------------------------------------------------
def verdict(r: dict) -> str:
    if "checks" not in r:
        return "no"
    if r["passes"]:
        return "PASS"
    if r["lead"] and r["checks"]["trades_100"]:
        return "works"
    return "weak" if r["mean_R"] > 0 and (r["t"] or 0) >= 1.5 else "no"


def judge(t: pd.DataFrame, years: float, n_trials: int) -> dict:
    r = r5.judge(t, years, n_trials)
    r["verdict"] = verdict(r)
    if "checks" in r:
        r["R_per_year"] = round(float(t.R.sum()) / years, 1)
        eq = t.sort_values("exit").R.cumsum()
        r["max_dd_R"] = round(float((eq.cummax() - eq).max()), 1)
        r["long_R"] = round(float(t.R[t.dir > 0].mean()), 4) if (t.dir > 0).any() else None
        r["short_R"] = round(float(t.R[t.dir < 0].mean()), 4) if (t.dir < 0).any() else None
    return r


def main() -> int:
    warnings.filterwarnings("ignore")
    ev = pd.read_csv(DATA / "events_history.csv")
    fomc = pd.DatetimeIndex(pd.to_datetime(ev[ev.event == "FOMC"].time_utc, utc=True)).floor("h")
    n_sessions = 2 + 3 + 2 * 4 + 3 + 4 + len(MARKETS)
    n_trials = r3.ledger_n() + r3.PRIOR_TRIALS + 2 * len(RULE_KIND) * len(MARKETS) + n_sessions
    res, meta, bad = {}, {}, None
    for s in MARKETS:
        cost = r6.cost(s)
        df, src = h1(s)
        yrs = r6.years_of(df)
        dd = d1(s)
        yd = (dd.time.iloc[-1] - dd.time.iloc[0]).days / 365.25
        meta[s] = dict(h1_source=src, h1_from=str(df.time.iloc[0].date()), h1_years=round(yrs, 1),
                       d1_from=str(dd.time.iloc[0].date()), d1_years=round(yd, 1))
        if bad is None:
            bad = lookahead_ok(fr.segments(df)[-1])
        for k, t in sweep(df, cost, "H1").items():
            res[f"H1|{k}|{s}"] = judge(t, yrs, n_trials)
        for k, t in sweep(dd, cost, "D1").items():
            res[f"D1|{k}|{s}"] = judge(t, yd, n_trials)
        for k, t in session_ideas(s, df, cost, fomc).items():
            res[f"S|{k}|{s}"] = judge(t, yrs, n_trials)
        res[f"S|turn_of_month|{s}"] = judge(turn_of_month(dd, cost), yd, n_trials)
        ok = [k for k, r in res.items() if k.endswith("|" + s) and r["verdict"] in ("PASS", "works")]
        print(s, meta[s], "works:", ok, file=sys.stderr)

    tested = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if "--ledger" in sys.argv:
        r3.append_ledger([dict(tested_utc=tested, name="r7_" + k.replace("|", "_"), params="{}", lookahead_ok=k.split("|")[1] not in bad,
                               n_trials=n_trials, deflated_sharpe=r.get("dsr", ""), passes=bool(r.get("passes")),
                               useful=bool(r.get("passes"))) for k, r in res.items()])
    rep = dict(meta=dict(n_trials=n_trials, generated=tested, data=meta, lookahead_failed=bad, rules=RULE_KIND), results=res)
    (ROOT / "reports" / "strategy_round7.json").write_text(json.dumps(rep, indent=2, default=str))
    md = _md(rep)
    (ROOT / "STRATEGY_ROUND7.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def _md(rep: dict) -> str:
    m, res = rep["meta"], rep["results"]
    mark = {"PASS": "**PASS**", "works": "**works**", "weak": "weak", "no": "-"}
    L = ["# Strategy round 7 -- every indicator rule and session idea, seven markets, H1 and D1", "",
         f"{len(res)} tests; deflated-Sharpe N = {m['n_trials']}.  FundingPips costs and swaps.  Rules pre-registered in "
         "`backtest/strategy_round7.py`.",
         "Look-ahead check: " + ("all rules OK" if not m["lookahead_failed"] else "FAILED for " + ", ".join(m["lookahead_failed"])), "",
         "| market | H1 source | H1 from | H1 years | D1 from | D1 years |", "|---|---|---|--:|---|--:|"]
    L += [f"| {s} | {d['h1_source']} | {d['h1_from']} | {d['h1_years']} | {d['d1_from']} | {d['d1_years']} |" for s, d in m["data"].items()]
    for tf, title in (("H1", "Hourly"), ("D1", "Daily")):
        L += ["", f"## {title} indicator rules: verdict grid", "", "| rule | type | " + " | ".join(MARKETS) + " | markets + |",
              "|---|:-:|" + ":-:|" * len(MARKETS) + "--:|"]
        for k, kind in m["rules"].items():
            pos = sum(1 for s in MARKETS if (res[f"{tf}|{k}|{s}"].get("mean_R") or -1) > 0)
            L.append(f"| {k} | {kind} | " + " | ".join(mark[res[f"{tf}|{k}|{s}"]["verdict"]] for s in MARKETS) + f" | {pos}/7 |")
    L += ["", "## Session and calendar ideas", "",
          "| idea | market | trades/yr | mean R | t | R per year | first 70 % | last 30 % | last 2 y | WF+ | 2x cost | verdict |",
          "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|"]
    row = lambda name, s, r: (f"| {name} | {s} | {r['trades_per_year']} | {r['mean_R']} | {r['t']} | {r['R_per_year']} | {r['first70_R']} | "
                              f"{r['last30_R']} | {r['last2y_R']} | {r['wf_positive']}/8 | {r['cost2x_R']} | {mark[r['verdict']]} |")
    for k, r in res.items():
        tf, name, s = k.split("|")
        if tf == "S":
            L.append(row(name, s, r) if "checks" in r else f"| {name} | {s} |  |  |  |  |  |  |  |  |  | - |")
    for s in MARKETS:
        L += ["", f"## {s}: best 12 by R per year (mean R > 0)", "",
              "| timeframe | rule | trades/yr | mean R | t | R per year | max DD R | buy R | sell R | first 70 % | last 30 % | last 2 y | WF+ | 2x cost | verdict |",
              "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|"]
        rows = [(k, r) for k, r in res.items() if k.endswith("|" + s) and "checks" in r and r["mean_R"] > 0]
        for k, r in sorted(rows, key=lambda kr: -kr[1]["R_per_year"])[:12]:
            tf, name, _ = k.split("|")
            L.append(f"| {tf} | {name} | {r['trades_per_year']} | {r['mean_R']} | {r['t']} | {r['R_per_year']} | {r['max_dd_R']} | {r['long_R']} | "
                     f"{r['short_R']} | {r['first70_R']} | {r['last30_R']} | {r['last2y_R']} | {r['wf_positive']}/8 | {r['cost2x_R']} | {mark[r['verdict']]} |")
    good = {v: [k for k, r in res.items() if r["verdict"] == v] for v in ("PASS", "works")}
    L += ["", f"**PASS ({len(good['PASS'])}):** {', '.join(good['PASS']) or 'none'}", "",
          f"**Works ({len(good['works'])}):** {', '.join(good['works']) or 'none'}", ""]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
