"""
Strategy round 5 -- the families never tested in rounds 1-4: volume profile, session breakout,
liquidity sweep, fair value gap, Bollinger / VWAP / pivot fades, Ichimoku, intraday volatility breakout.
(2026-10-10)

    python -m backtest.strategy_round5            -> STRATEGY_ROUND5.md, reports/strategy_round5.json

PRE-REGISTERED (written before the first run; one fixed textbook parameter set per idea, nothing tuned):

Data     the 7 long-history H1 markets of backtest/long_history_backtest.h1 (BTC Bitstamp, ETH + USDJPY
         FundingPips, gold / NDX100 / GER40 / EURUSD Dukascopy).  Volume for the profile and the VWAP:
         BTC Bitstamp traded volume, ETH Binance ETHUSDT traded volume, Dukascopy volume, USDJPY tick count.
         Day = broker day (rollover 21:00 UTC), week = Monday-Friday broker days.
Entry    signal on a CLOSED H1 bar, market order at the next open (the fair value gap uses a resting limit).
         One position per (strategy, symbol).  Skipped when the stop is closer than 0.5 ATR or further than
         6 ATR, or when the target is closer than half the risk.
Costs    FundingPips 1-year median spread + 0.25 x spread slippage per side + FundingPips swap (fb.cost_of).

Ideas (ATR = ATR14 of the signal bar)
  vp_fade_d      previous DAY's volume profile (40 price bins, value area = 70 % of volume around the POC).
                 Close back inside the value area from below (prev close < VAL <= close < POC): buy,
                 target VAH ("80 % rule"), stop 2 ATR, 24 bars.  Mirror from above.
  vp_fade_w      same with the previous WEEK's profile, 96 bars.
  vp_break_d     two closes in a row above yesterday's VAH after a close at / below it: buy, stop 2 ATR,
                 target 6 ATR, 96 bars.  Mirror below VAL.
  asia_break     range of 00:00-06:59 UTC.  First H1 close outside it between 07:00 and 15:59 UTC: trade the
                 break, stop at the other side of the range, target 1.5 x range, flat at 21:00 UTC.
  sweep_rev      bar trades below yesterday's low, closes back above it and above its own open: buy, stop
                 0.1 ATR under the bar's low, target 2 R, 48 bars.  Mirror at yesterday's high.
  fvg_retrace    bullish 3-bar gap (low > high two bars earlier, gap >= 0.25 ATR) above EMA200: buy limit at
                 the top of the gap for 12 bars, stop 0.25 ATR under the gap, target 2 R, 48 bars.  Mirror.
  bb_fade        close under the lower Bollinger band (20, 2) while ADX14 < 20: buy, target the 20-bar mean,
                 stop 2 ATR, 24 bars.  Mirror.
  vwap_fade      close 2 ATR or more under the day's VWAP, at least 6 bars into the day: buy, target the VWAP,
                 stop 2 ATR, 12 bars.  Mirror.
  pivot_bounce   floor pivots of yesterday.  Bar touches S1 and closes above it: buy, target the pivot,
                 stop 1.5 ATR, 24 bars.  Mirror at R1.
  ichimoku       fresh signal: close above the cloud, Tenkan (9) > Kijun (26), close above the close 26 bars
                 ago: buy, stop 2 ATR, target 6 ATR, 96 bars.  Mirror.
  vbo_day        Williams volatility breakout: first H1 close above day open + 0.5 x yesterday's range: buy,
                 stop at the day's open, no target, flat at the next day's first bar.  Mirror.
  nr7_vbo        vbo_day only on days after an NR7 day (narrowest range of the last 7).

PASS RULE (r3.evaluate, unchanged): >= 10 years, >= 100 trades, mean R > 0 on full / first 70 % / last 30 %,
  >= 6 of 8 folds, > 0 at 2x costs, deflated Sharpe >= 0.95 with N = ledger + every test here.
LEAD (reported, not a pass): every check except years_10 and dsr_95, and t >= 2.75.
Every (idea, symbol) and every pooled idea goes into research/ledger.csv.  Read-only; no orders.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from backtest import fp_strategy_build as fb
from backtest import full_reassessment as fr
from backtest import long_history_backtest as lh
from backtest import strategy_round3_daily as r3
from backtest.btc_strategies import base_atr
from strategy.btc_features import _ema

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BINANCE = DATA / "binance_H1"
SYMS = lh.SYMS
DUKA = {"XAUUSD": "XAUUSD.vx", "NDX100": "NAS100.vx", "GER40": "DAX40.vx", "EURUSD": "EURUSD.vx"}
BINS, VALUE_AREA = 40, 0.70
MIN_RISK_ATR, MAX_RISK_ATR, MIN_RR = 0.5, 6.0, 0.5
LEAD_T = 2.75


# -- data ----------------------------------------------------------------------------------------
def binance_h1(pair: str) -> pd.DataFrame:
    """Binance spot 1h candles (public market-data endpoint), cached in data/binance_H1."""
    p = BINANCE / f"{pair}.csv"
    if p.exists():
        d = pd.read_csv(p)
        d["time"] = pd.to_datetime(d["time"], utc=True)
        return d
    BINANCE.mkdir(parents=True, exist_ok=True)
    rows, start = [], int(pd.Timestamp("2017-08-01", tz="UTC").timestamp() * 1000)
    while True:
        url = f"https://data-api.binance.vision/api/v3/klines?symbol={pair}&interval=1h&limit=1000&startTime={start}"
        with urllib.request.urlopen(url, timeout=30) as r:
            k = json.loads(r.read())
        if not k:
            break
        rows += [(x[0], float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])) for x in k]
        if len(k) < 1000:
            break
        start = k[-1][0] + 1
        time.sleep(0.15)
    d = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
    d["time"] = pd.to_datetime(d["time"], unit="ms", utc=True)
    d.to_csv(p, index=False)
    return d


def volume_of(sym: str) -> tuple[pd.Series, str]:
    """Hourly volume indexed by bar time, and what it is."""
    if sym == "BTCUSD":
        d, col, what = pd.read_csv(DATA / "btcusd_bitstamp_H1.csv"), "real_volume", "Bitstamp traded volume"
    elif sym == "ETHUSD":
        d, col, what = binance_h1("ETHUSDT"), "volume", "Binance traded volume"
    elif sym in DUKA:
        d, col, what = pd.read_csv(DATA / "dukascopy_H1" / f"{DUKA[sym]}.csv"), "volume", "Dukascopy volume"
    else:
        d, col, what = pd.read_csv(fb.FP / f"{sym}_H1.csv"), "tick_volume", "tick count"
    t = pd.to_datetime(d["time"], utc=True)
    return pd.Series(d[col].to_numpy(float), index=t).groupby(level=0).last(), what


def load(sym: str) -> tuple[pd.DataFrame, str, str]:
    df, src = lh.h1(sym)
    v, what = volume_of(sym)
    return df.assign(vol=v.reindex(df.time).to_numpy()), src, what


# -- indicators (every value at bar j uses bars <= j only) ---------------------------------------
def broker_day(t: pd.Series) -> np.ndarray:
    return (t + pd.Timedelta(hours=3)).dt.tz_localize(None).dt.normalize().to_numpy()


def week_of(day: np.ndarray) -> np.ndarray:
    d = pd.DatetimeIndex(day)
    return (d - pd.to_timedelta(d.dayofweek, unit="D")).to_numpy()


def prev_group(values: dict, key: np.ndarray) -> dict:
    """Per-group values -> for each bar, the value of the PREVIOUS group (known when the group starts)."""
    uniq = pd.unique(key)
    pos = pd.Series(np.arange(len(uniq)), index=uniq).reindex(key).to_numpy()
    out = {}
    for name, per in values.items():
        prev = np.concatenate([[np.nan], per[:-1]])
        out[name] = prev[pos]
    return out


def profile(h, l, v, key) -> dict:
    """POC / value-area high / low of each group's volume-at-price histogram, mapped to the next group."""
    uniq, start = np.unique(key, return_index=True)
    order = np.argsort(start)
    start = np.append(start[order], len(key))
    poc, vah, val = (np.full(len(order), np.nan) for _ in range(3))
    for g in range(len(order)):
        a, b = start[g], start[g + 1]
        hh, ll, vv = h[a:b], l[a:b], np.nan_to_num(v[a:b])
        lo, hi = ll.min(), hh.max()
        if b - a < 12 or hi <= lo or vv.sum() <= 0:
            continue
        edges = np.linspace(lo, hi, BINS + 1)
        ov = np.clip(np.minimum(hh[:, None], edges[None, 1:]) - np.maximum(ll[:, None], edges[None, :-1]), 0, None)
        rng = (hh - ll)[:, None]
        share = np.where(rng > 0, ov / np.where(rng > 0, rng, 1), 0)
        flat = rng[:, 0] <= 0                                   # a bar with no range: all volume in its bin
        if flat.any():
            idx = np.clip(np.searchsorted(edges, hh[flat], side="right") - 1, 0, BINS - 1)
            share[np.flatnonzero(flat), idx] = 1.0
        hist = (share * vv[:, None]).sum(axis=0)
        p = int(hist.argmax())
        lo_i = hi_i = p
        got, need = hist[p], VALUE_AREA * hist.sum()
        while got < need and (lo_i > 0 or hi_i < BINS - 1):
            up = hist[hi_i + 1] if hi_i < BINS - 1 else -1.0
            dn = hist[lo_i - 1] if lo_i > 0 else -1.0
            if up >= dn:
                hi_i += 1; got += up
            else:
                lo_i -= 1; got += dn
        poc[g], vah[g], val[g] = (edges[p] + edges[p + 1]) / 2, edges[hi_i + 1], edges[lo_i]
    return prev_group(dict(poc=poc, vah=vah, val=val), key)


def prev_day_ohlc(o, h, l, c, day) -> dict:
    g = pd.DataFrame(dict(o=o, h=h, l=l, c=c)).groupby(day, sort=False)
    per = dict(pdo=g.o.first().to_numpy(), pdh=g.h.max().to_numpy(), pdl=g.l.min().to_numpy(), pdc=g.c.last().to_numpy())
    return prev_group(per, day)


def adx(h, l, c, n=14) -> np.ndarray:
    up, dn = np.diff(h, prepend=h[0]), -np.diff(l, prepend=l[0])
    plus, minus = np.where((up > dn) & (up > 0), up, 0.0), np.where((dn > up) & (dn > 0), dn, 0.0)
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    w = lambda x: pd.Series(x).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    a = w(tr)
    pdi, mdi = 100 * w(plus) / a, 100 * w(minus) / a
    dx = 100 * np.abs(pdi - mdi) / np.where(pdi + mdi > 0, pdi + mdi, np.nan)
    return pd.Series(dx).ewm(alpha=1 / n, adjust=False).mean().to_numpy()


def _sig(n: int) -> dict:
    return dict(d=np.zeros(n, np.int8), stop=np.full(n, np.nan), tgt=np.full(n, np.nan))


def _put(s: dict, long: np.ndarray, short: np.ndarray, stop_l, stop_s, tgt_l, tgt_s) -> dict:
    s["d"][long], s["d"][short & ~long] = 1, -1
    for k, a, b in (("stop", stop_l, stop_s), ("tgt", tgt_l, tgt_s)):
        s[k] = np.where(s["d"] > 0, a, np.where(s["d"] < 0, b, np.nan))
    return s


def _prev(a: np.ndarray, k: int = 1) -> np.ndarray:
    return np.concatenate([np.full(k, np.nan), a[:-k]])


def _first_of_group(mask: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Keep only the first True of each group."""
    m = pd.Series(mask.astype(int))
    return mask & (m.groupby(key, sort=False).cumsum().to_numpy() == 1)


# -- the ideas: signal at bar j, order for bar j + 1 ----------------------------------------------
def build(df: pd.DataFrame) -> dict:
    o, h, l, c, v = (df[k].to_numpy(float) for k in ("open", "high", "low", "close", "vol"))
    n, t = len(c), df["time"]
    A = base_atr(h, l, c, 14)
    day, hour = broker_day(t), t.dt.hour.to_numpy()
    utc_day = t.dt.tz_localize(None).dt.normalize().to_numpy()
    c1, c2 = _prev(c), _prev(c, 2)
    inf = np.full(n, np.inf)
    out = {}

    def fade(P, hold):
        long = (c1 < P["val"]) & (c >= P["val"]) & (c < P["poc"])
        short = (c1 > P["vah"]) & (c <= P["vah"]) & (c > P["poc"])
        return dict(_put(_sig(n), long, short, c - 2 * A, c + 2 * A, P["vah"], P["val"]), hold=hold)

    Pd, Pw = profile(h, l, v, day), profile(h, l, v, week_of(day))
    out["vp_fade_d"], out["vp_fade_w"] = fade(Pd, 24), fade(Pw, 96)
    long = (c > Pd["vah"]) & (c1 > Pd["vah"]) & (c2 <= Pd["vah"])
    short = (c < Pd["val"]) & (c1 < Pd["val"]) & (c2 >= Pd["val"])
    out["vp_break_d"] = dict(_put(_sig(n), long, short, c - 2 * A, c + 2 * A, c + 6 * A, c - 6 * A), hold=96)

    asia = hour < 7
    g = pd.DataFrame(dict(h=np.where(asia, h, np.nan), l=np.where(asia, l, np.nan))).groupby(utc_day, sort=False)
    ah, al = g.h.transform("max").to_numpy(), g.l.transform("min").to_numpy()   # only read on bars after 07:00
    full = pd.Series(asia.astype(int)).groupby(utc_day, sort=False).transform("sum").to_numpy() >= 5
    win = (hour >= 7) & (hour < 16) & full
    brk = _first_of_group(win & ((c > ah) | (c < al)), utc_day)
    w = ah - al
    s = _put(_sig(n), brk & (c > ah), brk & (c < al), al, ah, c + 1.5 * w, c - 1.5 * w)
    out["asia_break"] = dict(s, hold=24, flat=hour == 21)

    D = prev_day_ohlc(o, h, l, c, day)
    long = (l < D["pdl"]) & (c > D["pdl"]) & (c > o)
    short = (h > D["pdh"]) & (c < D["pdh"]) & (c < o)
    sl, ss = l - 0.1 * A, h + 0.1 * A
    out["sweep_rev"] = dict(_put(_sig(n), long, short, sl, ss, c + 2 * (c - sl), c - 2 * (ss - c)), hold=48)

    e200, h2, l2 = _ema(c, 200), _prev(h, 2), _prev(l, 2)
    long = (l > h2) & (l - h2 >= 0.25 * A) & (c > e200)
    short = (h < l2) & (l2 - h >= 0.25 * A) & (c < e200)
    sl, ss = h2 - 0.25 * A, l2 + 0.25 * A
    s = _put(_sig(n), long, short, sl, ss, l + 2 * (l - sl), h - 2 * (ss - h))
    out["fvg_retrace"] = dict(s, hold=48, limit=np.where(s["d"] > 0, l, np.where(s["d"] < 0, h, np.nan)), valid=12)

    cs = pd.Series(c)
    ma, sd = cs.rolling(20).mean().to_numpy(), cs.rolling(20).std(ddof=0).to_numpy()
    quiet = adx(h, l, c) < 20
    out["bb_fade"] = dict(_put(_sig(n), quiet & (c < ma - 2 * sd), quiet & (c > ma + 2 * sd),
                               c - 2 * A, c + 2 * A, ma, ma), hold=24)

    wv = np.where(np.isfinite(v) & (v > 0), v, 0.0)
    wv = np.where(pd.Series(wv).groupby(day, sort=False).transform("sum").to_numpy() > 0, wv, 1.0)   # no volume: equal weight
    gp = pd.DataFrame(dict(pv=(h + l + c) / 3 * wv, v=wv)).groupby(day, sort=False)
    vw = gp.pv.cumsum().to_numpy() / np.where(gp.v.cumsum().to_numpy() > 0, gp.v.cumsum().to_numpy(), np.nan)
    late = pd.Series(np.ones(n)).groupby(day, sort=False).cumsum().to_numpy() >= 6
    out["vwap_fade"] = dict(_put(_sig(n), late & (c <= vw - 2 * A), late & (c >= vw + 2 * A),
                                 c - 2 * A, c + 2 * A, vw, vw), hold=12)

    P = (D["pdh"] + D["pdl"] + D["pdc"]) / 3
    S1, R1 = 2 * P - D["pdh"], 2 * P - D["pdl"]
    out["pivot_bounce"] = dict(_put(_sig(n), (l <= S1) & (c > S1), (h >= R1) & (c < R1),
                                    c - 1.5 * A, c + 1.5 * A, P, P), hold=24)

    mid = lambda k: (pd.Series(h).rolling(k).max().to_numpy() + pd.Series(l).rolling(k).min().to_numpy()) / 2
    ten, kij = mid(9), mid(26)
    spa, spb = _prev((ten + kij) / 2, 26), _prev(mid(52), 26)
    top, bot = np.maximum(spa, spb), np.minimum(spa, spb)
    up = (c > top) & (ten > kij) & (c > _prev(c, 26))
    dn = (c < bot) & (ten < kij) & (c < _prev(c, 26))
    fresh = lambda m: m & ~np.concatenate([[False], m[:-1]])
    out["ichimoku"] = dict(_put(_sig(n), fresh(up), fresh(dn), c - 2 * A, c + 2 * A, c + 6 * A, c - 6 * A), hold=96)

    dopen = pd.Series(o).groupby(day, sort=False).transform("first").to_numpy()
    prng = D["pdh"] - D["pdl"]
    brk = _first_of_group((c > dopen + 0.5 * prng) | (c < dopen - 0.5 * prng), day)
    newday = np.concatenate([[True], day[1:] != day[:-1]])
    vbo = dict(_put(_sig(n), brk & (c > dopen + 0.5 * prng), brk & (c < dopen - 0.5 * prng), dopen, dopen, inf, -inf),
               hold=30, flat=newday)
    out["vbo_day"] = vbo
    dr = pd.DataFrame(dict(h=h, l=l)).groupby(day, sort=False)
    rng_d = (dr.h.max() - dr.l.min())
    nr7 = (rng_d <= rng_d.rolling(7).min()).to_numpy()
    after_nr7 = prev_group(dict(x=nr7.astype(float)), day)["x"] == 1
    s = {k: (a.copy() if isinstance(a, np.ndarray) else a) for k, a in vbo.items()}
    s["d"] = np.where(after_nr7, s["d"], 0).astype(np.int8)
    out["nr7_vbo"] = s
    return out


IDEAS = ["vp_fade_d", "vp_fade_w", "vp_break_d", "asia_break", "sweep_rev", "fvg_retrace", "bb_fade", "vwap_fade",
         "pivot_bounce", "ichimoku", "vbo_day", "nr7_vbo"]


# -- trade simulation ----------------------------------------------------------------------------
@njit(cache=True)
def _run(o, h, l, c, atr, d, stop, tgt, limit, flat, hold, valid, min_risk, max_risk, min_rr):
    n = len(c)
    ei, xi, dd, why_ = np.zeros(n, np.int64), np.zeros(n, np.int64), np.zeros(n, np.int64), np.zeros(n, np.int64)
    ep_, xp_, rk_ = np.zeros(n), np.zeros(n), np.zeros(n)
    m = 0
    i = 1
    pd_, pl, ps, pt, pa, pe = 0, 0.0, 0.0, 0.0, 0.0, -1          # resting limit order
    while i < n - 1:
        j = i - 1
        dj = d[j]
        market = False
        if dj != 0:
            if np.isnan(limit[j]):
                market = True
            else:
                pd_, pl, ps, pt, pa, pe = int(dj), limit[j], stop[j], tgt[j], atr[j], i + valid - 1
        di, ep, sp, tp, a = 0, 0.0, 0.0, 0.0, 0.0
        if market:
            di, ep, sp, tp, a = int(dj), o[i], stop[j], tgt[j], atr[j]
        elif pd_ != 0 and i <= pe:
            if pd_ * (o[i] - pl) <= 0:
                di, ep = pd_, o[i]
            elif (pd_ > 0 and l[i] <= pl) or (pd_ < 0 and h[i] >= pl):
                di, ep = pd_, pl
            if di != 0:
                sp, tp, a = ps, pt, pa
        if di == 0:
            i += 1
            continue
        risk = di * (ep - sp)
        if not (a > 0) or not (risk >= min_risk * a) or risk > max_risk * a or di * (tp - ep) < min_rr * risk:
            if not market:
                pd_ = 0
            i += 1
            continue
        pd_ = 0
        x, xp, why, at_open = -1, 0.0, 0, False                 # why: 0 time, 1 stop, 2 target, 3 flat
        last = min(n, i + hold)
        for k in range(i, last):
            if k > i and flat[k]:
                x, xp, why, at_open = k, o[k], 3, True
                break
            if k > i and di * (o[k] - sp) <= 0:
                x, xp, why, at_open = k, o[k], 1, True
                break
            if (di > 0 and l[k] <= sp) or (di < 0 and h[k] >= sp):
                x, xp, why = k, sp, 1
                break
            if k > i and di * (o[k] - tp) >= 0:
                x, xp, why, at_open = k, o[k], 2, True
                break
            if (market or k > i) and ((di > 0 and h[k] >= tp) or (di < 0 and l[k] <= tp)):
                x, xp, why = k, tp, 2
                break
        if x < 0:
            x = min(n - 1, i + hold)
            xp, at_open = o[x], True
        ei[m], xi[m], dd[m], ep_[m], xp_[m], rk_[m], why_[m] = i, x, di, ep, xp, risk, why
        m += 1
        i = x if at_open else x + 1
    return ei[:m], xi[:m], dd[:m], ep_[:m], xp_[:m], rk_[:m], why_[:m]


def simulate_piece(p: pd.DataFrame, s: dict, cost: dict) -> pd.DataFrame:
    o, h, l, c = (p[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n = len(c)
    limit = s.get("limit", np.full(n, np.nan))
    flat = s.get("flat", np.zeros(n, bool))
    ei, xi, dd, ep, xp, rk, why = _run(o, h, l, c, base_atr(h, l, c, 14), s["d"], s["stop"], s["tgt"], limit, flat,
                                       int(s["hold"]), int(s.get("valid", 1)), MIN_RISK_ATR, MAX_RISK_ATR, MIN_RR)
    if not len(ei):
        return pd.DataFrame(columns=["entry", "exit", "dir", "reason", "R", "cost_R"])
    t = p["time"]
    ent, ext = t.iloc[ei].reset_index(drop=True), t.iloc[xi].reset_index(drop=True)
    nt = ((ext + pd.Timedelta(hours=3)).dt.normalize() - (ent + pd.Timedelta(hours=3)).dt.normalize()).dt.days.to_numpy()
    rate = np.where(dd > 0, cost["swap_long"], cost["swap_short"])
    cost_R = cost["cost_frac"] * ep / rk
    R = (dd * (xp - ep) + ep * rate / 365 * nt) / rk - cost_R
    return pd.DataFrame(dict(entry=ent, exit=ext, dir=dd, reason=np.array(["time", "stop", "target", "flat"])[why],
                             R=R, cost_R=cost_R))


def simulate(df: pd.DataFrame, cost: dict) -> dict:
    parts = {k: [] for k in IDEAS}
    for p in fr.segments(df):
        sig = build(p)
        for k in IDEAS:
            parts[k].append(simulate_piece(p, sig[k], cost))
    return {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}


def lookahead_ok(df: pd.DataFrame, cut: int = 20000) -> dict:
    """Signals on the first `cut` bars must not change when later bars are added."""
    a, b = build(df.iloc[:cut].reset_index(drop=True)), build(df.iloc[:cut + 3000].reset_index(drop=True))
    ok = {}
    for k in IDEAS:
        same = True
        for f in ("d", "stop", "tgt", "limit"):
            if f in a[k]:
                x, y = np.asarray(a[k][f], float)[:cut - 30], np.asarray(b[k][f], float)[:cut - 30]
                m = a[k]["d"][:cut - 30] != 0 if f != "d" else np.ones(cut - 30, bool)
                same &= bool(np.allclose(x[m], y[m], equal_nan=True))
        ok[k] = same
    return ok


def t_stat(R: np.ndarray) -> float | None:
    return round(float(R.mean() / R.std() * np.sqrt(len(R))), 2) if len(R) > 1 and R.std() > 0 else None


def judge(t: pd.DataFrame, years: float, n_trials: int) -> dict:
    r = r3.evaluate(t, t.assign(R=t.R - t.cost_R), years, n_trials, None)
    if "checks" in r:
        r["t"] = t_stat(t.R.to_numpy())
        r["cost_R"] = round(float(t.cost_R.mean()), 3)
        r["last2y_R"] = round(float(t.R[t.entry >= t.entry.max() - pd.Timedelta(days=730)].mean()), 4)
        r["lead"] = bool(all(v for k, v in r["checks"].items() if k not in ("years_10", "dsr_95")) and (r["t"] or 0) >= LEAD_T)
    return r


def main() -> int:
    n_trials = r3.ledger_n() + r3.PRIOR_TRIALS + len(IDEAS) * (len(SYMS) + 1)
    res, pooled, meta, look = {}, {k: [] for k in IDEAS}, {}, None
    for s in SYMS:
        df, src, what = load(s)
        years = sum((p.time.iloc[-1] - p.time.iloc[0]).days for p in fr.segments(df)) / 365.25
        meta[s] = dict(source=src, volume=what, years=round(years, 1), bars=len(df),
                       volume_bars=int(np.isfinite(df.vol).sum()))
        if look is None:
            look = lookahead_ok(fr.segments(df)[-1])
        tr = simulate(df, fb.cost_of(lh.SPECS[s]))
        for k in IDEAS:
            res[f"{k}|{s}"] = judge(tr[k], years, n_trials)
            pooled[k].append(tr[k].assign(sym=s))
        print(s, {k: res[f"{k}|{s}"].get("mean_R") for k in IDEAS}, file=sys.stderr)
    pool = {}
    for k in IDEAS:
        t = pd.concat(pooled[k], ignore_index=True).sort_values("entry").reset_index(drop=True)
        pool[k] = judge(t, max(m["years"] for m in meta.values()), n_trials)
        pool[k]["symbols_positive"] = sum(1 for s in SYMS if (res[f"{k}|{s}"].get("mean_R") or -1) > 0)

    tested = datetime.now(timezone.utc).isoformat(timespec="seconds")
    r3.append_ledger([dict(tested_utc=tested, name=f"r5_{k.replace('|', '_')}", params="{}", lookahead_ok=look[k.split('|')[0]],
                           n_trials=n_trials, deflated_sharpe=r.get("dsr", ""), passes=bool(r.get("passes")),
                           useful=bool(r.get("passes"))) for k, r in res.items()]
                     + [dict(tested_utc=tested, name=f"r5_{k}_pooled", params="{}", lookahead_ok=look[k], n_trials=n_trials,
                             deflated_sharpe=r.get("dsr", ""), passes=bool(r.get("passes")), useful=bool(r.get("passes")))
                        for k, r in pool.items()])
    rep = dict(meta=dict(n_trials=n_trials, generated=tested, data=meta, lookahead_ok=look), results=res, pooled=pool)
    (ROOT / "reports" / "strategy_round5.json").write_text(json.dumps(rep, indent=2, default=str))
    md = _md(rep)
    (ROOT / "STRATEGY_ROUND5.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


HEAD = "| trades | /yr | mean R | t | win % | PF | cost R | first 70 % | last 30 % | last 2 y | WF+ | 2x cost | DSR | PASS | lead |"
SEP = "--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|:-:|"


def _row(r: dict) -> str:
    if "checks" not in r:
        return f"| {r.get('trades')} |  |  |  |  |  |  |  |  |  |  |  |  | no |  |"
    return (f"| {r['trades']} | {r['trades_per_year']} | {r['mean_R']} | {r['t']} | {round(100 * r['win_rate'])} | "
            f"{r['profit_factor']} | {r['cost_R']} | {r['first70_R']} | {r['last30_R']} | {r['last2y_R']} | "
            f"{r['wf_positive']}/8 | {r['cost2x_R']} | {r['dsr']} | {'**yes**' if r['passes'] else 'no'} | "
            f"{'**lead**' if r['lead'] else ''} |")


def _md(rep: dict) -> str:
    m = rep["meta"]
    L = ["# Strategy round 5 -- volume profile, sessions, sweeps, gaps, fades, Ichimoku, volatility breakout", "",
         f"Deflated-Sharpe N = {m['n_trials']}.  FundingPips costs and swaps.  Rules pre-registered in "
         "`backtest/strategy_round5.py`.", "",
         "Look-ahead check (signals unchanged when later bars are added): "
         + ("all ideas OK" if all(m["lookahead_ok"].values()) else "FAILED for " + ", ".join(k for k, v in m["lookahead_ok"].items() if not v)),
         "", "| symbol | H1 source | years | volume used |", "|---|---|--:|---|"]
    L += [f"| {s} | {d['source']} | {d['years']} | {d['volume']} ({round(100 * d['volume_bars'] / d['bars'])} % of bars) |"
          for s, d in m["data"].items()]
    L += ["", "## Pooled per idea (all 7 markets)", "", "| idea | markets + " + HEAD, "|---|--:|" + SEP]
    for k, r in sorted(rep["pooled"].items(), key=lambda kv: -(kv[1].get("mean_R") or -9)):
        L.append(f"| {k} | {r.get('symbols_positive')}/7 " + _row(r))
    for k in IDEAS:
        L += ["", f"## {k}", "", "| symbol " + HEAD, "|---|" + SEP]
        for s in SYMS:
            L.append(f"| {s} " + _row(rep["results"][f"{k}|{s}"]))
    passing = [k for k, r in {**rep["results"], **{f"{k}|pooled": v for k, v in rep["pooled"].items()}}.items() if r.get("passes")]
    leads = [k for k, r in {**rep["results"], **{f"{k}|pooled": v for k, v in rep["pooled"].items()}}.items() if r.get("lead")]
    L += ["", f"**Passing ({len(passing)}):** {', '.join(passing) or 'none'}", "",
          f"**Leads ({len(leads)}):** {', '.join(leads) or 'none'}", ""]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
