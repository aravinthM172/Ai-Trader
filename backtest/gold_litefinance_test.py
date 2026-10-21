"""
LiteFinance "Gold Trading Strategies: How to Trade XAU/USD Profitably" (owner, 2026-10-21): the four rule-based
strategies coded as the article describes them (mirror rules for sells).

    python -m backtest.gold_litefinance_test

  1 trend  D1   close > EMA190; the low of this or the previous bar touched EMA20, 50 or 190; bullish price-action bar
                (bullish engulfing, pin bar with lower wick >= 2x body closing in its top third, or bullish outside bar)
                -> BUY next open; stop = lowest low of the last 2 bars - 0.1 ATR; target = previous 20-bar high ("swing")
                or 2R
  2 break  D1   close above the highest high of the previous 20 bars, close > EMA190, 50 < RSI14 < 70 -> BUY; stop =
                breakout-bar low (or the 10-bar low if the bar is < 0.5 ATR tall) - 0.1 ATR; targets 2R / 3R
  3 swing  H4   close > EMA21 and EMA21 rising; low touched EMA21 in the last 3 bars; MACD(12,26,9) histogram turns up
                (rises after falling) -> BUY; stop = 10-bar low - 0.1 ATR; target 2R
  4 scalp  M5/M1  07-20 UTC; close > EMA21 and EMA21 rising; low touched EMA21 in the last 3 bars; impulsive bullish bar
                (body >= 0.8 ATR) and ATR14 above its value 5 bars ago -> BUY; stop = 5-bar low - 0.1 ATR; targets 1R / 2R
Data / costs: D1 Yahoo gold futures 2000- (second source: Dukascopy H1 built into days, 2005-); H4 Dukascopy 2005-
(second: FundingPips 2010-) at FundedNext-like cost 0.016 % of price; M5 FundedNext 2025-26 (second: other broker),
M1 FundedNext Jul-Oct 2026, at each data set's own median spread.  60 / 40 split; pass = unseen mean R > 0, t >= 2,
> 0 at 2x costs, > 0 on the second source.  Read-only.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import fp_strategy_build as fb
from backtest import full_reassessment as fr
from backtest import gold_cheatsheet_test as cs
from backtest import grand_sweep as gs
from backtest import strategy_round5 as r5
from backtest import strategy_round7 as r7
from backtest.btc_strategies import base_atr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
FN_COST = (0.40 * 1.5 + 0.065) / 4100
HOLD = {"D1": 30, "H4": 60, "M5": 48, "M1": 60}


def rsi(c, n=14):
    return r7.rsi(c, n)


def build(p: pd.DataFrame, kind: str) -> dict:
    """variant -> (buy, sell, stop_b, stop_s, tgt_b, tgt_s)"""
    o, h, l, c = (p[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    A = base_atr(h, l, c, 14)
    o1, c1, h1, l1 = r7.prev(o), r7.prev(c), r7.prev(h), r7.prev(l)
    out = {}
    if kind == "trend":
        e20, e50, e190 = r7.ema(c, 20), r7.ema(c, 50), r7.ema(c, 190)
        lo2, hi2 = np.fmin(l, l1), np.fmax(h, h1)
        tu = (lo2 <= np.fmax(np.fmax(e20, e50), e190)) & (lo2 >= np.fmin(np.fmin(e20, e50), e190) - A)
        td = (hi2 >= np.fmin(np.fmin(e20, e50), e190)) & (hi2 <= np.fmax(np.fmax(e20, e50), e190) + A)
        body = np.abs(c - o)
        bull = ((c > o) & (c1 < o1) & (o <= c1) & (c >= o1)) | \
               ((np.fmin(o, c) - l >= 2 * body) & (c >= l + 2 / 3 * (h - l))) | ((h > h1) & (l < l1) & (c > o))
        bear = ((c < o) & (c1 > o1) & (o >= c1) & (c <= o1)) | \
               ((h - np.fmax(o, c) >= 2 * body) & (c <= h - 2 / 3 * (h - l))) | ((h > h1) & (l < l1) & (c < o))
        buy, sell = (c > e190) & tu & bull, (c < e190) & td & bear
        sb, ss = lo2 - 0.1 * A, hi2 + 0.1 * A
        hh, ll = r7.prev(r7.hi(h, 20)), r7.prev(r7.lo(l, 20))
        out["swing high"] = (buy, sell, sb, ss, np.where(hh > c, hh, np.nan), np.where(ll < c, ll, np.nan))
        out["2R"] = (buy, sell, sb, ss, c + 2 * (c - sb), c - 2 * (ss - c))
    elif kind == "break":
        e190, r = r7.ema(c, 190), rsi(c)
        hh, ll = r7.prev(r7.hi(h, 20)), r7.prev(r7.lo(l, 20))
        buy = (c > hh) & (c > e190) & (r > 50) & (r < 70)
        sell = (c < ll) & (c < e190) & (r < 50) & (r > 30)
        small = (h - l) < 0.5 * A
        sb = np.where(small, r7.lo(l, 10), l) - 0.1 * A
        ss = np.where(small, r7.hi(h, 10), h) + 0.1 * A
        for k in (2, 3):
            out[f"{k}R"] = (buy, sell, sb, ss, c + k * (c - sb), c - k * (ss - c))
    else:                                        # swing (H4, MACD) or scalp (M1 / M5, impulse + rising ATR)
        e21 = r7.ema(c, 21)
        rise, fall = e21 > r7.prev(e21), e21 < r7.prev(e21)
        tu = pd.Series(l <= e21).rolling(3).max().to_numpy() > 0
        td = pd.Series(h >= e21).rolling(3).max().to_numpy() > 0
        if kind == "swing":
            hist = (r7.ema(c, 12) - r7.ema(c, 26)) - r7.ema(r7.ema(c, 12) - r7.ema(c, 26), 9)
            hp, hpp = r7.prev(hist), r7.prev(hist, 2)
            buy = (c > e21) & rise & tu & (hist > hp) & (hp <= hpp)
            sell = (c < e21) & fall & td & (hist < hp) & (hp >= hpp)
            sb, ss = r7.lo(l, 10) - 0.1 * A, r7.hi(h, 10) + 0.1 * A
            out["2R"] = (buy, sell, sb, ss, c + 2 * (c - sb), c - 2 * (ss - c))
        else:
            hr = p.time.dt.hour.to_numpy()
            sess = (hr >= 7) & (hr < 20)
            atr_up = A > r7.prev(A, 5)
            buy = sess & (c > e21) & rise & tu & (c - o >= 0.8 * A) & atr_up
            sell = sess & (c < e21) & fall & td & (o - c >= 0.8 * A) & atr_up
            sb, ss = r7.lo(l, 5) - 0.1 * A, r7.hi(h, 5) + 0.1 * A
            for k in (1, 2):
                out[f"{k}R"] = (buy, sell, sb, ss, c + k * (c - sb), c - k * (ss - c))
    return out


def trades(df: pd.DataFrame, kind: str, tf: str, cost_frac: float) -> dict:
    res = {}
    pieces = [df] if tf == "D1" else fr.segments(df)
    for p in pieces:
        if len(p) < 300:
            continue
        for v, (b, s_, sb, ss, tb, ts) in build(p, kind).items():
            b = np.nan_to_num(np.asarray(b, float)) > 0
            s_ = np.nan_to_num(np.asarray(s_, float)) > 0
            b, s_ = b & np.isfinite(tb), s_ & np.isfinite(ts)
            sig = r5._put(r5._sig(len(p)), b, s_, sb, ss, tb, ts)
            t = r5.simulate_piece(p, dict(sig, hold=HOLD[tf]), dict(cost_frac=cost_frac, swap_long=0.0, swap_short=0.0))
            if len(t):
                res.setdefault(v, []).append(t)
    return {k: pd.concat(v, ignore_index=True) for k, v in res.items()}


def to(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    out = []
    for p in fr.segments(df):
        g = p.groupby(p.time.dt.floor(rule))
        out.append(pd.DataFrame({"time": g.time.first().index, "open": g.open.first().values, "high": g.high.max().values,
                                 "low": g.low.min().values, "close": g.close.last().values}))
    return pd.concat(out, ignore_index=True)


def yahoo_gold() -> pd.DataFrame:
    y = pd.read_csv(ROOT / "data" / "yahoo_D1" / "GC_F.csv")
    y["time"] = pd.to_datetime(y["date"], utc=True) + pd.Timedelta(hours=21)
    y = y.dropna(subset=["open", "high", "low", "close"])
    return y[(y.high > y.low) & (y.close > 0)][["time", "open", "high", "low", "close"]].sort_values("time").reset_index(drop=True)


def main() -> int:
    duka = fr._dense_from(fr._read(ROOT / "data" / "dukascopy_H1" / "XAUUSD.vx.csv"))
    fp = fb.load_h1("XAUUSD")
    m5, m5b, m1 = (cs.load_m5(ROOT / "data" / "gold" / "fn_XAUUSD_M5.csv"), cs.load_m5(ROOT / "data" / "xauusd_m5_5y.csv"),
                   cs.load_m5(ROOT / "data" / "gold" / "fn_XAUUSD_M1.csv"))
    spread_cost = lambda d: (d.spr.median() * 1.5 + 0.065) / d.close.median()
    tests = [("1 trend-following", "trend", "D1", yahoo_gold(), to(duka, "1D"), FN_COST, FN_COST),
             ("2 breakout", "break", "D1", yahoo_gold(), to(duka, "1D"), FN_COST, FN_COST),
             ("3 swing", "swing", "H4", to(duka, "4h"), to(fp, "4h"), FN_COST, FN_COST),
             ("4 scalping", "scalp", "M5", m5, m5b, spread_cost(m5), spread_cost(m5b)),
             ("4 scalping", "scalp", "M1", m1, None, spread_cost(m1), None)]
    rows = []
    for name, kind, tf, df, alt, cf, cfa in tests:
        cut = gs.split_at(df)
        yrs = (df.time.iloc[-1] - df.time.iloc[0]).days / 365.25
        t1, t2 = trades(df, kind, tf, cf), trades(df, kind, tf, 2 * cf)
        ta = trades(alt, kind, tf, cfa) if alt is not None else {}
        cut_a = gs.split_at(alt) if alt is not None else None
        for v, t in t1.items():
            d_, c_ = t[t.entry < cut], t[t.entry >= cut]
            a = ta.get(v)
            rows.append({"strategy": name, "tf": tf, "exit": v, "trades/yr": round(len(t) / yrs, 1),
                         "win %": round(100 * float((t.R > 0).mean()), 1), "first 60 %": round(float(d_.R.mean()), 3) if len(d_) else None,
                         "unseen 40 %": round(float(c_.R.mean()), 3) if len(c_) else None,
                         "unseen t": round(float(c_.R.mean() / c_.R.std() * np.sqrt(len(c_))), 2) if len(c_) > 2 else None,
                         "2x cost unseen": round(float(t2[v][t2[v].entry >= cut].R.mean()), 3) if v in t2 and (t2[v].entry >= cut).any() else None,
                         "2nd source unseen": round(float(a[a.entry >= cut_a].R.mean()), 3) if a is not None and (a.entry >= cut_a).any() else None,
                         "$/yr @0.35% 15K": round(len(t) / yrs * float(t.R.mean()) * 0.0035 * 15000),
                         "data from": str(df.time.iloc[0].date())})
    P = pd.DataFrame(rows)
    P["PASS"] = (P["unseen 40 %"] > 0) & (P["unseen t"] >= 2) & (P["2x cost unseen"] > 0) & \
        (P["2nd source unseen"].isna() | (P["2nd source unseen"] > 0))
    with pd.option_context("display.width", 260, "display.max_columns", 20):
        print(P.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
