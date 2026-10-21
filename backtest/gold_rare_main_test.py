"""
Owner's "XAUUSD Gold Trading - Rare & Main Cheat Sheet" (2026-10-21): Turtle Soup, the main sweep -> structure shift ->
displacement / FVG -> pullback setup, and Asian range expansion with acceptance.

    python -m backtest.gold_rare_main_test

PRE-REGISTERED (mirror rules for sells):
  turtle_soup   bar makes a new 20-bar low (below the previous 20-bar low, which is at least 4 bars old) and closes back
                above that old low -> BUY next open; stop = this bar's low - 0.1 ATR
  main_setup    sweep (as turtle_soup) -> within 8 bars a bullish displacement bar (body >= 1 ATR) whose close breaks the
                highest high of the 10 bars before the sweep (structure shift) and leaves a bullish FVG (low of the next
                bar > high of the bar before the displacement) -> limit BUY at the middle of the FVG, valid 8 bars;
                stop = sweep low - 0.1 ATR; H1 context: last completed hour's close above its EMA50
  asia_accept   Asian range = 00:00-06:00 UTC high / low; between 07:00 and 16:00 UTC two consecutive closes above the
                Asian high (acceptance, not a wick) -> BUY next open; stop = middle of the Asian range; one trade a day
  targets 1.5R / 2R / 3R; time exit 32 bars (M15) / 24 bars (H1)
Data / cost: M15 FundedNext 2025-05..2026-10 (own spread; second source: other broker M15); H1 Dukascopy gold 2005-
(FundedNext-like cost 0.016 %; second source FundingPips 2010-).  For H1 the H1 context filter is the same hour series.
60 / 40 split; pass = unseen mean R > 0, t >= 2, > 0 at 2x costs and on the second source.  Read-only.
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
TARGETS = (1.5, 2.0, 3.0)
FN_COST = (0.40 * 1.5 + 0.065) / 4100


def h1_ctx(p: pd.DataFrame) -> np.ndarray:
    hc = p.set_index("time").close.resample("1h").last().dropna()
    s = np.sign(hc - hc.ewm(span=50, adjust=False).mean())
    s.index = s.index + pd.Timedelta(hours=1)
    return s.reindex(p.time, method="ffill").fillna(0).to_numpy()


def build(p: pd.DataFrame, setup: str) -> list[tuple]:
    """(signal index, direction, stop, limit or nan)"""
    o, h, l, c = (p[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n = len(c)
    A = base_atr(h, l, c, 14)
    lo20, hi20 = r7.prev(r7.lo(l, 20)), r7.prev(r7.hi(h, 20))
    age_lo = pd.Series(l).rolling(20).apply(lambda x: 19 - int(np.argmin(x)), raw=True).shift(1).to_numpy()
    age_hi = pd.Series(h).rolling(20).apply(lambda x: 19 - int(np.argmax(x)), raw=True).shift(1).to_numpy()
    bull_sw = (l < lo20) & (c > lo20) & (age_lo >= 4)
    bear_sw = (h > hi20) & (c < hi20) & (age_hi >= 4)
    out = []
    if setup == "turtle_soup":
        for i in np.flatnonzero(bull_sw):
            out.append((i, 1, l[i] - 0.1 * A[i], np.nan))
        for i in np.flatnonzero(bear_sw):
            out.append((i, -1, h[i] + 0.1 * A[i], np.nan))
    elif setup == "main_setup":
        ctx = h1_ctx(p)
        hi10, lo10 = r7.prev(r7.hi(h, 10)), r7.prev(r7.lo(l, 10))
        for i in range(25, n - 10):
            for d, sw in ((1, bull_sw), (-1, bear_sw)):
                if not sw[i] or ctx[i] != d:
                    continue
                for k in range(i + 1, min(n - 1, i + 9)):
                    body = c[k] - o[k]
                    if d > 0 and body >= A[k] and c[k] > hi10[i] and l[k + 1] > h[k - 1]:
                        out.append((k + 1, 1, l[i:k + 1].min() - 0.1 * A[k], (l[k + 1] + h[k - 1]) / 2))
                        break
                    if d < 0 and -body >= A[k] and c[k] < lo10[i] and h[k + 1] < l[k - 1]:
                        out.append((k + 1, -1, h[i:k + 1].max() + 0.1 * A[k], (h[k + 1] + l[k - 1]) / 2))
                        break
    else:                                                       # asia_accept
        hr = p.time.dt.hour.to_numpy()
        day = p.time.dt.normalize()
        asia = p[hr < 6].groupby(day[hr < 6])
        ah, al = asia.high.max().reindex(day).to_numpy(), asia.low.min().reindex(day).to_numpy()
        done = set()
        for i in range(1, n):
            if not (7 <= hr[i] < 16) or not np.isfinite(ah[i]) or day.iloc[i] in done:
                continue
            mid = (ah[i] + al[i]) / 2
            if c[i] > ah[i] and c[i - 1] > ah[i] and hr[i - 1] >= 7:
                out.append((i, 1, mid, np.nan)); done.add(day.iloc[i])
            elif c[i] < al[i] and c[i - 1] < al[i] and hr[i - 1] >= 7:
                out.append((i, -1, mid, np.nan)); done.add(day.iloc[i])
    return out


def trades(df: pd.DataFrame, setup: str, tp: float, cost_frac: float, hold: int) -> pd.DataFrame:
    res = []
    for p in fr.segments(df):
        if len(p) < 300:
            continue
        p = p.reset_index(drop=True)
        n, c = len(p), p.close.to_numpy(float)
        s, limit = r5._sig(n), np.full(n, np.nan)
        for i, d, stop, lim in build(p, setup):
            if i >= n - 1:
                continue
            ref = lim if np.isfinite(lim) else c[i]
            risk = d * (ref - stop)
            if risk <= 0:
                continue
            s["d"][i], s["stop"][i], s["tgt"][i], limit[i] = d, stop, ref + d * tp * risk, lim
        t = r5.simulate_piece(p, dict(s, hold=hold, limit=limit, valid=8),
                              dict(cost_frac=cost_frac, swap_long=0.0, swap_short=0.0))
        if len(t):
            res.append(t)
    return pd.concat(res, ignore_index=True) if res else pd.DataFrame(columns=["entry", "R"])


def to(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    out = []
    for p in fr.segments(df):
        g = p.groupby(p.time.dt.floor(rule))
        out.append(pd.DataFrame({"time": g.time.first().index, "open": g.open.first().values, "high": g.high.max().values,
                                 "low": g.low.min().values, "close": g.close.last().values}))
    return pd.concat(out, ignore_index=True)


def main() -> int:
    m5, m5b = cs.load_m5(ROOT / "data" / "gold" / "fn_XAUUSD_M5.csv"), cs.load_m5(ROOT / "data" / "xauusd_m5_5y.csv")
    sc = lambda d: (d.spr.median() * 1.5 + 0.065) / d.close.median()
    m15, m15b = to(m5, "15min"), to(m5b, "15min")
    h1 = fr._dense_from(fr._read(ROOT / "data" / "dukascopy_H1" / "XAUUSD.vx.csv"))
    fp = fb.load_h1("XAUUSD")
    sets = [("M15", m15, m15b, sc(m5), sc(m5b), 32), ("H1", h1, fp, FN_COST, FN_COST, 24)]
    rows = []
    for tf, df, alt, cf, cfa, hold in sets:
        cut, cut_a = gs.split_at(df), gs.split_at(alt)
        yrs = (df.time.iloc[-1] - df.time.iloc[0]).days / 365.25
        for setup in ("turtle_soup", "main_setup", "asia_accept"):
            for tp in TARGETS:
                t = trades(df, setup, tp, cf, hold)
                if len(t) < 10:
                    continue
                t2, ta = trades(df, setup, tp, 2 * cf, hold), trades(alt, setup, tp, cfa, hold)
                d_, c_ = t[t.entry < cut], t[t.entry >= cut]
                rows.append({"tf": tf, "setup": setup, "target": f"{tp:g}R", "trades/yr": round(len(t) / yrs, 1),
                             "win %": round(100 * float((t.R > 0).mean()), 1),
                             "first 60 %": round(float(d_.R.mean()), 3) if len(d_) else None,
                             "unseen 40 %": round(float(c_.R.mean()), 3) if len(c_) else None,
                             "unseen t": round(float(c_.R.mean() / c_.R.std() * np.sqrt(len(c_))), 2) if len(c_) > 2 else None,
                             "2x cost unseen": round(float(t2[t2.entry >= cut].R.mean()), 3) if len(t2) and (t2.entry >= cut).any() else None,
                             "2nd source unseen": round(float(ta[ta.entry >= cut_a].R.mean()), 3) if len(ta) and (ta.entry >= cut_a).any() else None,
                             "$/yr @0.35% 15K": round(len(t) / yrs * float(t.R.mean()) * 0.0035 * 15000)})
        print(f"{tf} done ({df.time.iloc[0].date()} .. {df.time.iloc[-1].date()})", flush=True)
    P = pd.DataFrame(rows)
    P["PASS"] = (P["unseen 40 %"] > 0) & (P["unseen t"] >= 2) & (P["2x cost unseen"] > 0) & (P["2nd source unseen"] > 0)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(P.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
