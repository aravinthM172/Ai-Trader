"""
Gold methods found in the 2026-10-21 internet sweep (ForexFactory, YouTube, TradingView, MQL5 sellers, papers) that had
not been tested exactly -- on low timeframes with FundedNext costs.

    python -m backtest.gold_internet_methods

PRE-REGISTERED (mirror rules for sells; times in New York or London local time; written before the first run)
  ny_orb15 / ny_orb30   range = first 15 / 30 min after 09:30 New York; first close outside it before 15:30 -> enter;
                        stop = other side of the range; target 1R / 2R / session end (flat at 15:30 New York)
  ldn_orb30             the same around the 08:00 London open, flat at 12:00 London
  ema9_21_rsi           07-20 UTC: EMA9 crosses EMA21, RSI14 > 50 (< 50 for sells); stop = 5-bar low; target 1.5R / 2R
  ema4_stack            EMA9 > 21 > 34 > 50; low touched EMA21 in the last 3 bars; close back above EMA9; stop 5-bar low;
                        target 1.5R / 2R
  vwap_pullback         session VWAP from 00:00 UTC (tick volume); VWAP rising and price above it; low touches VWAP and
                        closes above; stop = VWAP - 1 ATR; target 2R
  vwap_fade             close beyond VWAP +/- 2 sd of price around VWAP -> fade back; target = VWAP; stop 1 ATR beyond
  pivot_bounce          classic daily pivot of the previous UTC day; low touches S1 and closes above -> BUY to the pivot;
                        stop = S1 - 1 ATR (mirror at R1)
  pivot_break           close above R1 -> BUY, target R2, stop = pivot (mirror below S1)
  ny_momentum           sign of the 09:30-10:00 New York return -> enter at 10:00 in that direction, flat at 15:30;
                        stop = the other side of the first-30-minute range
  st_adx                Supertrend(10, 3) flip only while ADX14 > 25; stop 2 ATR, target 6 ATR
  hour_drift (H1)       20-yr Dukascopy H1: per UTC hour average return; hours with |t| >= 2 in the first 60 % are
                        traded (hold one hour, long if positive) in the last 40 %
  grid_basket (M15)     H1 trend up (close > EMA50): buy 0.01; each further $3 against adds 0.01 (max 15); close the
                        basket at +$2 average profit per ounce held; mirror in a down trend.  Measured in $ on $15K.
Data: FundedNext M1 (Jul-Oct 2026) and M5 (May 2025-Oct 2026) with their own spreads; M15 built from M5; second source =
other broker M5/M15.  60 / 40 split; pass = unseen mean R > 0, t >= 2, > 0 at 2x costs and on the second source.
Read-only.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import full_reassessment as fr
from backtest import gold_cheatsheet_test as cs
from backtest import grand_sweep as gs
from backtest import strategy_round5 as r5
from backtest import strategy_round7 as r7
from backtest.btc_strategies import base_atr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> pd.DataFrame:
    d = cs.load_m5(path)
    if "tick_volume" not in d:
        d["tick_volume"] = 1.0
    return d


def to(d: pd.DataFrame, rule: str) -> pd.DataFrame:
    g = d.groupby(d.time.dt.floor(rule))
    return pd.DataFrame({"time": g.time.first().index, "open": g.open.first().values, "high": g.high.max().values,
                         "low": g.low.min().values, "close": g.close.last().values, "spr": g.spr.median().values,
                         "tick_volume": g.tick_volume.sum().values})


def local_min(p: pd.DataFrame, tz: str) -> np.ndarray:
    t = p.time.dt.tz_convert(tz)
    return (t.dt.hour * 60 + t.dt.minute).to_numpy(), t.dt.date.to_numpy()


def orb(p, tz, start, minutes, end, kind):
    o, h, l, c = (p[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    mins, day = local_min(p, tz)
    n = len(c)
    sig = []
    flat = (mins >= end) & np.concatenate([[False], mins[:-1] < end])
    i = 0
    while i < n:
        if mins[i] < start or mins[i] >= start + minutes:
            i += 1
            continue
        d0, j = day[i], i
        while j < n and day[j] == d0 and mins[j] < start + minutes:
            j += 1
        hi, lo = h[i:j].max(), l[i:j].min()
        k = j
        while k < n and day[k] == d0 and mins[k] < end - 5:
            if kind == "mom":
                pass
            if c[k] > hi:
                sig.append((k, 1, lo, hi - lo)); break
            if c[k] < lo:
                sig.append((k, -1, hi, hi - lo)); break
            k += 1
        while k < n and day[k] == d0:
            k += 1
        i = k
    return sig, flat


def momentum(p):
    o, h, l, c = (p[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    mins, day = local_min(p, "America/New_York")
    flat = (mins >= 930) & np.concatenate([[False], mins[:-1] < 930])
    sig = []
    for d0 in pd.unique(day):
        idx = np.flatnonzero(day == d0)
        first = idx[(mins[idx] >= 570) & (mins[idx] < 600)]
        after = idx[mins[idx] >= 600]
        if len(first) < 2 or not len(after):
            continue
        r = c[first[-1]] - o[first[0]]
        if r == 0:
            continue
        hi, lo = h[first].max(), l[first].min()
        sig.append((first[-1], int(np.sign(r)), lo if r > 0 else hi, hi - lo))
    return sig, flat


def vwap(p):
    tp = (p.high + p.low + p.close) / 3
    v = p.tick_volume.clip(lower=1)
    day = p.time.dt.normalize()
    cv = (tp * v).groupby(day).cumsum() / v.groupby(day).cumsum()
    var = ((tp - cv) ** 2 * v).groupby(day).cumsum() / v.groupby(day).cumsum()
    return cv.to_numpy(), np.sqrt(var).to_numpy()


def signals(p: pd.DataFrame, method: str):
    """-> list of (i, d, stop, target or None for R-multiple targets), flat array or None"""
    o, h, l, c = (p[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    A = base_atr(h, l, c, 14)
    hr = p.time.dt.hour.to_numpy()
    sess = (hr >= 7) & (hr < 20)
    out, flat = [], None
    lo5, hi5 = r7.lo(l, 5), r7.hi(h, 5)                       # computed once (was per signal)
    if method in ("ny_orb15", "ny_orb30", "ldn_orb30"):
        tz, st, mn, en = {"ny_orb15": ("America/New_York", 570, 15, 930), "ny_orb30": ("America/New_York", 570, 30, 930),
                          "ldn_orb30": ("Europe/London", 480, 30, 720)}[method]
        s, flat = orb(p, tz, st, mn, en, "orb")
        out = [(i, d, stop, None) for i, d, stop, _ in s]
    elif method == "ny_momentum":
        s, flat = momentum(p)
        out = [(i, d, stop, None) for i, d, stop, _ in s]
    elif method == "ema9_21_rsi":
        e9, e21, r = r7.ema(c, 9), r7.ema(c, 21), r7.rsi(c, 14)
        up, dn = r7.cross_up(e9, e21) & (r > 50) & sess, r7.cross_up(e21, e9) & (r < 50) & sess
        out = [(i, 1, lo5[i] - 0.1 * A[i], None) for i in np.flatnonzero(up)] + \
              [(i, -1, hi5[i] + 0.1 * A[i], None) for i in np.flatnonzero(dn)]
    elif method == "ema4_stack":
        e9, e21, e34, e50 = (r7.ema(c, n) for n in (9, 21, 34, 50))
        tu = pd.Series(l <= e21).rolling(3).max().to_numpy() > 0
        td = pd.Series(h >= e21).rolling(3).max().to_numpy() > 0
        up = (e9 > e21) & (e21 > e34) & (e34 > e50) & tu & (c > e9) & (r7.prev(c) <= r7.prev(e9)) & sess
        dn = (e9 < e21) & (e21 < e34) & (e34 < e50) & td & (c < e9) & (r7.prev(c) >= r7.prev(e9)) & sess
        out = [(i, 1, lo5[i] - 0.1 * A[i], None) for i in np.flatnonzero(up)] + \
              [(i, -1, hi5[i] + 0.1 * A[i], None) for i in np.flatnonzero(dn)]
    elif method in ("vwap_pullback", "vwap_fade"):
        vw, sd = vwap(p)
        if method == "vwap_pullback":
            rise, fall = vw > r7.prev(vw), vw < r7.prev(vw)
            up = rise & (l <= vw) & (c > vw) & (r7.prev(c) > r7.prev(vw)) & sess
            dn = fall & (h >= vw) & (c < vw) & (r7.prev(c) < r7.prev(vw)) & sess
            out = [(i, 1, vw[i] - A[i], None) for i in np.flatnonzero(up)] + [(i, -1, vw[i] + A[i], None) for i in np.flatnonzero(dn)]
        else:
            up, dn = (c < vw - 2 * sd) & sess, (c > vw + 2 * sd) & sess
            up = up & ~np.concatenate([[False], up[:-1]]); dn = dn & ~np.concatenate([[False], dn[:-1]])
            out = [(i, 1, c[i] - A[i], vw[i]) for i in np.flatnonzero(up)] + [(i, -1, c[i] + A[i], vw[i]) for i in np.flatnonzero(dn)]
    elif method in ("pivot_bounce", "pivot_break"):
        day = p.time.dt.normalize()
        g = p.groupby(day)
        H, L, C = (g.high.max().shift(1).reindex(day).to_numpy(), g.low.min().shift(1).reindex(day).to_numpy(),
                   g.close.last().shift(1).reindex(day).to_numpy())
        P = (H + L + C) / 3
        R1, S1, R2, S2 = 2 * P - L, 2 * P - H, P + (H - L), P - (H - L)
        if method == "pivot_bounce":
            up = (l <= S1) & (c > S1) & sess
            dn = (h >= R1) & (c < R1) & sess
            out = [(i, 1, S1[i] - A[i], P[i]) for i in np.flatnonzero(up)] + [(i, -1, R1[i] + A[i], P[i]) for i in np.flatnonzero(dn)]
        else:
            up = r7.cross_up(c, R1) & sess
            dn = r7.cross_up(S1, c) & sess
            out = [(i, 1, P[i], R2[i]) for i in np.flatnonzero(up)] + [(i, -1, P[i], S2[i]) for i in np.flatnonzero(dn)]
    elif method == "st_adx":
        st = r7._supertrend(h, l, c, base_atr(h, l, c, 10))
        adx = r7.dmi(h, l, c)[2]
        stf = np.nan_to_num(st.astype(float))
        up = (stf == 1) & (r7.prev(stf) == 0) & (adx > 25)
        dn = (stf == 0) & (r7.prev(stf) == 1) & (adx > 25)
        out = [(i, 1, c[i] - 2 * A[i], c[i] + 6 * A[i]) for i in np.flatnonzero(up)] + \
              [(i, -1, c[i] + 2 * A[i], c[i] - 6 * A[i]) for i in np.flatnonzero(dn)]
    return out, flat


def trades(p: pd.DataFrame, method: str, tp, cost_frac: float, hold: int) -> pd.DataFrame:
    n, c = len(p), p.close.to_numpy(float)
    s = r5._sig(n)
    sigs, flat = signals(p, method)
    for i, d, stop, tgt in sorted(sigs):
        if i >= n - 1 or not np.isfinite(stop):
            continue
        risk = d * (c[i] - stop)
        if risk <= 0:
            continue
        if tgt is None:
            tgt = c[i] + d * (1e9 if tp == "end" else tp) * risk
        if not np.isfinite(tgt) or d * (tgt - c[i]) <= 0:
            continue
        s["d"][i], s["stop"][i], s["tgt"][i] = d, stop, tgt
    spec = dict(s, hold=hold)
    if flat is not None:
        spec["flat"] = flat
    return r5.simulate_piece(p, spec, dict(cost_frac=cost_frac, swap_long=0.0, swap_short=0.0))


def grid(p: pd.DataFrame, start_bal: float = 15000.0) -> dict:
    """Grid basket with H1 trend filter, in dollars (0.01 lot = $1 per $1 move)."""
    c, hi, lo, spr = (p[k].to_numpy(float) for k in ("close", "high", "low", "spr"))
    h1 = p.set_index("time").close.resample("1h").last().dropna()
    tr = np.sign(h1 - h1.ewm(span=50, adjust=False).mean()); tr.index = tr.index + pd.Timedelta(hours=1)
    trend = tr.reindex(p.time, method="ffill").fillna(0).to_numpy()
    bal, legs, d, eq_low, peak = start_bal, [], 0, start_bal, start_bal
    day_pl, worst_day, blown, closes = {}, 0.0, False, 0
    days = p.time.dt.normalize().to_numpy()
    for i in range(len(c)):
        if not legs and trend[i] != 0:
            d = int(trend[i]); legs = [c[i] + (spr[i] if d > 0 else 0)]
        elif legs:
            nxt = legs[-1] - d * 3.0
            if len(legs) < 15 and ((d > 0 and lo[i] <= nxt) or (d < 0 and hi[i] >= nxt)):
                legs.append(nxt + (spr[i] if d > 0 else 0))
            avg = np.mean(legs)
            exit_px = c[i] if d > 0 else c[i] + spr[i]
            fl = d * (exit_px - avg) * len(legs) * 1.0 - 0.065 * len(legs)
            worst = d * ((lo[i] if d > 0 else hi[i] + spr[i]) - avg) * len(legs)
            eq_low = min(eq_low, bal + worst)
            if bal + worst <= start_bal * 0.90:
                blown = True
            if d * (exit_px - avg) >= 2.0:
                bal += fl; closes += 1
                day_pl[days[i]] = day_pl.get(days[i], 0.0) + fl
                legs = []
        peak = max(peak, bal)
    dd = pd.Series(day_pl)
    months = (p.time.iloc[-1] - p.time.iloc[0]).days / 30.44
    return {"baskets closed": closes, "net $": round(bal - start_bal), "$/month": round((bal - start_bal) / months),
            "lowest equity $": round(eq_low), "max floating drop $": round(start_bal - eq_low),
            "breaks 10 % max loss": blown, "open basket at end": len(legs)}


def hour_drift() -> pd.DataFrame:
    d = fr._dense_from(fr._read(ROOT / "data" / "dukascopy_H1" / "XAUUSD.vx.csv"))
    d["r"] = (d.close - d.open) / d.open
    cut = gs.split_at(d)
    cost = (0.40 * 1.5 + 0.065) / 4100
    rows = []
    for hr, g in d.groupby(d.time.dt.hour):
        a, b = g[g.time < cut].r, g[g.time >= cut].r
        ta = a.mean() / a.std() * np.sqrt(len(a))
        rows.append({"UTC hour": hr, "first 60 % avg %": round(100 * a.mean(), 4), "t": round(ta, 2),
                     "unseen avg %": round(100 * b.mean(), 4), "unseen t": round(b.mean() / b.std() * np.sqrt(len(b)), 2),
                     "cost %": round(100 * cost, 4)})
    return pd.DataFrame(rows)


def main() -> int:
    m5, m5b = load(ROOT / "data" / "gold" / "fn_XAUUSD_M5.csv"), load(ROOT / "data" / "xauusd_m5_5y.csv")
    m1 = load(ROOT / "data" / "gold" / "fn_XAUUSD_M1.csv")
    sets = {"M1": (m1, None), "M5": (m5, m5b), "M15": (to(m5, "15min"), to(m5b, "15min"))}
    cf = lambda d: (d.spr.median() * 1.5 + 0.065) / d.close.median()
    plan = {"ny_orb15": [1.0, 2.0, "end"], "ny_orb30": [1.0, 2.0, "end"], "ldn_orb30": [1.0, 2.0, "end"],
            "ny_momentum": ["end"], "ema9_21_rsi": [1.5, 2.0], "ema4_stack": [1.5, 2.0], "vwap_pullback": [2.0],
            "vwap_fade": [None], "pivot_bounce": [None], "pivot_break": [None], "st_adx": [None]}
    hold = {"M1": 240, "M5": 96, "M15": 64}
    rows = []
    for tf, (df, alt) in sets.items():
        cut = gs.split_at(df)
        cut_a = gs.split_at(alt) if alt is not None else None
        months = (df.time.iloc[-1] - df.time.iloc[0]).days / 30.44
        for m, tps in plan.items():
            if m in ("ny_orb15",) and tf == "M15":
                continue
            for tp in tps:
                t = trades(df, m, tp, cf(df), hold[tf])
                if len(t) < 10:
                    continue
                t2 = trades(df, m, tp, 2 * cf(df), hold[tf])
                ta = trades(alt, m, tp, cf(alt), hold[tf]) if alt is not None else None
                d_, c_ = t[t.entry < cut], t[t.entry >= cut]
                rows.append({"tf": tf, "method": m, "target": "fixed" if tp is None else (f"{tp:g}R" if tp != "end" else "session end"),
                             "trades/month": round(len(t) / months, 1), "win %": round(100 * float((t.R > 0).mean()), 1),
                             "first 60 %": round(float(d_.R.mean()), 3) if len(d_) else None,
                             "unseen 40 %": round(float(c_.R.mean()), 3) if len(c_) else None,
                             "unseen t": round(float(c_.R.mean() / c_.R.std() * np.sqrt(len(c_))), 2) if len(c_) > 2 else None,
                             "2x cost unseen": round(float(t2[t2.entry >= cut].R.mean()), 3) if len(t2) and (t2.entry >= cut).any() else None,
                             "2nd source unseen": round(float(ta[ta.entry >= cut_a].R.mean()), 3) if ta is not None and len(ta) and (ta.entry >= cut_a).any() else None})
        print(f"{tf} done", flush=True)
    P = pd.DataFrame(rows)
    P["PASS"] = (P["unseen 40 %"] > 0) & (P["unseen t"] >= 2) & (P["2x cost unseen"] > 0) & \
        (P["2nd source unseen"].isna() | (P["2nd source unseen"] > 0))
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.max_rows", 200):
        print(P.to_string(index=False))
        print("\nHOUR-OF-DAY DRIFT, gold H1 Dukascopy 2005-2026 (avg % per hour vs cost per trade):")
        print(hour_drift().to_string(index=False))
        print("\nGRID BASKET (M15, H1 trend filter, 0.01 per $3 against, max 15, close at +$2 avg) on $15K:")
        for lab, d in (("FundedNext M15", to(m5, "15min")), ("other broker M15", to(m5b, "15min"))):
            print(lab, grid(d))
    return 0


if __name__ == "__main__":
    sys.exit(main())
