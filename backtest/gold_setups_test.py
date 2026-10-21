"""
Gold setups the owner asked about (2026-10-21): comparative (SMT) divergence, trend pullback, mid-range / range-bound.

    python -m backtest.gold_setups_test

PRE-REGISTERED (written before the first run):
  New setups
    smt_silver      gold makes a new 20-bar high while silver does NOT -> SELL; new 20-bar low while silver does not -> BUY
    smt_dxy (D1)    gold new 20-day low while the dollar index is NOT at a new 20-day high -> BUY; mirror -> SELL
    pullback_strict trend = close > EMA200, EMA50 > EMA200, ADX14 > 20; the low touched EMA20 (within 0.25 ATR) in the
                    last 3 bars; trigger = close above the previous bar's high -> BUY; mirror -> SELL
    range_mid_bb    only when ADX14 < 20 (no trend): close below the lower Bollinger band (20, 2) -> BUY, above the upper
                    -> SELL; target = the 20-bar average (the middle of the range)
    range_mid_dc    only when ADX14 < 20: close in the bottom 10 % of the 20-bar Donchian range -> BUY, top 10 % -> SELL;
                    target = the middle of that range
  Existing rules of the same families (strategy_round7): rsi_divergence; ema20_pullback, fib_618_pullback, rsi2_trend_dip;
    bollinger_reentry, keltner_fade, rsi14_30_70, zscore_2, stretch_from_sma50, williams_r, cci_100_reversal.
  Exits    ATR setups: stop 1 / 1.5 / 2 / 3 x target 1 / 2 / 3 / 4 / 6 / 8 ATR.  Mid-range setups: stop 1 / 1.5 / 2 / 3 ATR,
           target = the range middle.  Time exit 96 (H1), 48 (H4), 20 (D1) bars.
  Data     gold + silver H1 Dukascopy 2005-, H4 built from them; D1 gold / silver futures (Yahoo 2000-) and DX-Y.NYB.
           Second supplier: FundingPips gold H1 2010- (not for the silver / dollar setups).
  Costs    FundedNext-like: $0.40 spread x 1.5 + $6.5 / lot commission = 0.016 % of price per trade; also at 2x.
  Split    60 / 40 per setup family x timeframe; best discovery variant must have confirm mean R > 0, t >= 2, > 0 at 2x
           costs, and > 0 on the second supplier where it exists.
Read-only; writes reports/gold_setups_test.json.
"""
from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import fp_strategy_build as fb
from backtest import full_reassessment as fr
from backtest import grand_sweep as gs
from backtest import strategy_round5 as r5
from backtest import strategy_round7 as r7
from backtest.btc_strategies import base_atr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
COST = dict(cost_frac=(0.40 * 1.5 + 0.065) / 4100, swap_long=0.0, swap_short=0.0)
HOLD = {"H1": 96, "H4": 48, "D1": 20}
EXISTING = {"divergence": ["rsi_divergence"], "pullback": ["ema20_pullback", "fib_618_pullback", "rsi2_trend_dip"],
            "range": ["bollinger_reentry", "keltner_fade", "rsi14_30_70", "zscore_2", "stretch_from_sma50", "williams_r",
                      "cci_100_reversal"]}
NEW = {"divergence": ["smt_silver", "smt_dxy"], "pullback": ["pullback_strict"], "range": ["range_mid_bb", "range_mid_dc"]}
MID = {"range_mid_bb", "range_mid_dc"}


def adx(h, l, c, n=14):
    return r7.dmi(h, l, c, n)[2]


def new_setups(p: pd.DataFrame, other: np.ndarray | None, dxy: np.ndarray | None) -> dict:
    """name -> (buy, sell, mid-target or None); arrays use bars <= j only."""
    o, h, l, c = (p[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    A = base_atr(h, l, c, 14)
    ax = adx(h, l, c)
    e20, e50, e200 = r7.ema(c, 20), r7.ema(c, 50), r7.ema(c, 200)
    out = {}
    hh, ll = r7.prev(r7.hi(h, 20)), r7.prev(r7.lo(l, 20))
    if other is not None:
        oh, ol = r7.prev(r7.hi(other, 20)), r7.prev(r7.lo(other, 20))
        out["smt_silver"] = ((l < ll) & ~(other < ol), (h > hh) & ~(other > oh), None)
    if dxy is not None:
        dh, dl = r7.prev(r7.hi(dxy, 20)), r7.prev(r7.lo(dxy, 20))
        out["smt_dxy"] = ((l < ll) & ~(dxy > dh), (h > hh) & ~(dxy < dl), None)
    up = (c > e200) & (e50 > e200) & (ax > 20)
    dn = (c < e200) & (e50 < e200) & (ax > 20)
    touch_up = pd.Series(l <= e20 + 0.25 * A).rolling(3).max().to_numpy() > 0
    touch_dn = pd.Series(h >= e20 - 0.25 * A).rolling(3).max().to_numpy() > 0
    out["pullback_strict"] = (up & touch_up & (c > r7.prev(h)), dn & touch_dn & (c < r7.prev(l)), None)
    flat = ax < 20
    s20 = r7.sma(c, 20)
    sd = pd.Series(c).rolling(20).std(ddof=0).to_numpy()
    out["range_mid_bb"] = (flat & (c < s20 - 2 * sd), flat & (c > s20 + 2 * sd), s20)
    dhi, dlo = r7.hi(h, 20), r7.lo(l, 20)
    pos = (c - dlo) / np.where(dhi > dlo, dhi - dlo, np.nan)
    out["range_mid_dc"] = (flat & (pos <= 0.10), flat & (pos >= 0.90), (dhi + dlo) / 2)
    return out


def run(p: pd.DataFrame, tf: str, other, dxy, cost) -> dict:
    o, h, l, c = (p[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    A = base_atr(h, l, c, 14)
    R7 = r7.rules(o, h, l, c)
    sets = {k: (R7[k][0], R7[k][1], None) for fam in EXISTING.values() for k in fam}
    sets.update(new_setups(p, other, dxy))
    res = {}
    for name, (buy, sell, mid) in sets.items():
        buy, sell = np.nan_to_num(np.asarray(buy, float)) > 0, np.nan_to_num(np.asarray(sell, float)) > 0
        buy, sell = buy & ~np.concatenate([[False], buy[:-1]]), sell & ~np.concatenate([[False], sell[:-1]])
        exits = [(sl, None) for sl in gs.STOPS] if name in MID else [(sl, tp) for sl in gs.STOPS for tp in gs.TARGETS]
        for sl, tp in exits:
            if tp is None:
                tl, ts = np.where(mid > c, mid, np.nan), np.where(mid < c, mid, np.nan)
                b2, s2 = buy & np.isfinite(tl), sell & np.isfinite(ts)
                s = r5._put(r5._sig(len(c)), b2, s2, c - sl * A, c + sl * A, tl, ts)
            else:
                s = r5._put(r5._sig(len(c)), buy, sell, c - sl * A, c + sl * A, c + tp * A, c - tp * A)
            t = r5.simulate_piece(p, dict(s, hold=HOLD[tf]), cost)
            if len(t):
                res.setdefault((name, sl, "mid" if tp is None else tp), []).append(t)
    return res


def joined(gold: pd.DataFrame, other: pd.DataFrame | None) -> pd.DataFrame:
    if other is None:
        return gold.assign(oth=np.nan)
    o = other[["time", "high", "low", "close"]].rename(columns={"high": "oh", "low": "ol", "close": "oth"})
    return gold.merge(o, on="time", how="inner")


def all_trades(df: pd.DataFrame, tf: str, cost: dict, use_other: bool, dxy: pd.Series | None = None) -> dict:
    res: dict = {}
    parts = [df] if tf == "D1" else fr.segments(df)
    for p in parts:
        if tf == "H4":
            g = p.groupby(p.time.dt.floor("4h"))
            p = pd.DataFrame({"time": g.time.first().index, "open": g.open.first().values, "high": g.high.max().values,
                              "low": g.low.min().values, "close": g.close.last().values, "oth": g.oth.last().values})
            if len(p) < 300:
                continue
        oth = p["oth"].to_numpy(float) if use_other and p["oth"].notna().any() else None
        dx = dxy.reindex(p.time.dt.normalize()).ffill().to_numpy(float) if dxy is not None else None
        for k, v in run(p.reset_index(drop=True), tf, oth, dx, cost).items():
            res.setdefault(k, []).extend(v)
    return {k: pd.concat(v, ignore_index=True) for k, v in res.items()}


def main() -> int:
    duka = lambda n: fr._dense_from(fr._read(ROOT / "data" / "dukascopy_H1" / f"{n}.csv"))
    g_h1, s_h1 = duka("XAUUSD.vx"), duka("XAGUSD.vx")
    gs_h1 = joined(g_h1, s_h1)
    fp_h1 = joined(fb.load_h1("XAUUSD"), None)
    d1 = lambda f: r7.d1.__wrapped__(f) if hasattr(r7.d1, "__wrapped__") else None
    yg = pd.read_csv(ROOT / "data" / "yahoo_D1" / "GC_F.csv"); ys = pd.read_csv(ROOT / "data" / "yahoo_D1" / "SI_F.csv")
    yd = pd.read_csv(ROOT / "data" / "yahoo_D1" / "DX-Y.NYB.csv")
    for y in (yg, ys, yd):
        y["time"] = pd.to_datetime(y["date"], utc=True) + pd.Timedelta(hours=21)
    yg = yg.dropna(subset=["open", "high", "low", "close"])
    yg = yg[(yg.high > yg.low) & (yg.close > 0)][["time", "open", "high", "low", "close"]].sort_values("time").reset_index(drop=True)
    gd1 = joined(yg, ys.dropna(subset=["close"])[["time", "high", "low", "close"]])
    dxy = yd.set_index(yd.time.dt.normalize()).close
    sets = {("H1", "Dukascopy"): (gs_h1, True, None), ("H4", "Dukascopy"): (gs_h1, True, None),
            ("D1", "Yahoo futures"): (gd1, True, dxy)}
    alts = {"H1": fp_h1, "H4": fp_h1}
    rows = []
    for (tf, src), (df, use_o, dx) in sets.items():
        cut = gs.split_at(df)
        tr = all_trades(df, tf, COST, use_o, dx)
        tr2 = all_trades(df, tf, dict(COST, cost_frac=COST["cost_frac"] * 2), use_o, dx)
        ta = all_trades(alts[tf], tf, COST, False) if tf in alts else {}
        for key, t in tr.items():
            disc, conf = t[t.entry < cut], t[t.entry >= cut]
            c2, a = tr2.get(key), ta.get(key)
            fam = next(f for f, names in {**{k: v for k, v in EXISTING.items()}}.items() if key[0] in names) \
                if any(key[0] in v for v in EXISTING.values()) else next(f for f, v in NEW.items() if key[0] in v)
            rows.append({"tf": tf, "family": fam, "setup": key[0], "new": key[0] in sum(NEW.values(), []), "sl": key[1], "tp": key[2],
                         **{f"disc_{k}": v for k, v in gs.summ(disc).items()}, **{f"conf_{k}": v for k, v in gs.summ(conf).items()},
                         "conf_2x": round(float(c2[c2.entry >= cut].R.mean()), 4) if c2 is not None and (c2.entry >= cut).any() else None,
                         "alt_conf": round(float(a[a.entry >= cut].R.mean()), 4) if a is not None and (a.entry >= cut).sum() >= 5 else None})
        print(f"{tf}: {len(tr)} variants", flush=True)
    C = pd.DataFrame(rows)
    N = len(C)
    min_n = {"H1": 100, "H4": 60, "D1": 30}
    picks = []
    for (tf, fam), g in C.groupby(["tf", "family"]):
        ok = g[g.disc_n >= min_n[tf]]
        if ok.empty:
            continue
        b = ok.sort_values("disc_meanR", ascending=False).iloc[0].to_dict()
        alt = b.get("alt_conf")
        b["PASS"] = bool((b.get("conf_meanR") or -1) > 0 and (b.get("conf_t") or 0) >= 2 and (b.get("conf_2x") or -1) > 0
                         and (alt is None or (isinstance(alt, float) and math.isnan(alt)) or alt > 0))
        sr = (b.get("conf_t") or 0.0) / math.sqrt(max(1, b.get("conf_n") or 1))
        b["DSR"] = round(gs.dsr(sr, int(b.get("conf_n") or 0), N), 3)
        picks.append(b)
    P = pd.DataFrame(picks)
    cols = ["tf", "family", "setup", "sl", "tp", "disc_n", "disc_meanR", "conf_n", "conf_meanR", "conf_t", "conf_2x", "alt_conf", "DSR", "PASS"]
    Cc = C[C.conf_n >= 20]
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 200):
        print(f"\nvariants tried {N}\nBEST-IN-DISCOVERY PER FAMILY x TIMEFRAME:"); print(P[cols].to_string(index=False))
        print("\nEVERY SETUP, unseen 40 % (avg over its exits): mean R / share profitable / variants")
        s = Cc.groupby(["family", "setup", "tf"]).conf_meanR.agg(["mean", lambda x: (x > 0).mean(), "size"]).round(3)
        print(s.unstack("tf").to_string())
    (ROOT / "reports" / "gold_setups_test.json").write_text(json.dumps({"variants": N, "picks": picks}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
