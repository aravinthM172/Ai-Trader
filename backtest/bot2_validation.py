"""
Bot 2 (swing book: gold + NDX100 + ETH) -- maximum validation on data the components were NOT chosen on (owner 2026-10-22:
"paper trading is not possible, do max tests with previously unseen data").

    python -m backtest.bot2_validation

Components (fixed now; exits = the cells used in the Book):
  G_pb    gold  H4 ema20_pullback            stop 1 / target 6 ATR
  G_macd  gold  H4 macd_zero_trend           stop 2 / target 6
  G_rr    gold  H4 role reversal + daily trend, target 3R
  G_kc    gold  D1 keltner_breakout          stop 1.5 / target 4
  G_dc    gold  D1 donchian55_breakout       stop 1.5 / target 4
  N_rsi2  NDX   D1 rsi2_trend_dip            stop 2 / target 4
  N_pbs   NDX   D1 pullback_strict           stop 3 / target 4
  E_200   ETH   D1 price_sma200_cross        stop 2 / target 3
  E_rr    ETH   D1 role reversal + daily trend, target 3R
Tests
  T1 untouched older history: NDX D1 1985-10..2010-11 (Yahoo ^NDX), gold D1 2000-08..2009-03 (Yahoo GC=F), gold H4 2005-01..2009-03
     (Dukascopy) -- before the verified data the components were chosen on
  T2 independent source over the selection period: Yahoo GC=F / ^NDX / ETH-USD daily
  T3 related markets never used: gold rules on silver (Dukascopy XAGUSD H4, Yahoo SI=F D1); NDX rules on S&P 500 (^GSPC 1950-) and
     Dow (^DJI); ETH rules on BTC (Yahoo BTC-USD)
  T4 share of calendar years with positive R (verified data)
  T5 random-entry benchmark: 200 runs with the same number of entries at random bars, same exits -> share >= the real result
  T6 3x costs (verified data)
  T7 Bot 1 (live) + Bot 2 together, FundedNext 2-step, from 2018
Costs: FundedNext fractions (gold 0.018 %, NDX 0.008 %, ETH 0.071 %; silver 0.04 %, S&P / Dow 0.008 %, BTC 0.075 %).  Read-only.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import all_pairs_verified as ap
from backtest import book_test as bt
from backtest import full_reassessment as fr
from backtest import gold_setups_test as gst
from backtest import next_account_sim as nas
from backtest import role_reversal_test as rr
from backtest import strategy_round5 as r5
from backtest import strategy_round7 as r7
from backtest.btc_strategies import base_atr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
COMP = {"G_pb": ("XAUUSD", "H4", "ema20_pullback", 1.0, 6.0), "G_macd": ("XAUUSD", "H4", "macd_zero_trend", 2.0, 6.0),
        "G_rr": ("XAUUSD", "H4", "rr_trend", None, 3.0), "G_kc": ("XAUUSD", "D1", "keltner_breakout", 1.5, 4.0),
        "G_dc": ("XAUUSD", "D1", "donchian55_breakout", 1.5, 4.0), "N_rsi2": ("NDX100", "D1", "rsi2_trend_dip", 2.0, 4.0),
        "N_pbs": ("NDX100", "D1", "pullback_strict", 3.0, 4.0), "E_200": ("ETHUSD", "D1", "price_sma200_cross", 2.0, 3.0),
        "E_rr": ("ETHUSD", "D1", "rr_trend", None, 3.0)}
HOLD = {"H4": 48, "D1": 20}


def yahoo(name: str) -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data" / "yahoo_D1" / f"{name}.csv")
    d["time"] = pd.to_datetime(d["date"], utc=True) + pd.Timedelta(hours=21)
    d = d.dropna(subset=["open", "high", "low", "close"])
    d = d[(d.high > d.low) & (d.close > 0) & (d.open > 0)]
    d = d.assign(high=d[["open", "high", "close"]].max(axis=1), low=d[["open", "low", "close"]].min(axis=1))
    return d[["time", "open", "high", "low", "close"]].sort_values("time").drop_duplicates("time").reset_index(drop=True)


def h4_from_h1(name: str) -> pd.DataFrame:
    d = fr._dense_from(fr._read(ROOT / "data" / "dukascopy_H1" / f"{name}.csv"))
    out = []
    for p in fr.segments(d):
        g = p.groupby(p.time.dt.floor("4h"))
        out.append(pd.DataFrame({"time": g.time.first().index, "open": g.open.first().values, "high": g.high.max().values,
                                 "low": g.low.min().values, "close": g.close.last().values}))
    return pd.concat(out, ignore_index=True)


def signals(p: pd.DataFrame, tf: str, rule: str, sl, tp):
    o, h, l, c = (p[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    A = base_atr(h, l, c, 14)
    if rule == "rr_trend":
        d, stop, risk = rr.signals(p, tf, True)
        return d, stop, c + d * tp * risk
    if rule == "pullback_strict":
        buy, sell, _ = gst.new_setups(p, None, None)["pullback_strict"]
    else:
        buy, sell, _ = r7.rules(o, h, l, c)[rule]
    buy, sell = np.nan_to_num(np.asarray(buy, float)) > 0, np.nan_to_num(np.asarray(sell, float)) > 0
    buy, sell = buy & ~np.concatenate([[False], buy[:-1]]), sell & ~np.concatenate([[False], sell[:-1]])
    d = np.where(buy, 1, np.where(sell, -1, 0))
    return d, c - d * sl * A, c + d * tp * A


def run(df: pd.DataFrame, tf: str, rule: str, sl, tp, cf: float, d_override=None) -> pd.DataFrame:
    out = []
    for p in ([df] if tf == "D1" else [p for p in fr.segments(df) if len(p) > 300]):
        p = p.reset_index(drop=True)
        d, stop, tgt = signals(p, tf, rule, sl, tp)
        if d_override is not None:
            d = d_override(p, d)
            c, A = p.close.to_numpy(float), base_atr(p.high.to_numpy(float), p.low.to_numpy(float), p.close.to_numpy(float), 14)
            stop, tgt = c - d * (sl or 2.0) * A, c + d * (tp if sl else 2 * tp) * A
        s = r5._sig(len(p))
        ok = d != 0
        s["d"][ok], s["stop"][ok], s["tgt"][ok] = d[ok], stop[ok], tgt[ok]
        t = r5.simulate_piece(p, dict(s, hold=HOLD[tf]), dict(cost_frac=cf, swap_long=0.0, swap_short=0.0))
        if len(t):
            out.append(t)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["entry", "exit", "R", "cost_R"])


def stat(t: pd.DataFrame) -> str:
    if len(t) < 8:
        return f"n={len(t)}"
    tt = t.R.mean() / t.R.std() * np.sqrt(len(t)) if t.R.std() > 0 else 0
    return f"{t.R.mean():+.3f}R t{tt:+.1f} n{len(t)}"


def window(df, a=None, b=None):
    m = np.ones(len(df), bool)
    if a is not None:
        m &= df.time >= pd.Timestamp(a, tz="UTC")
    if b is not None:
        m &= df.time < pd.Timestamp(b, tz="UTC")
    return df[m].reset_index(drop=True)


def main() -> int:
    ver = {("XAUUSD", "H4"): ap.data("XAUUSD", "H4"), ("XAUUSD", "D1"): ap.data("XAUUSD", "D1"),
           ("NDX100", "D1"): ap.data("NDX100", "D1"), ("ETHUSD", "D1"): ap.data("ETHUSD", "D1")}
    y = {k: yahoo(k) for k in ("GC_F", "SI_F", "_NDX", "_GSPC", "_DJI", "ETH-USD", "BTC-USD")}
    duk_g4, duk_s4 = h4_from_h1("XAUUSD.vx"), h4_from_h1("XAGUSD.vx")
    sources = {   # component market -> (T1 untouched, T2 independent, T3 related list)
        ("XAUUSD", "H4"): (window(duk_g4, None, "2009-03-15"), None, [("silver H4", duk_s4, 0.0004)]),
        ("XAUUSD", "D1"): (window(y["GC_F"], None, "2009-03-15"), window(y["GC_F"], "2009-03-15"), [("silver D1", y["SI_F"], 0.0004)]),
        ("NDX100", "D1"): (window(y["_NDX"], None, "2010-11-14"), window(y["_NDX"], "2010-11-14"),
                           [("S&P500 D1 1950-", window(y["_GSPC"], "1950-01-01"), 0.00008), ("Dow D1", y["_DJI"], 0.00008)]),
        ("ETHUSD", "D1"): (None, y["ETH-USD"], [("BTC D1", y["BTC-USD"], 0.00075)])}
    rows, comps_ver = [], {}
    rng = np.random.default_rng(7)
    cache = ROOT / "reports" / "bot2_validation"
    cache.mkdir(parents=True, exist_ok=True)
    for name, (sym, tf, rule, sl, tp) in COMP.items():
        cf = ap.COST[sym]
        v = run(ver[(sym, tf)], tf, rule, sl, tp, cf)
        comps_ver[name] = v
        f = cache / f"{name}.json"
        if f.exists():
            rows.append(pd.read_json(f, typ="series").to_dict())
            continue
        t1, t2, t3 = sources[(sym, tf)]
        yr = v.groupby(v.entry.dt.year).R.sum()
        # T5 random entries: same count, random bars, same exits (direction kept from the real signals' mix)
        real = v.R.mean()
        df = ver[(sym, tf)]
        better = 0
        for _ in range(200):
            def rand(p, d):
                k = int((d != 0).sum())
                r_ = np.zeros(len(p), np.int64)
                idx = rng.choice(np.arange(60, len(p) - 2), size=min(k, len(p) - 62), replace=False)
                r_[idx] = np.where(rng.random(len(idx)) < (d[d != 0] > 0).mean() if k else 0.5, 1, -1)
                return r_
            rt = run(df, tf, rule if rule != "rr_trend" else "ema20_pullback", sl or 2.0, tp if sl else 2 * tp, cf, d_override=rand)
            better += int(len(rt) and rt.R.mean() >= real)
        row = {"component": name, "market": f"{sym} {tf}", "verified (selection data)": stat(v),
               "T1 untouched history": stat(run(t1, tf, rule, sl, tp, cf)) if t1 is not None and len(t1) > 260 else "n/a",
               "T2 independent source": stat(run(t2, tf, rule, sl, tp, cf)) if t2 is not None else "n/a",
               "T3 related markets": "; ".join(f"{lab}: {stat(run(d, tf, rule, sl, tp, c))}" for lab, d, c in t3),
               "T4 years +": f"{int((yr > 0).sum())}/{len(yr)}",
               "T5 random beats it": f"{100 * better / 200:.1f} %",
               "T6 3x cost": f"{(v.R - 2 * v.cost_R).mean():+.3f}R"}
        rows.append(row)
        pd.Series(row).to_json(f)
        print(f"{name} done", flush=True)
    P = pd.DataFrame(rows)
    with pd.option_context("display.width", 320, "display.max_columns", 20, "display.max_colwidth", 70):
        print(P.to_string(index=False))
    # T7 Bot 1 + Bot 2 on the FundedNext challenge
    live = bt.live_trades()
    b2 = bt.book({k: v for k, v in comps_ver.items()}, max_open=4)
    both = bt.book({"live": live, **comps_ver}, max_open=8)
    res = [bt.evaluate(bt.book({"live": live}), 0.0035, "Bot 1 alone 0.35 %"),
           bt.evaluate(b2, 0.0025, "Bot 2 alone 0.25 %"),
           bt.evaluate(both, 0.0025, "Bot 1 + Bot 2, 0.25 %")]
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(pd.DataFrame(res).to_string(index=False))
    P.to_csv(ROOT / "reports" / "bot2_validation.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
