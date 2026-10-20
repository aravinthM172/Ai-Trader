"""
Dedicated test of the grand-sweep lead: gold H4 "MACD zero-line trend" (owner request 2026-10-20).

    python -m backtest.gold_h4_macd_test

Rule (as in strategy_round7.rules, nothing changed): on a closed H4 bar, MACD(12,26) = EMA12 - EMA26 crosses above 0
while close > EMA200 -> BUY at the next open; crosses below 0 while close < EMA200 -> SELL.  One position at a time.

PRE-REGISTERED (written before any result of this script was seen):
  Exit fixed at stop 2 ATR(14) / target 6 ATR, time exit 48 H4 bars -- the middle of the region the sweep liked, not
  its best cell.  FundingPips gold / silver costs (spread + slippage + swap).
  PASS needs ALL of:
    1  silver (XAGUSD Dukascopy 2005-, a market never used for this rule): mean R > 0
    2  gold Dukascopy 2005-, H4 bars built from hour offsets 0 / 1 / 2 / 3: every offset mean R > 0
    3  parameter neighbours MACD (8,21) (12,26) (16,34) x EMA 100 / 200 / 300 on gold: >= 7 of 9 mean R > 0
    4  calendar years with >= 5 trades: >= 60 % positive
    5  third supplier (Valetax gold 2009-, not used in the sweep): mean R > 0
    6  2x costs on gold Dukascopy: mean R > 0
  Then (reported, not part of PASS): the live 5-market H1 bot with and without this rule added as a 6th strategy,
  same caps as live (max 4 open = 1.4 % at 0.35 %, max 2 per group, NDX / USDJPY buy-only), $15K from 2018-03.
Writes reports/gold_h4_macd_test.json.  Read-only; no orders.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import fp_strategy_build as fb
from backtest import full_reassessment as fr
from backtest import long_history_backtest as lh
from backtest import strategy_round5 as r5
from backtest import strategy_round7 as r7
from backtest import tp_weekday_study as tw
from backtest.btc_strategies import base_atr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
SL, TP, HOLD = 2.0, 6.0, 48


def h4(df: pd.DataFrame, offset: int = 0) -> list[pd.DataFrame]:
    """H4 bars per data segment; offset = hours added to the bar grid (0..3)."""
    out = []
    for p in fr.segments(df):
        key = (p.time - pd.Timedelta(hours=offset)).dt.floor("4h") + pd.Timedelta(hours=offset)
        g = p.groupby(key)
        b = pd.DataFrame({"time": g.time.first().index, "open": g.open.first().values, "high": g.high.max().values,
                          "low": g.low.min().values, "close": g.close.last().values})
        if len(b) > 300:
            out.append(b)
    return out


def trades(df: pd.DataFrame, sym: str, offset: int = 0, fast: int = 12, slow: int = 26, trend: int = 200,
           cost_mult: float = 1.0) -> pd.DataFrame:
    cost = fb.cost_of(lh.SPECS[sym])
    cost = dict(cost, cost_frac=cost["cost_frac"] * cost_mult)
    out = []
    for p in h4(df, offset):
        o, h, l, c = (p[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        A = base_atr(h, l, c, 14)
        macd = r7.ema(c, fast) - r7.ema(c, slow)
        e = r7.ema(c, trend).copy()
        e[:trend] = np.nan                                       # no trend verdict before the EMA has warmed up
        zero = 0 * c
        buy, sell = r7.cross_up(macd, zero) & (c > e), r7.cross_up(zero, macd) & (c < e)
        s = r5._put(r5._sig(len(c)), buy, sell, c - SL * A, c + SL * A, c + TP * A, c - TP * A)
        t = r5.simulate_piece(p, dict(s, hold=HOLD), cost)
        if len(t):
            out.append(t)
    t = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["entry", "exit", "dir", "reason", "R"])
    return t.sort_values("entry").reset_index(drop=True)


def st(t: pd.DataFrame) -> dict:
    if len(t) < 5:
        return {"n": len(t)}
    yrs = max((t.entry.max() - t.entry.min()).days / 365.25, 0.1)
    return {"n": len(t), "per_yr": round(len(t) / yrs, 1), "meanR": round(float(t.R.mean()), 3),
            "t": round(float(t.R.mean() / t.R.std() * np.sqrt(len(t))), 2), "win%": round(100 * float((t.R > 0).mean()), 1),
            "from": str(t.entry.min().date())}


# -- portfolio: the live 5-market H1 bot plus this rule ------------------------------------------
GROUP = {"BTCUSD": "crypto", "ETHUSD": "crypto", "XAUUSD": "comm", "GOLD_H4": "comm", "NDX100": "idx", "USDJPY": "fx"}


def capped(t: pd.DataFrame, max_total: int = 4, per_group: int = 2) -> pd.DataFrame:
    keep, open_ = [], []
    for r in t.sort_values(["entry", "sym"]).itertuples():
        open_ = [o for o in open_ if o[1] > r.entry or (o[1] == r.entry and not o[2])]
        if any(o[0] == r.sym for o in open_):
            continue
        if sum(GROUP[o[0]] == GROUP[r.sym] for o in open_) >= per_group or len(open_) >= max_total:
            continue
        keep.append(r.Index)
        open_.append((r.sym, r.exit, r.at_open))
    return t.loc[keep]


def portfolio(gold_h4: pd.DataFrame | None) -> dict:
    parts = []
    for k in tw.SYMS:
        t = tw.limit_open(tw.all_trades(lh.h1(k)[0], k, 6.0), 1)
        if k in ("NDX100", "USDJPY"):
            t = t[t.dir > 0]
        parts.append(t[["entry", "exit", "R", "at_open", "sym", "reason"]])
    if gold_h4 is not None:
        g = gold_h4.assign(sym="GOLD_H4", at_open=gold_h4.reason == "time")
        parts.append(g[["entry", "exit", "R", "at_open", "sym", "reason"]])
    p = pd.concat(parts, ignore_index=True)
    p = capped(p[p.entry >= tw.COMMON_FROM].reset_index(drop=True))
    return tw.money(p, 0.0035) | {"trades/yr": round(len(p) / ((p.exit.max() - tw.COMMON_FROM).days / 365.25), 1)}


def main() -> int:
    gold = r7.h1("XAUUSD")[0]
    silver = fr._dense_from(fr._read(ROOT / "data" / "dukascopy_H1" / "XAGUSD.vx.csv"))
    valetax = fr._dense_from(fr._read(ROOT / "data" / "xauusd_vx_H1.csv"))
    base = trades(gold, "XAUUSD")
    res = {"gold_dukascopy": st(base)}
    s1 = trades(silver, "XAGUSD")
    res["1_silver"] = st(s1)
    res["2_offsets"] = {o: st(trades(gold, "XAUUSD", offset=o)) for o in range(4)}
    res["3_neighbours"] = {f"{f}/{s} ema{e}": st(trades(gold, "XAUUSD", fast=f, slow=s, trend=e))
                           for f, s in ((8, 21), (12, 26), (16, 34)) for e in (100, 200, 300)}
    yr = base.groupby(base.entry.dt.year).R.agg(["count", "sum", "mean"])
    yr = yr[yr["count"] >= 5]
    res["4_years"] = {int(y): {"n": int(r["count"]), "R": round(float(r["sum"]), 2)} for y, r in yr.iterrows()}
    res["5_valetax"] = st(trades(valetax, "XAUUSD"))
    res["6_double_cost"] = st(trades(gold, "XAUUSD", cost_mult=2.0))
    res["fundingpips_2010"] = st(trades(fb.load_h1("XAUUSD"), "XAUUSD"))
    checks = {
        "1 silver > 0": (res["1_silver"].get("meanR") or -1) > 0,
        "2 all 4 offsets > 0": all((v.get("meanR") or -1) > 0 for v in res["2_offsets"].values()),
        "3 >= 7 of 9 neighbours > 0": sum((v.get("meanR") or -1) > 0 for v in res["3_neighbours"].values()) >= 7,
        "4 >= 60 % years positive": float((yr["sum"] > 0).mean()) >= 0.6,
        "5 Valetax > 0": (res["5_valetax"].get("meanR") or -1) > 0,
        "6 2x costs > 0": (res["6_double_cost"].get("meanR") or -1) > 0,
    }
    res["checks"], res["PASS"] = checks, all(checks.values())
    res["years_positive_share"] = round(float((yr["sum"] > 0).mean()), 2)
    res["portfolio_live"] = portfolio(None)
    res["portfolio_live_plus_gold_h4"] = portfolio(base)
    (ROOT / "reports" / "gold_h4_macd_test.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
