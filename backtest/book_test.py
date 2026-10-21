"""
"The Book" (owner 2026-10-22): the swing clusters that held up on the verified data, run together with the live bot.

    python -m backtest.book_test

PRE-REGISTERED components (exit = a middle cell of each passing cluster, not its best one):
  live    the live H1 bot: 5 markets, live rules (strong-signal gold / NDX, daily EMA200 filter gold / NDX / USDJPY, NDX / USDJPY
          buy-only, weekend-flat crypto entries), stop 2 / target 4.5 ATR, one crypto trade at a time
  g_pb    gold H4 ema20_pullback, stop 1 / target 6 ATR          g_macd  gold H4 macd_zero_trend, stop 2 / target 6
  g_kc    gold D1 keltner_breakout, stop 1.5 / target 4           g_dc55  gold D1 donchian55_breakout, stop 1.5 / target 4
  n_rsi2  NDX D1 rsi2_trend_dip, stop 2 / target 4                n_pbs   NDX D1 pullback_strict, stop 3 / target 4
  e_200   ETH D1 price_sma200_cross, stop 2 / target 3
Data: verified zip history (tools/zip_history); FundedNext costs (all_pairs_verified.COST).  Each component holds at most one
position; the book holds at most 6 positions at once.  Risk per trade 0.25 % or 0.35 % of the balance at entry.
Reported from 2018-01 (all markets live): $ / year, max drop, worst day, monthly correlation between components, and the
FundedNext 2-step challenge (5 % daily of initial, 10 % static, 8 % then 5 %) started every month, as in next_account_sim.
CAVEAT: the components were chosen with the same data's last 40 %, so this is an optimistic in-sample check; the real test is
paper trading.  Read-only.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

from backtest import all_pairs_verified as ap
from backtest import fp_strategy_build as fb
from backtest import full_reassessment as fr
from backtest import gold_setups_test as gst
from backtest import next_account_sim as nas
from backtest import strategy_round5 as r5
from backtest import strategy_round7 as r7
from backtest import tp_weekday_study as tw
from backtest.btc_strategies import base_atr
from tools import zip_history as zh

warnings.filterwarnings("ignore")
FROM = pd.Timestamp("2018-01-01", tz="UTC")
ACC = nas.ACCOUNTS["FN-2step-15K"]


def rule_trades(sym, tf, rule, sl, tp):
    df = ap.data(sym, tf)
    cf = dict(cost_frac=ap.COST[sym], swap_long=0.0, swap_short=0.0)
    out = []
    for p in ([df] if tf == "D1" else [p for p in fr.segments(df) if len(p) > 300]):
        p = p.reset_index(drop=True)
        o, h, l, c = (p[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        A = base_atr(h, l, c, 14)
        if rule == "pullback_strict":
            buy, sell, _ = gst.new_setups(p, None, None)["pullback_strict"]
        else:
            buy, sell, _ = r7.rules(o, h, l, c)[rule]
        buy, sell = np.nan_to_num(np.asarray(buy, float)) > 0, np.nan_to_num(np.asarray(sell, float)) > 0
        buy, sell = buy & ~np.concatenate([[False], buy[:-1]]), sell & ~np.concatenate([[False], sell[:-1]])
        s = r5._put(r5._sig(len(c)), buy, sell, c - sl * A, c + sl * A, c + tp * A, c - tp * A)
        t = r5.simulate_piece(p, dict(s, hold=ap.HOLD[tf]), cf)
        if len(t):
            out.append(t)
    return pd.concat(out, ignore_index=True)


def live_trades():
    parts = []
    orig = tw.fb.cost_of
    for s in tw.SYMS:
        df = zh.resample(zh.load_m1(s), "1h")
        tw.fb.cost_of = lambda sp, s=s: dict(cost_frac=ap.COST[s], swap_long=0.0, swap_short=0.0)
        t = tw.limit_open(tw.all_trades(df, s, 4.5), 1)
        if s in ("NDX100", "USDJPY"):
            t = t[t.dir > 0]
        parts.append(t[["entry", "exit", "R", "at_open", "sym"]])
    tw.fb.cost_of = orig
    t = pd.concat(parts, ignore_index=True).sort_values(["entry", "sym"]).reset_index(drop=True)
    return nas.capped(t[t.entry >= FROM - pd.Timedelta(days=30)], crypto=1)


def book(components: dict, max_open: int = 6) -> pd.DataFrame:
    rows = []
    for name, t in components.items():
        t = t[t.entry >= FROM]
        rows.append(pd.DataFrame({"entry": t.entry.reset_index(drop=True), "exit": t.exit.reset_index(drop=True),
                                  "R": t.R.reset_index(drop=True), "comp": name}))
    allt = pd.concat(rows, ignore_index=True).sort_values("entry").reset_index(drop=True)
    keep, open_ = [], []
    for r in allt.itertuples():
        open_ = [x for x in open_ if x > r.entry]
        if len(open_) >= max_open:
            continue
        keep.append(r.Index); open_.append(r.exit)
    return allt.loc[keep].reset_index(drop=True)


def evaluate(t: pd.DataFrame, risk: float, label: str) -> dict:
    t = t.sort_values("exit")
    yrs = (t.exit.max() - FROM).days / 365.25
    usd = t.R * risk * 15000
    eq = usd.cumsum()
    day = usd.groupby(t.exit.dt.tz_convert(None).dt.normalize()).sum()
    half = FROM + (t.exit.max() - FROM) * 0.6
    sz = dict(base=risk, cushion=False, taper=False, crypto=1)
    tt = t.assign(sym="X", at_open=False)
    starts = pd.date_range(FROM, t.exit.max() - pd.Timedelta(days=365), freq="MS")
    out = pd.DataFrame([nas.attempt(tt, s, ACC, sz) for s in starts])
    vc = out.outcome.value_counts(normalize=True)
    return {"setup": label, "risk": f"{100 * risk:g} %", "trades/yr": round(len(t) / yrs), "avg R": round(float(t.R.mean()), 3),
            "first 60 % R": round(float(t[t.entry < half].R.mean()), 3), "last 40 % R": round(float(t[t.entry >= half].R.mean()), 3),
            "$/yr": round(float(usd.sum()) / yrs), "last 12 m $": round(float(usd[t.exit >= t.exit.max() - pd.Timedelta(days=365)].sum())),
            "max drop $": round(float((eq.cummax() - eq).max())), "worst day $": round(float(day.min())),
            "pass %": round(100 * float(vc.get("pass", 0)), 1),
            "fail %": round(100 * float(vc.get("fail daily", 0) + vc.get("fail max loss", 0)), 1),
            "median days": int(out[out.outcome == "pass"].days.median()) if (out.outcome == "pass").any() else None}


def main() -> int:
    comps = {"live": live_trades(),
             "g_pb": rule_trades("XAUUSD", "H4", "ema20_pullback", 1.0, 6.0),
             "g_macd": rule_trades("XAUUSD", "H4", "macd_zero_trend", 2.0, 6.0),
             "g_kc": rule_trades("XAUUSD", "D1", "keltner_breakout", 1.5, 4.0),
             "g_dc55": rule_trades("XAUUSD", "D1", "donchian55_breakout", 1.5, 4.0),
             "n_rsi2": rule_trades("NDX100", "D1", "rsi2_trend_dip", 2.0, 4.0),
             "n_pbs": rule_trades("NDX100", "D1", "pullback_strict", 3.0, 4.0),
             "e_200": rule_trades("ETHUSD", "D1", "price_sma200_cross", 2.0, 3.0)}
    print("components (from 2018):")
    for k, t in comps.items():
        t = t[t.entry >= FROM]
        yrs = (t.exit.max() - FROM).days / 365.25
        print(f"  {k:7} {len(t) / yrs:6.1f} trades/yr  avg {t.R.mean():+.3f}R  win {100 * (t.R > 0).mean():.0f} %")
    m = pd.DataFrame({k: t[t.entry >= FROM].groupby(t[t.entry >= FROM].exit.dt.to_period("M")).R.sum() for k, t in comps.items()}).fillna(0)
    c = m.corr().values[np.triu_indices(len(m.columns), 1)]
    print(f"monthly R correlation between components: average {c.mean():.2f}, highest {c.max():.2f}")
    B = book(comps)
    L = book({"live": comps["live"]})
    rows = [evaluate(L, 0.0035, "live bot only"), evaluate(B, 0.0025, "BOOK"), evaluate(B, 0.0035, "BOOK")]
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
