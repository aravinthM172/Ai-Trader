"""
Gold-only strategy study on CLEAN FundingPips data (data/gold, tools/collect_gold_data).

    python -m backtest.gold_strategy_study

Why: on clean data the live H1 momentum has no edge on gold (+0.002 R, FULL_REASSESSMENT.md), and
GOLD_BEHAVIOUR.md shows gold's hourly/daily returns do not persist.  Gold is known to trend on
slower horizons (round 3: daily trend systems were positive on gold).  This tests a small,
PRE-REGISTERED set (book rules, nothing tuned; written before the first run):

  G1 turtle55        Curtis Faith, "Way of the Turtle" System 2 (55/20, 2N stop), daily, long + short
  G2 clenow_trend    Andreas Clenow, "Following the Trend" (50/100 EMA, 50-day breakout, 3 ATR trail)
  G3 tsmom12         Moskowitz-Ooi-Pedersen 2012: sign of the 12-month return, monthly, 3 ATR stop
  G4 faber_10m       Mebane Faber 2007, "A Quantitative Approach to Tactical Asset Allocation":
                     long while the month-end close is above the 10-month SMA, else flat; long only;
                     3 x ATR(20) protective stop (to define 1 R)
  G5 pdh_breakout    GOLD_BEHAVIOUR.md section 4 (the one effect real in both halves): in an uptrend
                     (yesterday's close > SMA200), buy stop at yesterday's high on the H1 chart; stop
                     = 1 daily ATR(14) below the entry; exit at the broker day's close (21:00 UTC).
                     Long only; intraday (no swap).
  G6 h1_mom_uptrend  the live H1 momentum, LONGS only, only when yesterday's close > SMA200 (daily),
                     6 ATR target (GOLD_BEHAVIOUR.md section 7: longs in an uptrend were the best split)

Data: daily = FundingPips D1 2004-today, Friday's last-hour stub merged into Friday; H1 = FundingPips
H1 2010-today.  Second source (G1-G4): Yahoo GC=F futures daily 2000-today.
Costs: FundingPips gold spread (or the Valetax median, if larger) + slippage; FundingPips swap
(long -6 %/yr, short +2.2 %/yr) per night held.
PASS RULE: round-3 rule (r3.evaluate) -- >= 10 years, >= 100 trades, mean R > 0 on full / first 70 %
/ last 30 %, >= 6/8 folds, positive at 2x costs, deflated Sharpe >= 0.95 (N = ledger), second source
> 0.  Also reported: discovery (to 2016) vs confirmation (2017-) mean R.
Writes reports/gold_strategy_study.json and GOLD_STRATEGY_STUDY.md.  Read-only; no orders.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import full_reassessment as fr
from backtest import strategy_round3_daily as r3

ROOT = Path(__file__).resolve().parents[1]
G = ROOT / "data" / "gold"
SPLIT = pd.Timestamp("2017-01-01")


# -- data -----------------------------------------------------------------------------------
def gold_daily() -> pd.DataFrame:
    d = pd.read_csv(G / "fp_XAUUSD_D1.csv")
    t = pd.to_datetime(d["time"], utc=True) + pd.Timedelta(hours=3)          # broker day opens 21:00 UTC
    date = t.dt.tz_localize(None).dt.normalize()
    sat = date.dt.dayofweek == 5                                            # Friday's last hour -> Friday
    date = date.where(~sat, date - pd.Timedelta(days=1))
    d = d.assign(date=date).groupby("date").agg(open=("open", "first"), high=("high", "max"),
                                                low=("low", "min"), close=("close", "last"))
    return r3._clean(d[d.index.dayofweek < 5])


def futures_daily() -> pd.DataFrame:
    d = pd.read_csv(G / "yahoo_gold_futures_D1.csv", parse_dates=["date"]).set_index("date")
    return r3._clean(d[["open", "high", "low", "close"]])


def gold_h1() -> pd.DataFrame:
    d = pd.read_csv(G / "fp_XAUUSD_H1.csv")
    d["time"] = pd.to_datetime(d["time"], utc=True)
    return d[d.time >= "2010-01-01"][["time", "open", "high", "low", "close"]].reset_index(drop=True)


# -- G4 Faber ------------------------------------------------------------------------------
def s_faber_10m(o, h, l, c, dates, state=None):
    n = len(c)
    A = r3.atr(h, l, c, 20)
    s = pd.Series(c, index=dates)
    month_end = s.groupby([dates.year, dates.month]).tail(1).index
    me = np.isin(dates, month_end)
    mclose = s[me]
    sma10 = mclose.rolling(10).mean()
    sig = pd.Series(np.nan, index=dates)
    sig[mclose.index] = (mclose > sma10).astype(float).where(sma10.notna())
    sig = sig.ffill().to_numpy()                       # regime known at the month-end close
    out, pos = [], 0
    for i in range(1, n):
        if pos:
            x = r3._stop_hit(1, o[i], l[i], h[i], stop)
            if x is not None:
                out.append((ei, i, 1, ep, x, risk)); pos = 0
                continue
            if me[i - 1] and sig[i - 1] == 0:          # month-end exit at the next open
                out.append((ei, i, 1, ep, o[i], risk)); pos = 0
            continue
        if me[i - 1] and sig[i - 1] == 1 and np.isfinite(A[i - 1]):
            pos, ei, ep, risk = 1, i, o[i], 3 * A[i - 1]
            stop = ep - risk
            if l[i] <= stop:
                out.append((ei, i, 1, ep, stop, risk)); pos = 0
    return out


DAILY = {"G1_turtle55": r3.s_turtle55, "G2_clenow_trend": r3.s_clenow_trend,
         "G3_tsmom12": r3.s_tsmom12, "G4_faber_10m": s_faber_10m}


# -- G5 previous-day-high breakout (H1 path) -----------------------------------------------
def pdh_breakout(h1: pd.DataFrame, daily: pd.DataFrame, cost_frac: float, cost_mult: float = 1.0) -> pd.DataFrame:
    sma200 = daily.close.rolling(200).mean()
    datr = r3.atr(daily.high.to_numpy(), daily.low.to_numpy(), daily.close.to_numpy(), 14)
    prev = pd.DataFrame(dict(ph=daily.high, up=daily.close > sma200, atr=datr), index=daily.index).shift(1)
    day = (h1.time + pd.Timedelta(hours=3)).dt.tz_localize(None).dt.normalize()
    day = day.where(day.dt.dayofweek != 5, day - pd.Timedelta(days=1))
    rows = []
    for dte, g in h1.groupby(day):
        if dte not in prev.index:
            continue
        p = prev.loc[dte]
        if not (p.up is True or p.up == 1.0) or not np.isfinite(p.atr) or not np.isfinite(p.ph):
            continue
        o, hi, lo = g.open.to_numpy(), g.high.to_numpy(), g.low.to_numpy()
        lvl, e = p.ph, None
        for k in range(len(g)):
            if hi[k] >= lvl:
                e = k
                break
        if e is None:
            continue
        ep = max(o[e], lvl)
        stop = ep - p.atr
        xp = None
        for k in range(e, len(g)):
            if (k > e and o[k] <= stop) or lo[k] <= stop:
                xp = min(o[k], stop) if k > e else stop
                break
        if xp is None:
            xp = g.close.iloc[-1]
        R = (xp - ep - cost_mult * cost_frac * ep) / p.atr
        rows.append((g.time.iloc[e], g.time.iloc[-1], 1, R))
    return pd.DataFrame(rows, columns=["entry", "exit", "dir", "R"])


# -- G6 H1 momentum, longs in an uptrend ---------------------------------------------------
def h1_mom_uptrend(h1: pd.DataFrame, daily: pd.DataFrame, cost: dict, cost_mult: float = 1.0) -> pd.DataFrame:
    t = fr.simulate(h1, 6.0, cost, cost_mult)
    up = (daily.close > daily.close.rolling(200).mean()).shift(1)
    day = t.entry.dt.tz_convert(None).dt.normalize()
    flag = up.reindex(day, method="ffill").to_numpy()
    keep = (t.dir.to_numpy() > 0) & (flag == True)        # noqa: E712  (NaN -> False)
    return t[keep].reset_index(drop=True)


def split_R(t: pd.DataFrame) -> tuple:
    e = pd.to_datetime(t.entry)
    e = e.dt.tz_convert(None) if e.dt.tz is not None else e
    a, b = t.R[e < SPLIT], t.R[e >= SPLIT]
    return (round(float(a.mean()), 4) if len(a) else None, len(a), round(float(b.mean()), 4) if len(b) else None, len(b))


def main() -> int:
    daily, fut, h1 = gold_daily(), futures_daily(), gold_h1()
    cst = fr.costs()["XAUUSD"]
    cf, sl, ss = cst["cost_frac"] / (1 + 2 * r3.SLIP_FRAC), cst["swap_long"], cst["swap_short"]  # trade_table adds slippage
    n_trials = r3.ledger_n() + r3.PRIOR_TRIALS + 6
    res = {}
    yrs_d = (daily.index[-1] - daily.index[0]).days / 365.25
    for name, fn in DAILY.items():
        t = r3.trade_table(daily, fn, cf, sl, ss)
        t2 = r3.trade_table(daily, fn, cf, sl, ss, cost_mult=2.0)
        tb = r3.trade_table(fut, fn, cf, sl, ss)
        r = r3.evaluate(t, t2, yrs_d, n_trials, float(tb.R.mean()) if len(tb) else None)
        r["futures_trades"] = len(tb)
        res[name] = (r, t)
    yrs_h = (h1.time.iloc[-1] - h1.time.iloc[0]).days / 365.25
    for name, f, args in (("G5_pdh_breakout", pdh_breakout, (cst["cost_frac"],)),
                          ("G6_h1_mom_uptrend", h1_mom_uptrend, (cst,))):
        t = f(h1, daily, *args)
        t2 = f(h1, daily, *args, cost_mult=2.0)
        r = r3.evaluate(t, t2, yrs_h, n_trials, None)
        res[name] = (r, t)
    out = {}
    for name, (r, t) in res.items():
        d_R, d_n, c_R, c_n = split_R(t)
        e = pd.to_datetime(t.entry)
        e = e.dt.tz_convert(None) if e.dt.tz is not None else e
        yr = t.assign(y=e.dt.year).groupby("y").R.agg(["count", "mean"]).round(3)
        r.update(discovery_R=d_R, discovery_n=d_n, confirmation_R=c_R, confirmation_n=c_n, **fr.risk_stats(t),
                 per_year={int(y): dict(n=int(v["count"]), R=float(v["mean"])) for y, v in yr.iterrows()})
        out[name] = r
    tested = datetime.now(timezone.utc).isoformat(timespec="seconds")
    r3.append_ledger([dict(tested_utc=tested, name=f"gold_{k}", params=json.dumps({"data": "FundingPips clean"}),
                           lookahead_ok=True, xau_exp_R=r.get("mean_R", ""), n_trials=n_trials,
                           deflated_sharpe=r.get("dsr", ""), passes=bool(r.get("passes")), useful=bool(r.get("passes")))
                      for k, r in out.items()])
    rep = dict(meta=dict(n_trials=n_trials, daily_span=f"{daily.index[0].date()}..{daily.index[-1].date()}",
                         h1_span=f"{h1.time.iloc[0].date()}..{h1.time.iloc[-1].date()}", costs=cst), results=out)
    (ROOT / "reports" / "gold_strategy_study.json").write_text(json.dumps(rep, indent=2, default=str))
    md = _md(rep)
    (ROOT / "GOLD_STRATEGY_STUDY.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def _md(rep) -> str:
    m = rep["meta"]
    L = ["# Gold strategy study -- clean FundingPips data", "",
         f"Daily {m['daily_span']}, H1 {m['h1_span']}.  Pre-registered in `backtest/gold_strategy_study.py`; ledger N = {m['n_trials']}.", "",
         "| strategy | trades | /yr | mean R | win % | PF | to 2016 (n) | 2017- (n) | first 70 % | last 30 % | WF+ | 2x cost | futures | max DD R | streak | DSR | PASS |",
         "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|"]
    for k, r in rep["results"].items():
        if r.get("mean_R") is None:
            L.append(f"| {k} | {r.get('trades')} | | too few trades | | | | | | | | | | | | | no |")
            continue
        L.append(f"| {k} | {r['trades']} | {r['trades_per_year']} | {r['mean_R']} | {round(100 * r['win_rate'])} | "
                 f"{r['profit_factor']} | {r['discovery_R']} ({r['discovery_n']}) | {r['confirmation_R']} ({r['confirmation_n']}) | "
                 f"{r['first70_R']} | {r['last30_R']} | {r['wf_positive']}/8 | {r['cost2x_R']} | {r['broker_R']} | "
                 f"{r['max_dd_R']} | {r['worst_streak']} | {r['dsr']} | {'**yes**' if r['passes'] else 'no'} |")
    L += ["", "Failed checks:", ""]
    for k, r in rep["results"].items():
        bad = [c for c, v in r.get("checks", {}).items() if not v]
        L.append(f"- {k}: {', '.join(bad) or 'none'}")
    L += ["", "## Mean R per year", ""]
    keys = list(rep["results"])
    years = sorted({y for r in rep["results"].values() for y in r.get("per_year", {})})
    L += ["| year | " + " | ".join(keys) + " |", "|---|" + "--:|" * len(keys)]
    for y in years:
        L.append(f"| {y} | " + " | ".join(
            (f"{rep['results'][k]['per_year'][y]['R']} ({rep['results'][k]['per_year'][y]['n']})"
             if y in rep["results"][k].get("per_year", {}) else "") for k in keys) + " |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
