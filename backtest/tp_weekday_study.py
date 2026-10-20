"""
Target size x weekday x trades-at-a-time x risk study for the LIVE multi bot (owner request 2026-10-20).

    python -m backtest.tp_weekday_study

Owner asked: smaller take-profit, more trades at once (e.g. 2 BTC trades from the start), bigger lots, and the
average chance of a trade reaching its target -- per weekday -- on the longest data we have.

Rules = live: momentum_rsi_mtf H1 signal, entry next open, stop 2 ATR(14), 96-bar time exit, daily EMA200 trend
filter for gold / NDX100 / USDJPY, strong-signal RSI >= 65 / <= 35 for gold / NDX100, no crypto entries Fri 20:00 ->
Sun 21:00 UTC (the forced Friday close is NOT modelled), FundingPips spread + slippage + swap.
Grid: target 1 / 1.5 / 2 / 3 / 4 / 5 / 6 ATR  x  max open trades per market 1 / 2  x  risk 0.35 / 0.5 / 0.7 %.
Data: H1 longest clean history (gold 2005-, NDX 2011-, BTC 2014-, USDJPY / ETH FundingPips) and D1 (Yahoo, NDX 1985-)
for the weekday table.  Portfolio $ and challenge results use the common period 2018-03 -> (all 5 markets live).
Writes reports/tp_weekday_study.json and TP_WEEKDAY_STUDY.md.  Read-only; no orders.
"""
from __future__ import annotations

import json
from pathlib import Path

import numba
import numpy as np
import pandas as pd

from backtest import fp_strategy_build as fb
from backtest import full_reassessment as fr
from backtest import long_history_backtest as lh
from backtest.btc_strategies import base_atr, build_momentum
from strategy.btc_features import _rsi
from strategy.btc_h1_signal import PARAMS

ROOT = Path(__file__).resolve().parents[1]
SYMS = ("BTCUSD", "ETHUSD", "XAUUSD", "USDJPY", "NDX100")
STRONG = {"XAUUSD", "NDX100"}
CRYPTO = {"BTCUSD", "ETHUSD"}
TPS = (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)
MAX_OPEN = (1, 2)
RISKS = (0.0035, 0.005, 0.007)
ACCOUNT = 15000.0
COMMON_FROM = pd.Timestamp("2018-03-01", tz="UTC")
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


@numba.njit(cache=True)
def _outcomes(o, h, l, ent, A, sl_atr, tp_atr, hold):
    """Every signal bar's trade on its own (no one-position rule): exit bar, exit price, reason, exit-at-open."""
    n = len(o)
    m = 0
    for i in range(1, n - 1):
        if ent[i] != 0 and A[i - 1] > 0:
            m += 1
    ii, xx, why, ao = np.empty(m, np.int64), np.empty(m, np.int64), np.empty(m, np.int8), np.empty(m, np.bool_)
    xp = np.empty(m)
    j = 0
    for i in range(1, n - 1):
        d, a = ent[i], A[i - 1]
        if d == 0 or not a > 0:
            continue
        ep = o[i]
        stop, tgt = ep - d * sl_atr * a, ep + d * tp_atr * a
        x, p, w, at = -1, 0.0, 0, False                        # w: 0 time, 1 stop, 2 target
        for k in range(i, min(n, i + hold)):
            if k > i and d * (o[k] - stop) <= 0:
                x, p, w, at = k, o[k], 1, True
                break
            if (l[k] <= stop) if d > 0 else (h[k] >= stop):
                x, p, w = k, stop, 1
                break
            if k > i and d * (o[k] - tgt) >= 0:
                x, p, w, at = k, o[k], 2, True
                break
            if (h[k] >= tgt) if d > 0 else (l[k] <= tgt):
                x, p, w = k, tgt, 2
                break
        if x < 0:
            x = min(n - 1, i + hold)
            p, at = o[x], True
        ii[j], xx[j], xp[j], why[j], ao[j] = i, x, p, w, at
        j += 1
    return ii, xx, xp, why, ao


def signals(p: pd.DataFrame, sym: str, intraday: bool) -> tuple[np.ndarray, np.ndarray]:
    o, h, l, c = (p[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    ent, _ = build_momentum(c, h, l, **PARAMS)
    ent = ent.astype(np.int64)
    if sym in STRONG:                                          # live strong-signal rule (signal bar = i - 1)
        r = np.r_[np.nan, _rsi(c, 14)[:-1]]
        ent[(ent > 0) & ~(r >= 65)] = 0
        ent[(ent < 0) & ~(r <= 35)] = 0
    if sym in CRYPTO and intraday:                             # live weekend-flat: no entries Fri 20:00 -> Sun 21:00 UTC
        t = p.time
        wk = (t.dt.weekday * 24 + t.dt.hour).to_numpy()
        ent[(wk >= 4 * 24 + 20) & (wk < 6 * 24 + 21)] = 0
    return ent, base_atr(h, l, c, 14)


def all_trades(df: pd.DataFrame, sym: str, tp: float, intraday: bool = True) -> pd.DataFrame:
    """Every signal as its own trade (R after costs), with the daily-trend filter applied for the filtered markets."""
    cost = fb.cost_of(lh.SPECS[sym])
    hold = fr.HOLD
    out = []
    for p in (fr.segments(df) if intraday else [df]):
        ent, A = signals(p, sym, intraday)
        o, h, l = (p[k].to_numpy(float) for k in ("open", "high", "low"))
        ii, xx, xp, why, ao = _outcomes(o, h, l, ent, A, fr.SL_ATR, tp, hold)
        if not len(ii):
            continue
        t0, t1 = p.time.iloc[ii].reset_index(drop=True), p.time.iloc[xx].reset_index(drop=True)
        d = ent[ii].astype(float)
        ep, sl_d = o[ii], fr.SL_ATR * A[ii - 1]
        nights = ((t1 + pd.Timedelta(hours=3)).dt.normalize() - (t0 + pd.Timedelta(hours=3)).dt.normalize()).dt.days.to_numpy()
        rate = np.where(d > 0, cost["swap_long"], cost["swap_short"])
        R = (d * (xp - ep) - cost["cost_frac"] * ep + ep * rate / 365 * nights) / sl_d
        out.append(pd.DataFrame({"entry": t0, "exit": t1, "i": ii, "x": xx, "at_open": ao, "dir": d,
                                 "reason": np.array(["time", "stop", "target"])[why], "R": R, "seg": len(out)}))
    t = pd.concat(out, ignore_index=True)
    if sym in lh.LIVE_FILTER:
        reg = lh.daily_regime(df, intraday)
        day = (t.entry + pd.Timedelta(hours=3)).dt.tz_localize(None).dt.normalize() if intraday \
            else t.entry.dt.tz_localize(None).dt.normalize()
        side = reg.side.reindex(day, method="ffill").to_numpy()
        t = t[side == t.dir.to_numpy()].reset_index(drop=True)
    return t.assign(sym=sym)


def limit_open(t: pd.DataFrame, k: int) -> pd.DataFrame:
    """Keep trades in entry order while fewer than k of the kept ones are still open (k = 1 is the live bot)."""
    keep, open_ = [], []
    for row in t.sort_values(["seg", "i"]).itertuples():
        open_ = [(s, x, a) for s, x, a in open_ if s == row.seg and (row.i < x or (row.i == x and not a))]
        if len(open_) < k:
            keep.append(row.Index)
            open_.append((row.seg, row.x, row.at_open))
    return t.loc[keep].sort_values("entry").reset_index(drop=True)


def weekday_table(t: pd.DataFrame, daily: bool = False) -> dict:
    wd = (t.entry - pd.Timedelta(hours=21)).dt.weekday if daily else t.entry.dt.weekday
    g = t.groupby(wd)
    return {DAYS[d]: {"trades": int(len(x)), "target %": round(100 * (x.reason == "target").mean(), 1),
                      "stop %": round(100 * (x.reason == "stop").mean(), 1), "avg R": round(float(x.R.mean()), 3)}
            for d, x in g if len(x) >= 10}


def basic(t: pd.DataFrame) -> dict:
    yrs = max((t.entry.max() - t.entry.min()).days / 365.25, 0.1)
    return {"trades": len(t), "trades/yr": round(len(t) / yrs, 1), "target %": round(100 * (t.reason == "target").mean(), 1),
            "stop %": round(100 * (t.reason == "stop").mean(), 1), "avg R": round(float(t.R.mean()), 3),
            "R/yr": round(float(t.R.sum()) / yrs, 1)}


def money(port: pd.DataFrame, risk: float) -> dict:
    """$ on a $15K account at a fixed risk per trade, plus the FundedNext 2-step (5 % daily / 10 % max) result."""
    p = port[port.entry >= COMMON_FROM].sort_values("exit")
    yrs = (p.exit.max() - COMMON_FROM).days / 365.25
    usd = p.R * risk * ACCOUNT
    eq = usd.cumsum()
    day = usd.groupby(p.exit.dt.tz_convert(None).dt.normalize()).sum()
    last = p[p.exit >= p.exit.max() - pd.Timedelta(days=365)]
    ps = fr.prop_sim(p[["exit", "R"]], risk, daily_limit=0.05, max_loss=0.10)
    return {"$/yr": round(float(usd.sum()) / yrs), "last 12 m $": round(float(last.R.sum()) * risk * ACCOUNT),
            "max drop $": round(float((eq.cummax() - eq).max())), "worst day $": round(float(day.min())),
            "challenge pass %": round(100 * ps["outcomes"].get("pass", 0.0), 1),
            "challenge fail %": round(100 * (ps["outcomes"].get("daily", 0.0) + ps["outcomes"].get("max_loss", 0.0)), 1),
            "median days to pass": ps["median_days_to_pass"]}


def main() -> int:
    data = {s: lh.h1(s)[0] for s in SYMS}
    d1 = {s: lh.d1(s) for s in SYMS}
    res = {"h1": {}, "weekday_h1": {}, "weekday_d1": {}, "portfolio": {}, "spans": {}}
    for s in SYMS:
        res["spans"][s] = {"H1": f"{data[s].time.iloc[0].date()}..{data[s].time.iloc[-1].date()}",
                           "D1": f"{d1[s].time.iloc[0].date()}..{d1[s].time.iloc[-1].date()}"}
    for tp in TPS:
        port = {k: [] for k in MAX_OPEN}
        for s in SYMS:
            allt = all_trades(data[s], s, tp)
            for k in MAX_OPEN:
                t = limit_open(allt, k)
                port[k].append(t)
                res["h1"].setdefault(s, {}).setdefault(f"tp{tp:g}", {})[f"max{k}"] = basic(t)
                if k == 1:
                    res["weekday_h1"].setdefault(s, {})[f"tp{tp:g}"] = weekday_table(t)
            td = limit_open(all_trades(d1[s], s, tp, intraday=False), 1)
            res["weekday_d1"].setdefault(s, {})[f"tp{tp:g}"] = weekday_table(td, daily=True)
        for k in MAX_OPEN:
            pt = pd.concat(port[k], ignore_index=True)
            res["portfolio"][f"tp{tp:g}_max{k}"] = {f"risk{100 * r:g}%": money(pt, r) for r in RISKS}
            pooled = pt[pt.entry >= COMMON_FROM]
            res["portfolio"][f"tp{tp:g}_max{k}"]["pooled"] = basic(pooled)
            res["portfolio"][f"tp{tp:g}_max{k}"]["weekday"] = weekday_table(pooled) if k == 1 else None
        print(f"tp {tp:g} done", flush=True)
    (ROOT / "reports" / "tp_weekday_study.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res["portfolio"], indent=1, default=str)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
