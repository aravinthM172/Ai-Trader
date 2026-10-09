"""
Longest-history backtest of the LIVE multi strategy (added 2026-10-09).

Strategy = execution/live_multi.py: momentum_rsi_mtf on H1 (signal bar: RSI14 >= 60, 8-bar momentum > 0,
close > EMA96 -> BUY next open; mirror for SELL), stop 2 ATR, target 6 ATR, 96-bar time exit,
optional daily EMA200 trend filter (side200), FundingPips spread + slippage + swap.

Two tests:
  1. H1, the longest clean hourly history per symbol (no source has 30 years of hourly data):
       XAUUSD Dukascopy 2005-, NDX100 Dukascopy 2011-, GER40 Dukascopy 2013-, BTCUSD Bitstamp 2014-,
       ETHUSD FundingPips 2018-.
  2. D1, the same rules on DAILY candles, as far back as Yahoo goes (^NDX 1985, ^GDAXI 1987, gold futures
     2000, BTC 2014, ETH 2017).  A different holding period (96 days), so a robustness check, not the live bot.

Results are also split into BUY and SELL trades.  Every trade carries the indicator values on its signal bar.  Writes reports/long_history_trades.csv,
LONG_HISTORY_BACKTEST.md and a JSON payload for the chart page.  Read-only otherwise.

Run: python -m backtest.long_history_backtest [--export folder]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import full_reassessment as fr
from backtest import fp_strategy_build as fb
from backtest.btc_strategies import base_atr, build_momentum
from strategy.btc_features import _ema, _rsi
from strategy.btc_h1_signal import PARAMS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TP = 6.0
SYMS = ("BTCUSD", "ETHUSD", "XAUUSD", "USDJPY", "NDX100", "EURUSD", "GER40")
LIVE_FILTER = {"XAUUSD", "NDX100", "GER40", "USDJPY", "EURUSD"}   # MULTI_REGIME_SYMBOLS on the VPS (+ candidates)
CORE = ["BTCUSD", "ETHUSD", "XAUUSD", "USDJPY"]
SCENARIOS = {"Core 4": CORE, "Core + NDX100": CORE + ["NDX100"], "Core + EURUSD": CORE + ["EURUSD"],
             "Core + NDX100 + EURUSD": CORE + ["NDX100", "EURUSD"], "All 7": list(SYMS),
             "Live now (BTC ETH gold NDX)": ["BTCUSD", "ETHUSD", "XAUUSD", "NDX100"]}
RISK_USD = {"ETHUSD": 6.25}                            # $5k x 0.25 % = $12.50 per R, ETH half risk
EMA_SIG = PARAMS["ema_htf"] * 4                        # 96 -- the trend EMA build_momentum uses
SPECS = json.loads((fb.FP / "SPECS.json").read_text())


# -- data ----------------------------------------------------------------------------------------
def _duka(name: str) -> pd.DataFrame:
    return fr._dense_from(fr._read(DATA / "dukascopy_H1" / f"{name}.csv"))


def h1(sym: str) -> tuple[pd.DataFrame, str]:
    if sym == "XAUUSD":
        return _duka("XAUUSD.vx"), "Dukascopy"
    if sym == "NDX100":
        return _duka("NAS100.vx"), "Dukascopy"
    if sym == "GER40":
        return _duka("DAX40.vx"), "Dukascopy"
    if sym == "EURUSD":
        return _duka("EURUSD.vx"), "Dukascopy"
    if sym == "BTCUSD":
        return fr._dense_from(fr._read(DATA / "btcusd_bitstamp_H1.csv")), "Bitstamp"
    if sym == "USDJPY":
        return fb.load_h1(sym), "FundingPips"
    d = fb.load_h1(sym)
    return d[d.time >= "2018-01-01"].reset_index(drop=True), "FundingPips"


YAHOO = {"NDX100": "yahoo_D1/_NDX.csv", "GER40": "yahoo_D1/_GDAXI.csv", "XAUUSD": "gold/yahoo_gold_futures_D1.csv",
         "BTCUSD": "yahoo_D1/BTC-USD.csv", "ETHUSD": "yahoo_D1/ETH-USD.csv", "USDJPY": "yahoo_D1/USDJPY_X.csv",
         "EURUSD": "yahoo_D1/EURUSD_X.csv"}


def d1(sym: str) -> pd.DataFrame:
    d = pd.read_csv(DATA / YAHOO[sym])
    d["time"] = pd.to_datetime(d["date"], utc=True) + pd.Timedelta(hours=21)     # daily close ~ rollover
    d = d.dropna(subset=["open", "high", "low", "close"])
    d = d[(d.high > d.low) & (d.close > 0)]                                       # drop flat / placeholder rows
    return d[["time", "open", "high", "low", "close"]].sort_values("time").drop_duplicates("time").reset_index(drop=True)


# -- simulation + indicators ---------------------------------------------------------------------
def daily_regime(df: pd.DataFrame, intraday: bool) -> pd.DataFrame:
    """Daily close vs EMA200, known from the NEXT day on (as live: last completed daily bar)."""
    if intraday:
        day = (df.time + pd.Timedelta(hours=3)).dt.tz_localize(None).dt.normalize()
        c = df.groupby(day).close.last()
    else:
        c = df.set_index(df.time.dt.tz_localize(None).dt.normalize()).close
    e = c.ewm(span=200, adjust=False).mean()
    r = pd.DataFrame({"d_close": c, "d_ema200": e, "side": np.sign(c - e)}).iloc[200:]
    return r.shift(1).dropna()


def run(df: pd.DataFrame, sym: str, intraday: bool = True) -> pd.DataFrame:
    cost = fb.cost_of(SPECS[sym])
    out = []
    for p in fr.segments(df) if intraday else [df]:
        t = fr.simulate_piece(p, TP, cost)
        if not len(t):
            continue
        h, l, c = (p[k].to_numpy(float) for k in ("high", "low", "close"))
        ind = pd.DataFrame({"rsi": _rsi(c, 14), "mom": pd.Series(c).diff(PARAMS["mom_win"]).to_numpy(),
                            "ema96": _ema(c, EMA_SIG), "atr": base_atr(h, l, c, 14), "sig_close": c})
        i = pd.Index(p.time).get_indexer(t.entry) - 1                              # signal bar = bar before entry
        t = pd.concat([t.reset_index(drop=True), ind.iloc[i].reset_index(drop=True)], axis=1)
        o = p.open.to_numpy(float)
        x = pd.Index(p.time).get_indexer(t.exit)
        t["entry_px"], t["exit_px_open"] = o[i + 1], o[x]
        out.append(t)
    t = pd.concat(out, ignore_index=True)
    reg = daily_regime(df, intraday)
    day = (t.entry + pd.Timedelta(hours=3)).dt.tz_localize(None).dt.normalize() if intraday \
        else t.entry.dt.tz_localize(None).dt.normalize()
    rg = reg.reindex(day, method="ffill").reset_index(drop=True)
    t = pd.concat([t, rg], axis=1)
    t["trend_ok"] = t.side.to_numpy() == t.dir.to_numpy()
    return t.assign(sym=sym)


# -- reporting -----------------------------------------------------------------------------------
def stats(t: pd.DataFrame) -> dict:
    if len(t) < 20:
        return {"trades": len(t)}
    t = t.sort_values("entry")
    yrs = max((t.entry.max() - t.entry.min()).days / 365.25, 0.1)
    half = t.entry.min() + (t.entry.max() - t.entry.min()) / 2
    eq = t.R.cumsum()
    yr = t.groupby(t.entry.dt.year).R.sum()
    return {"from": str(t.entry.min().date()), "years": round(yrs, 1), "trades": len(t), "trades/yr": round(len(t) / yrs),
            "win %": round(100 * (t.R > 0).mean(), 1), "avg R": round(t.R.mean(), 3), "total R": round(t.R.sum(), 1),
            "t": round(t.R.mean() / t.R.std() * np.sqrt(len(t)), 2), "max DD R": round(float((eq.cummax() - eq).max()), 1),
            "1st half": round(t[t.entry < half].R.mean(), 3), "2nd half": round(t[t.entry >= half].R.mean(), 3),
            "last 2y": round(t[t.entry >= t.entry.max() - pd.Timedelta(days=730)].R.mean(), 3),
            "years +": f"{(yr > 0).sum()}/{len(yr)}"}


def live_rows(t: pd.DataFrame) -> pd.DataFrame:
    return t[t.trend_ok] if t.sym.iloc[0] in LIVE_FILTER else t


def windows(sym: str, df: pd.DataFrame, t: pd.DataFrame, n_last: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """H1 candles around chosen trades (live config: last n + 5 best + 5 worst) with EMA96 and RSI14.
    Returns (candles, picked trades with their window indexes)."""
    c, h, l = (df[k].to_numpy(float) for k in ("close", "high", "low"))
    ema, rsi = _ema(c, EMA_SIG), _rsi(c, 14)
    pos = pd.Index(df.time)
    lt = live_rows(t).sort_values("entry")
    pick = pd.concat([lt.tail(n_last), lt.nlargest(5, "R"), lt.nsmallest(5, "R")]).drop_duplicates("entry")
    bars, picked = [], []
    for r in pick.sort_values("entry").itertuples():
        a, b = pos.get_indexer([r.entry])[0], pos.get_indexer([r.exit])[0]
        if a < 0 or b < 0:
            continue
        lo, hi = max(0, a - 60), min(len(df), b + 12)
        if hi - lo > 220:                                    # long holds: keep the end, cut the lead-in
            lo = max(0, hi - 220)
        tid = f"{sym}-{r.entry:%Y%m%d%H}"
        bars.append(pd.DataFrame({"trade_id": tid, "i": np.arange(hi - lo),
                                  "time": df.time.iloc[lo:hi].dt.strftime("%Y-%m-%d %H:%M").to_numpy(),
                                  "open": df.open.to_numpy(float)[lo:hi], "high": h[lo:hi], "low": l[lo:hi],
                                  "close": c[lo:hi], "ema96": ema[lo:hi], "rsi": rsi[lo:hi]}))
        picked.append((r.entry, tid, int(a - lo), int(b - lo)))
    return (pd.concat(bars, ignore_index=True),
            pd.DataFrame(picked, columns=["entry", "trade_id", "win_entry", "win_exit"]))


def thin(x: pd.DataFrame, n: int = 700) -> pd.DataFrame:
    k = max(1, len(x) // n)
    return pd.concat([x.iloc[::k], x.iloc[[-1]]]).drop_duplicates()


def export(out: Path, res: dict, d1res: dict, alltr: list, wins: list, py: pd.DataFrame, pr: dict) -> None:
    """Chart-ready CSVs for the dashboard page."""
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for test, dct in (("H1", res), ("D1", d1res)):
        for k, v in dct.items():
            sym, src, filt = k if test == "H1" else (k[0], "Yahoo", k[1])
            live = test == "H1" and (filt == "EMA200 filter") == (sym in LIVE_FILTER)
            rows.append({"key": f"{test}|{sym}|{filt}", "test": test, "symbol": sym, "data": src, "filter": filt,
                         "live_setup": live, **v})
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    tr = pd.concat(alltr, ignore_index=True)
    sides(tr).to_csv(out / "sides.csv", index=False)
    cur = []
    for (test, sym), g in tr.groupby(["test", "sym"]):
        g = g.sort_values("exit")
        for filt, x in (("no filter", g), ("EMA200 filter", g[g.trend_ok])):
            x = x.assign(cumR=x.R.cumsum())
            cur.append(thin(x)[["exit", "cumR"]].assign(test=test, symbol=sym, filter=filt))
    cur = pd.concat(cur)
    cur["date"] = cur.exit.dt.strftime("%Y-%m-%d")
    cur[["test", "symbol", "filter", "date", "cumR"]].to_csv(out / "curves.csv", index=False, float_format="%.2f")
    yr = tr.assign(year=tr.entry.dt.year)
    y1 = yr.groupby(["test", "sym", "year"]).R.sum().rename("R_no_filter")
    y2 = yr[yr.trend_ok].groupby(["test", "sym", "year"]).R.sum().rename("R_filter")
    pd.concat([y1, y2], axis=1).fillna(0).reset_index().rename(columns={"sym": "symbol"}).to_csv(
        out / "yearly.csv", index=False, float_format="%.2f")
    py.drop(columns=["total"]).stack().rename("usd").reset_index().rename(
        columns={"exit": "year", "sym": "symbol"}).to_csv(out / "portfolio.csv", index=False)
    pd.DataFrame([{"scenario": k, **v} for k, v in pr.items()]).to_csv(out / "prop.csv", index=False)
    pd.concat([w[0] for w in wins], ignore_index=True).to_csv(out / "windows.csv", index=False, float_format="%.6g")
    picked = pd.concat([w[1].assign(sym=w[2], test="H1") for w in wins], ignore_index=True)
    t = tr.merge(picked, on=["test", "sym", "entry"], how="left")
    t["trade_id"] = t.trade_id.fillna(t.test + "-" + t.sym + "-" + t.entry.dt.strftime("%Y%m%d%H"))
    t["side"] = np.where(t.dir > 0, "BUY", "SELL")
    t["entry_time"], t["exit_time"] = t.entry.dt.strftime("%Y-%m-%d %H:%M"), t.exit.dt.strftime("%Y-%m-%d %H:%M")
    t["live_setup"] = np.where(t.sym.isin(LIVE_FILTER), t.trend_ok, True)
    t.rename(columns={"sym": "symbol", "sig_close": "close", "d_close": "daily_close", "d_ema200": "daily_ema200"})[
        ["trade_id", "test", "source", "symbol", "entry_time", "exit_time", "side", "reason", "R", "close", "rsi", "mom",
         "ema96", "atr", "entry_px", "daily_close", "daily_ema200", "trend_ok", "live_setup", "win_entry", "win_exit"]
    ].to_csv(out / "trades.csv", index=False, float_format="%.6g")
    dl = []
    for s in SYMS:
        d = d1(s)
        d = d.assign(ema200=d.close.ewm(span=200, adjust=False).mean())
        dl.append(d.assign(symbol=s, date=d.time.dt.strftime("%Y-%m-%d"))[
            ["symbol", "date", "open", "high", "low", "close", "ema200"]])
    pd.concat(dl).to_csv(out / "daily.csv", index=False, float_format="%.6g")


def sides(tr: pd.DataFrame) -> pd.DataFrame:
    """BUY and SELL results apart, per timeframe, market and filter setting."""
    rows = []
    for (test, sym), g in tr.groupby(["test", "sym"]):
        for filt, x in (("no filter", g), ("EMA200 filter", g[g.trend_ok])):
            for d, side in ((1, "BUY"), (-1, "SELL")):
                rows.append({"key": f"{test}|{sym}|{filt}|{side}", "test": test, "symbol": sym, "filter": filt, "side": side,
                             "live_setup": test == "H1" and (filt == "EMA200 filter") == (sym in LIVE_FILTER),
                             **stats(x[x.dir == d])})
    return pd.DataFrame(rows)


def _md(df: pd.DataFrame) -> str:
    return "```\n" + df.to_string() + "\n```"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", help="write chart-ready CSVs into this folder")
    args = ap.parse_args()

    res, d1res, alltr, wins = {}, {}, [], []
    for s in SYMS:
        df, src = h1(s)
        t = run(df, s)
        alltr.append(t.assign(test="H1", source=src))
        res[(s, src, "no filter")] = stats(t)
        res[(s, src, "EMA200 filter")] = stats(t[t.trend_ok])
        td = run(d1(s), s, intraday=False)
        alltr.append(td.assign(test="D1", source="Yahoo"))
        d1res[(s, "no filter")] = stats(td)
        d1res[(s, "EMA200 filter")] = stats(td[td.trend_ok])
        if args.export:
            wins.append((*windows(s, df, t), s))
        print(f"{s}: {src} H1 {len(t)} trades, D1 {len(td)} trades", file=sys.stderr)

    tr = pd.concat(alltr, ignore_index=True)
    cols = ["test", "source", "sym", "entry", "exit", "dir", "reason", "R", "nights", "sig_close", "rsi", "mom",
            "ema96", "atr", "entry_px", "d_close", "d_ema200", "trend_ok"]
    tr[cols].to_csv(ROOT / "reports" / "long_history_trades.csv", index=False, float_format="%.6g")

    # portfolio in $ ($12.50 per R, ETH $6.25; min-lot rounding ignored), filter where LIVE_FILTER says
    h1tr = tr[tr.test == "H1"]
    port = []
    for s in SYMS:
        x = h1tr[h1tr.sym == s]
        x = x[x.trend_ok] if s in LIVE_FILTER else x
        port.append(x.assign(usd=x.R * RISK_USD.get(s, 12.5)))
    port = pd.concat(port)
    py = port.pivot_table(index=port.exit.dt.year, columns="sym", values="usd", aggfunc="sum").fillna(0).round(0)
    py["total"] = py.sum(axis=1)
    pr = {}
    for name, syms in SCENARIOS.items():
        sub = port[port.sym.isin(syms)].sort_values("exit")
        for since in ("2014-01-01", "2019-01-01"):
            s2 = sub[sub.exit >= since]
            p = fr.prop_sim(s2[["exit"]].assign(R=s2.usd / 5000), 1.0)
            eq = s2.usd.cumsum()
            yrs = (s2.exit.max() - pd.Timestamp(since, tz="UTC")).days / 365.25
            yr = s2.groupby(s2.exit.dt.year).usd.sum()
            pr[f"{name}, since {since[:4]}"] = {"$ per year": round(s2.usd.sum() / yrs), "max DD $": round(float((eq.cummax().clip(lower=0) - eq).max())),
                                                "years up": f"{(yr > 0).sum()}/{len(yr)}", "pass": p["outcomes"].get("pass", 0),
                                                "fail": round(p["outcomes"].get("daily", 0) + p["outcomes"].get("max_loss", 0), 3),
                                                "median days": p["median_days_to_pass"]}

    h1t = pd.DataFrame(res).T
    h1t.index.names = ["symbol", "data", "filter"]
    d1t = pd.DataFrame(d1res).T
    d1t.index.names = ["symbol", "filter"]
    pd.set_option("display.width", 250)
    print("\nH1 (live timeframe), longest history\n", h1t.to_string())
    print("\nD1 candles, same rules, up to 40 years\n", d1t.to_string())
    print("\nPortfolio $ per year (live config + GER40 filtered)\n", py.to_string())
    print("\nProp simulation\n", pd.DataFrame(pr).T.to_string())
    if args.export:
        export(Path(args.export), res, d1res, alltr, wins, py, pr)
    (ROOT / "LONG_HISTORY_BACKTEST.md").write_text(
        "# Longest-history backtest of the live strategy (2026-10-09)\n\n"
        "Run: `python -m backtest.long_history_backtest`.  Trades with indicator values: `reports/long_history_trades.csv`.\n\n"
        "## H1 (live timeframe)\n\n" + _md(h1t) + "\n\n## D1 candles, same rules (robustness only)\n\n"
        + _md(d1t) + "\n\n## Portfolio $ per year ($12.50 per R, ETH $6.25; GER40 with filter)\n\n"
        + _md(py) + "\n\n## Buy and sell apart\n\n"
        + _md(sides(tr).set_index("key")[["trades", "win %", "avg R", "total R", "t", "1st half", "2nd half", "last 2y"]])
        + "\n\n## Prop simulation\n\n" + _md(pd.DataFrame(pr).T) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
