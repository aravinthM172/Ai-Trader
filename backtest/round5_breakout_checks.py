"""
Round-5 follow-up: is the volatility breakout (vbo_day) real, and what does it add to the live bot?  (2026-10-10)

    python -m backtest.round5_breakout_checks      -> STRATEGY_ROUND5_BREAKOUT.md

Same rules as backtest/strategy_round5.py, nothing re-tuned: other data sources, a 00:00 UTC day boundary,
per year, long / short, daily-R correlation with the live strategy, and the combined $5k prop simulation.
Analysis only; no orders.
"""
from __future__ import annotations

import contextlib
import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import fp_strategy_build as fb
from backtest import full_reassessment as fr
from backtest import long_history_backtest as lh
from backtest import strategy_round5 as r5

ROOT = Path(__file__).resolve().parents[1]
CAND = ["BTCUSD", "ETHUSD", "XAUUSD", "USDJPY", "NDX100"]


def vbo(df, sym, key="vbo_day"):
    if "vol" not in df:
        df = df.assign(vol=np.nan)
    return r5.simulate(df, fb.cost_of(lh.SPECS[sym]))[key].assign(sym=sym)


def line(t):
    R = t.R.to_numpy()
    yrs = (t.entry.max() - t.entry.min()).days / 365.25
    return dict(trades=len(t), per_yr=round(len(t) / yrs), mean_R=round(R.mean(), 4), t=r5.t_stat(R),
                win=round(100 * (R > 0).mean()), last2y=round(t.R[t.entry >= t.entry.max() - pd.Timedelta(days=730)].mean(), 4))


def report() -> None:
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 250)
    trades = {s: vbo(r5.load(s)[0], s) for s in CAND}

    print("=== 1. other data sources (same rules) ===")
    bn = lambda p: r5.binance_h1(p)[["time", "open", "high", "low", "close"]]
    alt = {"BTCUSD FundingPips 2018+": ("BTCUSD", (lambda d: d[d.time >= "2018-01-01"].reset_index(drop=True))(fb.load_h1("BTCUSD"))),
           "BTCUSD Binance": ("BTCUSD", bn("BTCUSDT")),
           "ETHUSD Binance": ("ETHUSD", bn("ETHUSDT")),
           "XAUUSD FundingPips": ("XAUUSD", fb.load_h1("XAUUSD")),
           "NDX100 FundingPips": ("NDX100", fb.load_h1("NDX100"))}
    rows = {f"{s} main": line(trades[s]) for s in CAND}
    rows.update({k: line(vbo(d, s)) for k, (s, d) in alt.items()})
    print(pd.DataFrame(rows).T.sort_index().to_string())

    print("\n=== 2. day starts at 00:00 UTC instead of 21:00 UTC ===")
    orig = r5.broker_day
    r5.broker_day = lambda t: t.dt.tz_localize(None).dt.normalize().to_numpy()
    print(pd.DataFrame({s: line(vbo(r5.load(s)[0], s)) for s in CAND}).T.to_string())
    r5.broker_day = orig

    print("\n=== 3. per year (mean R, trades) ===")
    py = pd.concat({s: t.groupby(t.entry.dt.year).R.agg(["mean", "count"]).round(3) for s, t in trades.items()}, axis=1)
    print(py.to_string())

    print("\n=== 4. long / short / exit reason ===")
    for s, t in trades.items():
        print(s, "long", round(t.R[t.dir > 0].mean(), 3), (t.dir > 0).sum(), "| short", round(t.R[t.dir < 0].mean(), 3), (t.dir < 0).sum(),
              "|", t.groupby("reason").R.agg(["mean", "count"]).round(3).to_dict("index"))

    print("\n=== 5. overlap with the live strategy (daily R correlation) ===")
    live = {}
    for s in CAND:
        t = lh.live_rows(lh.run(lh.h1(s)[0], s))
        if s in ("NDX100", "USDJPY"):
            t = t[t.dir > 0]
        live[s] = t[["entry", "exit", "R"]].assign(sym=s)
    dR = lambda t: t.groupby(t.exit.dt.tz_convert(None).dt.normalize()).R.sum()
    for s in CAND:
        a, b = dR(live[s]), dR(trades[s])
        j = pd.concat([a, b], axis=1).fillna(0)
        j = j[j.index >= max(a.index.min(), b.index.min())]
        print(s, "corr", round(j.corr().iloc[0, 1], 3), "| live mean R", round(live[s].R.mean(), 3), "trades/yr",
              round(len(live[s]) / ((live[s].entry.max() - live[s].entry.min()).days / 365.25)))

    print("\n=== 6. portfolio on a $5k account, 2018+ (risk in % of balance per trade) ===")
    since = pd.Timestamp("2018-03-01", tz="UTC")
    LIVE_W = {"BTCUSD": 1.0, "ETHUSD": 0.5, "XAUUSD": 1.0, "USDJPY": 0.5, "NDX100": 1.0}      # x 0.25 %


    def port(parts):
        p = pd.concat([t[t.entry >= since][["entry", "exit", "sym"]].assign(R=t.R[t.entry >= since] * w) for t, w in parts],
                      ignore_index=True).sort_values("exit")
        yrs = (p.exit.max() - since).days / 365.25
        usd = p.R * 12.5
        eq = usd.cumsum()
        ps = fr.prop_sim(p[["exit"]].assign(R=p.R.to_numpy()), 0.0025)
        o = ps["outcomes"]
        last = p[p.exit >= p.exit.max() - pd.Timedelta(days=730)]
        return {"trades/yr": round(len(p) / yrs), "$ / yr": round(usd.sum() / yrs), "$ / yr last 2y": round((last.R * 12.5).sum() / 2),
                "max DD $": round(float((eq.cummax() - eq).max())), "pass": o.get("pass", 0),
                "fail": round(o.get("daily", 0) + o.get("max_loss", 0), 3), "open": o.get("open", 0), "median days": ps["median_days_to_pass"]}


    live_parts = [(live[s], LIVE_W[s]) for s in CAND]
    core = [(live[s], LIVE_W[s]) for s in ("BTCUSD", "ETHUSD")]
    sc = {"live now (5 markets)": live_parts,
          "live + breakout BTC/ETH/XAU at 0.125 %": live_parts + [(trades[s], 0.5) for s in ("BTCUSD", "ETHUSD", "XAUUSD")],
          "live + breakout BTC/ETH/XAU at 0.25 %": live_parts + [(trades[s], 1.0) for s in ("BTCUSD", "ETHUSD", "XAUUSD")],
          "breakout only BTC/ETH/XAU at 0.25 %": [(trades[s], 1.0) for s in ("BTCUSD", "ETHUSD", "XAUUSD")],
          "live + breakout BTC/ETH only at 0.125 %": live_parts + [(trades[s], 0.5) for s in ("BTCUSD", "ETHUSD")],
          "live + breakout BTC/ETH only at 0.25 %": live_parts + [(trades[s], 1.0) for s in ("BTCUSD", "ETHUSD")],
          "live BTC+ETH only": core,
          "live BTC+ETH + breakout BTC/ETH/XAU 0.25 %": core + [(trades[s], 1.0) for s in ("BTCUSD", "ETHUSD", "XAUUSD")]}
    print(pd.DataFrame({k: port(v) for k, v in sc.items()}).T.to_string())

    print("\n=== 7. typical stop distance of the breakout (to size the lot) ===")
    for s in ("BTCUSD", "ETHUSD", "XAUUSD"):
        df = r5.load(s)[0]
        df = df[df.time >= df.time.max() - pd.Timedelta(days=365)].reset_index(drop=True)
        sig = r5.build(df)["vbo_day"]
        m = sig["d"] != 0
        dist = np.abs(df.close.to_numpy()[m] - sig["stop"][m])
        print(s, "median stop distance last year:", round(float(np.median(dist)), 2), "| price", round(float(df.close.iloc[-1]), 2),
              "| contract", lh.SPECS[s]["contract_size"])


def main() -> int:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report()
    fence = chr(96) * 3
    out = f"# Round 5 follow-up: the volatility breakout (vbo_day){chr(10) * 2}{fence}{chr(10)}{buf.getvalue()}{fence}{chr(10)}"
    (ROOT / "STRATEGY_ROUND5_BREAKOUT.md").write_text(out, encoding="utf-8")
    print(buf.getvalue())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
