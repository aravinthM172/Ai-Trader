"""
Indian market sweep (owner request 2026-10-21): the same process as the forex grand sweep (backtest/grand_sweep.py) on
Indian indices and the Nifty 50 stocks.

    python -m backtest.india_sweep           (resumable: one result file per market in reports/india_sweep/)
    python -m backtest.india_sweep --report  (summary only)

Data (tools/fetch_yahoo_india.py, Yahoo): D1 as far back as Yahoo has it (Sensex 1997-, Nifty / Bank Nifty / IT / Midcap
2007-, Pharma 2011-, stocks mostly 1996-2002-); H1 for the last ~2 years (Yahoo limit; no free 30-year intraday exists).
Stock list = today's Nifty 50 -> survivor-biased, stock results are optimistic; index results are not.

PRE-REGISTERED (written before the first run):
  Rules    the ~42 indicator rules of strategy_round7.rules() incl. the live momentum rule; signal on a closed bar when it
           first becomes true, entry next open, one position per (rule, market, exit).
  Exits    stop 1 / 1.5 / 2 / 3 ATR(14) x target 1 / 2 / 3 / 4 / 6 / 8 ATR; time exit 20 bars (D1), 96 (H1).
  Modes    D1 swing; H1 swing (overnight allowed); H1 INTRADAY (no new entry on the day's last bar, every trade closed at
           the open of the day's last hourly bar ~14:45 IST -- no overnight risk, like an MIS / day trade).
  Costs    index / stock futures round trip 0.07 % of price (STT 0.02 % sell, exchange + SEBI + stamp + GST ~0.01 %,
           brokerage ~0, slippage 0.04 %); also run at 2x.  No carry (futures price it in).
  Split    per market x mode: DISCOVERY first 60 % of the data, CONFIRM last 40 %.  Min discovery trades 40 (D1), 80 (H1).
  PASS     the best-in-discovery cell must have confirm mean R > 0 with t >= 2, > 0 at 2x costs, and for Nifty / Sensex the
           same cell > 0 on the other index's confirm period.  Every pick also gets a deflated Sharpe vs all cells tried.
Read-only.
"""
from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import full_reassessment as fr
from backtest import grand_sweep as gs
from backtest import strategy_round5 as r5
from backtest import strategy_round7 as r7
from backtest.btc_strategies import base_atr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "india"
OUT = ROOT / "reports" / "india_sweep"
COST = dict(cost_frac=0.0007, swap_long=0.0, swap_short=0.0)
HOLD = {"D1": 20, "H1": 96, "H1i": 96}
MIN_TRADES = {"D1": 40, "H1": 80, "H1i": 80}
PAIR = {"_NSEI": "_BSESN", "_BSESN": "_NSEI"}


def load(name: str, tf: str) -> pd.DataFrame | None:
    p = DATA / ("D1" if tf == "D1" else "H1") / f"{name}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d["time"] = pd.to_datetime(d["time"], utc=True)
    d = d.dropna(subset=["open", "high", "low", "close"])
    d = d[(d.high >= d.low) & (d.close > 0) & (d.high > 0)]
    d = d.assign(high=d[["open", "high", "close"]].max(axis=1), low=d[["open", "low", "close"]].min(axis=1))
    d = d.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    return d if len(d) > 300 else None


def run_all(df: pd.DataFrame, tf: str, cost_mult: float = 1.0) -> dict:
    cost = dict(COST, cost_frac=COST["cost_frac"] * cost_mult)
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    A = base_atr(h, l, c, 14)
    flat = None
    if tf == "H1i":                                   # close at the open of each session's last hourly bar
        day = (df.time + pd.Timedelta(hours=5, minutes=30)).dt.date
        last = (day != day.shift(-1)).to_numpy()
        flat = last.copy()
    res = {}
    for name, (buy, sell, _k) in r7.rules(o, h, l, c).items():
        buy, sell = np.nan_to_num(buy.astype(float)) > 0, np.nan_to_num(sell.astype(float)) > 0
        buy, sell = buy & ~np.concatenate([[False], buy[:-1]]), sell & ~np.concatenate([[False], sell[:-1]])
        if flat is not None:                          # no entry decided on the bar before the last one (entry would be the last)
            nxt_last = np.concatenate([flat[1:], [True]])
            buy, sell = buy & ~nxt_last & ~flat, sell & ~nxt_last & ~flat
        for sl in gs.STOPS:
            for tp in gs.TARGETS:
                s = r5._put(r5._sig(len(c)), buy, sell, c - sl * A, c + sl * A, c + tp * A, c - tp * A)
                spec = dict(s, hold=HOLD[tf])
                if flat is not None:
                    spec["flat"] = flat
                t = r5.simulate_piece(df, spec, cost)
                if len(t):
                    res[(name, sl, tp)] = t
    return res


def market(name: str) -> pd.DataFrame:
    path = OUT / f"{name}.csv"
    if path.exists():
        return pd.read_csv(path)
    rows = []
    for tf in ("D1", "H1", "H1i"):
        df = load(name, tf)
        if df is None:
            continue
        cut = gs.split_at(df)
        tr, tr2 = run_all(df, tf), run_all(df, tf, 2.0)
        alt = None
        if name in PAIR:
            adf = load(PAIR[name], tf)
            alt = run_all(adf, tf) if adf is not None else None
        for key, t in tr.items():
            disc, conf = t[t.entry < cut], t[t.entry >= cut]
            c2 = tr2.get(key)
            a = alt.get(key) if alt is not None else None
            rows.append({"market": name, "tf": tf, "rule": key[0], "sl": key[1], "tp": key[2],
                         **{f"disc_{k}": v for k, v in gs.summ(disc).items()},
                         **{f"conf_{k}": v for k, v in gs.summ(conf).items()},
                         "conf_2xcost_meanR": round(float(c2[c2.entry >= cut].R.mean()), 4) if c2 is not None and (c2.entry >= cut).any() else None,
                         "alt_conf_meanR": round(float(a[a.entry >= cut].R.mean()), 4) if a is not None and (a.entry >= cut).sum() >= 5 else None,
                         "full_meanR": round(float(t.R.mean()), 4), "full_n": len(t),
                         "from": str(df.time.iloc[0].date())})
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def report() -> None:
    C = pd.concat([pd.read_csv(p) for p in sorted(OUT.glob("*.csv"))], ignore_index=True)
    N = len(C)
    picks = []
    for (m, tf), g in C.groupby(["market", "tf"]):
        ok = g[g.disc_n >= MIN_TRADES[tf]]
        if ok.empty:
            continue
        b = ok.sort_values("disc_meanR", ascending=False).iloc[0].to_dict()
        live = g[(g.rule == "live_momentum_rsi60") & (g.sl == 2.0) & (g.tp == 6.0)]
        sr = (b.get("conf_t") or 0.0) / math.sqrt(max(1, b.get("conf_n") or 1))
        b["DSR"] = round(gs.dsr(sr, int(b.get("conf_n") or 0), N), 3)
        alt = b.get("alt_conf_meanR")
        b["PASS"] = bool((b.get("conf_meanR") or -1) > 0 and (b.get("conf_t") or 0) >= 2 and (b.get("conf_2xcost_meanR") or -1) > 0
                         and (alt is None or (isinstance(alt, float) and math.isnan(alt)) or alt > 0))
        b["live_rule_conf_meanR"] = float(live.conf_meanR.iloc[0]) if len(live) else None
        picks.append(b)
    P = pd.DataFrame(picks)
    print(f"cells tried {N}, market x mode picks {len(P)}, PASS {int(P.PASS.sum())}")
    cols = ["market", "tf", "rule", "sl", "tp", "disc_n", "disc_meanR", "conf_n", "conf_meanR", "conf_t", "conf_2xcost_meanR",
            "alt_conf_meanR", "DSR", "PASS", "live_rule_conf_meanR"]
    with pd.option_context("display.width", 260, "display.max_columns", 30, "display.max_rows", 300):
        idx = P[P.market.str.startswith("_")]
        print("\nINDICES"); print(idx[cols].to_string(index=False))
        print("\nSTOCKS: passes only"); print(P[(~P.market.str.startswith("_")) & P.PASS][cols].to_string(index=False))
        Cc = C[C.conf_n >= 30]
        print("\nALL CELLS by mode (unseen 40 %): avg R / share profitable")
        print(Cc.groupby("tf").conf_meanR.agg(["mean", lambda x: (x > 0).mean(), "size"]).round(3).to_string())
        print("\nby target:"); print(Cc.groupby(["tf", "tp"]).conf_meanR.agg(["mean", lambda x: (x > 0).mean()]).round(3).unstack(0).to_string())
        g = Cc.groupby(["tf", "rule"]).conf_meanR.mean().unstack(0)
        print("\nbest rules by mode (avg unseen R over all markets and exits):")
        for tf in g.columns:
            print(tf, g[tf].sort_values(ascending=False).head(5).round(3).to_dict())
        lv = Cc[(Cc.rule == "live_momentum_rsi60") & (Cc.sl == 2.0) & (Cc.tp == 6.0)]
        print("\nlive forex rule (stop 2 / target 6) on Indian markets, unseen 40 %:",
              lv.groupby("tf").conf_meanR.agg(["mean", lambda x: (x > 0).mean(), "size"]).round(3).to_dict())
    (ROOT / "reports" / "india_sweep.json").write_text(json.dumps({"cells": N, "picks": picks}, indent=1, default=str))


def main() -> int:
    if "--report" not in sys.argv:
        names = sorted({p.stem for p in (DATA / "D1").glob("*.csv")} | {p.stem for p in (DATA / "H1").glob("*.csv")})
        for n in names:
            if (OUT / f"{n}.csv").exists():
                continue
            m = market(n)
            print(f"{n}: {len(m)} cells", flush=True)
    report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
