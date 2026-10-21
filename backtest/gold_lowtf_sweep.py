"""
Gold on 1-minute and 5-minute charts (owner request 2026-10-21): the grand-sweep process on low timeframes.

    python -m backtest.gold_lowtf_sweep

Data (broker MT5 history, read-only copy_rates):
  M1  FundedNext   2026-07-13 .. 10-21  (data/gold/fn_XAUUSD_M1.csv, median spread $0.45)
  M5  FundedNext   2025-05-30 .. 2026-10-21 (data/gold/fn_XAUUSD_M5.csv, median spread $0.37)
  M5  second broker 2025-03-24 .. 2026-08-28 (data/xauusd_m5_5y.csv, median spread $0.08) -- second-source check
PRE-REGISTERED (same as backtest/grand_sweep.py): ~42 indicator rules x stop 1/1.5/2/3 ATR x target 1/2/3/4/6/8 ATR,
time exit 96 bars, signal when it first becomes true, entry next open, one position per cell.  Cost per trade = that
data set's median spread x 1.5 (slippage) + $6.5 / lot commission, as a fraction of the median price; also at 2x.
Split 60 / 40 per data set; the best discovery cell must have confirm mean R > 0 with t >= 2, > 0 at 2x costs and
(M5) > 0 on the second broker's confirm period.  Read-only; writes reports/gold_lowtf_sweep.json.
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
HOLD = 96
COMM_PER_OZ = 0.065                     # $6.5 per lot round turn / 100 oz


def load(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    d["time"] = pd.to_datetime(d["time"], unit="s", utc=True) if pd.api.types.is_numeric_dtype(d["time"]) \
        else pd.to_datetime(d["time"], utc=True)
    d["spr"] = d["spread"] / 100.0
    d = d[(d.high >= d.low) & (d.close > 0)].sort_values("time").drop_duplicates("time").reset_index(drop=True)
    return d


def cost_of(d: pd.DataFrame, mult: float = 1.0) -> dict:
    frac = (d.spr.median() * 1.5 + COMM_PER_OZ) / d.close.median()
    return dict(cost_frac=frac * mult, swap_long=0.0, swap_short=0.0)


def run_all(df: pd.DataFrame, cost: dict) -> dict:
    res: dict = {}
    for p in fr.segments(df):
        o, h, l, c = (p[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        A = base_atr(h, l, c, 14)
        for name, (buy, sell, _k) in r7.rules(o, h, l, c).items():
            buy, sell = np.nan_to_num(buy.astype(float)) > 0, np.nan_to_num(sell.astype(float)) > 0
            buy, sell = buy & ~np.concatenate([[False], buy[:-1]]), sell & ~np.concatenate([[False], sell[:-1]])
            for sl in gs.STOPS:
                for tp in gs.TARGETS:
                    s = r5._put(r5._sig(len(c)), buy, sell, c - sl * A, c + sl * A, c + tp * A, c - tp * A)
                    t = r5.simulate_piece(p, dict(s, hold=HOLD), cost)
                    if len(t):
                        res.setdefault((name, sl, tp), []).append(t)
    return {k: pd.concat(v, ignore_index=True) for k, v in res.items()}


def sweep(label: str, df: pd.DataFrame, alt: pd.DataFrame | None) -> pd.DataFrame:
    cut = gs.split_at(df)
    tr, tr2 = run_all(df, cost_of(df)), run_all(df, cost_of(df, 2.0))
    ta = run_all(alt, cost_of(alt)) if alt is not None else None
    rows = []
    for key, t in tr.items():
        disc, conf = t[t.entry < cut], t[t.entry >= cut]
        c2 = tr2.get(key)
        a = ta.get(key) if ta is not None else None
        rows.append({"data": label, "rule": key[0], "sl": key[1], "tp": key[2],
                     **{f"disc_{k}": v for k, v in gs.summ(disc).items()}, **{f"conf_{k}": v for k, v in gs.summ(conf).items()},
                     "conf_2xcost_meanR": round(float(c2[c2.entry >= cut].R.mean()), 4) if c2 is not None and (c2.entry >= cut).any() else None,
                     "alt_conf_meanR": round(float(a[a.entry >= cut].R.mean()), 4) if a is not None and (a.entry >= cut).sum() >= 5 else None,
                     "full_meanR": round(float(t.R.mean()), 4), "full_n": len(t)})
    print(f"{label}: {len(rows)} cells, cost {100 * cost_of(df)['cost_frac']:.4f} % of price per trade", flush=True)
    return pd.DataFrame(rows)


def main() -> int:
    g = ROOT / "data" / "gold"
    m1, m5 = load(g / "fn_XAUUSD_M1.csv"), load(g / "fn_XAUUSD_M5.csv")
    m5b = load(ROOT / "data" / "xauusd_m5_5y.csv")
    C = pd.concat([sweep("M1 FundedNext", m1, None), sweep("M5 FundedNext", m5, m5b), sweep("M5 other broker", m5b, m5)],
                  ignore_index=True)
    N, picks = len(C), []
    for lab, gdf in C.groupby("data"):
        ok = gdf[gdf.disc_n >= 150]
        b = ok.sort_values("disc_meanR", ascending=False).iloc[0].to_dict()
        sr = (b.get("conf_t") or 0.0) / math.sqrt(max(1, b.get("conf_n") or 1))
        b["DSR"] = round(gs.dsr(sr, int(b.get("conf_n") or 0), N), 3)
        alt = b.get("alt_conf_meanR")
        b["PASS"] = bool((b.get("conf_meanR") or -1) > 0 and (b.get("conf_t") or 0) >= 2 and (b.get("conf_2xcost_meanR") or -1) > 0
                         and (alt is None or (isinstance(alt, float) and math.isnan(alt)) or alt > 0))
        live = gdf[(gdf.rule == "live_momentum_rsi60") & (gdf.sl == 2.0) & (gdf.tp == 6.0)]
        b["live_rule_conf_meanR"] = float(live.conf_meanR.iloc[0]) if len(live) else None
        picks.append(b)
    P = pd.DataFrame(picks)
    cols = ["data", "rule", "sl", "tp", "disc_n", "disc_meanR", "conf_n", "conf_meanR", "conf_t", "conf_2xcost_meanR",
            "alt_conf_meanR", "DSR", "PASS", "live_rule_conf_meanR"]
    Cc = C[C.conf_n >= 30]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(f"\ncells tried {N}"); print(P[cols].to_string(index=False))
        print("\nALL CELLS, unseen 40 %: avg R / share profitable / trades")
        print(Cc.groupby("data").conf_meanR.agg(["mean", lambda x: (x > 0).mean(), "size"]).round(3).to_string())
        print("\nby target:"); print(Cc.groupby(["data", "tp"]).conf_meanR.agg(lambda x: (x > 0).mean()).round(3).unstack(0).to_string())
        print("\nbest rules (avg unseen R over all exits):")
        for lab, gdf in Cc.groupby("data"):
            print(lab, gdf.groupby("rule").conf_meanR.mean().sort_values(ascending=False).head(4).round(3).to_dict())
        both = Cc[(Cc.data == "M5 FundedNext") & (Cc.conf_t >= 2) & (Cc.conf_2xcost_meanR > 0) & (Cc.alt_conf_meanR > 0)]
        print(f"\nM5 cells profitable unseen on FundedNext (t>=2, 2x cost) AND on the other broker: {len(both)}")
        print(both.sort_values("conf_t", ascending=False)[["rule", "sl", "tp", "disc_meanR", "conf_n", "conf_meanR", "conf_t",
                                                           "alt_conf_meanR"]].head(8).to_string(index=False))
    (ROOT / "reports" / "gold_lowtf_sweep.json").write_text(json.dumps({"cells": N, "picks": picks}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
