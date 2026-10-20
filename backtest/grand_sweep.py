"""
Grand sweep: every indicator rule x every stop / target x three timeframes x eight markets, checked on a second
data supplier (owner request 2026-10-20: "multiple charts, multiple solutions, 30 years, all data suppliers, all
indicators" -- not only the live 6 ATR strategy).

    python -m backtest.grand_sweep

PRE-REGISTERED (written before the first run):
  Rules      the ~40 standard indicator rules of backtest/strategy_round7.rules() (trend, breakout, oscillator,
             candle and mean-reversion rules; textbook parameters) incl. the live momentum rule.  A signal counts on
             the bar it first becomes true; entry at the next open; one position per (rule, market, exit).
  Exits      stop 1 / 1.5 / 2 / 3 ATR(14)  x  target 1 / 2 / 3 / 4 / 6 / 8 ATR  (24 pairs, R = P/L / stop distance);
             time exit 96 bars (H1), 48 (H4), 20 (D1).
  Timeframes H1 (longest clean hourly data), H4 (built from the same H1), D1 (Yahoo, up to 30 years).
  Markets    BTCUSD ETHUSD XAUUSD USDJPY NDX100 EURUSD GBPUSD GER40.
  Costs      FundingPips 1-year median spread + slippage + swap (backtest/fp_strategy_build.cost_of).
  Split      per market and timeframe: DISCOVERY = first 60 % of the data, CONFIRM = last 40 % (untouched while
             choosing).  Minimum trades in discovery: 150 (H1), 80 (H4), 40 (D1).
  Choose     per market x timeframe, the best discovery mean R.
  PASS       confirm mean R > 0 with t >= 2.0, > 0 at 2x costs, and (if a second supplier exists for H1 / H4)
             mean R > 0 on that supplier's confirm period too.  Every cell is also reported with its deflated Sharpe
             (N = all cells tried), so a lucky pick among thousands shows up as such.
Writes reports/grand_sweep.json and reports/grand_sweep_cells.csv.  Read-only; no orders.
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
from backtest import long_history_backtest as lh
from backtest import strategy_round5 as r5
from backtest import strategy_round7 as r7
from backtest.btc_strategies import base_atr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
MARKETS = ["BTCUSD", "ETHUSD", "XAUUSD", "USDJPY", "NDX100", "EURUSD", "GBPUSD", "GER40"]
STOPS = (1.0, 1.5, 2.0, 3.0)
TARGETS = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
HOLD = {"H1": 96, "H4": 48, "D1": 20}
MIN_TRADES = {"H1": 150, "H4": 80, "D1": 40}
SPLIT = 0.6


def second_h1(sym: str) -> tuple[pd.DataFrame | None, str]:
    if sym in ("BTCUSD", "ETHUSD"):
        p = ROOT / "data" / "binance_H1" / f"{sym[:3]}USDT.csv"
        return fr._dense_from(fr._read(p)), "Binance"
    if sym == "USDJPY":
        return None, ""
    return fb.load_h1(sym), "FundingPips"


def to_h4(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for p in fr.segments(df):
        g = p.groupby(p.time.dt.floor("4h"))
        out.append(pd.DataFrame({"time": g.time.first().index, "open": g.open.first().values, "high": g.high.max().values,
                                 "low": g.low.min().values, "close": g.close.last().values}))
    return pd.concat(out, ignore_index=True)


def pieces(df: pd.DataFrame, tf: str) -> list[pd.DataFrame]:
    return [df] if tf == "D1" else fr.segments(df) if tf == "H1" else \
        [x for x in (df.iloc[a:b].reset_index(drop=True) for a, b in _h4_edges(df)) if len(x) > 300]


def _h4_edges(df):
    cut = np.flatnonzero(df["time"].diff().to_numpy() > fr.GAP.to_timedelta64())
    e = [0, *cut, len(df)]
    return zip(e[:-1], e[1:])


def run_all(df: pd.DataFrame, tf: str, cost: dict, cost_mult: float = 1.0) -> dict:
    """(rule, sl, tp) -> trades table for one data set."""
    c2 = dict(cost, cost_frac=cost["cost_frac"] * cost_mult)
    res: dict = {}
    for p in pieces(df, tf):
        o, h, l, c = (p[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        A = base_atr(h, l, c, 14)
        for name, (buy, sell, _kind) in r7.rules(o, h, l, c).items():
            buy, sell = np.nan_to_num(buy.astype(float)) > 0, np.nan_to_num(sell.astype(float)) > 0
            buy, sell = buy & ~np.concatenate([[False], buy[:-1]]), sell & ~np.concatenate([[False], sell[:-1]])
            for sl in STOPS:
                for tp in TARGETS:
                    s = r5._put(r5._sig(len(c)), buy, sell, c - sl * A, c + sl * A, c + tp * A, c - tp * A)
                    t = r5.simulate_piece(p, dict(s, hold=HOLD[tf]), c2)
                    if len(t):
                        res.setdefault((name, sl, tp), []).append(t)
    return {k: pd.concat(v, ignore_index=True) for k, v in res.items()}


def split_at(df: pd.DataFrame) -> pd.Timestamp:
    return df.time.iloc[0] + (df.time.iloc[-1] - df.time.iloc[0]) * SPLIT


def summ(t: pd.DataFrame) -> dict:
    n = len(t)
    if n < 5:
        return {"n": n}
    m, sd = float(t.R.mean()), float(t.R.std())
    return {"n": n, "meanR": round(m, 4), "t": round(m / sd * math.sqrt(n), 2) if sd > 0 else 0.0,
            "win%": round(100 * float((t.R > 0).mean()), 1)}


def dsr(sr: float, n: int, trials: int, skew: float = 0.0, kurt: float = 3.0) -> float:
    """Deflated Sharpe (Bailey & Lopez de Prado): probability the per-trade Sharpe beats the best of `trials` nulls."""
    from scipy.stats import norm
    if n < 10 or trials < 2:
        return float("nan")
    emax = (1 - 0.5772) * norm.ppf(1 - 1 / trials) + 0.5772 * norm.ppf(1 - 1 / (trials * math.e))
    sr0 = emax / math.sqrt(n)
    den = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr * sr))
    return float(norm.cdf((sr - sr0) * math.sqrt(n - 1) / den))


def main() -> int:
    cells, picks = [], []
    for sym in MARKETS:
        if sym not in lh.SPECS:
            print(f"{sym}: no cost spec, skipped", flush=True)
            continue
        cost = fb.cost_of(lh.SPECS[sym])
        h1df, src = r7.h1(sym)
        alt, alt_src = second_h1(sym)
        data = {"H1": h1df, "H4": to_h4(h1df), "D1": r7.d1(sym)}
        for tf, df in data.items():
            cut = split_at(df)
            tr = run_all(df, tf, cost)
            tr2 = run_all(df, tf, cost, cost_mult=2.0)
            alt_tr = None
            if alt is not None and tf != "D1":
                adf = alt if tf == "H1" else to_h4(alt)
                alt_tr = run_all(adf, tf, cost)
            for (rule, sl, tp), t in tr.items():
                disc, conf = t[t.entry < cut], t[t.entry >= cut]
                c2 = tr2.get((rule, sl, tp))
                a = alt_tr.get((rule, sl, tp)) if alt_tr is not None else None
                row = {"market": sym, "tf": tf, "rule": rule, "sl": sl, "tp": tp, "source": src,
                       **{f"disc_{k}": v for k, v in summ(disc).items()}, **{f"conf_{k}": v for k, v in summ(conf).items()},
                       "conf_2xcost_meanR": round(float(c2[c2.entry >= cut].R.mean()), 4) if c2 is not None and len(c2[c2.entry >= cut]) else None,
                       "alt_source": alt_src if a is not None else "",
                       "alt_conf_meanR": round(float(a[a.entry >= cut].R.mean()), 4) if a is not None and len(a[a.entry >= cut]) >= 5 else None,
                       "full_meanR": round(float(t.R.mean()), 4), "full_n": len(t),
                       "years": round((df.time.iloc[-1] - df.time.iloc[0]).days / 365.25, 1)}
                cells.append(row)
            print(f"{sym} {tf}: {len(tr)} cells ({src}{', ' + alt_src if alt_tr is not None else ''})", flush=True)
    C = pd.DataFrame(cells)
    N = len(C)
    for (sym, tf), g in C.groupby(["market", "tf"]):
        ok = g[g.disc_n >= MIN_TRADES[tf]]
        if ok.empty:
            continue
        best = ok.sort_values("disc_meanR", ascending=False).iloc[0].to_dict()
        live = g[(g.rule == "live_momentum_rsi60") & (g.sl == 2.0) & (g.tp == 6.0)]
        sr = (best.get("conf_t") or 0.0) / math.sqrt(max(1, best.get("conf_n") or 1))     # per-trade Sharpe = t / sqrt(n)
        best["conf_DSR_vs_all_cells"] = round(dsr(sr, int(best.get("conf_n") or 0), N), 3)
        best["PASS"] = bool((best.get("conf_meanR") or -1) > 0 and (best.get("conf_t") or 0) >= 2.0
                            and (best.get("conf_2xcost_meanR") or -1) > 0
                            and (best["alt_conf_meanR"] is None or best["alt_conf_meanR"] > 0))
        best["live_6ATR_conf_meanR"] = float(live.conf_meanR.iloc[0]) if len(live) and "conf_meanR" in live else None
        picks.append(best)
    out = ROOT / "reports"
    C.to_csv(out / "grand_sweep_cells.csv", index=False)
    (out / "grand_sweep.json").write_text(json.dumps({"cells": N, "picks": picks}, indent=1, default=str))
    P = pd.DataFrame(picks)
    cols = ["market", "tf", "rule", "sl", "tp", "disc_n", "disc_meanR", "conf_n", "conf_meanR", "conf_t",
            "conf_2xcost_meanR", "alt_conf_meanR", "conf_DSR_vs_all_cells", "PASS", "live_6ATR_conf_meanR"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(P[cols].to_string(index=False))
    print("cells tried:", N)
    return 0


if __name__ == "__main__":
    sys.exit(main())
