"""
Every test from the 2026-10-20/21 gold work, on all five live markets, swing and scalping, on the VERIFIED zip data
(tools/zip_history: Binance BTC / ETH, HistData gold / USDJPY / NDX100).

    python -m backtest.all_pairs_verified            (resumable: one result file per market x timeframe x family)
    python -m backtest.all_pairs_verified --report

Timeframes: swing H1 / H4 / D1 on the full history; scalping M1 (last 1 year), M5 (last 3 years), M15 (last 5 years).
Costs: FundedNext, measured 2026-10-21 (median M5 spread x 1.5 for slippage + commission), as a fraction of price per trade:
  BTC 0.075 %, ETH 0.071 %, gold 0.018 %, USDJPY 0.011 %, NDX100 0.008 %.  No swap.
Families
  grid        strategy_round7.rules (~42 rules incl. the live one) x stops 1 / 1.5 / 2 / 3 x targets 1-8 ATR (H1 / H4 / D1);
              stops 1 / 2 x targets 1 / 2 / 3 / 4 / 6 / 8 on M5 / M15; stops 1 / 2 x targets 1 / 2 / 4 on M1
  setups      gold_setups_test new setups: pullback_strict, range_mid_bb, range_mid_dc
  cheatsheet  gold_cheatsheet_test A breakout-retest, B liquidity sweep, C trend continuation (H1 context)
  smc         gold_smc_test sweep -> displacement -> FVG limit entry in the kill zones
  raremain    gold_rare_main_test turtle_soup, main_setup, asia_accept
  internet    gold_internet_methods ORBs, EMA scalps, VWAP (time-weighted: the zip data has no volume), pivots,
              NY momentum, Supertrend + ADX
  litefinance gold_litefinance_test trend / breakout (D1), swing (H4), scalp (M1 / M5)
  hourdrift   average return per UTC hour, both halves
Judge (pre-set): 60 / 40 split per data set; ROBUST = first 60 % mean R > 0 AND unseen 40 % mean R > 0 with t >= 2 AND unseen
mean R still > 0 at 2x costs.  Read-only.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import full_reassessment as fr
from backtest import gold_cheatsheet_test as cs
from backtest import gold_internet_methods as gim
from backtest import gold_litefinance_test as glf
from backtest import gold_rare_main_test as grm
from backtest import gold_setups_test as gst
from backtest import gold_smc_test as gsm
from backtest import grand_sweep as gs
from backtest import strategy_round5 as r5
from backtest import strategy_round7 as r7
from backtest.btc_strategies import base_atr
from tools import zip_history as zh

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "all_pairs_verified"
COST = {"BTCUSD": 0.00075, "ETHUSD": 0.00071, "XAUUSD": 0.00018, "USDJPY": 0.00011, "NDX100": 0.00008}
SYMS = list(COST)
SCALP_YEARS = {"M1": 1, "M5": 3, "M15": 5}
RULE = {"M1": "1min", "M5": "5min", "M15": "15min", "H1": "1h", "H4": "4h", "D1": "1D"}
HOLD = {"M1": 240, "M5": 96, "M15": 64, "H1": 96, "H4": 48, "D1": 20}
EXITS = {"M1": [(s, t) for s in (1.0, 2.0) for t in (1.0, 2.0, 4.0)],
         "M5": [(s, t) for s in (1.0, 2.0) for t in (1.0, 2.0, 3.0, 4.0, 6.0, 8.0)],
         "M15": [(s, t) for s in (1.0, 2.0) for t in (1.0, 2.0, 3.0, 4.0, 6.0, 8.0)]}
for tf in ("H1", "H4", "D1"):
    EXITS[tf] = [(s, t) for s in gs.STOPS for t in gs.TARGETS]
cs.COMM_PER_OZ = 0.0                                        # commission is inside COST (via the spr column)


def data(sym: str, tf: str) -> pd.DataFrame:
    m1 = zh.load_m1(sym)
    if tf in SCALP_YEARS:
        m1 = m1[m1.time >= m1.time.iloc[-1] - pd.Timedelta(days=365 * SCALP_YEARS[tf])]
    d = m1 if tf == "M1" else zh.resample(m1, RULE[tf])
    d = d.copy()
    d["spr"] = COST[sym] * d.close / 1.5                    # modules charge spr x 1.5 -> exactly COST of price
    d["tick_volume"] = 1.0
    return d.reset_index(drop=True)


def summarise(t: pd.DataFrame, cut: pd.Timestamp, years: float) -> dict:
    if t is None or len(t) < 20:
        return {}
    a, b = t[t.entry < cut], t[t.entry >= cut]
    if len(a) < 5 or len(b) < 5:
        return {}
    r2 = b.R - b.cost_R if "cost_R" in b else b.R
    return {"trades/yr": round(len(t) / years, 1), "win %": round(100 * float((t.R > 0).mean()), 1),
            "first60 R": round(float(a.R.mean()), 4), "unseen R": round(float(b.R.mean()), 4),
            "unseen t": round(float(b.R.mean() / b.R.std() * np.sqrt(len(b))), 2) if b.R.std() > 0 else 0.0,
            "2x cost unseen R": round(float(r2.mean()), 4)}


def fam_grid(df, tf, cf):
    res = {}
    pieces = [df] if tf == "D1" else [p for p in fr.segments(df) if len(p) > 300]
    for p in pieces:
        p = p.reset_index(drop=True)
        o, h, l, c = (p[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        A = base_atr(h, l, c, 14)
        for name, (buy, sell, _k) in r7.rules(o, h, l, c).items():
            buy, sell = np.nan_to_num(buy.astype(float)) > 0, np.nan_to_num(sell.astype(float)) > 0
            buy, sell = buy & ~np.concatenate([[False], buy[:-1]]), sell & ~np.concatenate([[False], sell[:-1]])
            for sl, tp in EXITS[tf]:
                s = r5._put(r5._sig(len(c)), buy, sell, c - sl * A, c + sl * A, c + tp * A, c - tp * A)
                t = r5.simulate_piece(p, dict(s, hold=HOLD[tf]), dict(cost_frac=cf, swap_long=0.0, swap_short=0.0))
                if len(t):
                    res.setdefault(f"{name} sl{sl:g} tp{tp:g}", []).append(t)
    return {k: pd.concat(v, ignore_index=True) for k, v in res.items()}


def fam_setups(df, tf, cf):
    gst.HOLD.update(HOLD)
    res = {}
    for p in ([df] if tf == "D1" else [p for p in fr.segments(df) if len(p) > 300]):
        out = gst.run(p.reset_index(drop=True), tf, None, None, dict(cost_frac=cf, swap_long=0.0, swap_short=0.0))
        for (name, sl, tp), ts in out.items():
            if name in ("pullback_strict", "range_mid_bb", "range_mid_dc"):
                res.setdefault(f"{name} sl{sl:g} tp{tp}", []).extend(ts)
    return {k: pd.concat(v, ignore_index=True) for k, v in res.items()}


def fam_cheatsheet(df, tf, cf):
    trend = cs.h1_trend(df, cs.resample(df, "1h"))
    return {f"{k} {tp:g}R": t for (k, tp), t in cs.trades(df, trend).items()}


def fam_smc(df, tf, cf):
    return {f"smc {e} {x}": gsm.trades(df, e, x) for e in ("50%", "3rd") for x in ("liquidity", "2R", "3R")}


def fam_raremain(df, tf, cf):
    return {f"{s} {tp:g}R": grm.trades(df, s, tp, cf, HOLD[tf]) for s in ("turtle_soup", "main_setup", "asia_accept")
            for tp in (1.5, 2.0, 3.0)}


_SIG_CACHE: dict = {}
_orig_signals = gim.signals


def _cached_signals(p, method):                     # gim.trades() recomputes signals for every target: compute once
    key = (id(p), len(p), method)
    if key not in _SIG_CACHE:
        _SIG_CACHE.clear() if len(_SIG_CACHE) > 50 else None
        _SIG_CACHE[key] = _orig_signals(p, method)
    return _SIG_CACHE[key]


gim.signals = _cached_signals


def fam_internet(df, tf, cf):
    plan = {"ny_orb15": [1.0, 2.0, "end"], "ny_orb30": [1.0, 2.0, "end"], "ldn_orb30": [1.0, 2.0, "end"], "ny_momentum": ["end"],
            "ema9_21_rsi": [1.5, 2.0], "ema4_stack": [1.5, 2.0], "vwap_pullback": [2.0], "vwap_fade": [None],
            "pivot_bounce": [None], "pivot_break": [None], "st_adx": [None]}
    out = {}
    for m, tps in plan.items():
        for tp in tps:
            out[f"{m} {tp}"] = gim.trades(df, m, tp, cf, HOLD[tf])
    return out


def fam_litefinance(df, tf, cf):
    kinds = {"D1": ["trend", "break"], "H4": ["swing"], "M5": ["scalp"], "M1": ["scalp"]}.get(tf, [])
    glf.HOLD.update(HOLD)
    out = {}
    for k in kinds:
        for v, t in glf.trades(df, k, tf, cf).items():
            out[f"lf_{k} {v}"] = t
    return out


PLAN = {"M1": ["grid", "internet", "litefinance"],
        "M5": ["grid", "setups", "cheatsheet", "smc", "raremain", "internet", "litefinance"],
        "M15": ["grid", "setups", "cheatsheet", "smc", "raremain", "internet"],
        "H1": ["grid", "setups", "cheatsheet", "raremain", "hourdrift"],
        "H4": ["grid", "setups", "litefinance"],
        "D1": ["grid", "setups", "litefinance"]}
FAMS = {"grid": fam_grid, "setups": fam_setups, "cheatsheet": fam_cheatsheet, "smc": fam_smc, "raremain": fam_raremain,
        "internet": fam_internet, "litefinance": fam_litefinance}


def hourdrift(df, cf):
    d = df.assign(r=(df.close - df.open) / df.open)
    cut = gs.split_at(d)
    rows = []
    for hr, g in d.groupby(d.time.dt.hour):
        a, b = g[g.time < cut].r, g[g.time >= cut].r
        if len(a) < 50 or len(b) < 50:
            continue
        rows.append({"method": f"hour {hr:02d} UTC", "first60 R": round(float(a.mean() / cf), 4),
                     "unseen R": round(float(b.mean() / cf), 4),
                     "unseen t": round(float(b.mean() / b.std() * np.sqrt(len(b))), 2),
                     "note": "R here = average hourly move / cost per trade (> 1 means the move beats the cost)"})
    return rows


def run_one(sym: str, tf: str, fam: str) -> Path:
    path = OUT / f"{sym}_{tf}_{fam}.csv"
    if path.exists():
        return path
    df = data(sym, tf)
    cf = COST[sym]
    if fam == "hourdrift":
        rows = hourdrift(df, cf)
    else:
        cut = gs.split_at(df)
        years = (df.time.iloc[-1] - df.time.iloc[0]).days / 365.25
        rows = []
        for name, t in FAMS[fam](df, tf, cf).items():
            s = summarise(t, cut, years)
            if s:
                rows.append({"method": name, **s})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).assign(symbol=sym, tf=tf, family=fam).to_csv(path, index=False)
    print(f"{sym} {tf} {fam}: {len(rows)} variants", flush=True)
    return path


def report() -> None:
    C = pd.concat([pd.read_csv(p) for p in sorted(OUT.glob("*.csv")) if p.stat().st_size > 50], ignore_index=True)
    T = C[C.family != "hourdrift"].copy()
    T["ROBUST"] = (T["first60 R"] > 0) & (T["unseen R"] > 0) & (T["unseen t"] >= 2) & (T["2x cost unseen R"] > 0)
    T["style"] = T.tf.map(lambda x: "scalping" if x in SCALP_YEARS else "swing")
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 400):
        print(f"variants tested: {len(T):,}")
        print("\nSHARE OF VARIANTS PROFITABLE ON UNSEEN DATA (and number of ROBUST ones):")
        g = T.groupby(["symbol", "tf"]).agg(variants=("unseen R", "size"), profitable=("unseen R", lambda x: round(100 * (x > 0).mean(), 1)),
                                           avg_R=("unseen R", lambda x: round(x.mean(), 3)), robust=("ROBUST", "sum"))
        print(g.unstack("tf").to_string())
        R = T[T.ROBUST].sort_values("unseen t", ascending=False)
        print(f"\nROBUST VARIANTS ({len(R)}): positive in both halves, unseen t >= 2, positive at 2x costs")
        print(R[["symbol", "tf", "family", "method", "trades/yr", "win %", "first60 R", "unseen R", "unseen t", "2x cost unseen R"]]
              .head(60).to_string(index=False))
        H = C[C.family == "hourdrift"]
        if len(H):
            h = H[(H["first60 R"].abs() > 1) & (H["unseen R"].abs() > 1) & (np.sign(H["first60 R"]) == np.sign(H["unseen R"]))]
            print("\nHOURS WHOSE AVERAGE MOVE BEATS THE COST IN BOTH HALVES (R = move / cost):")
            print(h[["symbol", "method", "first60 R", "unseen R", "unseen t"]].to_string(index=False) if len(h) else "none")
    T.to_csv(ROOT / "reports" / "all_pairs_verified_all.csv", index=False)


def main() -> int:
    if "--report" not in sys.argv:
        only = [a for a in sys.argv[1:] if not a.startswith("--")]
        for tf in ("H1", "H4", "D1", "M15", "M5", "M1"):
            for sym in SYMS:
                if only and sym not in only and tf not in only:
                    continue
                for fam in PLAN[tf]:
                    run_one(sym, tf, fam)
    report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
