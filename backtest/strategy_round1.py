"""
Round-1 test of the research strategies (backtest/research_strategies.py) against
the live momentum_rsi_mtf, on all three datasets the live strategy was judged on:

  1. Valetax BTCUSD.vx H1   -- TRAIN/VALIDATION/FINAL splits + 8-fold walk-forward
  2. Bitstamp BTC H1 2014-2023 -- years the parameters never saw (scaled costs)
  3. Valetax XAUUSD.vx H1   -- a second, unrelated market

    python -m backtest.strategy_round1

PRE-REGISTERED PASS RULE (fixed before the first run; do not edit after seeing results):
  a candidate PASSES only if ALL hold, at realistic costs:
    - Valetax BTC: combined expectancy_R > 0, FINAL split expectancy_R > 0,
      walk-forward >= 6/8 folds positive
    - Bitstamp:    pooled unseen-year expectancy_R > 0, t-stat >= 2.5
                   (Bonferroni-style bar for 4 candidates), majority of unseen years positive
    - XAU:         combined expectancy_R > 0, walk-forward >= 5/8
  and it is USEFUL only if, in addition, either
    - Bitstamp pooled expectancy_R beats the live strategy's, or
    - daily-R correlation with the live strategy on Valetax BTC is < 0.5
      (a diversifier rather than a copy).

Swap/financing is NOT yet in the cost model (see ADD_LIST Phase 0) -- applies to all
strategies equally, so the comparison is fair but absolute numbers are optimistic.

Writes reports/strategy_round1.json and STRATEGY_ROUND1.md.  Research only -- NO LIVE ORDER.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.logging_setup import get_logger
from backtest import btc_lab as lab
from backtest import btc_strategies as strat
from backtest import btc_h1_validation as bv
from backtest import btc_longhistory_validation as lh
from backtest import xau_h1_validation as xv
from backtest.research_strategies import ROUND1, LIVE_MOMENTUM

log = get_logger("strategy.round1")
REPORTS = ROOT / "reports"
LIVE = "momentum_rsi_mtf"
ALL = {LIVE: (strat.build_momentum, LIVE_MOMENTUM), **ROUND1}
N_CANDIDATES = len(ROUND1)
T_BAR = 2.5


def daily_R(trades, times):
    t = pd.Series(np.where(trades["risk"] > 0, trades["pnl"] / trades["risk"], 0.0),
                  index=pd.to_datetime(times[trades["exit_i"]]).floor("D"))
    return t.groupby(level=0).sum()


def bitstamp(builder, kw, df, ref):
    c, h, l = (df[k].to_numpy(float) for k in ("close", "high", "low"))
    px = {k: df[k].to_numpy(float) for k in ("open", "high", "low", "close")}
    ent, wu = builder(c, h, l, **kw)
    atr = strat.base_atr(h, l, c, 14)
    rows, _, R_unseen = lh.run_years(px, atr, ent, wu, lh.year_folds(df, wu),
                                     bv.COST_SCENARIOS["realistic"], ref)
    return dict(pooled_unseen=lh.pooled(R_unseen), years=rows, **lh.summarise(rows))


def verdict(r, live):
    v, b, x = r["valetax"], r["bitstamp"], r["xau"]
    pu = b["pooled_unseen"]
    checks = {
        "btc_combined_pos": (v["combined"].get("expectancy_R") or -1) > 0,
        "btc_final_pos": (v["per_split"]["FINAL"].get("expectancy_R") or -1) > 0,
        "btc_wf_6of8": (v["walk_forward"].get("folds_positive_expectancy") or 0) >= 6,
        "bitstamp_pos": (pu.get("expectancy_R") or -1) > 0,
        "bitstamp_t_2_5": (pu.get("t_stat") or 0) >= T_BAR,
        "bitstamp_majority_years": bool(b.get("majority_unseen_years_positive")),
        "xau_combined_pos": (x["combined"].get("expectancy_R") or -1) > 0,
        "xau_wf_5of8": (x["walk_forward"].get("folds_positive_expectancy") or 0) >= 5,
    }
    passes = all(checks.values())
    beats = (pu.get("expectancy_R") or -1) > (live["bitstamp"]["pooled_unseen"].get("expectancy_R") or 9)
    diversifies = r.get("corr_with_live") is not None and r["corr_with_live"] < 0.5
    return dict(checks=checks, passes=passes, beats_live_on_bitstamp=beats,
                diversifies=diversifies, useful=passes and (beats or diversifies))


def main() -> int:
    t0 = time.perf_counter()
    vdf, bdf, xdf = bv.load(), lh.load(lh.DATA), xv.load()
    ref = lh.ref_price()
    log.info("valetax %d rows, bitstamp %d rows, xau %d rows", len(vdf), len(bdf), len(xdf))

    res = {}
    for name, (bld, kw) in ALL.items():
        v = bv.evaluate(name, bld, vdf, kw)
        x = xv.evaluate(name, bld, xdf, kw)
        b = bitstamp(bld, kw, bdf, ref)
        res[name] = dict(valetax={k: v[k] for k in ("combined", "per_split", "walk_forward")},
                         xau={k: x[k] for k in ("combined", "per_split", "walk_forward")},
                         bitstamp=b, _vt=v["_trades"])
        log.info("%-20s BTC expR=%s WF=%s/8 | Bitstamp unseen expR=%s t=%s | XAU expR=%s WF=%s/8",
                 name, v["combined"].get("expectancy_R"), v["walk_forward"].get("folds_positive_expectancy"),
                 b["pooled_unseen"].get("expectancy_R"), b["pooled_unseen"].get("t_stat"),
                 x["combined"].get("expectancy_R"), x["walk_forward"].get("folds_positive_expectancy"))

    times = vdf["time"].to_numpy()
    live_d = daily_R(res[LIVE]["_vt"], times)
    for name, r in res.items():
        d = daily_R(r["_vt"], times)
        j = pd.concat([live_d, d], axis=1, sort=True).fillna(0.0)
        r["corr_with_live"] = None if name == LIVE else round(float(j.corr().iloc[0, 1]), 3)
        live_entries = set(res[LIVE]["_vt"]["entry_i"].tolist())
        own = r["_vt"]["entry_i"].tolist()
        r["share_of_entries_also_live"] = round(sum(e in live_entries for e in own) / max(1, len(own)), 3)
    for name, r in res.items():
        if name != LIVE:
            r["verdict"] = verdict(r, res[LIVE])
        r.pop("_vt")

    rep = dict(meta=dict(rule="pre-registered in module docstring", n_candidates=N_CANDIDATES,
                         t_bar=T_BAR, swap_in_costs=False, runtime_s=round(time.perf_counter() - t0, 1)),
               results=res)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "strategy_round1.json").write_text(json.dumps(rep, indent=2, default=str))
    (ROOT / "STRATEGY_ROUND1.md").write_text(_md(rep), encoding="utf-8")
    print(_md(rep))
    return 0


def _md(rep):
    L = ["# Strategy round 1 -- research candidates vs live momentum_rsi_mtf", "",
         "Realistic costs, swap NOT included (applies to all equally). Pass rule pre-registered in "
         "`backtest/strategy_round1.py`.", "",
         "| strategy | BTC trades | BTC exp R | BTC FINAL | BTC WF+ | Bitstamp unseen exp R | t | yrs + | XAU exp R | XAU WF+ | corr w/ live | PASS | USEFUL |",
         "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|:-:|"]
    for n, r in rep["results"].items():
        v, b, x = r["valetax"], r["bitstamp"], r["xau"]
        pu = b["pooled_unseen"]; vd = r.get("verdict", {})
        L.append(f"| {n} | {v['combined'].get('trades')} | {v['combined'].get('expectancy_R')} | "
                 f"{v['per_split']['FINAL'].get('expectancy_R')} | {v['walk_forward'].get('folds_positive_expectancy')}/8 | "
                 f"{pu.get('expectancy_R')} | {pu.get('t_stat')} | {b.get('unseen_years_positive')}/{b.get('unseen_years')} | "
                 f"{x['combined'].get('expectancy_R')} | {x['walk_forward'].get('folds_positive_expectancy')}/8 | "
                 f"{r.get('corr_with_live') if r.get('corr_with_live') is not None else '-'} | "
                 f"{('yes' if vd.get('passes') else 'no') if vd else 'live'} | {('yes' if vd.get('useful') else 'no') if vd else '-'} |")
    L += ["", "Failed checks per candidate:", ""]
    for n, r in rep["results"].items():
        if "verdict" in r:
            bad = [k for k, ok in r["verdict"]["checks"].items() if not ok]
            L.append(f"- **{n}**: {', '.join(bad) if bad else 'none'}")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
