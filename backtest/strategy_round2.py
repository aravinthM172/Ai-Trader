"""
Round-2 test: the classic open-source strategy families (backtest/research_strategies.py ROUND2:
Supertrend, Turtle/Donchian + Supertrend, EMA cross, MACD, TTM squeeze, Connors RSI-2) against the
live momentum_rsi_mtf, on the same three datasets as round 1:

  1. Valetax BTCUSD.vx H1   -- TRAIN/VALIDATION/FINAL splits + 8-fold walk-forward
  2. Bitstamp BTC H1 2014-2023 -- years never seen by any parameter
  3. Valetax XAUUSD.vx H1   -- a second, unrelated market

    python -m backtest.strategy_round2

PRE-REGISTERED PASS RULE (fixed 2026-10-05 before the first run; do not edit after seeing results):
  identical to round 1 (backtest/strategy_round1.verdict) except the Bitstamp t-stat bar, raised from
  2.5 to 2.75 for 6 candidates (Bonferroni, one-sided alpha 0.05 / 6 ~ z 2.64, rounded up):
    - Valetax BTC: combined expectancy_R > 0, FINAL expectancy_R > 0, walk-forward >= 6/8
    - Bitstamp:    pooled unseen expectancy_R > 0, t >= 2.75, majority of unseen years positive
    - XAU:         combined expectancy_R > 0, walk-forward >= 5/8
  USEFUL only if it also beats live on Bitstamp OR its daily-R correlation with live is < 0.5.
  A USEFUL candidate then still has to pass backtest.multi_symbol_scan (swap included) before any
  paper or live use.

Appends to research/ledger.csv.  Writes reports/strategy_round2.json and STRATEGY_ROUND2.md.
Research only -- NO LIVE ORDER.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.logging_setup import get_logger
from backtest import btc_strategies as strat
from backtest import btc_h1_validation as bv
from backtest import btc_longhistory_validation as lh
from backtest import xau_h1_validation as xv
from backtest import strategy_round1 as r1
from backtest.research_strategies import ROUND2, LIVE_MOMENTUM

log = get_logger("strategy.round2")
LIVE = r1.LIVE
ALL = {LIVE: (strat.build_momentum, LIVE_MOMENTUM), **ROUND2}
T_BAR = 2.75
LEDGER = ROOT / "research" / "ledger.csv"


def verdict(r, live):
    v = r1.verdict(r, live)
    t = (r["bitstamp"]["pooled_unseen"].get("t_stat") or 0)
    v["checks"].pop("bitstamp_t_2_5")
    v["checks"]["bitstamp_t_2_75"] = t >= T_BAR
    v["passes"] = all(v["checks"].values())
    v["useful"] = v["passes"] and (v["beats_live_on_bitstamp"] or v["diversifies"])
    return v


def _ledger(res):
    prior = 0
    if LEDGER.exists():
        with LEDGER.open(encoding="utf-8") as f:
            prior = sum(1 for _ in f) - 1
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with LEDGER.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for i, (n, r) in enumerate((k, v) for k, v in res.items() if k != LIVE):
            pu, vd = r["bitstamp"]["pooled_unseen"], r["verdict"]
            w.writerow([now, n, "{}", True, r["valetax"]["combined"].get("expectancy_R"),
                        r["valetax"]["walk_forward"].get("folds_positive_expectancy"), pu.get("expectancy_R"),
                        pu.get("t_stat"), r["xau"]["combined"].get("expectancy_R"), r["corr_with_live"],
                        14 + prior + i + 1, "", vd["passes"], vd["useful"]])


def main() -> int:
    t0 = time.perf_counter()
    vdf, bdf, xdf = bv.load(), lh.load(lh.DATA), xv.load()
    ref = lh.ref_price()
    res = {}
    for name, (bld, kw) in ALL.items():
        v = bv.evaluate(name, bld, vdf, kw)
        x = xv.evaluate(name, bld, xdf, kw)
        b = r1.bitstamp(bld, kw, bdf, ref)
        res[name] = dict(valetax={k: v[k] for k in ("combined", "per_split", "walk_forward")},
                         xau={k: x[k] for k in ("combined", "per_split", "walk_forward")},
                         bitstamp=b, _vt=v["_trades"])
        log.info("%-20s BTC expR=%s WF=%s/8 | Bitstamp unseen expR=%s t=%s | XAU expR=%s WF=%s/8",
                 name, v["combined"].get("expectancy_R"), v["walk_forward"].get("folds_positive_expectancy"),
                 b["pooled_unseen"].get("expectancy_R"), b["pooled_unseen"].get("t_stat"),
                 x["combined"].get("expectancy_R"), x["walk_forward"].get("folds_positive_expectancy"))
    times = vdf["time"].to_numpy()
    live_d = r1.daily_R(res[LIVE]["_vt"], times)
    for name, r in res.items():
        j = pd.concat([live_d, r1.daily_R(r["_vt"], times)], axis=1, sort=True).fillna(0.0)
        r["corr_with_live"] = None if name == LIVE else round(float(j.corr().iloc[0, 1]), 3)
    for name, r in res.items():
        if name != LIVE:
            r["verdict"] = verdict(r, res[LIVE])
        r.pop("_vt")
    rep = dict(meta=dict(rule="pre-registered in module docstring", n_candidates=len(ROUND2), t_bar=T_BAR,
                         swap_in_costs=False, runtime_s=round(time.perf_counter() - t0, 1)), results=res)
    (ROOT / "reports" / "strategy_round2.json").write_text(json.dumps(rep, indent=2, default=str))
    md = r1._md(rep).replace("Strategy round 1", "Strategy round 2 (classic open-source families)") \
                    .replace("strategy_round1.py", "strategy_round2.py")
    (ROOT / "STRATEGY_ROUND2.md").write_text(md, encoding="utf-8")
    _ledger(res)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
