"""
Research gate: the hard, code-enforced checks every new strategy idea must pass.
Used by the research agent (agents/research_agent.md) and by hand.

For each builder in backtest/research_strategies.ROUND1 that is NOT yet in the ledger:
  1. CRITIC (automatic look-ahead test): the signal recomputed on data truncated at
     several points must equal the full-data signal before each cut.  Any mismatch = REJECT.
  2. EVIDENCE: Valetax BTC splits + walk-forward, Bitstamp 2014-2023 unseen years,
     Valetax XAU -- the same evidence strategy_round1.py uses.
  3. STATISTICIAN: deflated Sharpe on the Bitstamp unseen trades with
     N = every idea ever recorded in research/ledger.csv (+ this one).  Must be >= 0.95.
  4. USEFULNESS: beats the live strategy on Bitstamp OR daily correlation with it < 0.5.
Every tested idea is appended to the ledger, pass or fail, so N only grows.

    python -m backtest.research_gate

Research only -- never touches live files.
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest import btc_h1_validation as bv
from backtest import btc_longhistory_validation as lh
from backtest import xau_h1_validation as xv
from backtest import btc_strategies as strat
from backtest import strategy_round1 as r1
from backtest.multi_symbol_scan import deflated_sharpe
from backtest.research_strategies import ROUND1, LIVE_MOMENTUM

LEDGER = ROOT / "research" / "ledger.csv"
FIELDS = ["tested_utc", "name", "params", "lookahead_ok", "btc_exp_R", "btc_wf", "bitstamp_exp_R",
          "bitstamp_t", "xau_exp_R", "corr_with_live", "n_trials", "deflated_sharpe", "passes", "useful"]
# Ideas tested before the ledger existed (honest count): 5 H1 candidates + 9 robustness
# neighbours (btc_h1_validation).  The 4 strategy_round1 ideas are seeded into the ledger.
PRIOR_TRIALS = 14


def ledger_rows() -> list[dict]:
    if not LEDGER.exists():
        return []
    with LEDGER.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append(row: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    new = not LEDGER.exists()
    with LEDGER.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k) for k in FIELDS})


def lookahead_ok(builder, kw, df, cuts=(0.4, 0.6, 0.8)) -> bool:
    c, h, l = (df[k].to_numpy(float) for k in ("close", "high", "low"))
    full, _ = builder(c, h, l, **kw)
    for f in cuts:
        k = int(len(c) * f)
        part, _ = builder(c[:k], h[:k], l[:k], **kw)
        # compare THROUGH the last bar before the cut: entries[k-1] may use bars <= k-2 only,
        # so a one-bar peek (using close[i] for the entry at open[i]) shows up exactly here
        if not np.array_equal(full[:k], part[:k]):
            return False
    return True


def bitstamp_unseen_R(builder, kw, df, ref) -> np.ndarray:
    c, h, l = (df[k].to_numpy(float) for k in ("close", "high", "low"))
    px = {k: df[k].to_numpy(float) for k in ("open", "high", "low", "close")}
    ent, wu = builder(c, h, l, **kw)
    atr = strat.base_atr(h, l, c, 14)
    _, _, R_unseen = lh.run_years(px, atr, ent, wu, lh.year_folds(df, wu), bv.COST_SCENARIOS["realistic"], ref)
    return R_unseen


def main() -> int:
    done = {r["name"] for r in ledger_rows()}
    todo = {k: v for k, v in ROUND1.items() if k not in done}
    if not todo:
        print("nothing new to test -- add a builder to backtest/research_strategies.ROUND1")
        return 0
    vdf, bdf, xdf = bv.load(), lh.load(lh.DATA), xv.load()
    ref = lh.ref_price()
    live_v = bv.evaluate("live", strat.build_momentum, vdf, LIVE_MOMENTUM)
    live_b = r1.bitstamp(strat.build_momentum, LIVE_MOMENTUM, bdf, ref)
    times = vdf["time"].to_numpy()
    live_d = r1.daily_R(live_v["_trades"], times)

    results = {}
    for name, (bld, kw) in todo.items():
        n_trials = PRIOR_TRIALS + len(ledger_rows()) + 1
        la = lookahead_ok(bld, kw, vdf)
        row = dict(tested_utc=datetime.now(timezone.utc).isoformat(), name=name, params=json.dumps(kw),
                   lookahead_ok=la, n_trials=n_trials)
        if not la:
            row.update(passes=False, useful=False)
            append(row)
            results[name] = row
            print(f"{name}: REJECTED -- look-ahead detected (signal changes when future bars are removed)")
            continue
        v = bv.evaluate(name, bld, vdf, kw)
        x = xv.evaluate(name, bld, xdf, kw)
        b = r1.bitstamp(bld, kw, bdf, ref)
        Ru = bitstamp_unseen_R(bld, kw, bdf, ref)
        dsr = deflated_sharpe(Ru, n_trials) if len(Ru) else 0.0
        j = pd.concat([live_d, r1.daily_R(v["_trades"], times)], axis=1, sort=True).fillna(0.0)
        corr = float(j.corr().iloc[0, 1])
        rec = dict(valetax={k: v[k] for k in ("combined", "per_split", "walk_forward")},
                   xau={k: x[k] for k in ("combined", "per_split", "walk_forward")}, bitstamp=b, corr_with_live=corr)
        verdict = r1.verdict(rec, dict(bitstamp=live_b))
        passes = verdict["passes"] and dsr >= 0.95
        useful = passes and (verdict["beats_live_on_bitstamp"] or corr < 0.5)
        row.update(btc_exp_R=v["combined"].get("expectancy_R"), btc_wf=v["walk_forward"].get("folds_positive_expectancy"),
                   bitstamp_exp_R=b["pooled_unseen"].get("expectancy_R"), bitstamp_t=b["pooled_unseen"].get("t_stat"),
                   xau_exp_R=x["combined"].get("expectancy_R"), corr_with_live=round(corr, 3),
                   deflated_sharpe=round(dsr, 3), passes=passes, useful=useful)
        append(row)
        results[name] = dict(row, failed_checks=[k for k, ok in verdict["checks"].items() if not ok])
        print(f"{name}: lookahead ok | BTC {row['btc_exp_R']} WF {row['btc_wf']}/8 | Bitstamp {row['bitstamp_exp_R']} "
              f"t={row['bitstamp_t']} | XAU {row['xau_exp_R']} | corr {row['corr_with_live']} | "
              f"DSR {row['deflated_sharpe']} (N={n_trials}) | PASS={passes} USEFUL={useful}")
    out = ROOT / "reports" / "research_gate_last.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
