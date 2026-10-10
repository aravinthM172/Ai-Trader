"""
EURUSD / GBPUSD build on the longest history: daily, weekly and 4-hour candles.  (2026-10-10)

    python -m backtest.fx_build             -> FX_BUILD.md, reports/fx_build.json
    python -m backtest.fx_build --ledger    same, and record the weekly and 4-hour tests in research/ledger.csv

A currency pair has no long-run drift, so unlike the index builds this trades both directions.
Data   D1: GBPUSD October 1996 -, EURUSD January 1999 - (the euro's first day); Federal Reserve daily rates
       before December 2003 (one price a day), Yahoo candles after (backtest/round7_d1_30y.long_d1).
       W1: the same history as weekly candles.  H4: built from the hourly files (EURUSD Dukascopy 2005-,
       GBPUSD FundingPips 2011-).
Rules  the 43 round-7 indicator rules, both directions, plus buy-only and sell-only splits of each result.
       D1 exits as round 7 (already recorded there).  PRE-REGISTERED for the new timeframes:
         W1  trend 2 ATR stop / 6 ATR target / 12 weeks;  mean reversion 2 / 2 / 3 weeks
         H4  trend 2 ATR stop / 6 ATR target / 48 bars;   mean reversion 2 / 2 / 12 bars
Costs  FundingPips spread + slippage + swap.  PASS RULE and "works" as in backtest/strategy_round7.py.
Read-only; no orders.
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backtest import full_reassessment as fr
from backtest import round7_d1_30y as d30
from backtest import strategy_round3_daily as r3
from backtest import strategy_round5 as r5
from backtest import strategy_round6 as r6
from backtest import strategy_round7 as r7

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ["EURUSD", "GBPUSD"]
r7.EXITS["W1"] = dict(T=(2.0, 6.0, 12), M=(2.0, 2.0, 3))
r7.EXITS["H4"] = dict(T=(2.0, 6.0, 48), M=(2.0, 2.0, 12))


def resample(d: pd.DataFrame, rule: str) -> pd.DataFrame:
    g = d.set_index("time").resample(rule, label="left", closed="left")
    out = pd.DataFrame(dict(open=g.open.first(), high=g.high.max(), low=g.low.min(), close=g.close.last())).dropna()
    return out[out.high > out.low].reset_index()


def run(df: pd.DataFrame, cost: dict, tf: str) -> dict:
    parts = {k: [] for k in r7.RULE_KIND}
    for p in (fr.segments(df) if tf == "H4" else [df]):
        s = r7.signals(p, tf)
        for k in r7.RULE_KIND:
            parts[k].append(r5.simulate_piece(p, s[k], cost))
    return {k: r7._cat(v) for k, v in parts.items()}


def main() -> int:
    warnings.filterwarnings("ignore")
    fr.MIN_SEGMENT = 300                                          # 4-hour pieces between data holes
    n_trials = r3.ledger_n() + r3.PRIOR_TRIALS + 2 * len(PAIRS) * len(r7.RULE_KIND)
    res, meta = {}, {}
    for s in PAIRS:
        cost = r6.cost(s)
        d = d30.long_d1(s)
        w = resample(d, "W-MON")
        h, src = r7.h1(s)
        h4 = resample(h, "4h")
        frames = {"D1": d, "W1": w, "H4": h4}
        for tf, f in frames.items():
            yrs = sum((p.time.iloc[-1] - p.time.iloc[0]).days for p in fr.segments(f)) / 365.25 if tf == "H4" \
                else (f.time.iloc[-1] - f.time.iloc[0]).days / 365.25
            meta[f"{s} {tf}"] = f"{f.time.iloc[0].date()} to {f.time.iloc[-1].date()}, {yrs:.1f} years, {len(f)} candles" + (f" ({src})" if tf == "H4" else "")
            for k, t in run(f, cost, tf).items():
                res[f"{tf}|{k}|{s}"] = r7.judge(t, yrs, n_trials)
        print(s, {tf: [k.split("|")[1] for k, r in res.items() if k.startswith(tf) and k.endswith(s) and r["verdict"] in ("PASS", "works")] for tf in frames}, file=sys.stderr)
    if "--ledger" in sys.argv:
        tested = datetime.now(timezone.utc).isoformat(timespec="seconds")
        r3.append_ledger([dict(tested_utc=tested, name="fx_" + k.replace("|", "_"), params="{}", lookahead_ok=True, n_trials=n_trials,
                               deflated_sharpe=r.get("dsr", ""), passes=bool(r.get("passes")), useful=bool(r.get("passes")))
                          for k, r in res.items() if not k.startswith("D1")])
    rep = dict(meta=dict(n_trials=n_trials, data=meta), results=res)
    (ROOT / "reports" / "fx_build.json").write_text(json.dumps(rep, indent=2, default=str))
    md = _md(rep)
    (ROOT / "FX_BUILD.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def _md(rep: dict) -> str:
    res = rep["results"]
    mark = {"PASS": "**PASS**", "works": "**works**", "weak": "weak", "no": "-"}
    L = ["# EURUSD / GBPUSD build: daily, weekly and 4-hour candles on the longest history", "",
         f"Deflated-Sharpe N = {rep['meta']['n_trials']}.  Rules pre-registered in `backtest/fx_build.py`.", ""]
    L += [f"- {k}: {v}" for k, v in rep["meta"]["data"].items()]
    L += ["", "## Count of rules per verdict (43 rules each)", "", "| pair | timeframe | PASS | works | weak | no edge | rules with mean R > 0 |", "|---|---|--:|--:|--:|--:|--:|"]
    for s in PAIRS:
        for tf in ("D1", "W1", "H4"):
            v = [r for k, r in res.items() if k.startswith(tf + "|") and k.endswith("|" + s)]
            c = lambda x: sum(1 for r in v if r["verdict"] == x)
            L.append(f"| {s} | {tf} | {c('PASS')} | {c('works')} | {c('weak')} | {c('no')} | {sum(1 for r in v if (r.get('mean_R') or -1) > 0)} |")
    for s in PAIRS:
        for tf in ("D1", "W1", "H4"):
            L += ["", f"## {s} {tf}: best 8 by R per year", "",
                  "| rule | trades/yr | mean R | t | R per year | max DD R | buy R | sell R | first 70 % | last 30 % | last 2 y | WF+ | 2x cost | verdict |",
                  "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|"]
            rows = [(k.split("|")[1], r) for k, r in res.items() if k.startswith(tf + "|") and k.endswith("|" + s) and "checks" in r]
            for k, r in sorted(rows, key=lambda kr: -kr[1]["R_per_year"])[:8]:
                L.append(f"| {k} | {r['trades_per_year']} | {r['mean_R']} | {r['t']} | {r['R_per_year']} | {r['max_dd_R']} | {r['long_R']} | {r['short_R']} | "
                         f"{r['first70_R']} | {r['last30_R']} | {r['last2y_R']} | {r['wf_positive']}/8 | {r['cost2x_R']} | {mark[r['verdict']]} |")
    good = [k for k, r in res.items() if r["verdict"] in ("PASS", "works")]
    L += ["", f"**PASS or works ({len(good)}):** {', '.join(good) or 'none'}", ""]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
