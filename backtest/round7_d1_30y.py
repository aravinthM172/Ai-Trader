"""
Round-7 daily sweep on the longest daily history that exists for the FX pairs.  (2026-10-10)

    python -m backtest.round7_d1_30y             -> STRATEGY_ROUND7_D1_30Y.md
    python -m backtest.round7_d1_30y --ledger    same, and record every test in research/ledger.csv

GBPUSD back to October 1996 (30 years) and EURUSD back to January 1999 (the euro's first day): Federal
Reserve daily rates (FRED DEXUSUK / DEXUSEU, one price per day) before Yahoo's first day (December 2003),
Yahoo daily candles after it.  A day with only one price gets open = previous close and high / low = the
larger / smaller of open and close, so the ATR is smaller and stops are only seen at the close in that part.
Same 43 rules, exits, costs and pass rule as backtest/strategy_round7.py.  Read-only; no orders.
"""
from __future__ import annotations

import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backtest import strategy_round3_daily as r3
from backtest import strategy_round6 as r6
from backtest import strategy_round7 as r7

ROOT = Path(__file__).resolve().parents[1]
FRED = {"GBPUSD": "DEXUSUK", "EURUSD": "DEXUSEU"}


def long_d1(sym: str) -> pd.DataFrame:
    y = r7.d1(sym)
    c = r6.fred(FRED[sym])
    c = c[c.index >= pd.Timestamp.now().normalize() - pd.Timedelta(days=30 * 365.25)]
    old = pd.DataFrame({"time": pd.DatetimeIndex(c.index).tz_localize("UTC") + pd.Timedelta(hours=21), "close": c.to_numpy()})
    old["open"] = old.close.shift(1)
    old = old.dropna()
    old["high"], old["low"] = old[["open", "close"]].max(axis=1), old[["open", "close"]].min(axis=1)
    old = old[(old.time < y.time.iloc[0]) & (old.high > old.low)]
    return pd.concat([old[["time", "open", "high", "low", "close"]], y], ignore_index=True)


def main() -> int:
    warnings.filterwarnings("ignore")
    n_trials = r3.ledger_n() + r3.PRIOR_TRIALS + len(FRED) * (len(r7.RULE_KIND) + 1)
    res, meta = {}, {}
    for s in FRED:
        d, cost = long_d1(s), r6.cost(s)
        yrs = (d.time.iloc[-1] - d.time.iloc[0]).days / 365.25
        meta[s] = f"{d.time.iloc[0].date()} to {d.time.iloc[-1].date()} ({yrs:.1f} years, {len(d)} days)"
        for k, t in r7.sweep(d, cost, "D1").items():
            res[f"{k}|{s}"] = r7.judge(t, yrs, n_trials)
        res[f"turn_of_month|{s}"] = r7.judge(r7.turn_of_month(d, cost), yrs, n_trials)
    if "--ledger" in sys.argv:
        tested = datetime.now(timezone.utc).isoformat(timespec="seconds")
        r3.append_ledger([dict(tested_utc=tested, name="r7d30_" + k.replace("|", "_"), params="{}", lookahead_ok=True, n_trials=n_trials,
                               deflated_sharpe=r.get("dsr", ""), passes=bool(r.get("passes")), useful=bool(r.get("passes")))
                          for k, r in res.items()])
    mark = {"PASS": "**PASS**", "works": "**works**", "weak": "weak", "no": "-"}
    L = ["# Round 7 daily sweep on the longest FX history", "",
         f"Deflated-Sharpe N = {n_trials}.  Rules, exits and costs as in `backtest/strategy_round7.py`.", ""]
    L += [f"- {s}: {m}" for s, m in meta.items()]
    for s in FRED:
        L += ["", f"## {s}", "",
              "| rule | trades/yr | mean R | t | R per year | max DD R | buy R | sell R | first 70 % | last 30 % | last 2 y | WF+ | 2x cost | verdict |",
              "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|"]
        rows = [(k.split("|")[0], r) for k, r in res.items() if k.endswith("|" + s) and "checks" in r]
        for k, r in sorted(rows, key=lambda kr: -kr[1]["R_per_year"]):
            L.append(f"| {k} | {r['trades_per_year']} | {r['mean_R']} | {r['t']} | {r['R_per_year']} | {r['max_dd_R']} | {r['long_R']} | {r['short_R']} | "
                     f"{r['first70_R']} | {r['last30_R']} | {r['last2y_R']} | {r['wf_positive']}/8 | {r['cost2x_R']} | {mark[r['verdict']]} |")
    good = [k for k, r in res.items() if r["verdict"] in ("PASS", "works")]
    weak = [k for k, r in res.items() if r["verdict"] == "weak"]
    L += ["", f"**PASS or works ({len(good)}):** {', '.join(good) or 'none'}", "", f"**Weak ({len(weak)}):** {', '.join(weak) or 'none'}", ""]
    md = "\n".join(L)
    (ROOT / "STRATEGY_ROUND7_D1_30Y.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
