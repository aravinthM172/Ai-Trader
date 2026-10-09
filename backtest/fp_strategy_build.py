"""
Build the trading plan from VERIFIED data only: the FundingPips feed for every tradable CFD
(data/fp, tools/collect_fp_history -- checked against TradingView OANDA and Dukascopy 2026-10-07).

    python -m backtest.fp_strategy_build

PRE-REGISTERED (written before the first run, while the data was still downloading):

Eligibility per symbol: >= 10 dense years of H1 (first dense year <= today - 10 y) and < 50 'daily bar
mixed into H1' suspects (tools/collect_fp_history.quality).  Everything else is reported, not traded.

Sleeves
  H1  the live H1 momentum, simulated exactly as execution/live_multi.py (backtest/full_reassessment.
      simulate: entry at the next open, 2 ATR stop, one position, 96-bar time exit, entry bar counts,
      data holes split), exits target 3 ATR (live) and 6 ATR.
  D1  Turtle System 2 (55/20, 2N stop; r3.s_turtle55) on the FundingPips D1 bars (Friday stub merged),
      per symbol and pooled over all eligible symbols.
Costs: FundingPips 1-year median spread + slippage 0.25 x spread per side; FundingPips swap per night
  (annualised from the raw broker value; points mode = points x point x 365 / price).
PASS RULE (r3.evaluate): >= 10 years, >= 100 trades, mean R > 0 on full / first 70 % / last 30 %,
  >= 6 of 8 folds, > 0 at 2x costs, deflated Sharpe >= 0.95 with N = ledger + every test here,
  second source > 0 where one exists (Valetax H1 2024- for the H1 sleeve, Yahoo daily for D1).
Plan = the passing (symbol, sleeve, exit) set.  Prop simulation of the plan: FundingPips 2-step
  (8 % / 5 %, 3 % daily, 10 % max, closed trades), risk 0.25 / 0.35 / 0.5 % per trade.
Writes reports/fp_strategy_build.json and FP_STRATEGY_BUILD.md.  Read-only; no orders.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import full_reassessment as fr
from backtest import strategy_round3_daily as r3
from common.logging_setup import get_logger

log = get_logger("fp.build", filename="fp_strategy_build.log")
ROOT = Path(__file__).resolve().parents[1]
FP = ROOT / "data" / "fp"
MIN_YEARS, MAX_SUSPECTS = 10, 50
H1_EXITS = {"tp3": 3.0, "tp6": 6.0, "tp8": 8.0}   # tp8 added 2026-10-07 after backtest/sl_tp_grid.py chose 2 / 8 (before any build run)
RISKS = (0.0025, 0.0035, 0.005)
VALETAX = {**{s: f"{s}.vx" for s in r3.FX}, "GER40": "DAX40.vx", "NDX100": "NAS100.vx", "SPX500": "SP500.vx",
           "DJI30": "US30.vx", "JP225": "JPN225.vx", "FTSE100": "UK100.vx", "STX50": "EU50.vx",
           "USOIL": "XTIUSD.vx", "UKOIL": "XBRUSD.vx", "XAGUSD": "XAGUSD.vx", "BTCUSD": "BTCUSD.vx",
           "ETHUSD": "ETHUSD.vx"}               # XAUUSD.vx deliberately absent: corrupt (daily bars mixed in)
YAHOO = {**{s: f"{s}=X" for s in r3.FX}, "GER40": "^GDAXI", "NDX100": "^NDX", "SPX500": "^GSPC", "DJI30": "^DJI",
         "JP225": "^N225", "FTSE100": "^FTSE", "STX50": "^STOXX50E", "USOIL": "CL=F", "UKOIL": "BZ=F",
         "XAGUSD": "SI=F", "XAUUSD": "GC=F", "BTCUSD": "BTC-USD", "ETHUSD": "ETH-USD"}


def load_h1(sym: str) -> pd.DataFrame:
    d = pd.read_csv(FP / f"{sym}_H1.csv")
    d["time"] = pd.to_datetime(d["time"], utc=True)
    return fr._dense_from(d[["time", "open", "high", "low", "close"]].sort_values("time").reset_index(drop=True))


def load_d1(sym: str) -> pd.DataFrame:
    d = pd.read_csv(FP / f"{sym}_D1.csv")
    date = (pd.to_datetime(d["time"], utc=True) + pd.Timedelta(hours=3)).dt.tz_localize(None).dt.normalize()
    date = date.where(date.dt.dayofweek != 5, date - pd.Timedelta(days=1))
    d = d.assign(date=date).groupby("date").agg(open=("open", "first"), high=("high", "max"),
                                                low=("low", "min"), close=("close", "last"))
    d = d[d.index.dayofweek < 5]
    return r3._clean(d)


def cost_of(sp: dict) -> dict:
    return dict(cost_frac=sp["spread_price"] / sp["price"] * (1 + 2 * r3.SLIP_FRAC),
                swap_long=sp["swap_long"] or 0.0, swap_short=sp["swap_short"] or 0.0)


def second_h1(sym: str) -> pd.DataFrame | None:
    v = VALETAX.get(sym)
    p = r3.H1CACHE / f"{v}.csv" if v else None
    if p is None or not p.exists():
        return None
    d = fr._read(p)
    d = d[d.time >= "2024-01-01"].reset_index(drop=True)
    return d if len(d) > 3000 else None


def main() -> int:
    specs = json.loads((FP / "SPECS.json").read_text())
    qual = json.loads((FP / "QUALITY.json").read_text())
    this_year = datetime.now(timezone.utc).year
    elig = [s for s, q in qual.items() if s in specs and q["first_dense_year"] is not None
            and q["first_dense_year"] <= this_year - MIN_YEARS and q["daily_bar_suspects"] < MAX_SUSPECTS]
    skipped = {s: q for s, q in qual.items() if s not in elig}
    n_trials = r3.ledger_n() + r3.PRIOR_TRIALS + len(elig) * (len(H1_EXITS) + 1) + 1
    log.info("%d eligible symbols, %d skipped; N=%d", len(elig), len(skipped), n_trials)

    res, trades, pool = {}, {}, dict(t=[], t2=[], b=[])
    for s in elig:
        c = cost_of(specs[s])
        h1 = load_h1(s)
        years = sum((p.time.iloc[-1] - p.time.iloc[0]).days for p in fr.segments(h1)) / 365.25
        sec = second_h1(s)
        for ex, tp in H1_EXITS.items():
            t = fr.simulate(h1, tp, c)
            t2 = fr.simulate(h1, tp, c, cost_mult=2.0)
            bR = None
            if sec is not None:
                tb = fr.simulate(sec, tp, c)
                bR = float(tb.R.mean()) if len(tb) else None
            r = r3.evaluate(t, t2, years, n_trials, bR)
            r.update(sleeve="H1", exit=ex, group=specs[s]["path"].split("\\")[0], **fr.risk_stats(t),
                     last2y_R=round(float(t.R[t.entry >= t.entry.max() - pd.Timedelta(days=730)].mean()), 4) if len(t) else None)
            res[f"{s}|H1|{ex}"] = r
            trades[f"{s}|H1|{ex}"] = t[["entry", "exit", "R"]]
        if not (FP / f"{s}_D1.csv").exists():
            log.warning("%s: no D1 file -- D1 sleeve skipped", s)
            continue
        d1 = load_d1(s)
        sp = specs[s]
        cf = c["cost_frac"] / (1 + 2 * r3.SLIP_FRAC)
        t = r3.trade_table(d1, r3.s_turtle55, cf, c["swap_long"], c["swap_short"])
        t2 = r3.trade_table(d1, r3.s_turtle55, cf, c["swap_long"], c["swap_short"], cost_mult=2.0)
        y = r3.yahoo(YAHOO[s], offline=False) if s in YAHOO else None
        bR = None
        if y is not None and len(y) > 1000:
            tb = r3.trade_table(y, r3.s_turtle55, cf, c["swap_long"], c["swap_short"])
            bR = float(tb.R.mean()) if len(tb) else None
            pool["b"].append(tb.R)
        r = r3.evaluate(t, t2, (d1.index[-1] - d1.index[0]).days / 365.25, n_trials, bR)
        r.update(sleeve="D1", exit="turtle55", group=sp["path"].split("\\")[0])
        res[f"{s}|D1|turtle55"] = r
        trades[f"{s}|D1|turtle55"] = t[["entry", "exit", "R"]]
        pool["t"].append(t); pool["t2"].append(t2)
        log.info("%-8s H1 tp3 %s tp6 %s | D1 turtle %s", s, res[f"{s}|H1|tp3"].get("mean_R"),
                 res[f"{s}|H1|tp6"].get("mean_R"), r.get("mean_R"))

    pt, pt2 = pd.concat(pool["t"], ignore_index=True), pd.concat(pool["t2"], ignore_index=True)
    pooled = r3.evaluate(pt, pt2, max((x.entry.max() - x.entry.min()).days for x in pool["t"] if len(x)) / 365.25,
                         n_trials, float(pd.concat(pool["b"]).mean()) if pool["b"] else None)

    plan = [k for k, r in res.items() if r.get("passes")]
    prop = {}
    if plan:
        pt_ = []
        for k in plan:
            x = trades[k].copy()
            x["exit"] = pd.to_datetime(x["exit"], utc=True) if pd.to_datetime(x["exit"]).dt.tz is None else x["exit"]
            pt_.append(x)
        port = pd.concat(pt_, ignore_index=True)
        port = port[port.exit >= pd.Timestamp("2016-01-01", tz="UTC")]
        for rk in RISKS:
            prop[f"{rk:.2%}"] = fr.prop_sim(port, rk)
    tested = datetime.now(timezone.utc).isoformat(timespec="seconds")
    r3.append_ledger([dict(tested_utc=tested, name=f"fp_{k.replace('|', '_')}", params=json.dumps({"data": "FundingPips"}),
                           lookahead_ok=True, n_trials=n_trials, deflated_sharpe=r.get("dsr", ""),
                           passes=bool(r.get("passes")), useful=bool(r.get("passes"))) for k, r in res.items()]
                     + [dict(tested_utc=tested, name="fp_D1_turtle55_pooled", lookahead_ok=True, n_trials=n_trials,
                             deflated_sharpe=pooled.get("dsr", ""), passes=bool(pooled.get("passes")),
                             useful=bool(pooled.get("passes")))])
    rep = dict(meta=dict(n_trials=n_trials, eligible=elig, generated=tested),
               skipped={s: {k: q[k] for k in ("first_dense_year", "daily_bar_suspects", "start")} for s, q in skipped.items()},
               results=res, turtle_pooled=pooled, plan=plan, prop=prop)
    (ROOT / "reports" / "fp_strategy_build.json").write_text(json.dumps(rep, indent=2, default=str))
    md = _md(rep)
    (ROOT / "FP_STRATEGY_BUILD.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def _md(rep) -> str:
    L = ["# Trading plan from verified FundingPips data", "",
         f"{len(rep['meta']['eligible'])} eligible symbols (>= 10 dense H1 years, clean); ledger N = {rep['meta']['n_trials']}.  "
         "Rules pre-registered in `backtest/fp_strategy_build.py`.", "",
         f"## Plan (passing): {', '.join(rep['plan']) or 'nothing passed'}", ""]
    if rep["prop"]:
        L += ["| risk per trade | pass | daily fail | max-loss fail | open | median days |", "|---|--:|--:|--:|--:|--:|"]
        for k, p in rep["prop"].items():
            o = p["outcomes"]
            L.append(f"| {k} | {100 * o.get('pass', 0):.0f} % | {100 * o.get('daily', 0):.0f} % | "
                     f"{100 * o.get('max_loss', 0):.0f} % | {100 * o.get('open', 0):.0f} % | {p['median_days_to_pass']} |")
    tp = rep["turtle_pooled"]
    L += ["", f"Turtle D1 pooled over all eligible symbols: {tp.get('trades')} trades, mean R {tp.get('mean_R')}, "
              f"last 30 % {tp.get('last30_R')}, WF {tp.get('wf_positive')}/8, DSR {tp.get('dsr')}, "
              f"PASS {'yes' if tp.get('passes') else 'no'}", "",
          "## All results (sorted by mean R)", "",
          "| symbol | sleeve | exit | group | trades | mean R | first 70 % | last 30 % | WF+ | 2x cost | 2nd src | DSR | failed checks |",
          "|---|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|---|"]
    for k, r in sorted(rep["results"].items(), key=lambda kv: -(kv[1].get("mean_R") or -9)):
        s, sl, ex = k.split("|")
        bad = ", ".join(c for c, v in r.get("checks", {}).items() if not v) or "**PASS**"
        L.append(f"| {s} | {sl} | {ex} | {r.get('group')} | {r.get('trades')} | {r.get('mean_R')} | {r.get('first70_R')} | "
                 f"{r.get('last30_R')} | {r.get('wf_positive')}/8 | {r.get('cost2x_R')} | {r.get('broker_R')} | {r.get('dsr')} | {bad} |")
    if rep["skipped"]:
        L += ["", "Skipped (history too short or corrupt): " +
              ", ".join(f"{s} (dense from {q['first_dense_year']}, suspects {q['daily_bar_suspects']})" for s, q in rep["skipped"].items())]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
