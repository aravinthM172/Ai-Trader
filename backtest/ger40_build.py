"""
GER40 strategy build on 30 years of daily candles (October 1996 - October 2026).  (2026-10-10)

    python -m backtest.ger40_build             -> GER40_BUILD.md, reports/ger40_build.json
    python -m backtest.ger40_build --ledger    same, and record every test in research/ledger.csv
    python -m backtest.ger40_build NDX100      the same build for another index (-> NDX100_BUILD.md)

Why another look: round 7 traded every rule in both directions and nothing worked.  A stock index drifts up,
so this build is BUY-ONLY, the way the live bot already trades NDX100.

Data   D1: Yahoo ^GDAXI cash index, real candles for the whole 30 years.  H1: Dukascopy 2014-2026 with a hole
       in 2016-2020 (7 dense years) -- reported, too short to pass.  FundingPips D1 (2021-) as second source.
Costs  FundingPips spread + slippage, and the FundingPips swap: holding GER40 long costs 5.4 % a year.

PRE-REGISTERED (written before the first run; textbook parameters, nothing tuned):
  drift_trend / drift_dip   BASELINES: buy every 5th day regardless of the chart, with the trend exit
                 (2 ATR stop, 6 ATR target, 20 days) and the dip exit (2 ATR stop, 2 ATR target, 5 days).
                 A rule only has timing value if it beats its baseline.
  43 rules       every round-7 indicator rule, buy signals only, same exits as round 7.
  43 rules + up  the same, only while the close is above the 200-day average.
  connors_rsi2   RSI(2) < 5 above the 200-day average: buy next open, sell at the open after a close above
                 the 5-day average, protective stop 3 ATR10.
  ibs            close in the lowest 20 % of the day's range, above the 200-day average; out after a close
                 in the upper half or 5 days; stop 3 ATR10.
  turn_of_month  buy the open of the last trading day, sell the open of the 4th trading day of the next month.
  halloween      buy the first trading day of November, sell the first trading day of May; stop 3 ATR20 x 3.
  overnight      buy the close, sell the next open (1 R = 1 ATR14).
  H1 live rule   live momentum (RSI14 >= 60, 8-bar momentum, EMA96; 2 ATR stop, 6 ATR target), buys only,
                 with and without the daily EMA200 filter.
PASS RULE (r3.evaluate, unchanged) with N = ledger + every test here; "works" = all checks except dsr_95,
and t >= 2.75.  Read-only; no orders.
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import fp_strategy_build as fb
from backtest import full_reassessment as fr
from backtest import long_history_backtest as lh
from backtest import strategy_round3_daily as r3
from backtest import strategy_round5 as r5
from backtest import strategy_round6 as r6
from backtest import strategy_round7 as r7
from backtest.btc_strategies import base_atr

ROOT = Path(__file__).resolve().parents[1]
SYM = next((a for a in sys.argv[1:] if not a.startswith("-")), "GER40")      # python -m backtest.ger40_build NDX100


def long_only(s: dict, allow: np.ndarray | None = None) -> dict:
    d = np.where(s["d"] > 0, 1, 0).astype(np.int8)
    if allow is not None:
        d = np.where(allow, d, 0).astype(np.int8)
    return dict(s, d=d)


def sweep_long(p: pd.DataFrame, cost: dict, up_only: bool) -> dict:
    c = p["close"].to_numpy(float)
    allow = (c > r7.sma(c, 200)) if up_only else None
    return {k: r5.simulate_piece(p, long_only(s, allow), cost) for k, s in r7.signals(p, "D1").items()}


def drift(p: pd.DataFrame, cost: dict, kind: str) -> pd.DataFrame:
    h, l, c = (p[k].to_numpy(float) for k in ("high", "low", "close"))
    A, n = base_atr(h, l, c, 14), len(c)
    sl, tp, hold = r7.EXITS["D1"][kind]
    buy = (np.arange(n) % 5 == 0) & (np.arange(n) > 250)
    return r5.simulate_piece(p, dict(r5._put(r5._sig(n), buy, np.zeros(n, bool), c - sl * A, c + sl * A, c + tp * A, c - tp * A), hold=hold), cost)


def daily_fn(p: pd.DataFrame, fn, cost: dict, mult: float = 1.0) -> pd.DataFrame:
    d = p.set_index(p["time"].dt.tz_localize(None).dt.normalize())[["open", "high", "low", "close"]]
    t = r3.trade_table(d, fn, cost["cost_frac"] / (1 + 2 * r3.SLIP_FRAC), cost["swap_long"], cost["swap_short"], cost_mult=mult)
    return t


def halloween(p: pd.DataFrame, cost: dict) -> pd.DataFrame:
    t = p["time"]
    first = np.concatenate([[False], t.dt.month.to_numpy()[1:] != t.dt.month.to_numpy()[:-1]])
    mon = t.dt.month.to_numpy()
    h, l, c = (p[k].to_numpy(float) for k in ("high", "low", "close"))
    n = len(c)
    A = base_atr(h, l, c, 20)
    sig_d = np.zeros(n, np.int8)
    sig_d[:-1] = np.where(first[1:] & (mon[1:] == 11), 1, 0)
    s = r5._sig(n)
    s["d"], s["stop"], s["tgt"] = sig_d, np.where(sig_d > 0, c - 9 * A, np.nan), np.where(sig_d > 0, np.inf, np.nan)
    # the generic simulator skips stops wider than 6 ATR14, so run with its own limits
    old = r5.MAX_RISK_ATR
    r5.MAX_RISK_ATR = 50.0
    try:
        return r5.simulate_piece(p, dict(s, hold=200, flat=first & (mon == 5)), cost)
    finally:
        r5.MAX_RISK_ATR = old


def overnight(p: pd.DataFrame, cost: dict) -> pd.DataFrame:
    o, h, l, c = (p[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    A = base_atr(h, l, c, 14)
    i = np.arange(250, len(c) - 1)
    risk = A[i]
    cost_R = cost["cost_frac"] * c[i] / risk
    R = (o[i + 1] - c[i] + c[i] * cost["swap_long"] / 365 * (p["time"].iloc[i + 1].to_numpy() - p["time"].iloc[i].to_numpy()).astype("timedelta64[D]").astype(int)) / risk - cost_R
    return pd.DataFrame(dict(entry=p["time"].iloc[i].to_numpy(), exit=p["time"].iloc[i + 1].to_numpy(), dir=1, reason="flat", R=R, cost_R=cost_R)) \
        .assign(entry=lambda x: pd.to_datetime(x.entry, utc=True), exit=lambda x: pd.to_datetime(x.exit, utc=True))


def h1_live(df: pd.DataFrame, cost: dict, filt: bool) -> pd.DataFrame:
    t = pd.concat([r6.momentum_piece(p, cost, 6.0, 0.0) for p in fr.segments(df)], ignore_index=True)
    t = t[t.dir > 0]
    if filt:
        reg = lh.daily_regime(df, True)
        day = (t.entry + pd.Timedelta(hours=3)).dt.tz_localize(None).dt.normalize()
        t = t[reg.side.reindex(day, method="ffill").to_numpy() == 1]
    return t.reset_index(drop=True)


def judge(t: pd.DataFrame, years: float, n: int, t2: pd.DataFrame | None = None, broker_R: float | None = None) -> dict:
    if "cost_R" in t:
        t2 = t.assign(R=t.R - t.cost_R)
    r = r6.judge(t, t2, years, n, broker_R)
    if "checks" in r:
        r["R_per_year"] = round(float(t.R.sum()) / years, 1)
        eq = t.sort_values("exit").R.cumsum()
        r["max_dd_R"] = round(float((eq.cummax() - eq).max()), 1)
        dec = t.groupby((t.entry.dt.year // 5) * 5).R.mean().round(3)
        r["by_5y"] = {int(k): float(v) for k, v in dec.items()}
        ok = all(v for k, v in r["checks"].items() if k != "dsr_95") and (r["t"] or 0) >= r5.LEAD_T
        r["verdict"] = "PASS" if r["passes"] else "works" if ok else "weak" if r["mean_R"] > 0 and (r["t"] or 0) >= 1.5 else "no"
    else:
        r["verdict"] = "no"
    return r


def main() -> int:
    warnings.filterwarnings("ignore")
    cost = r6.cost(SYM)
    d = r7.d1(SYM)
    yrs = (d.time.iloc[-1] - d.time.iloc[0]).days / 365.25
    fp = fb.load_d1(SYM).reset_index().rename(columns={"date": "time"})
    fp["time"] = pd.to_datetime(fp["time"], utc=True) + pd.Timedelta(hours=21)
    n_rules = len(r7.RULE_KIND)
    n_trials = r3.ledger_n() + r3.PRIOR_TRIALS + 2 * n_rules + 9
    res = {}
    res["drift_trend"], res["drift_dip"] = judge(drift(d, cost, "T"), yrs, n_trials), judge(drift(d, cost, "M"), yrs, n_trials)
    for tag, up in (("", False), ("+up", True)):
        for k, t in sweep_long(d, cost, up).items():
            res[f"{k}{tag}"] = judge(t, yrs, n_trials)
            res[f"{k}{tag}"]["kind"] = r7.RULE_KIND[k]
    for name, fn in (("connors_rsi2", r3.s_connors_rsi2), ("ibs", r6.s_ibs)):
        t, t2 = daily_fn(d, fn, cost), daily_fn(d, fn, cost, 2.0)
        tb = daily_fn(fp, fn, cost)
        res[name] = judge(t.assign(entry=t.entry.dt.tz_localize("UTC"), exit=t.exit.dt.tz_localize("UTC")), yrs, n_trials,
                          t2.assign(entry=t2.entry.dt.tz_localize("UTC"), exit=t2.exit.dt.tz_localize("UTC")),
                          float(tb.R.mean()) if len(tb) >= 20 else None)
    res["turn_of_month"] = judge(r7.turn_of_month(d, cost), yrs, n_trials)
    res["halloween"] = judge(halloween(d, cost), yrs, n_trials)
    res["overnight"] = judge(overnight(d, cost), yrs, n_trials)
    h, src = r7.h1(SYM)
    hy = r6.years_of(h)
    res["H1_live_buy_only"] = judge(h1_live(h, cost, False), hy, n_trials)
    res["H1_live_buy_only+ema200"] = judge(h1_live(h, cost, True), hy, n_trials)

    if "--ledger" in sys.argv:
        tested = datetime.now(timezone.utc).isoformat(timespec="seconds")
        r3.append_ledger([dict(tested_utc=tested, name=SYM.lower() + "_" + k, params="{}", lookahead_ok=True, n_trials=n_trials,
                               deflated_sharpe=r.get("dsr", ""), passes=bool(r.get("passes")), useful=bool(r.get("passes")))
                          for k, r in res.items()])
    rep = dict(meta=dict(n_trials=n_trials, d1=f"{d.time.iloc[0].date()} to {d.time.iloc[-1].date()} ({yrs:.1f} years, {len(d)} days)",
                         h1=f"{src} {h.time.iloc[0].date()} to {h.time.iloc[-1].date()} ({hy:.1f} dense years)", cost=cost), results=res)
    (ROOT / "reports" / f"{SYM.lower()}_build.json").write_text(json.dumps(rep, indent=2, default=str))
    md = _md(rep)
    (ROOT / f"{SYM}_BUILD.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def _md(rep: dict) -> str:
    m, res = rep["meta"], rep["results"]
    mark = {"PASS": "**PASS**", "works": "**works**", "weak": "weak", "no": "-"}
    base = {"T": res["drift_trend"].get("mean_R"), "M": res["drift_dip"].get("mean_R")}
    L = [f"# {SYM} buy-only build on 30 years of daily candles", "",
         f"Daily: {m['d1']}.  Hourly: {m['h1']}.  Deflated-Sharpe N = {m['n_trials']}.  "
         f"Swap for holding long: {100 * m['cost']['swap_long']:.1f} % a year.  Rules pre-registered in `backtest/ger40_build.py`.", "",
         f"Baselines (buy every 5th day): trend exit {base['T']} R per trade, dip exit {base['M']} R per trade.", "",
         "| strategy | trades/yr | mean R | vs baseline | t | win % | R per year | max DD R | first 70 % | last 30 % | last 2 y | WF+ | 2x cost | 2nd src | DSR | verdict |",
         "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|"]
    rows = [(k, r) for k, r in res.items() if "checks" in r]
    for k, r in sorted(rows, key=lambda kr: -(kr[1]["t"] or -9))[:40]:
        b = base.get(r.get("kind"))
        vs = "" if b is None else round(r["mean_R"] - b, 4)
        L.append(f"| {k} | {r['trades_per_year']} | {r['mean_R']} | {vs} | {r['t']} | {round(100 * r['win_rate'])} | {r['R_per_year']} | {r['max_dd_R']} | "
                 f"{r['first70_R']} | {r['last30_R']} | {r['last2y_R']} | {r['wf_positive']}/8 | {r['cost2x_R']} | {r.get('broker_R')} | {r['dsr']} | {mark[r['verdict']]} |")
    L += ["", "(top 40 of " + str(len(res)) + " by t; the rest are weaker)", "", "## Mean R by 5-year block, best ten", ""]
    for k, r in sorted(rows, key=lambda kr: -(kr[1]["t"] or -9))[:10]:
        L.append(f"- {k}: {r['by_5y']}")
    good = {v: [k for k, r in res.items() if r["verdict"] == v] for v in ("PASS", "works", "weak")}
    L += ["", f"**PASS ({len(good['PASS'])}):** {', '.join(good['PASS']) or 'none'}", "",
          f"**Works ({len(good['works'])}):** {', '.join(good['works']) or 'none'}", "",
          f"**Weak ({len(good['weak'])}):** {', '.join(good['weak']) or 'none'}", ""]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
