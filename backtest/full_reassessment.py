"""
Full reassessment of the live H1 momentum on the symbols traded at FundingPips -- from scratch,
after the 2026-10-07 discovery that backtest/btc_lab.py re-entered on the bar a trade closed inside
(inflating every btc_lab validation) and that the broker's H1 history is dense only from 2024.

    python -m backtest.full_reassessment

ONE simulator (`simulate`), written to copy execution/live_multi.py exactly:
  - signal: strategy/btc_h1_signal PARAMS on the completed bar, entry at the NEXT bar's open
  - stop 2 ATR(14) / target k ATR from the fill, broker SL/TP live from the entry bar itself
    (its high/low count), stop before target inside a bar, gaps fill at the open
  - time exit at the open 96 bars after entry; one position per symbol; a new entry only on a bar
    that opened flat (an exit AT the open frees the slot, an exit inside the bar does not)
  - costs: FundingPips spread (or the Valetax median if larger) + slippage 0.25 x spread per side;
    FundingPips swap per night held (rollover 21:00 UTC), long or short rate
Checked against the live trades (tools/twin_check pairs) and against the fixed btc_lab.

Data -- the longest DENSE hourly history per symbol (second source = independent check):
  XAUUSD  Dukascopy 2005- (if downloaded to 2026) else broker 2010-;  second: the other
  BTCUSD  Bitstamp 2014-;                                               second: broker 2024-
  GER40   Dukascopy 2014-;                                              second: broker 2024-
  EURUSD  Dukascopy 2005- (reference: trend is expected to fail on FX); second: broker 2024-

PRE-REGISTERED PASS RULE per (symbol, exit) -- round-3 rule (r3.evaluate): >= 10 years, >= 100
trades, mean R > 0 on full / first 70 % / last 30 %, >= 6 of 8 time folds positive, positive at 2x
costs, deflated Sharpe >= 0.95 (N = research ledger), second source mean R > 0.
Exits: target 3 ATR (live) and 6 ATR (proposed 2026-10-07).

Prop simulation (FundingPips 2-step): portfolio of XAUUSD + BTCUSD + GER40 trades, fixed risk per
trade, phase 1 +8 % then phase 2 +5 %, fail at -3 % on a day (closed trades; open drawdown not
modelled -- optimistic) or -10 % total.  Started on the first trading day of every month.

Writes reports/full_reassessment.json and FULL_REASSESSMENT.md.  Read-only; no orders.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import strategy_round3_daily as r3
from backtest.btc_strategies import base_atr, build_momentum
from common.logging_setup import get_logger
from strategy.btc_h1_signal import PARAMS

log = get_logger("reassess", filename="full_reassessment.log")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FP_SPECS = ROOT / "reports" / "fundingpips_specs.json"
SL_ATR, HOLD, SLIP = 2.0, 96, 0.25
EXITS = {"tp3": 3.0, "tp6": 6.0}
VALETAX = {"XAUUSD": "XAUUSD.vx", "BTCUSD": "BTCUSD.vx", "GER40": "DAX40.vx", "EURUSD": "EURUSD.vx"}
PROP_SYMBOLS = ["XAUUSD", "BTCUSD", "GER40"]


# -- data -----------------------------------------------------------------------------------
def _read(p: Path) -> pd.DataFrame:
    d = pd.read_csv(p)
    d["time"] = pd.to_datetime(d["time"], utc=True)
    d = d.dropna(subset=["open", "high", "low", "close"]).sort_values("time").drop_duplicates("time")
    return d[["time", "open", "high", "low", "close"]].reset_index(drop=True)


def _dense_from(d: pd.DataFrame, frac: float = 0.7) -> pd.DataFrame:
    """Drop the sparse early years (fewer than frac x the bars of the last full year)."""
    per = d["time"].dt.year.value_counts().sort_index()
    ref = per.iloc[-2] if len(per) > 1 else per.iloc[-1]
    ok = [y for y, n in per.items() if n >= frac * ref]
    return d[d["time"].dt.year >= ok[0]].reset_index(drop=True) if ok else d


def datasets() -> dict:
    broker = {s: _dense_from(_read(r3.H1CACHE / f"{v}.csv")) for s, v in VALETAX.items()}
    duka = {s: _read(DATA / "dukascopy_H1" / f"{v}.csv") for s, v in VALETAX.items()
            if (DATA / "dukascopy_H1" / f"{v}.csv").exists()}
    out = {}
    # Gold: the FundingPips H1 feed (data/gold, tools/collect_gold_data).  NOT the Valetax H1 file -- it has a
    # full DAILY bar mixed in at 21:00 UTC on most days 2018-2023 (found 2026-10-07), which leaks the coming
    # day's range into the signal and faked most of gold's apparent 2018-2023 edge.
    fp = _read(DATA / "gold" / "fp_XAUUSD_H1.csv")
    fp = fp[fp.time >= "2010-01-01"].reset_index(drop=True)
    xd = duka.get("XAUUSD")
    full_duka = xd is not None and xd["time"].max() >= fp["time"].max() - pd.Timedelta(days=30)
    out["XAUUSD"] = (fp, "FundingPips", xd if full_duka else None, "Dukascopy" if full_duka else None)
    out["BTCUSD"] = (_dense_from(_read(DATA / "btcusd_bitstamp_H1.csv")), "Bitstamp", broker["BTCUSD"], "broker")
    out["GER40"] = (_dense_from(duka["GER40"]), "Dukascopy", broker["GER40"], "broker")
    out["EURUSD"] = (duka["EURUSD"], "Dukascopy", broker["EURUSD"], "broker")
    return out


def costs() -> dict:
    """Per symbol: round-trip cost as a fraction of price, annual swap long/short (fraction)."""
    fp = json.loads(FP_SPECS.read_text())
    vx = json.loads(r3.SPECS.read_text())
    out = {}
    for s, sp in fp.items():
        frac = sp["spread_now"] / sp["price"]
        v = vx.get(VALETAX[s])
        if v:
            frac = max(frac, r3.broker_spread_frac(v))
        out[s] = dict(cost_frac=frac * (1 + 2 * SLIP), swap_long=sp["swap_long"], swap_short=sp["swap_short"])
    # FundingPips reports XAUUSD tick value 100x too small (the trader sizes from the profit calculator
    # instead), which shrinks the annualised points swap 100x: recompute from points directly.
    raw = {"XAUUSD": (-67.986, 25.026, 0.01)}
    for s, (sl, ss, pt) in raw.items():
        px = fp[s]["price"]
        out[s]["swap_long"], out[s]["swap_short"] = sl * pt * 365 / px, ss * pt * 365 / px
    return out


# -- the simulator -----------------------------------------------------------------------------
def nights(t0: pd.Timestamp, t1: pd.Timestamp) -> int:
    """Rollovers (21:00 UTC) crossed between entry and exit."""
    a, b = (t0 + pd.Timedelta(hours=3)).normalize(), (t1 + pd.Timedelta(hours=3)).normalize()
    return int((b - a).days)


GAP = pd.Timedelta(days=5)
MIN_SEGMENT = 1000                  # bars; shorter pieces between gaps are skipped (indicator warm-up)


def segments(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Contiguous pieces split at data holes longer than GAP (downloads with skipped months).
    Indicators restart in each piece and no trade is held across a hole."""
    cut = np.flatnonzero(df["time"].diff().to_numpy() > GAP.to_timedelta64())
    edges = [0, *cut, len(df)]
    return [df.iloc[a:b].reset_index(drop=True) for a, b in zip(edges[:-1], edges[1:]) if b - a >= MIN_SEGMENT]


def simulate(df: pd.DataFrame, tp_atr: float, cost: dict, cost_mult: float = 1.0) -> pd.DataFrame:
    parts = [simulate_piece(p, tp_atr, cost, cost_mult) for p in segments(df)]
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else simulate_piece(df.iloc[:0], tp_atr, cost, cost_mult)


def simulate_piece(df: pd.DataFrame, tp_atr: float, cost: dict, cost_mult: float = 1.0) -> pd.DataFrame:
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    t = df["time"]
    ent, _ = build_momentum(c, h, l, **PARAMS)
    A = base_atr(h, l, c, 14)
    rows, i, n = [], 1, len(c)
    while i < n - 1:
        d, a = int(ent[i]), A[i - 1]
        if d == 0 or not np.isfinite(a) or a <= 0:
            i += 1
            continue
        ep = o[i]
        sl_d, tp_d = SL_ATR * a, tp_atr * a
        stop, tgt = ep - d * sl_d, ep + d * tp_d
        x, xp, why, at_open = None, None, "time", False
        for k in range(i, min(n, i + HOLD)):
            if k > i and d * (o[k] - stop) <= 0:
                x, xp, why, at_open = k, o[k], "stop", True; break
            if (l[k] <= stop) if d > 0 else (h[k] >= stop):
                x, xp, why = k, stop, "stop"; break
            if k > i and d * (o[k] - tgt) >= 0:
                x, xp, why, at_open = k, o[k], "target", True; break
            if (h[k] >= tgt) if d > 0 else (l[k] <= tgt):
                x, xp, why = k, tgt, "target"; break
        if x is None:
            x = min(n - 1, i + HOLD)
            xp, at_open = o[x], True
        nt = nights(t.iloc[i], t.iloc[x])
        rate = cost["swap_long"] if d > 0 else cost["swap_short"]
        R = (d * (xp - ep) - cost_mult * cost["cost_frac"] * ep + ep * rate / 365 * nt) / sl_d
        rows.append((t.iloc[i], t.iloc[x], d, why, nt, R))
        i = x if at_open else x + 1
    return pd.DataFrame(rows, columns=["entry", "exit", "dir", "reason", "nights", "R"])


# -- reporting helpers ------------------------------------------------------------------------
def risk_stats(t: pd.DataFrame) -> dict:
    s = w = 0
    for r in t.R:
        s = s + 1 if r < 0 else 0
        w = max(w, s)
    eq = t.R.cumsum()
    return dict(max_dd_R=round(float((eq - eq.cummax()).min()), 1), worst_streak=w)


def per_year(t: pd.DataFrame) -> dict:
    g = t.groupby(t.entry.dt.year).R
    return {int(y): dict(n=int(n), R=round(float(m), 3)) for y, n, m in zip(g.count().index, g.count(), g.mean())}


def prop_sim(trades: pd.DataFrame, risk: float, daily_limit: float = 0.03, max_loss: float = 0.10) -> dict:
    """trades: exit time + R for the whole portfolio.  Attempts start on each month's first day."""
    tr = trades.sort_values("exit").reset_index(drop=True)
    day = tr.exit.dt.tz_convert(None).dt.normalize()
    starts = pd.date_range(day.min(), day.max() - pd.Timedelta(days=60), freq="MS")
    res = []
    for s0 in starts:
        sub = tr[day >= s0]
        sd = day[day >= s0]
        phase, base, bal, outcome, days = 1, 1.0, 1.0, "open", None
        for dte, grp in sub.groupby(sd):
            pnl = float(grp.R.sum()) * risk * base         # fixed risk of the phase-start balance
            if pnl <= -daily_limit * bal:
                outcome = "daily"; break
            bal += pnl
            if bal <= base * (1 - max_loss):
                outcome = "max_loss"; break
            if bal >= base * (1 + (0.08 if phase == 1 else 0.05)):
                if phase == 1:
                    phase, base = 2, bal
                else:
                    outcome, days = "pass", (dte - s0).days; break
        res.append((outcome, days))
    out = pd.Series([r[0] for r in res]).value_counts(normalize=True).round(3).to_dict()
    d = [r[1] for r in res if r[1] is not None]
    return dict(attempts=len(res), outcomes=out, median_days_to_pass=int(np.median(d)) if d else None)


def main() -> int:
    data = datasets()
    cst = costs()
    n_trials = r3.ledger_n() + r3.PRIOR_TRIALS + len(data) * len(EXITS)
    res, port = {}, {ex: [] for ex in EXITS}
    for s, (d, src, d2, src2) in data.items():
        for ex, tp in EXITS.items():
            t = simulate(d, tp, cst[s])
            t2 = simulate(d, tp, cst[s], cost_mult=2.0)
            bR = None
            if d2 is not None and len(d2) > 2000:
                tb = simulate(d2, tp, cst[s])
                bR = float(tb.R.mean()) if len(tb) else None
            years = sum((p.time.iloc[-1] - p.time.iloc[0]).days for p in segments(d)) / 365.25   # covered, not span
            r = r3.evaluate(t, t2, years, n_trials, bR)
            holes = [f"{a.date()}..{b.date()}" for a, b in zip(d.time.shift(1)[d.time.diff() > GAP], d.time[d.time.diff() > GAP])
                     if (b - a) > pd.Timedelta(days=30)]
            r.update(source=src, second_source=src2, span=f"{d.time.iloc[0].date()}..{d.time.iloc[-1].date()}",
                     years_covered=round(years, 1), holes=holes,
                     per_year=per_year(t), **risk_stats(t),
                     last2y_R=round(float(t.R[t.entry >= t.entry.max() - pd.Timedelta(days=730)].mean()), 4),
                     long_R=round(float(t.R[t.dir > 0].mean()), 4), short_R=round(float(t.R[t.dir < 0].mean()), 4),
                     avg_nights=round(float(t.nights.mean()), 2),
                     exits=t.reason.value_counts().to_dict())
            res[f"{s}|{ex}"] = r
            if s in PROP_SYMBOLS:
                port[ex].append(t.assign(sym=s))
            log.info("%s|%s %s trades=%s R=%s first70=%s last30=%s WF=%s/8 2x=%s second=%s DSR=%s PASS=%s",
                     s, ex, src, r.get("trades"), r.get("mean_R"), r.get("first70_R"), r.get("last30_R"),
                     r.get("wf_positive"), r.get("cost2x_R"), r.get("broker_R"), r.get("dsr"), r.get("passes"))
    prop = {}
    for ex, ts in port.items():
        p = pd.concat(ts, ignore_index=True)
        common = max(x.entry.min() for x in ts)
        p = p[p.entry >= common]
        for rk in (0.0025, 0.005):
            prop[f"{ex}|{rk:.2%}"] = prop_sim(p, rk)
    tested = datetime.now(timezone.utc).isoformat(timespec="seconds")
    r3.append_ledger([dict(tested_utc=tested, name=f"reassess_{k.replace('|', '_')}",
                           params=json.dumps({"source": r["source"], "span": r["span"]}), lookahead_ok=True,
                           xau_exp_R=r.get("mean_R") if k.startswith("XAU") else "", n_trials=n_trials,
                           deflated_sharpe=r.get("dsr", ""), passes=bool(r.get("passes")), useful=bool(r.get("passes")))
                      for k, r in res.items()])
    rep = dict(meta=dict(n_trials=n_trials, costs=cst, generated=tested), results=res, prop=prop)
    (ROOT / "reports" / "full_reassessment.json").write_text(json.dumps(rep, indent=2, default=str))
    md = _md(rep)
    (ROOT / "FULL_REASSESSMENT.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def _md(rep) -> str:
    L = ["# Full reassessment -- live H1 momentum, longest dense history, FundingPips costs", "",
         "Simulator copies execution/live_multi.py (see module docstring).  Pass rule pre-registered; "
         f"ledger N = {rep['meta']['n_trials']}.", "",
         "| symbol | exit | data | span (years with data) | trades | mean R | win % | first 70 % | last 30 % | last 2 y | WF+ | 2x cost | "
         "2nd source | long / short R | max DD R | streak | DSR | PASS |",
         "|---|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|"]
    for k, r in rep["results"].items():
        s, ex = k.split("|")
        L.append(f"| {s} | {ex} | {r['source']} | {r['span']} ({r['years_covered']}) | {r.get('trades')} | {r.get('mean_R')} | "
                 f"{round(100 * r['win_rate']) if r.get('win_rate') is not None else ''} | {r.get('first70_R')} | "
                 f"{r.get('last30_R')} | {r['last2y_R']} | {r.get('wf_positive')}/8 | {r.get('cost2x_R')} | "
                 f"{r.get('broker_R')} | {r['long_R']} / {r['short_R']} | {r['max_dd_R']} | {r['worst_streak']} | "
                 f"{r.get('dsr')} | {'**yes**' if r.get('passes') else 'no'} |")
    holes = {k.split("|")[0]: r["holes"] for k, r in rep["results"].items() if r["holes"]}
    if holes:
        L += ["", "Data holes > 30 days (skipped, no trade held across): " +
              "; ".join(f"{s}: {', '.join(h)}" for s, h in holes.items())]
    L += ["", "## Mean R per year", ""]
    keys = list(rep["results"])
    years = sorted({y for r in rep["results"].values() for y in r["per_year"]})
    L += ["| year | " + " | ".join(keys) + " |", "|---|" + "--:|" * len(keys)]
    for y in years:
        L.append(f"| {y} | " + " | ".join(str(rep["results"][k]["per_year"].get(y, {}).get("R", "")) for k in keys) + " |")
    L += ["", "## FundingPips 2-step simulation (XAUUSD + BTCUSD + GER40, one attempt started every month)", "",
          "| exit / risk per trade | attempts | pass | daily-loss fail | max-loss fail | still open | median days to pass |",
          "|---|--:|--:|--:|--:|--:|--:|"]
    for k, p in rep["prop"].items():
        o = p["outcomes"]
        L.append(f"| {k} | {p['attempts']} | {100 * o.get('pass', 0):.0f} % | {100 * o.get('daily', 0):.0f} % | "
                 f"{100 * o.get('max_loss', 0):.0f} % | {100 * o.get('open', 0):.0f} % | {p['median_days_to_pass']} |")
    c = rep["meta"]["costs"]
    L += ["", "Costs used (round trip as % of price; swap as annual % of price): " +
          "; ".join(f"{s} {100 * v['cost_frac']:.3f} % / long {100 * v['swap_long']:+.1f} % short {100 * v['swap_short']:+.1f} %"
                    for s, v in c.items())]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
