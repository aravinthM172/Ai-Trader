"""
BTC H1 long-history OUT-OF-SAMPLE check of the FROZEN strategies on 2014 -> today.

Every candidate (incl. the paper-traded momentum_rsi_mtf) was chosen on the Valetax
dense window (2024-01 -> ).  Bitstamp H1 (backtest/btc_bitstamp_h1_dataset.py) gives
~10 earlier years the parameters never saw.  NOTHING is re-tuned here.

Same engine (btc_lab) + same builders (btc_strategies) + same params as
btc_h1_validation.py.  Folds = calendar years (each a different BTC regime).

Costs: Valetax quotes a FIXED $29.76 spread at today's ~$60-120k prices.  A fixed
dollar spread is meaningless at 2015's $300, so every $ cost (spread, slippage,
broker min stop) is scaled per year by  median_close(year) / REF_PRICE  -- i.e. the
same cost as a FRACTION of price.  (Real 2014-17 exchange spreads were wider; the
conservative/stressed scenarios cover that.)  R-multiples are scale-invariant, so
expectancy_R is comparable across years; dollar P/L is not.

    python -m backtest.btc_longhistory_validation

Writes reports/btc_longhistory_validation.json.  Research only -- NO LIVE ORDER.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.logging_setup import get_logger
from backtest import btc_lab as lab
from backtest import btc_strategies as strat
from backtest.btc_h1_validation import CANDIDATES, COST_SCENARIOS, EDGE_RISK

log = get_logger("btc.longhist")
DATA = ROOT / "data" / "btcusd_bitstamp_H1.csv"
VALETAX = ROOT / "data" / "btcusd_vx_H1_dense.csv"
REPORTS = ROOT / "reports"
LIVE = "momentum_rsi_mtf"                 # frozen, paper-traded (strategy/btc_h1_signal.py)
SEEN_FROM = 2024                          # parameters were selected on 2024-01 ->


def load(path):
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time").reset_index(drop=True)


def ref_price() -> float:
    """Price level at which the Valetax $ costs were observed."""
    return float(load(VALETAX)["close"].median())


def scaled(costs: lab.Costs, s: float) -> lab.Costs:
    return replace(costs, spread=costs.spread * s, slippage_per_side=costs.slippage_per_side * s)


def year_folds(df, warmup):
    yrs = df["time"].dt.year.to_numpy()
    out = []
    for y in np.unique(yrs):
        idx = np.where(yrs == y)[0]
        i0, i1 = max(int(idx[0]), warmup + 30), int(idx[-1]) + 1
        if i1 - i0 > 500:
            out.append((int(y), i0, i1))
    return out


def run_years(px, atr, entries, warmup, folds, costs, ref):
    rows, R_all, R_unseen = {}, [], []
    c = px["close"]
    for y, i0, i1 in folds:
        s = float(np.median(c[i0:i1])) / ref
        contract = replace(lab.BTC_CONTRACT, broker_stops_level=lab.BTC_CONTRACT.broker_stops_level * s)
        r = lab.run_backtest(px, atr, entries, i0, i1, warmup, scaled(costs, s), EDGE_RISK, contract)
        t, sm = r["trades"], r["summary"]
        R = np.where(t["risk"] > 0, t["pnl"] / t["risk"], 0.0)
        R_all.append(R)
        if y < SEEN_FROM:
            R_unseen.append(R)
        rows[y] = dict(trades=sm.get("trades", 0), expectancy_R=sm.get("expectancy_R"),
                       profit_factor=sm.get("profit_factor"), win_rate=sm.get("win_rate"),
                       max_drawdown_pct=sm.get("max_drawdown_pct"),
                       median_price=round(float(np.median(c[i0:i1])), 2),
                       round_trip_cost=round(scaled(costs, s).round_trip(), 4))
    return rows, np.concatenate(R_all), (np.concatenate(R_unseen) if R_unseen else np.array([]))


def pooled(R):
    if len(R) == 0:
        return dict(trades=0)
    se = R.std(ddof=1) / np.sqrt(len(R)) if len(R) > 1 else np.nan
    return dict(trades=int(len(R)), expectancy_R=round(float(R.mean()), 4),
                t_stat=round(float(R.mean() / se), 2) if se > 0 else None,
                win_rate=round(float((R > 0).mean()), 4),
                profit_factor=round(float(R[R > 0].sum() / -R[R < 0].sum()), 3) if (R < 0).any() else None)


def summarise(rows):
    unseen = {y: v for y, v in rows.items() if y < SEEN_FROM and v["trades"]}
    pos = sum(1 for v in unseen.values() if (v["expectancy_R"] or 0) > 0)
    return dict(unseen_years=len(unseen), unseen_years_positive=pos,
                majority_unseen_years_positive=bool(unseen and pos > len(unseen) / 2))


def main() -> int:
    t0 = time.perf_counter()
    df = load(DATA)
    N = len(df)
    c = df.close.to_numpy(float); h = df.high.to_numpy(float); l = df.low.to_numpy(float)
    PX = {k: df[k].to_numpy(float) for k in ("open", "high", "low", "close")}
    atr = strat.base_atr(h, l, c, 14)
    ref = ref_price()
    REAL = COST_SCENARIOS["realistic"]
    log.info("Bitstamp H1 rows=%d %s -> %s  ref price $%.0f", N, df.time.iloc[0], df.time.iloc[-1], ref)

    rep = {"meta": dict(source="Bitstamp BTC/USD H1 (backtest/btc_bitstamp_h1_dataset.py)",
                        rows=N, span=[str(df.time.iloc[0]), str(df.time.iloc[-1])],
                        ref_price=round(ref, 2), seen_from_year=SEEN_FROM,
                        cost_model="Valetax $ costs scaled by median_close(year)/ref_price",
                        risk=dict(initial_balance=EDGE_RISK.initial_balance, risk_per_trade=EDGE_RISK.risk_per_trade,
                                  sl_atr=EDGE_RISK.sl_atr_mult, tp_atr=EDGE_RISK.tp_atr_mult),
                        params="frozen -- identical to btc_h1_validation.CANDIDATES")}

    # sanity: on 2024+ Bitstamp should reproduce the Valetax result for the live strategy
    bld, kw = CANDIDATES[LIVE]
    ent, wu = bld(c, h, l, **kw)
    cut = int(N * 0.6)
    part, _ = bld(c[:cut], h[:cut], l[:cut], **kw)
    rep["look_ahead_probe"] = "no look-ahead detected" if np.array_equal(ent[:cut], part[:cut]) else "LEAKAGE"

    v = load(VALETAX)
    vc, vh, vl = (v[k].to_numpy(float) for k in ("close", "high", "low"))
    ve, vw = bld(vc, vh, vl, **kw)
    vpx = {k: v[k].to_numpy(float) for k in ("open", "high", "low", "close")}
    vs = lab.run_backtest(vpx, strat.base_atr(vh, vl, vc, 14), ve, vw + 30, len(vc), vw, REAL, EDGE_RISK)["summary"]
    b0 = int(df.time.searchsorted(v.time.iloc[vw + 30]))       # same first evaluated bar as Valetax
    bs = lab.run_backtest(PX, atr, ent, b0, N, wu, REAL, EDGE_RISK)["summary"]
    rep["feed_sanity_2024plus"] = dict(
        valetax=dict(trades=vs.get("trades"), expectancy_R=vs.get("expectancy_R"), profit_factor=vs.get("profit_factor")),
        bitstamp=dict(trades=bs.get("trades"), expectancy_R=bs.get("expectancy_R"), profit_factor=bs.get("profit_factor")),
        note="same frozen strategy, same fixed Valetax costs, same window; should be close")

    # all candidates, realistic costs, per calendar year
    res = {}
    for nm, (b, k) in CANDIDATES.items():
        e, w = b(c, h, l, **k)
        folds = year_folds(df, w)
        rows, R_all, R_unseen = run_years(PX, atr, e, w, folds, REAL, ref)
        res[nm] = dict(params=k, per_year=rows, pooled_all=pooled(R_all),
                       pooled_unseen_pre2024=pooled(R_unseen), **summarise(rows))
        s = res[nm]
        log.info("  %-22s unseen: n=%s expR=%s t=%s  years+ %s/%s",
                 nm, s["pooled_unseen_pre2024"].get("trades"), s["pooled_unseen_pre2024"].get("expectancy_R"),
                 s["pooled_unseen_pre2024"].get("t_stat"), s["unseen_years_positive"], s["unseen_years"])
        if nm == LIVE:
            live_e, live_w, live_folds = e, w, folds
    rep["candidates"] = res

    # live strategy: cost sensitivity + random-direction null on unseen years
    cost = {}
    for cn, cc in COST_SCENARIOS.items():
        rows, _, Ru = run_years(PX, atr, live_e, live_w, live_folds, cc, ref)
        cost[cn] = dict(pooled_unseen_pre2024=pooled(Ru), **summarise(rows))
    rep["live_cost_sensitivity"] = cost

    unseen_folds = [f for f in live_folds if f[0] < SEEN_FROM]
    obs = res[LIVE]["pooled_unseen_pre2024"].get("expectancy_R") or 0.0
    rng = np.random.default_rng(7)
    mask = live_e != 0
    null = []
    for _ in range(200):
        e2 = live_e.copy()
        e2[mask] = (rng.integers(0, 2, int(mask.sum())) * 2 - 1).astype(np.int8)
        _, _, Ru = run_years(PX, atr, e2, live_w, unseen_folds, REAL, ref)
        null.append(Ru.mean() if len(Ru) else 0.0)
    null = np.array(null)
    rep["live_random_direction_null_unseen"] = dict(
        trials=200, null_mean_expR=round(float(null.mean()), 4), null_std=round(float(null.std()), 4),
        observed_expR=round(obs, 4), observed_percentile=round(100 * float((null < obs).mean()), 1),
        direction_carries_information=bool((null < obs).mean() >= 0.95))
    rep["runtime_s"] = round(time.perf_counter() - t0, 1)

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "btc_longhistory_validation.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")

    L = res[LIVE]
    print("=" * 100)
    print("BTC H1 LONG-HISTORY OUT-OF-SAMPLE (Bitstamp 2014 ->, frozen params)  --  NO LIVE ORDER")
    print("=" * 100)
    fs = rep["feed_sanity_2024plus"]
    print(f"feed sanity 2024+ ({LIVE}): Valetax n={fs['valetax']['trades']} expR={fs['valetax']['expectancy_R']}   "
          f"Bitstamp n={fs['bitstamp']['trades']} expR={fs['bitstamp']['expectancy_R']}   look-ahead: {rep['look_ahead_probe']}")
    print(f"\n{'strategy':22s} {'unseen 2014-23: n':>18s} {'expR':>8s} {'t':>6s} {'PF':>6s} {'yrs+':>6s}   {'2024+ expR':>10s}")
    for nm, s in res.items():
        u = s["pooled_unseen_pre2024"]
        seen = [v["expectancy_R"] for y, v in s["per_year"].items() if y >= SEEN_FROM and v["trades"]]
        print(f"{nm:22s} {u.get('trades', 0):>18} {u.get('expectancy_R', 0):>8} {str(u.get('t_stat')):>6} "
              f"{str(u.get('profit_factor')):>6} {s['unseen_years_positive']:>3}/{s['unseen_years']:<2}   "
              f"{(np.mean(seen) if seen else float('nan')):>10.4f}")
    print(f"\n{LIVE} per year (realistic, price-scaled costs):")
    for y, v in L["per_year"].items():
        tag = "seen" if y >= SEEN_FROM else "UNSEEN"
        print(f"  {y} {tag:6s} n={v['trades']:>4}  expR={str(v['expectancy_R']):>8}  PF={str(v['profit_factor']):>7}  "
              f"win={v['win_rate']}  px~${v['median_price']:,.0f}")
    print("\ncost sensitivity (unseen 2014-23 pooled expR):  " +
          "  ".join(f"{k}={v['pooled_unseen_pre2024'].get('expectancy_R')}" for k, v in cost.items()))
    nl = rep["live_random_direction_null_unseen"]
    print(f"random-direction null: observed {nl['observed_expR']} vs null {nl['null_mean_expR']}±{nl['null_std']}  "
          f"-> {nl['observed_percentile']} pct  (direction informative={nl['direction_carries_information']})")
    print(f"report: reports/btc_longhistory_validation.json   ({rep['runtime_s']}s)")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
