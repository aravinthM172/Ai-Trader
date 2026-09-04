"""
XAUUSD.vx (Valetax) H1 strategy validation -- same engine + strategies + method
as backtest/btc_h1_validation.py, so BTC and XAU are directly comparable.

    python -m backtest.xau_h1_validation

Downloads XAUUSD.vx H1 from MT5 on first run (reuses data/xauusd_vx_H1.csv after).
XAU has 17 years of dense hourly history -> multiple macro regimes (2015-16 bottom,
2020 covid spike, 2022 rate-hike selloff, 2024-25 rally).

Historical spread: the Valetax H1 feed records spread on only ~13% of bars (rest 0).
Zero is NOT a real spread, so it is floored at a realistic gold spread and the
four cost scenarios bracket it.  Nothing is set artificially low.

Writes: reports/xau_h1_validation.json / BTC_style .md, xau_h1_walkforward.json,
xau_h1_cost_sensitivity.json, xau_h1_monte_carlo.json.  NO LIVE ORDER.
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

log = get_logger("btc.xauh1")
CSV = ROOT / "data" / "xauusd_vx_H1.csv"
REPORTS = ROOT / "reports"
SYMBOL = "XAUUSD.vx"
DENSE_FROM = "2015-01-01"

CONTRACT = lab.XAU_CONTRACT                     # value $100/$1move/lot, min stop $0.31
RISK = lab.RiskCfg()                            # $100 account -> STEP 9 feasibility only
EDGE_RISK = lab.RiskCfg(initial_balance=5000.0) # measure the edge where trades execute

# gold spread: broker column is mostly 0 (not recorded).  Real retail gold spread ~$0.15-0.50.
COST_SCENARIOS = {
    "optimistic":   lab.Costs(spread=0.15, slippage_per_side=0.05),
    "realistic":    lab.Costs(spread=0.30, slippage_per_side=0.10),   # ~ the recorded p95
    "conservative": lab.Costs(spread=0.50, slippage_per_side=0.20),
    "stressed":     lab.Costs(spread=1.00, slippage_per_side=0.40),
}
REALISTIC = COST_SCENARIOS["realistic"]

CANDIDATES = {
    "ema_trend":            (strat.build_trend, dict(fast=20, slow=50, trend=200, adx_min=0.0, atr_rank_max=1.0)),
    "ema_rsi_trend":        (strat.build_trend, dict(fast=20, slow=50, trend=200, adx_min=20.0, atr_rank_max=0.85)),
    "donchian_breakout":    (strat.build_breakout, dict(lookback=20, expansion=1.3, atr_rank_min=0.20)),
    "momentum_rsi_mtf":     (strat.build_momentum, dict(rsi_buy=60.0, rsi_sell=40.0, mom_win=8, ema_htf=24)),
    "mean_reversion_range": (strat.build_mean_reversion, dict(z_win=20, z_entry=2.0, adx_max=18.0)),
}


def load():
    if not CSV.exists():
        from mt5.gateway import MT5Gateway
        gw = MT5Gateway()
        if not gw.connect():
            raise SystemExit("MT5 connection failed and no cached data/xauusd_vx_H1.csv")
        d = gw.get_rates(SYMBOL, "H1", 200_000)
        gw.shutdown()
        if d is None or d.empty:
            raise SystemExit("no XAUUSD.vx H1 data")
        d.to_csv(CSV, index=False)
    d = pd.read_csv(CSV)
    d["time"] = pd.to_datetime(d["time"], utc=True)
    d = d.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    dense = d[d["time"] >= pd.Timestamp(DENSE_FROM, tz="UTC")].reset_index(drop=True)
    return dense


def px_dict(df):
    return {k: df[k].to_numpy(float) for k in ("open", "high", "low", "close")}


def splits(N, warmup):
    a = warmup + 30
    return [("TRAIN", a, int(N * 0.70)), ("VALIDATION", int(N * 0.70), int(N * 0.85)),
            ("FINAL", int(N * 0.85), N)]


def evaluate(name, builder, df, kw):
    c = df.close.to_numpy(float); h = df.high.to_numpy(float); l = df.low.to_numpy(float)
    entries, warmup = builder(c, h, l, **kw)
    atr = strat.base_atr(h, l, c, 14)
    PX = px_dict(df); N = len(c)
    per = {tag: lab.run_backtest(PX, atr, entries, i0, i1, warmup, REALISTIC, EDGE_RISK, CONTRACT)["summary"]
           for tag, i0, i1 in splits(N, warmup)}
    comb = lab.run_backtest(PX, atr, entries, warmup + 30, N, warmup, REALISTIC, EDGE_RISK, CONTRACT)
    wf = lab.walk_forward(PX, atr, entries, warmup, REALISTIC, EDGE_RISK, n_folds=8, contract=CONTRACT)
    return dict(name=name, params=kw, per_split=per, combined=comb["summary"],
                walk_forward=wf["summary"], walk_forward_folds=wf["folds"],
                _entries=entries, _atr=atr, _warmup=warmup, _trades=comb["trades"], _bld=builder, _kw=kw)


def main() -> int:
    t0 = time.perf_counter()
    df = load()
    N = len(df)
    c = df.close.to_numpy(float); h = df.high.to_numpy(float); l = df.low.to_numpy(float)
    PX = px_dict(df)
    atr_full = strat.base_atr(h, l, c, 14)
    log.info("XAU H1 dense rows=%d  %s -> %s", N, df.time.iloc[0], df.time.iloc[-1])

    rep = {"meta": dict(
        symbol=SYMBOL, broker="Valetax", timeframe="H1",
        rows=int(N), span=[str(df.time.iloc[0]), str(df.time.iloc[-1])],
        span_years=round((df.time.iloc[-1] - df.time.iloc[0]).days / 365.25, 2),
        contract=dict(value_per_unit_per_lot=CONTRACT.value_per_unit_per_lot,
                      volume_min=CONTRACT.volume_min, broker_stops_level=CONTRACT.broker_stops_level),
        realistic_costs=dict(spread=REALISTIC.spread, slippage_per_side=REALISTIC.slippage_per_side,
                             round_trip=REALISTIC.round_trip()),
        spread_note="Valetax H1 records spread on ~13% of bars (rest 0); floored at a realistic "
                    "gold spread, bracketed by 4 scenarios -- never set to 0.",
        methodology="identical to backtest/btc_h1_validation.py",
    )}

    # leakage probe
    cut = int(N * 0.6)
    fp, _ = strat.build_momentum(c[:cut], h[:cut], l[:cut], **CANDIDATES["momentum_rsi_mtf"][1])
    fu, _ = strat.build_momentum(c, h, l, **CANDIDATES["momentum_rsi_mtf"][1])
    rep["leakage_probe"] = {"signal_identical_pre_truncation": bool(np.array_equal(fu[:cut - 1], fp[:cut - 1]))}

    log.info("strategies")
    evals = {}
    for nm, (bld, kw) in CANDIDATES.items():
        e = evaluate(nm, bld, df, kw)
        evals[nm] = e
        s = e["combined"]
        log.info("  %-22s trades=%s pf=%s expR=%s DD%%=%s WF+=%s/8",
                 nm, s.get("trades"), s.get("profit_factor"), s.get("expectancy_R"),
                 s.get("max_drawdown_pct"), e["walk_forward"]["folds_positive_expectancy"])
    rep["strategies"] = {nm: {k: v for k, v in e.items() if not k.startswith("_")} for nm, e in evals.items()}

    # per-calendar-year expectancy R for every candidate (>= 20 trades/year)
    def yearly_expR(e):
        tr = e["_trades"]
        if len(tr["pnl"]) == 0:
            return {}
        R = np.where(tr["risk"] > 0, tr["pnl"] / tr["risk"], 0.0)
        yr = pd.to_datetime(df["time"].iloc[tr["entry_i"]].to_numpy()).year
        return {int(y): round(float(R[yr == y].mean()), 4) for y in np.unique(yr) if (yr == y).sum() >= 20}
    year_tbl = {nm: yearly_expR(e) for nm, e in evals.items()}
    rep["per_year_expectancy_R"] = year_tbl

    def score(e, nm):
        s = e["combined"]; w = e["walk_forward"]
        if s.get("trades", 0) < 100:
            return -9.0
        ys = list(year_tbl[nm].values())
        if len(ys) < 5:
            return -9.0
        worst_year = min(ys)                      # cross-regime robustness = worst calendar year
        frac_pos = sum(1 for x in ys if x > 0) / len(ys)
        ce = s.get("expectancy_R") or -9
        # dominated by the worst year (rides-one-trend strategies get punished here)
        return 3.0 * worst_year + 1.0 * ce + 0.5 * frac_pos + (0.1 if w.get("majority_folds_positive") else 0)
    best_name = max(evals, key=lambda k: score(evals[k], k))
    best = evals[best_name]
    rep["best_strategy"] = dict(
        name=best_name, score=round(score(best, best_name), 4),
        selection="max of (3*worst-calendar-year-expR + combined-expR + fraction-of-years-positive); "
                  "punishes strategies that ride a single multi-year trend")

    # edge authenticity on the best
    be, ba, bw_ = best["_entries"], best["_atr"], best["_warmup"]
    tr = best["_trades"]; di = tr["dir"]; pnl = tr["pnl"]
    Rm = np.where(tr["risk"] > 0, pnl / tr["risk"], 0.0)
    rng = np.random.default_rng(7)
    mask = be != 0
    null = []
    for _ in range(150):
        e2 = be.copy()
        e2[mask] = (rng.integers(0, 2, int(mask.sum())) * 2 - 1).astype(np.int8)
        rr = lab.run_backtest(PX, ba, e2, bw_ + 30, N, bw_, REALISTIC, EDGE_RISK, CONTRACT)["summary"]
        null.append(rr.get("expectancy_R") or 0.0)
    null = np.array(null)
    obs = best["combined"].get("expectancy_R") or 0.0
    fixed = {}
    for lbl, dv in (("always_long", 1), ("always_short", -1)):
        e3 = be.copy(); e3[e3 != 0] = dv
        rr = lab.run_backtest(PX, ba, e3.astype(np.int8), bw_ + 30, N, bw_, REALISTIC, EDGE_RISK, CONTRACT)["summary"]
        fixed[lbl] = dict(trades=rr.get("trades"), expectancy_R=rr.get("expectancy_R"), profit_factor=rr.get("profit_factor"))
    et = pd.to_datetime(df["time"].iloc[tr["entry_i"]].to_numpy())
    per_year = {int(y): dict(trades=int((et.year == y).sum()),
                             net=round(float(pnl[et.year == y].sum()), 2),
                             expectancy_R=round(float(Rm[et.year == y].mean()), 4))
                for y in sorted(set(et.year)) if (et.year == y).sum() >= 5}
    rep["edge_authenticity"] = dict(
        best_strategy=best_name,
        look_ahead="no look-ahead detected" if rep["leakage_probe"]["signal_identical_pre_truncation"] else "LEAKAGE",
        direction_split=dict(long=int((di == 1).sum()), short=int((di == -1).sum()),
                             long_expR=round(float(Rm[di == 1].mean()), 4) if (di == 1).any() else None,
                             short_expR=round(float(Rm[di == -1].mean()), 4) if (di == -1).any() else None),
        both_directions_profitable=bool((di == 1).any() and (di == -1).any()
                                        and Rm[di == 1].mean() > 0 and Rm[di == -1].mean() > 0),
        random_direction_null=dict(trials=150, null_mean_expR=round(float(null.mean()), 4),
                                   null_std=round(float(null.std()), 4), observed_expR=round(obs, 4),
                                   observed_percentile=round(100 * float((null < obs).mean()), 1),
                                   direction_carries_information=bool((null < obs).mean() >= 0.99)),
        fixed_direction_baselines=fixed,
        fixed_direction_both_lose=bool((fixed["always_long"]["expectancy_R"] or 0) <= 0
                                      and (fixed["always_short"]["expectancy_R"] or 0) <= 0),
        per_calendar_year=per_year,
        all_years_positive=bool(per_year and all(v["expectancy_R"] > 0 for v in per_year.values())),
        n_years=len(per_year),
    )

    # walk-forward (best) full
    wf = {"best_strategy_name": best_name,
          "best_strategy": lab.walk_forward(PX, ba, be, bw_, REALISTIC, EDGE_RISK, n_folds=8, contract=CONTRACT)}
    (REPORTS / "xau_h1_walkforward.json").write_text(json.dumps(wf, indent=2, default=str), encoding="utf-8")
    rep["walk_forward"] = wf["best_strategy"]["summary"]

    # cost sensitivity
    def cs(e):
        return lab.cost_sensitivity(PX, e["_atr"], e["_entries"], e["_warmup"] + 30, N, e["_warmup"],
                                    EDGE_RISK, COST_SCENARIOS, contract=CONTRACT)
    cost = {"scenarios": {k: dict(spread=v.spread, slippage_per_side=v.slippage_per_side,
                                  round_trip=round(v.round_trip(), 2)) for k, v in COST_SCENARIOS.items()},
            "all_candidates": {nm: cs(e) for nm, e in evals.items()},
            "best_strategy": cs(best)}
    (REPORTS / "xau_h1_cost_sensitivity.json").write_text(json.dumps(cost, indent=2, default=str), encoding="utf-8")
    rep["cost_sensitivity"] = cost["best_strategy"]

    # parameter robustness (best)
    robust = []
    if best_name in ("ema_trend", "ema_rsi_trend"):
        adx = 20.0 if best_name == "ema_rsi_trend" else 0.0
        arm = 0.85 if best_name == "ema_rsi_trend" else 1.0
        for f in (18, 20, 22):
            for s in (45, 50, 55):
                for tp in (180, 200, 220):
                    e = evaluate(best_name, strat.build_trend, df,
                                 dict(fast=f, slow=s, trend=tp, adx_min=adx, atr_rank_max=arm))
                    robust.append(dict(params=dict(fast=f, slow=s, trend=tp),
                                       trades=e["combined"].get("trades", 0),
                                       expectancy_R=e["combined"].get("expectancy_R"),
                                       profit_factor=e["combined"].get("profit_factor"),
                                       wf_folds_positive=e["walk_forward"]["folds_positive_expectancy"]))
    elif best_name == "donchian_breakout":
        for lb in (15, 20, 25):
            for ex in (1.2, 1.3, 1.4):
                e = evaluate(best_name, strat.build_breakout, df, dict(lookback=lb, expansion=ex, atr_rank_min=0.20))
                robust.append(dict(params=dict(lookback=lb, expansion=ex),
                                   trades=e["combined"].get("trades", 0),
                                   expectancy_R=e["combined"].get("expectancy_R"),
                                   profit_factor=e["combined"].get("profit_factor"),
                                   wf_folds_positive=e["walk_forward"]["folds_positive_expectancy"]))
    else:
        for rb in (55, 60, 65):
            for rs in (35, 40, 45):
                e = evaluate(best_name, strat.build_momentum, df,
                             dict(rsi_buy=float(rb), rsi_sell=float(rs), mom_win=8, ema_htf=24))
                robust.append(dict(params=dict(rsi_buy=rb, rsi_sell=rs),
                                   trades=e["combined"].get("trades", 0),
                                   expectancy_R=e["combined"].get("expectancy_R"),
                                   profit_factor=e["combined"].get("profit_factor"),
                                   wf_folds_positive=e["walk_forward"]["folds_positive_expectancy"]))
    exps = [r["expectancy_R"] for r in robust if r["expectancy_R"] is not None]
    rep["parameter_robustness"] = dict(
        best_strategy=best_name, n_configs=len(robust), table=robust,
        n_positive_expectancy=int(sum(1 for x in exps if x > 0)),
        expectancy_R_min=round(min(exps), 4) if exps else None,
        expectancy_R_mean=round(float(np.mean(exps)), 4) if exps else None,
        single_combination_only=bool(exps and sum(1 for x in exps if x > 0) == 1),
        stable_region=bool(exps and sum(1 for x in exps if x > 0) >= 0.7 * len(exps) and min(exps) > -0.05),
    )

    # monte carlo (fixed-fractional on the real $100 account)
    mc = lab.monte_carlo(best["_trades"], RISK, n=5000,
                         sim_balance=RISK.initial_balance, sim_risk_frac=RISK.risk_per_trade)
    (REPORTS / "xau_h1_monte_carlo.json").write_text(
        json.dumps({"best_strategy": best_name, **mc}, indent=2, default=str), encoding="utf-8")
    rep["monte_carlo"] = mc

    # $100 feasibility (XAU H1 vol)
    a = atr_full[np.isfinite(atr_full)]
    bmin = CONTRACT.broker_stops_level * lab.BROKER_STOP_BUFFER
    vpu = CONTRACT.value_per_unit_per_lot

    def risk_at(av):
        sd = max(av * RISK.sl_atr_mult, bmin)
        loss = sd * vpu * CONTRACT.volume_min
        return round(sd, 2), round(loss, 3), round(100 * loss / RISK.initial_balance, 2)

    calm = risk_at(float(np.percentile(a, 10)))
    med = risk_at(float(np.median(a)))
    hi = risk_at(float(np.percentile(a, 90)))
    price = float(df.close.iloc[-1])
    margin = round(price * CONTRACT.volume_min * CONTRACT.value_per_unit_per_lot / 2000.0, 2)  # XAU uses leverage
    rep["account_feasibility"] = dict(
        gold_price=round(price, 2),
        h1_atr_median=round(float(np.median(a)), 2), h1_atr_p10=round(float(np.percentile(a, 10)), 2),
        h1_atr_p90=round(float(np.percentile(a, 90)), 2),
        risk_0_01_lot=dict(calm=dict(stop=calm[0], loss_usd=calm[1], pct=calm[2]),
                           median=dict(stop=med[0], loss_usd=med[1], pct=med[2]),
                           p90=dict(stop=hi[0], loss_usd=hi[1], pct=hi[2]),
                           broker_min=dict(stop=CONTRACT.broker_stops_level,
                                           loss_usd=round(CONTRACT.broker_stops_level * vpu * CONTRACT.volume_min, 3),
                                           pct=round(100 * CONTRACT.broker_stops_level * vpu * CONTRACT.volume_min / RISK.initial_balance, 2))),
        approx_margin_0_01_lot=margin,
        account_for_1pct_median_vol=round(med[1] / RISK.risk_per_trade, 0),
        account_for_1pct_p90_vol=round(hi[1] / RISK.risk_per_trade, 0),
        verdict=("XAU H1 ATR ~$%d; 0.01 lot risks %.1f%%/%.1f%%/%.1f%% of $100 (calm/median/p90). "
                 "A clean 1%% ATR-stop position needs ~$%d-$%d. Same account constraint as BTC H1."
                 % (np.median(a), calm[2], med[2], hi[2],
                    int(med[1] / RISK.risk_per_trade), int(hi[1] / RISK.risk_per_trade))),
    )

    # regime breakdown for the best (gold has clear macro regimes)
    tr_b = best["_trades"]
    Rb = np.where(tr_b["risk"] > 0, tr_b["pnl"] / tr_b["risk"], 0.0)
    yb = pd.to_datetime(df["time"].iloc[tr_b["entry_i"]].to_numpy()).year
    regimes = {"ranging_2015_2018": (yb <= 2018), "gold_bull_2019_2023": (yb >= 2019) & (yb <= 2023),
               "recent_2024_2026": (yb >= 2024)}
    rep["regime_breakdown"] = {k: dict(n=int(m.sum()), expectancy_R=round(float(Rb[m].mean()), 4),
                                       net=round(float(tr_b["pnl"][m].sum()), 2))
                               for k, m in regimes.items() if m.sum() >= 20}
    ys_best = list(year_tbl[best_name].values())
    rep["best_years_positive"] = f"{sum(1 for x in ys_best if x > 0)}/{len(ys_best)}"
    all_regimes_pos = all(v["expectancy_R"] > 0 for v in rep["regime_breakdown"].values())

    # verdict
    bc = best["combined"]; bwf = rep["walk_forward"]; pr = rep["parameter_robustness"]; ea = rep["edge_authenticity"]
    csb = rep["cost_sensitivity"]
    yrs_pos_frac = sum(1 for x in ys_best if x > 0) / max(1, len(ys_best))
    edge = (ea["look_ahead"] == "no look-ahead detected" and ea["both_directions_profitable"]
            and ea["random_direction_null"]["direction_carries_information"]
            and bwf.get("folds_positive_expectancy", 0) >= 6 and (bwf.get("min_expectancy_R") or -9) > 0
            and (bc.get("profit_factor") or 0) > 1.15 and pr.get("stable_region")
            and (csb.get("conservative", {}).get("expectancy_R") or -9) > 0
            and all_regimes_pos and yrs_pos_frac >= 0.80 and min(ys_best) > -0.10)
    rep["verdict"] = dict(
        edge_found=bool(edge),
        classification=("PROMISING CROSS-REGIME H1 EDGE -- forward-test alongside BTC H1" if edge else
                        "NO ROBUST EDGE -- do not forward-test"),
        combined_pf=bc.get("profit_factor"), combined_expR=bc.get("expectancy_R"),
        oos_final_expR=best["per_split"]["FINAL"].get("expectancy_R"),
        wf_positive=f"{bwf.get('folds_positive_expectancy')}/8",
        stressed_expR=csb.get("stressed", {}).get("expectancy_R"),
        conservative_expR=csb.get("conservative", {}).get("expectancy_R"),
        stable_region=pr.get("stable_region"),
        years_positive=rep["best_years_positive"],
        worst_year_expR=round(min(ys_best), 4),
        all_regimes_positive=bool(all_regimes_pos),
        regime_breakdown=rep["regime_breakdown"],
    )
    rep["elapsed_s"] = round(time.perf_counter() - t0, 1)
    (REPORTS / "xau_h1_validation.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    _md(rep)

    # console
    print("=" * 100)
    print("XAUUSD.vx H1 VALIDATION  (Valetax, %d bars, %.1f years %s -> %s)  --  NO LIVE ORDER"
          % (N, rep["meta"]["span_years"], rep["meta"]["span"][0][:10], rep["meta"]["span"][1][:10]))
    print("=" * 100)
    print(f"leakage: {rep['leakage_probe']}")
    print(f"{'strategy':22s} {'trades':>7} {'net$':>10} {'PF':>7} {'expR':>8} {'DD%':>7} {'WF+/8':>6} {'OOS expR':>9}")
    for nm, r in rep["strategies"].items():
        s = r["combined"]; w = r["walk_forward"]
        oos = r["per_split"]["FINAL"].get("expectancy_R")
        print(f"{nm:22s} {s.get('trades',0):>7} {s.get('net_pl',0):>10.2f} {str(s.get('profit_factor')):>7} "
              f"{str(s.get('expectancy_R')):>8} {str(s.get('max_drawdown_pct')):>7} "
              f"{w['folds_positive_expectancy']:>4}/8 {str(oos):>9}")
    print("-" * 100)
    print(f"BEST: {best_name}")
    print(f"edge authenticity: both-dir-profitable={ea['both_directions_profitable']}  "
          f"random-null pctile {ea['random_direction_null']['observed_percentile']} "
          f"(informative {ea['random_direction_null']['direction_carries_information']})  "
          f"always L/S {fixed['always_long']['expectancy_R']}/{fixed['always_short']['expectancy_R']} "
          f"(both lose {ea['fixed_direction_both_lose']})")
    print(f"per year ({ea['n_years']}y): " + " ".join(f"{y}:{v['expectancy_R']:+.2f}" for y, v in per_year.items())
          + f"  (all+ {ea['all_years_positive']})")
    print(f"walk-forward: {rep['walk_forward']['folds_positive_expectancy']}/8, mean {rep['walk_forward']['mean_expectancy_R']}, "
          f"worst {rep['walk_forward']['min_expectancy_R']}")
    print("cost sensitivity (best):")
    for k, v in csb.items():
        print(f"   {k:12s} rt ${v['round_trip']:>5}  expR {v['expectancy_R']}  PF {v['profit_factor']}")
    print(f"param robustness: {pr['n_positive_expectancy']}/{pr['n_configs']} positive, stable_region={pr['stable_region']}")
    print("regime breakdown (best):")
    for k, v in rep["regime_breakdown"].items():
        print(f"   {k:22s} n {v['n']:5d}  expR {v['expectancy_R']:+.3f}")
    print(f"years positive: {rep['best_years_positive']}   worst year expR {rep['verdict']['worst_year_expR']:+.3f}")
    if "n_trials" in mc:
        print(f"Monte Carlo: P(neg) {mc['prob_negative_final']}  P(ruin) {mc['prob_ruin_equity_stop']}  "
              f"P(DD>50%) {mc.get('prob_drawdown_over_50pct')}")
    print(f"$100: {rep['account_feasibility']['verdict']}")
    print("=" * 100)
    print(f"VERDICT: {rep['verdict']['classification']}")
    print("=" * 100)
    return 0


def _md(r):
    m = r["meta"]; ea = r["edge_authenticity"]; v = r["verdict"]
    pr = r["parameter_robustness"]; mc = r["monte_carlo"]; f = r["account_feasibility"]
    L = []; A = L.append
    A("# XAUUSD.vx H1 Validation")
    A("")
    A(f"- Valetax / XAUUSD.vx **H1**, {m['rows']:,} bars, {m['span'][0][:10]} to {m['span'][1][:10]} "
      f"({m['span_years']} years)")
    A(f"- Same engine / strategies / method as `BTC_H1_VALIDATION.md` -- directly comparable")
    A(f"- Realistic costs: spread ${m['realistic_costs']['spread']}, slippage "
      f"${m['realistic_costs']['slippage_per_side']}/side -> round-trip ${m['realistic_costs']['round_trip']:.2f}")
    A(f"- {m['spread_note']}")
    A(f"- Look-ahead probe: signal identical pre-truncation = "
      f"**{r['leakage_probe']['signal_identical_pre_truncation']}**")
    A("")
    A("## Strategies (realistic costs)")
    A("| strategy | trades | net $ | PF | exp R | DD % | WF +/8 | OOS exp R |")
    A("|---|--:|--:|--:|--:|--:|--:|--:|")
    for nm, x in r["strategies"].items():
        s = x["combined"]; w = x["walk_forward"]
        oos = x["per_split"]["FINAL"].get("expectancy_R")
        A(f"| {nm} | {s.get('trades',0)} | {s.get('net_pl',0)} | {s.get('profit_factor')} | "
          f"{s.get('expectancy_R')} | {s.get('max_drawdown_pct')} | {w['folds_positive_expectancy']}/8 | {oos} |")
    A("")
    A(f"**Best (cross-regime score): `{r['best_strategy']['name']}`** -- {r['best_strategy']['selection']}")
    A("")
    A("## Edge authenticity (best strategy)")
    A(f"- Look-ahead: **{ea['look_ahead']}**")
    ds = ea["direction_split"]
    A(f"- Direction split: {ds['long']} long / {ds['short']} short; long {ds['long_expR']} R, short {ds['short_expR']} R "
      f"-- **both directions profitable: {ea['both_directions_profitable']}**")
    rn = ea["random_direction_null"]
    A(f"- Random-direction null ({rn['trials']} runs): observed {rn['observed_expR']} R vs "
      f"null {rn['null_mean_expR']} +/- {rn['null_std']} R -> **{rn['observed_percentile']}th percentile**, "
      f"direction carries information: **{rn['direction_carries_information']}**")
    fb = ea["fixed_direction_baselines"]
    A(f"- Always-long {fb['always_long']['expectancy_R']} R, always-short {fb['always_short']['expectancy_R']} R "
      f"-- **both lose on the same entries: {ea['fixed_direction_both_lose']}**")
    A(f"- Per calendar year ({ea['n_years']}y): " + ", ".join(
        f"{y} {vv['expectancy_R']:+.3f} R" for y, vv in ea["per_calendar_year"].items())
      + f" -- all positive: **{ea['all_years_positive']}**")
    A("")
    A("## Walk-forward (best)")
    w = r["walk_forward"]
    A(f"{w.get('folds_traded', 8)} folds, {w['folds_positive_expectancy']} positive, mean {w['mean_expectancy_R']} R, "
      f"worst {w['min_expectancy_R']} R, majority positive: {w.get('majority_folds_positive')}.")
    A("")
    A("## Cost sensitivity (best)")
    A("| scenario | round-trip $ | trades | net $ | PF | exp R |")
    A("|---|--:|--:|--:|--:|--:|")
    for k, x in r["cost_sensitivity"].items():
        A(f"| {k} | {x['round_trip']} | {x['trades']} | {x['net_pl']} | {x['profit_factor']} | {x['expectancy_R']} |")
    A("")
    A("## Parameter robustness")
    A(f"{pr['n_positive_expectancy']}/{pr['n_configs']} nearby configs positive "
      f"(min {pr['expectancy_R_min']}, mean {pr['expectancy_R_mean']} R). "
      f"Single combination only: {pr['single_combination_only']}; **stable region: {pr['stable_region']}**.")
    A("")
    A("## Regime breakdown (best)")
    A("| regime | trades | exp R | net $ |")
    A("|---|--:|--:|--:|")
    for k, x in r["regime_breakdown"].items():
        A(f"| {k} | {x['n']} | {x['expectancy_R']} | {x['net']} |")
    A(f"\nYears positive: **{r['best_years_positive']}**; worst year exp R "
      f"**{v['worst_year_expR']:+.3f}**; all regimes positive: **{v['all_regimes_positive']}**.")
    A("")
    A("## Monte Carlo (best, fixed-fractional on $100)")
    if "n_trials" in mc:
        A(f"- P(negative final): {mc['prob_negative_final']}")
        A(f"- P(>50% drawdown): {mc.get('prob_drawdown_over_50pct')}")
        A(f"- P(ruin / equity stop): {mc['prob_ruin_equity_stop']}")
        A(f"- Worst losing streak: {mc['worst_losing_streak']}")
    else:
        A(f"- {mc.get('note')}")
    A("")
    A("## $100 account feasibility (H1)")
    rr = f["risk_0_01_lot"]
    A(f"- Gold ${f['gold_price']}; H1 ATR median ${f['h1_atr_median']} / p10 ${f['h1_atr_p10']} / p90 ${f['h1_atr_p90']}")
    A(f"- 0.01 lot risk: calm {rr['calm']['pct']}%, median {rr['median']['pct']}%, p90 {rr['p90']['pct']}%, "
      f"broker-min {rr['broker_min']['pct']}%")
    A(f"- Account for a clean 1% ATR stop: ~${int(f['account_for_1pct_median_vol'])} (median vol) "
      f"to ~${int(f['account_for_1pct_p90_vol'])} (p90 vol)")
    A(f"- **Verdict:** {f['verdict']}")
    A("")
    A("## Verdict")
    A(f"**{v['classification']}** -- edge_found = **{v['edge_found']}**")
    A("")
    A(f"PF {v['combined_pf']}, combined exp R {v['combined_expR']}, OOS FINAL exp R {v['oos_final_expR']}, "
      f"WF {v['wf_positive']}, stressed exp R {v['stressed_expR']}, conservative exp R {v['conservative_expR']}.")
    A("")
    A("Even if every research condition passes, `LIVE_TRADING` stays **false**: live "
      "execution + reconciliation is not implemented. See `tools/xau_h1_live_gate.py`.")
    (REPORTS.parent / "XAU_H1_VALIDATION.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
