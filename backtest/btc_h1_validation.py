"""
STEPS 3-9 -- BTCUSD.vx H1 strategy validation.

Same engine (backtest/btc_lab.py) and same strategies (backtest/btc_strategies.py)
as the M5 pipeline, so H1 and M5 results are directly comparable.

    python -m backtest.btc_h1_validation

Data: data/btcusd_vx_H1_dense.csv  (run  python -m backtest.btc_h1_dataset  first).

Writes:
    reports/btc_h1_validation.json / BTC_H1_VALIDATION.md
    reports/btc_h1_walkforward.json
    reports/btc_h1_cost_sensitivity.json
    reports/btc_h1_monte_carlo.json

NO optimisation for a single period.  NO LIVE ORDER.  Historical spread is
broker-reported ($29.76 floor); it is never set artificially low.
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

log = get_logger("btc.h1val")
DATA = ROOT / "data" / "btcusd_vx_H1_dense.csv"
REPORTS = ROOT / "reports"
TF = "H1"

RISK = lab.RiskCfg()                       # $100 account -- used for STEP 9 feasibility only
# H1 ATR (~$500) makes an ATR stop huge vs $100, so the $100 sizer would skip almost
# every H1 setup.  To measure the EDGE itself (STEP 3-8) we run on a nominal $5,000
# balance so trades execute; expectancy-R / PF / walk-forward are scale-invariant.
EDGE_RISK = lab.RiskCfg(initial_balance=5000.0, risk_per_trade=0.01, max_risk_per_trade=0.02)
COST_SCENARIOS = {
    "optimistic":   lab.Costs(spread=15.0, slippage_per_side=0.5),
    "realistic":    lab.Costs(spread=29.76, slippage_per_side=1.0),   # broker minimum / nonzero median
    "conservative": lab.Costs(spread=40.0, slippage_per_side=2.0),
    "stressed":     lab.Costs(spread=60.0, slippage_per_side=4.0),
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
    if not DATA.exists():
        raise SystemExit("run  python -m backtest.btc_h1_dataset  first")
    df = pd.read_csv(DATA)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


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
    per = {tag: lab.run_backtest(PX, atr, entries, i0, i1, warmup, REALISTIC, EDGE_RISK)["summary"]
           for tag, i0, i1 in splits(N, warmup)}
    comb = lab.run_backtest(PX, atr, entries, warmup + 30, N, warmup, REALISTIC, EDGE_RISK)
    wf = lab.walk_forward(PX, atr, entries, warmup, REALISTIC, EDGE_RISK, n_folds=8)
    return dict(name=name, params=kw, warmup=warmup, per_split=per,
                combined=comb["summary"], walk_forward=wf["summary"], walk_forward_folds=wf["folds"],
                _entries=entries, _atr=atr, _warmup=warmup, _trades=comb["trades"])


def leakage_probe(df):
    """Re-derive the trend signal for a mid bar on a truncated series; must match."""
    c = df.close.to_numpy(float); h = df.high.to_numpy(float); l = df.low.to_numpy(float)
    full, _ = strat.build_trend(c, h, l, adx_min=20.0, atr_rank_max=0.85)
    cut = int(len(c) * 0.6)
    part, _ = strat.build_trend(c[:cut], h[:cut], l[:cut], adx_min=20.0, atr_rank_max=0.85)
    same = bool(np.array_equal(full[:cut - 1], part[:cut - 1]))
    return {"truncation_bar": cut, "signal_identical_pre_truncation": same,
            "verdict": "no look-ahead detected" if same else "LEAKAGE -- signal changed"}


def main() -> int:
    t0 = time.perf_counter()
    df = load()
    N = len(df)
    c = df.close.to_numpy(float); h = df.high.to_numpy(float); l = df.low.to_numpy(float)
    PX = px_dict(df)
    atr_full = strat.base_atr(h, l, c, 14)
    log.info("H1 dense rows=%d  %s -> %s", N, df.time.iloc[0], df.time.iloc[-1])

    rep = {"meta": dict(
        symbol="BTCUSD.vx", broker="Valetax", timeframe=TF,
        rows=int(N), span=[str(df.time.iloc[0]), str(df.time.iloc[-1])],
        span_days=int((df.time.iloc[-1] - df.time.iloc[0]).days),
        span_years=round((df.time.iloc[-1] - df.time.iloc[0]).days / 365.25, 2),
        realistic_costs=dict(spread=REALISTIC.spread, slippage_per_side=REALISTIC.slippage_per_side,
                             round_trip=REALISTIC.round_trip()),
        contract=dict(value_per_unit_per_lot=lab.VALUE_PER_UNIT_PER_LOT, volume_min=lab.VOLUME_MIN,
                      broker_stops_level=lab.BROKER_STOPS_LEVEL),
        risk=dict(initial_balance=RISK.initial_balance, risk_per_trade=RISK.risk_per_trade,
                  sl_atr=RISK.sl_atr_mult, tp_atr=RISK.tp_atr_mult, min_rr=RISK.min_rr),
        methodology="identical to the M5 pipeline (backtest/btc_validate.py) for comparability",
    )}

    rep["leakage_probe"] = leakage_probe(df)

    # -- STEP 3 baseline + strategies -----------------------------------
    log.info("step 3: baseline + strategies")
    evals = {}
    for nm, (bld, kw) in CANDIDATES.items():
        e = evaluate(nm, bld, df, kw)
        evals[nm] = e
        s = e["combined"]
        log.info("  %-22s trades=%s net=%s pf=%s expR=%s DD%%=%s WF+=%s/8",
                 nm, s.get("trades"), s.get("net_pl"), s.get("profit_factor"),
                 s.get("expectancy_R"), s.get("max_drawdown_pct"),
                 e["walk_forward"]["folds_positive_expectancy"])
    rep["step3_strategies"] = {nm: {k: v for k, v in e.items() if not k.startswith("_")}
                               for nm, e in evals.items()}
    rep["step3_baseline"] = rep["step3_strategies"]["ema_rsi_trend"]

    # regime breakdown
    tlab, vlab = strat.regime_labels(c, h, l)
    def regime_bd(e):
        tr = e["_trades"]
        if len(tr["pnl"]) == 0:
            return {}
        ei = tr["entry_i"]; out = {}
        for lname, arr in (("trend", tlab), ("vol", vlab)):
            for val in np.unique(arr[ei]):
                m = arr[ei] == val
                if m.sum() < 5:
                    continue
                R = np.where(tr["risk"][m] > 0, tr["pnl"][m] / tr["risk"][m], 0.0)
                out[f"{lname}:{val}"] = dict(n=int(m.sum()), net=round(float(tr["pnl"][m].sum()), 2),
                                             expectancy_R=round(float(R.mean()), 4))
        return out
    rep["step3_regime_breakdown"] = {nm: regime_bd(e) for nm, e in evals.items()}

    # -- pick best: weakest-link (combined expR, WF mean expR), NOT total return --
    def score(e):
        s = e["combined"]; w = e["walk_forward"]
        if s.get("trades", 0) < 25:
            return -9.0
        ce = s.get("expectancy_R") or -9
        we = w.get("mean_expectancy_R")
        we = -9 if we is None else we
        return min(ce, we) + (0.05 if w.get("majority_folds_positive") else -0.05)
    best_name = max(evals, key=lambda k: score(evals[k]))
    best = evals[best_name]
    rep["best_strategy"] = dict(name=best_name, score=round(score(best), 4),
                                selection="max weakest-link(combined expR, walk-forward mean expR)")

    # -- edge-authenticity probes on the best strategy ------------------
    log.info("edge-authenticity probes")
    bld, kw = CANDIDATES[best_name]
    be = best["_entries"]; ba = best["_atr"]; bw_ = best["_warmup"]
    tr = best["_trades"]
    di = tr["dir"]; pnl = tr["pnl"]
    Rm = np.where(tr["risk"] > 0, pnl / tr["risk"], 0.0)

    # leakage probe on THIS strategy
    cut = int(N * 0.6)
    fp, _ = bld(c[:cut], h[:cut], l[:cut], **kw)
    leak_ok = bool(np.array_equal(be[:cut - 1], fp[:cut - 1]))

    # random-direction null: same entry bars, coin-flip direction
    rng = np.random.default_rng(7)
    mask = be != 0
    null_expR = []
    for _ in range(200):
        e2 = be.copy()
        e2[mask] = (rng.integers(0, 2, int(mask.sum())) * 2 - 1).astype(np.int8)
        rr = lab.run_backtest(PX, ba, e2, bw_ + 30, N, bw_, REALISTIC, EDGE_RISK)["summary"]
        null_expR.append(rr.get("expectancy_R") or 0.0)
    null_expR = np.array(null_expR)
    obs = best["combined"].get("expectancy_R") or 0.0

    # always-long / always-short on the same entries
    fixed = {}
    for lbl, dv in (("always_long", 1), ("always_short", -1)):
        e3 = be.copy(); e3[e3 != 0] = dv
        rr = lab.run_backtest(PX, ba, e3.astype(np.int8), bw_ + 30, N, bw_, REALISTIC, EDGE_RISK)["summary"]
        fixed[lbl] = dict(trades=rr.get("trades"), expectancy_R=rr.get("expectancy_R"),
                          profit_factor=rr.get("profit_factor"))

    # per calendar year
    et = pd.to_datetime(df["time"].iloc[tr["entry_i"]].to_numpy())
    per_year = {}
    for y in sorted(set(et.year)):
        m = et.year == y
        per_year[int(y)] = dict(trades=int(m.sum()), net=round(float(pnl[m].sum()), 2),
                                expectancy_R=round(float(Rm[m].mean()), 4),
                                win_rate=round(float((pnl[m] > 0).mean()), 4))

    rep["edge_authenticity"] = dict(
        best_strategy=best_name,
        look_ahead_probe=("no look-ahead detected" if leak_ok else "LEAKAGE"),
        direction_split=dict(long=int((di == 1).sum()), short=int((di == -1).sum()),
                             long_expectancy_R=round(float(Rm[di == 1].mean()), 4) if (di == 1).any() else None,
                             short_expectancy_R=round(float(Rm[di == -1].mean()), 4) if (di == -1).any() else None),
        both_directions_profitable=bool((di == 1).any() and (di == -1).any()
                                        and Rm[di == 1].mean() > 0 and Rm[di == -1].mean() > 0),
        random_direction_null=dict(trials=200, null_mean_expR=round(float(null_expR.mean()), 4),
                                   null_std=round(float(null_expR.std()), 4),
                                   observed_expR=round(obs, 4),
                                   observed_percentile=round(100 * float((null_expR < obs).mean()), 1),
                                   direction_carries_information=bool(
                                       (null_expR < obs).mean() >= 0.99)),
        fixed_direction_baselines=fixed,
        fixed_direction_both_lose=bool((fixed["always_long"]["expectancy_R"] or 0) <= 0
                                       and (fixed["always_short"]["expectancy_R"] or 0) <= 0),
        per_calendar_year=per_year,
        all_years_positive=bool(per_year and all(v["expectancy_R"] > 0 for v in per_year.values())),
        exit_reason_counts={int(k): int((tr["reason"] == k).sum()) for k in (0, 1, 2, 3)},
        avg_holding_bars=round(float(tr["bars"].mean()), 1),
        caveat="2.59 years of H1 covers ONE BTC macro cycle (2024 accumulation -> 2025 bull -> "
               "2026). The edge has NOT been observed through a prolonged bear/chop regime, and "
               "live execution/slippage on Valetax is unverified.",
    )

    # -- STEP 5 walk-forward (best + baseline) --------------------------
    log.info("step 5: walk-forward")
    wf = {
        "best_strategy_name": best_name,
        "baseline": lab.walk_forward(PX, evals["ema_rsi_trend"]["_atr"], evals["ema_rsi_trend"]["_entries"],
                                     evals["ema_rsi_trend"]["_warmup"], REALISTIC, EDGE_RISK, n_folds=8),
        "best_strategy": lab.walk_forward(PX, best["_atr"], best["_entries"], best["_warmup"],
                                          REALISTIC, EDGE_RISK, n_folds=8),
    }
    (REPORTS / "btc_h1_walkforward.json").write_text(json.dumps(wf, indent=2, default=str), encoding="utf-8")
    rep["step5_walk_forward"] = {"baseline": wf["baseline"]["summary"],
                                 "best_strategy": wf["best_strategy"]["summary"]}

    # -- STEP 6 cost sensitivity --------------------------------------
    log.info("step 6: cost sensitivity")
    def cs(e):
        return lab.cost_sensitivity(PX, e["_atr"], e["_entries"], e["_warmup"] + 30, N,
                                    e["_warmup"], EDGE_RISK, COST_SCENARIOS)
    cost = {"scenarios": {k: dict(spread=v.spread, slippage_per_side=v.slippage_per_side,
                                  round_trip=round(v.round_trip(), 2)) for k, v in COST_SCENARIOS.items()},
            "all_candidates": {nm: cs(e) for nm, e in evals.items()},
            "best_strategy": cs(best)}
    (REPORTS / "btc_h1_cost_sensitivity.json").write_text(json.dumps(cost, indent=2, default=str), encoding="utf-8")
    rep["step6_cost_sensitivity"] = {"best_strategy": cost["best_strategy"]}

    # -- STEP 7 parameter robustness (small nearby grid) --------------
    log.info("step 7: parameter robustness")
    robust = []
    if best_name in ("ema_trend", "ema_rsi_trend"):
        adx = 20.0 if best_name == "ema_rsi_trend" else 0.0
        arm = 0.85 if best_name == "ema_rsi_trend" else 1.0
        for f in (18, 20, 22):
            for s in (45, 50, 55):
                for tp in (180, 200, 220):
                    e = evaluate(best_name, strat.build_trend, df,
                                 dict(fast=f, slow=s, trend=tp, adx_min=adx, atr_rank_max=arm))
                    robust.append(dict(params={"fast": f, "slow": s, "trend": tp},
                                       trades=e["combined"].get("trades", 0),
                                       expectancy_R=e["combined"].get("expectancy_R"),
                                       profit_factor=e["combined"].get("profit_factor"),
                                       wf_folds_positive=e["walk_forward"]["folds_positive_expectancy"]))
    elif best_name == "donchian_breakout":
        for lb in (15, 20, 25):
            for ex in (1.2, 1.3, 1.4):
                e = evaluate(best_name, strat.build_breakout, df,
                             dict(lookback=lb, expansion=ex, atr_rank_min=0.20))
                robust.append(dict(params={"lookback": lb, "expansion": ex},
                                   trades=e["combined"].get("trades", 0),
                                   expectancy_R=e["combined"].get("expectancy_R"),
                                   profit_factor=e["combined"].get("profit_factor"),
                                   wf_folds_positive=e["walk_forward"]["folds_positive_expectancy"]))
    else:
        for rb in (55, 60, 65):
            for rs in (35, 40, 45):
                e = evaluate(best_name, strat.build_momentum, df,
                             dict(rsi_buy=float(rb), rsi_sell=float(rs), mom_win=8, ema_htf=24))
                robust.append(dict(params={"rsi_buy": rb, "rsi_sell": rs},
                                   trades=e["combined"].get("trades", 0),
                                   expectancy_R=e["combined"].get("expectancy_R"),
                                   profit_factor=e["combined"].get("profit_factor"),
                                   wf_folds_positive=e["walk_forward"]["folds_positive_expectancy"]))
    exps = [r["expectancy_R"] for r in robust if r["expectancy_R"] is not None]
    rep["step7_parameter_robustness"] = dict(
        best_strategy=best_name, n_configs=len(robust), table=robust,
        n_positive_expectancy=int(sum(1 for x in exps if x > 0)),
        expectancy_R_min=round(min(exps), 4) if exps else None,
        expectancy_R_mean=round(float(np.mean(exps)), 4) if exps else None,
        expectancy_R_max=round(max(exps), 4) if exps else None,
        profitable_configs_cluster=bool(exps and sum(1 for x in exps if x > 0) >= 0.6 * len(exps)),
        single_combination_only=bool(exps and sum(1 for x in exps if x > 0) == 1),
        stable_region=bool(exps and sum(1 for x in exps if x > 0) >= 0.7 * len(exps) and min(exps) > -0.05),
    )

    # -- STEP 8 Monte Carlo (fixed-fractional on the real $100 account) ----
    log.info("step 8: monte carlo")
    mc = lab.monte_carlo(best["_trades"], RISK, n=5000,
                         sim_balance=RISK.initial_balance, sim_risk_frac=RISK.risk_per_trade)
    mc_report = {"best_strategy": best_name,
                 "note": "bootstrap of realised R-multiples, re-simulated fixed-fractional "
                         "(1% risk) on the real $100 account", **mc}
    (REPORTS / "btc_h1_monte_carlo.json").write_text(json.dumps(mc_report, indent=2, default=str), encoding="utf-8")
    rep["step8_monte_carlo"] = mc_report

    # -- STEP 9 $100 feasibility (H1 vol) --------------------------
    log.info("step 9: $100 feasibility")
    price = float(df.close.iloc[-1])
    a = atr_full[np.isfinite(atr_full)]
    bmin = lab.BROKER_STOPS_LEVEL * lab.BROKER_STOP_BUFFER

    def risk_at(av):
        sd = max(av * RISK.sl_atr_mult, bmin)
        loss = sd * lab.VALUE_PER_UNIT_PER_LOT * lab.VOLUME_MIN
        return round(sd, 1), round(loss, 3), round(100 * loss / RISK.initial_balance, 2)

    calm = risk_at(float(np.percentile(a, 10)))
    med = risk_at(float(np.median(a)))
    hi = risk_at(float(np.percentile(a, 90)))
    margin = round(price * lab.VOLUME_MIN / 100.0, 2)
    worst_streak = mc.get("worst_losing_streak", 0)
    survive_streak = bool(hi[1] * max(1, worst_streak) < RISK.initial_balance * 0.5)
    rep["step9_account_feasibility"] = dict(
        btc_price=round(price, 2),
        h1_atr_now=round(float(a[-1]), 1), h1_atr_median=round(float(np.median(a)), 1),
        h1_atr_p10=round(float(np.percentile(a, 10)), 1), h1_atr_p90=round(float(np.percentile(a, 90)), 1),
        min_executable_lot=lab.VOLUME_MIN,
        risk_0_01_lot=dict(
            calm=dict(stop=calm[0], loss_usd=calm[1], pct_of_100=calm[2]),
            median=dict(stop=med[0], loss_usd=med[1], pct_of_100=med[2]),
            p90=dict(stop=hi[0], loss_usd=hi[1], pct_of_100=hi[2]),
            broker_min_stop=dict(stop=lab.BROKER_STOPS_LEVEL,
                                 loss_usd=round(lab.BROKER_STOPS_LEVEL * lab.VOLUME_MIN, 3),
                                 pct_of_100=round(100 * lab.BROKER_STOPS_LEVEL * lab.VOLUME_MIN / RISK.initial_balance, 2)),
        ),
        margin_0_01_lot=margin, free_margin_after=round(RISK.initial_balance - margin, 2),
        stops_executable=bool(med[0] >= lab.BROKER_STOPS_LEVEL),
        worst_losing_streak_mc=int(worst_streak),
        can_survive_worst_streak=survive_streak,
        min_lot_risk_exceeds_2pct_at_p90=bool(hi[2] > 100 * RISK.max_risk_per_trade),
        account_size_for_1pct_median_vol=round(med[1] / RISK.risk_per_trade, 0),
        account_size_for_1pct_p90_vol=round(hi[1] / RISK.risk_per_trade, 0),
        verdict=(
            "H1 ATR is large ($%d median), so a proper 1%%-risk ATR stop at 0.01 lot needs "
            "~$%d (median vol) to ~$%d (p90 vol). On $100, 0.01 lot risks %.1f%%/%.1f%%/%.1f%% "
            "(calm/median/p90) -- above the 1%% target and often above the 2%% ceiling, so the "
            "sizer would SKIP most H1 setups. Margin ($%.2f) is not the constraint; the stop "
            "size vs a $100 account is."
            % (np.median(a), int(med[1] / RISK.risk_per_trade), int(hi[1] / RISK.risk_per_trade),
               calm[2], med[2], hi[2], margin)),
    )

    # -- console + json + md -------------------------------------
    rep["elapsed_s"] = round(time.perf_counter() - t0, 1)
    (REPORTS / "btc_h1_validation.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    _md(rep, cost)

    print("=" * 100)
    print("BTCUSD.vx H1 VALIDATION  (Valetax, %d bars, %.2f years, %s -> %s)  --  NO LIVE ORDER"
          % (N, rep["meta"]["span_years"], rep["meta"]["span"][0][:10], rep["meta"]["span"][1][:10]))
    print("=" * 100)
    print(f"leakage probe: {rep['leakage_probe']['verdict']}")
    print("-" * 100)
    print(f"{'strategy':22s} {'trades':>7} {'net$':>9} {'PF':>7} {'expR':>8} {'DD%':>7} {'WF+/8':>6} "
          f"{'OOS expR':>9} {'costOK':>7}")
    for nm, r in rep["step3_strategies"].items():
        s = r["combined"]; w = r["walk_forward"]
        oos = r["per_split"]["FINAL"].get("expectancy_R")
        c_ok = (cost["all_candidates"][nm]["conservative"]["expectancy_R"] or -9) > 0
        print(f"{nm:22s} {s.get('trades',0):>7} {s.get('net_pl',0):>9.2f} {str(s.get('profit_factor')):>7} "
              f"{str(s.get('expectancy_R')):>8} {str(s.get('max_drawdown_pct')):>7} "
              f"{w['folds_positive_expectancy']:>4}/8 {str(oos):>9} {str(c_ok):>7}")
    print("-" * 100)
    print(f"BEST (weakest-link, not total return): {best_name}")
    bw = rep["step5_walk_forward"]["best_strategy"]
    print(f"walk-forward (best): {bw['folds_positive_expectancy']}/8 positive, mean expR {bw['mean_expectancy_R']}, "
          f"worst {bw['min_expectancy_R']}")
    cb = rep["step6_cost_sensitivity"]["best_strategy"]
    print("cost sensitivity (best):")
    for k, v in cb.items():
        print(f"   {k:12s} rt ${v['round_trip']:>6}  net ${v['net_pl']:>9}  PF {v['profit_factor']}  expR {v['expectancy_R']}")
    pr = rep["step7_parameter_robustness"]
    print(f"param robustness: {pr['n_positive_expectancy']}/{pr['n_configs']} configs positive, "
          f"stable_region={pr['stable_region']}, single_combo_only={pr['single_combination_only']}")
    m = rep["step8_monte_carlo"]
    if "n_trials" in m:
        print(f"Monte Carlo (best, {m['n_trials']} sims): P(neg final) {m['prob_negative_final']}, "
              f"P(ruin) {m['prob_ruin_equity_stop']}, P(DD>50%) {m.get('prob_drawdown_over_50pct')}, "
              f"worst streak {m['worst_losing_streak']}")
    print(f"$100 feasibility: {rep['step9_account_feasibility']['verdict']}")
    ea = rep["edge_authenticity"]
    print("-" * 100)
    print("EDGE AUTHENTICITY (best strategy):")
    print(f"  look-ahead        : {ea['look_ahead_probe']}")
    print(f"  direction split   : {ea['direction_split']['long']}L / {ea['direction_split']['short']}S  "
          f"long expR {ea['direction_split']['long_expectancy_R']}  short expR {ea['direction_split']['short_expectancy_R']}  "
          f"(both profitable: {ea['both_directions_profitable']})")
    rn = ea["random_direction_null"]
    print(f"  random-dir null   : observed {rn['observed_expR']} vs null {rn['null_mean_expR']}+-{rn['null_std']}  "
          f"pctile {rn['observed_percentile']}  (direction informative: {rn['direction_carries_information']})")
    print(f"  always long/short : {ea['fixed_direction_baselines']['always_long']['expectancy_R']} / "
          f"{ea['fixed_direction_baselines']['always_short']['expectancy_R']}  (both lose: {ea['fixed_direction_both_lose']})")
    print(f"  per year          : " + "  ".join(f"{y}:{v['expectancy_R']:+.2f}R" for y, v in ea['per_calendar_year'].items())
          + f"  (all positive: {ea['all_years_positive']})")
    print(f"  CAVEAT            : {ea['caveat']}")
    print("=" * 100)
    print("reports: btc_h1_validation.{json}, BTC_H1_VALIDATION.md, btc_h1_walkforward.json, "
          "btc_h1_cost_sensitivity.json, btc_h1_monte_carlo.json")
    print("next: python -m tools.btc_h1_live_gate")
    return 0


def _md(r, cost):
    m = r["meta"]
    L = []
    A = L.append
    A("# BTCUSD.vx H1 Validation")
    A("")
    A(f"- Valetax / BTCUSD.vx **H1**, {m['rows']:,} bars, {m['span'][0][:10]} to {m['span'][1][:10]} "
      f"({m['span_days']} days / {m['span_years']} years)")
    A(f"- Realistic costs: spread ${m['realistic_costs']['spread']}, slippage "
      f"${m['realistic_costs']['slippage_per_side']}/side -> round-trip ${m['realistic_costs']['round_trip']:.2f}")
    A(f"- Round-trip cost is ~6% of median H1 ATR (vs ~72% on M5) -- the reason to test H1")
    A(f"- Look-ahead probe: **{r['leakage_probe']['verdict']}**")
    A("")
    A("## Strategies (realistic costs)")
    A("| strategy | trades | net $ | PF | exp R | DD % | WF +/8 | OOS exp R |")
    A("|---|--:|--:|--:|--:|--:|--:|--:|")
    for nm, x in r["step3_strategies"].items():
        s = x["combined"]; w = x["walk_forward"]
        oos = x["per_split"]["FINAL"].get("expectancy_R")
        A(f"| {nm} | {s.get('trades',0)} | {s.get('net_pl',0)} | {s.get('profit_factor')} | "
          f"{s.get('expectancy_R')} | {s.get('max_drawdown_pct')} | {w['folds_positive_expectancy']}/8 | {oos} |")
    A("")
    A(f"**Best (weakest-link score): `{r['best_strategy']['name']}`**")
    A("")
    A("## Edge authenticity (best strategy)")
    ea = r["edge_authenticity"]
    A(f"- Look-ahead probe: **{ea['look_ahead_probe']}**")
    A(f"- Direction split: {ea['direction_split']['long']} long / {ea['direction_split']['short']} short; "
      f"long expectancy {ea['direction_split']['long_expectancy_R']} R, short {ea['direction_split']['short_expectancy_R']} R "
      f"— **both directions profitable: {ea['both_directions_profitable']}**")
    rn = ea["random_direction_null"]
    A(f"- Random-direction null (200 runs, same entry bars): observed {rn['observed_expR']} R vs "
      f"null {rn['null_mean_expR']} ± {rn['null_std']} R → **{rn['observed_percentile']}th percentile**, "
      f"direction carries information: **{rn['direction_carries_information']}**")
    A(f"- Always-long expectancy {ea['fixed_direction_baselines']['always_long']['expectancy_R']} R, "
      f"always-short {ea['fixed_direction_baselines']['always_short']['expectancy_R']} R "
      f"— **both lose on the same entries: {ea['fixed_direction_both_lose']}**")
    A(f"- Per calendar year: " + ", ".join(f"{y} {v['expectancy_R']:+.3f} R ({v['trades']} tr)"
                                           for y, v in ea['per_calendar_year'].items())
      + f" — all positive: **{ea['all_years_positive']}**")
    A(f"- Exit mix (target/stop/time/eq-stop): {ea['exit_reason_counts']}; avg hold {ea['avg_holding_bars']} bars")
    A(f"- **Caveat:** {ea['caveat']}")
    A("")
    A("## Walk-forward (best)")
    w = r["step5_walk_forward"]["best_strategy"]
    A(f"{w['folds_traded']} folds, {w['folds_positive_expectancy']} positive, mean {w['mean_expectancy_R']} R, "
      f"worst {w['min_expectancy_R']} R, single-fold profit share {w['single_fold_profit_share']}, "
      f"majority positive: {w['majority_folds_positive']}.")
    A("")
    A("## Cost sensitivity (best)")
    A("| scenario | round-trip $ | trades | net $ | PF | exp R |")
    A("|---|--:|--:|--:|--:|--:|")
    for k, v in r["step6_cost_sensitivity"]["best_strategy"].items():
        A(f"| {k} | {v['round_trip']} | {v['trades']} | {v['net_pl']} | {v['profit_factor']} | {v['expectancy_R']} |")
    A("")
    A("## Parameter robustness")
    pr = r["step7_parameter_robustness"]
    A(f"{pr['n_positive_expectancy']}/{pr['n_configs']} nearby configs positive "
      f"(min {pr['expectancy_R_min']}, mean {pr['expectancy_R_mean']} R). "
      f"Cluster: {pr['profitable_configs_cluster']}; single combination only: {pr['single_combination_only']}; "
      f"**stable region: {pr['stable_region']}**.")
    A("")
    A("## Monte Carlo (best)")
    mc = r["step8_monte_carlo"]
    if "n_trials" in mc:
        A(f"- P(negative final): {mc['prob_negative_final']}")
        A(f"- P(>50% drawdown): {mc.get('prob_drawdown_over_50pct')}")
        A(f"- P(ruin / equity stop): {mc['prob_ruin_equity_stop']}")
        A(f"- Expected max DD ${mc['expected_max_drawdown_abs']}, p95 ${mc['p95_max_drawdown_abs']}")
        A(f"- Worst losing streak: {mc['worst_losing_streak']}")
    else:
        A(f"- {mc.get('note')}")
    A("")
    A("## $100 account feasibility (H1)")
    f = r["step9_account_feasibility"]
    rr = f["risk_0_01_lot"]
    A(f"- BTC ${f['btc_price']}; H1 ATR now ${f['h1_atr_now']} / median ${f['h1_atr_median']} / p90 ${f['h1_atr_p90']}")
    A(f"- 0.01 lot risk: calm {rr['calm']['pct_of_100']}%, median {rr['median']['pct_of_100']}%, "
      f"p90 {rr['p90']['pct_of_100']}%, broker-min stop {rr['broker_min_stop']['pct_of_100']}%")
    A(f"- margin for 0.01 lot: ${f['margin_0_01_lot']}; stops executable: {f['stops_executable']}")
    A(f"- can survive worst MC streak ({f['worst_losing_streak_mc']} losses): {f['can_survive_worst_streak']}")
    A(f"- account for a clean 1% ATR stop: ~${int(f['account_size_for_1pct_median_vol'])} (median vol) "
      f"to ~${int(f['account_size_for_1pct_p90_vol'])} (p90 vol)")
    A(f"- **Verdict:** {f['verdict']}")
    (REPORTS.parent / "BTC_H1_VALIDATION.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
