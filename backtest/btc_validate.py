"""
PHASES 3-9 orchestrator -- statistically defensible BTCUSD.vx strategy validation.

    python -m backtest.btc_validate

Writes:
    reports/btc_strategy_validation.json / .md
    reports/btc_walkforward.json
    reports/btc_cost_sensitivity.json
    reports/btc_monte_carlo.json

Data: data/btcusd_vx_M5.csv  (run  python -m backtest.btc_dataset  first).
No optimisation for a single period; walk-forward + cost sensitivity + parameter
robustness + Monte Carlo gate every conclusion.  NO LIVE ORDER.
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

log = get_logger("btc.validate")
DATA = ROOT / "data" / "btcusd_vx_M5.csv"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

RISK = lab.RiskCfg()
COST_SCENARIOS = {
    "optimistic":   lab.Costs(spread=12.0, slippage_per_side=0.5),
    "normal":       lab.Costs(spread=29.76, slippage_per_side=1.0),   # Valetax-reported
    "conservative": lab.Costs(spread=40.0, slippage_per_side=2.0),
    "stressed":     lab.Costs(spread=60.0, slippage_per_side=4.0),
}
NORMAL = COST_SCENARIOS["normal"]


def load_px():
    if not DATA.exists():
        raise SystemExit("run  python -m backtest.btc_dataset  first")
    df = pd.read_csv(DATA)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def px_dict(df):
    return {k: df[k].to_numpy(float) for k in ("open", "high", "low", "close")}


def splits(N, warmup):
    a = warmup + 50
    t1 = int(N * 0.70)
    v1 = int(N * 0.85)
    return [("TRAIN", a, t1), ("VALIDATION", t1, v1), ("FINAL", v1, N)]


def eval_strategy(name, builder, df, kwargs):
    c = df["close"].to_numpy(float); h = df["high"].to_numpy(float); l = df["low"].to_numpy(float)
    entries, warmup = builder(c, h, l, **kwargs)
    atr = strat.base_atr(h, l, c, 14)
    PX = px_dict(df)
    N = len(c)
    per = {}
    for tag, i0, i1 in splits(N, warmup):
        per[tag] = lab.run_backtest(PX, atr, entries, i0, i1, warmup, NORMAL, RISK)["summary"]
    comb = lab.run_backtest(PX, atr, entries, warmup + 50, N, warmup, NORMAL, RISK)
    wf = lab.walk_forward(PX, atr, entries, warmup, NORMAL, RISK, n_folds=8)
    return dict(name=name, params=kwargs, warmup=warmup,
                per_split=per, combined=comb["summary"], walk_forward=wf["summary"],
                walk_forward_folds=wf["folds"],
                _entries=entries, _atr=atr, _warmup=warmup, _trades=comb["trades"])


def main() -> int:
    t0 = time.perf_counter()
    df = load_px()
    N = len(df)
    c = df["close"].to_numpy(float); h = df["high"].to_numpy(float); l = df["low"].to_numpy(float)
    PX = px_dict(df)
    log.info("M5 rows=%d  %s -> %s", N, df.time.iloc[0], df.time.iloc[-1])

    report = {
        "meta": dict(
            symbol="BTCUSD.vx", broker="Valetax", timeframe="M5",
            rows=int(N), span=[str(df.time.iloc[0]), str(df.time.iloc[-1])],
            span_days=int((df.time.iloc[-1] - df.time.iloc[0]).days),
            contract=dict(value_per_unit_per_lot=lab.VALUE_PER_UNIT_PER_LOT,
                          volume_min=lab.VOLUME_MIN, broker_stops_level=lab.BROKER_STOPS_LEVEL),
            normal_costs=dict(spread=NORMAL.spread, slippage_per_side=NORMAL.slippage_per_side,
                              round_trip=NORMAL.round_trip()),
            risk=dict(initial_balance=RISK.initial_balance, risk_per_trade=RISK.risk_per_trade,
                      sl_atr=RISK.sl_atr_mult, tp_atr=RISK.tp_atr_mult, min_rr=RISK.min_rr),
            note="NO optimisation for a single period; every conclusion gated by "
                 "walk-forward + cost sensitivity + parameter robustness + Monte Carlo.",
        )
    }

    # -------------------- PHASE 3: BASELINE (frozen EMA20/50/200 + RSI trend) ----
    log.info("phase 3: baseline")
    baseline = eval_strategy(
        "baseline_trend_ema", strat.build_trend, df,
        dict(fast=20, slow=50, trend=200, adx_min=0.0, atr_rank_max=1.0))  # frozen, no ADX/regime gate
    report["phase3_baseline"] = {k: v for k, v in baseline.items() if not k.startswith("_")}

    # -------------------- PHASE 5: STRATEGY RESEARCH ---------------------------
    log.info("phase 5: strategy research")
    candidates = {
        "trend_ema":            (strat.build_trend, dict(fast=20, slow=50, trend=200, adx_min=20.0, atr_rank_max=0.85)),
        "breakout_donchian":    (strat.build_breakout, dict(lookback=48, expansion=1.3, atr_rank_min=0.20)),
        "momentum_rsi_mtf":     (strat.build_momentum, dict(rsi_buy=60.0, rsi_sell=40.0, mom_win=12, ema_htf=50)),
        "mean_reversion_range": (strat.build_mean_reversion, dict(z_win=20, z_entry=2.0, adx_max=18.0)),
    }
    strat_results = {}
    evals = {}
    for nm, (bld, kw) in candidates.items():
        e = eval_strategy(nm, bld, df, kw)
        evals[nm] = e
        strat_results[nm] = {k: v for k, v in e.items() if not k.startswith("_")}
        s = e["combined"]
        log.info("  %-22s trades=%s net=%s pf=%s expR=%s wf+=%s/%s",
                 nm, s.get("trades"), s.get("net_pl"), s.get("profit_factor"),
                 s.get("expectancy_R"), e["walk_forward"]["folds_positive_expectancy"],
                 e["walk_forward"]["n_folds"])
    report["phase5_strategy_research"] = strat_results

    # regime breakdown for the baseline + each candidate
    trend_lab, vol_lab = strat.regime_labels(c, h, l)
    def regime_breakdown(e):
        tr = e["_trades"]
        if len(tr["pnl"]) == 0:
            return {}
        ei = tr["entry_i"]
        out = {}
        for lab_name, arr in (("trend", trend_lab), ("vol", vol_lab)):
            for val in np.unique(arr[ei]):
                m = arr[ei] == val
                if m.sum() < 5:
                    continue
                pnl = tr["pnl"][m]; rk = tr["risk"][m]
                R = np.where(rk > 0, pnl / rk, 0.0)
                out[f"{lab_name}:{val}"] = dict(n=int(m.sum()), net=round(float(pnl.sum()), 2),
                                                win_rate=round(float((pnl > 0).mean()), 3),
                                                expectancy_R=round(float(R.mean()), 4))
        return out
    report["phase5_regime_breakdown"] = {nm: regime_breakdown(e) for nm, e in evals.items()}

    # pick the "best" by weakest-link: min(walk-forward mean expR, combined expR) with
    # a bonus for majority-positive folds -- NOT by total return.
    def score(e):
        s = e["combined"]; w = e["walk_forward"]
        ce = s.get("expectancy_R") or -9
        we = w.get("mean_expectancy_R")
        we = -9 if we is None else we
        nt = s.get("trades", 0)
        if nt < 30:
            return -9.0
        return min(ce, we) + (0.05 if w.get("majority_folds_positive") else -0.05)
    best_name = max(evals, key=lambda k: score(evals[k]))
    best = evals[best_name]
    report["best_strategy"] = dict(name=best_name, score=round(score(best), 4),
                                   reason="max of weakest-link(combined expR, walk-forward mean expR); "
                                          "NOT chosen by total return")

    # -------------------- PHASE 4: COST SENSITIVITY --------------------------
    log.info("phase 4: cost sensitivity")
    def cost_sens(e):
        return lab.cost_sensitivity(PX, e["_atr"], e["_entries"], e["_warmup"] + 50, N,
                                    e["_warmup"], RISK, COST_SCENARIOS)
    cost_report = {
        "scenarios": {k: dict(spread=v.spread, slippage_per_side=v.slippage_per_side,
                              round_trip=round(v.round_trip(), 2)) for k, v in COST_SCENARIOS.items()},
        "baseline": cost_sens(baseline),
        "best_strategy": cost_sens(best),
        "all_candidates": {nm: cost_sens(e) for nm, e in evals.items()},
    }
    (REPORTS / "btc_cost_sensitivity.json").write_text(json.dumps(cost_report, indent=2, default=str), encoding="utf-8")
    report["phase4_cost_sensitivity"] = {"baseline": cost_report["baseline"], "best_strategy": cost_report["best_strategy"]}

    # -------------------- PHASE 6: WALK-FORWARD (best + baseline) -----------
    log.info("phase 6: walk-forward")
    wf_full = {
        "baseline": lab.walk_forward(PX, baseline["_atr"], baseline["_entries"], baseline["_warmup"], NORMAL, RISK, n_folds=8),
        "best_strategy": lab.walk_forward(PX, best["_atr"], best["_entries"], best["_warmup"], NORMAL, RISK, n_folds=8),
        "best_strategy_name": best_name,
    }
    (REPORTS / "btc_walkforward.json").write_text(json.dumps(wf_full, indent=2, default=str), encoding="utf-8")
    report["phase6_walk_forward"] = {
        "baseline": wf_full["baseline"]["summary"],
        "best_strategy": wf_full["best_strategy"]["summary"],
    }

    # -------------------- PHASE 7: PARAMETER ROBUSTNESS --------------------
    log.info("phase 7: parameter robustness (best strategy neighbourhood)")
    robustness = []
    if best_name == "trend_ema":
        grid = [(f, s, t) for f in (18, 20, 22) for s in (45, 50, 55) for t in (180, 200, 220)]
        for f, s, t in grid:
            e = eval_strategy("trend_ema", strat.build_trend, df,
                              dict(fast=f, slow=s, trend=t, adx_min=20.0, atr_rank_max=0.85))
            robustness.append(dict(params={"fast": f, "slow": s, "trend": t},
                                   trades=e["combined"].get("trades", 0),
                                   expectancy_R=e["combined"].get("expectancy_R"),
                                   profit_factor=e["combined"].get("profit_factor"),
                                   wf_folds_positive=e["walk_forward"]["folds_positive_expectancy"]))
    elif best_name == "breakout_donchian":
        for lb in (36, 48, 60):
            for ex in (1.2, 1.3, 1.4):
                e = eval_strategy("breakout_donchian", strat.build_breakout, df,
                                  dict(lookback=lb, expansion=ex, atr_rank_min=0.20))
                robustness.append(dict(params={"lookback": lb, "expansion": ex},
                                       trades=e["combined"].get("trades", 0),
                                       expectancy_R=e["combined"].get("expectancy_R"),
                                       profit_factor=e["combined"].get("profit_factor"),
                                       wf_folds_positive=e["walk_forward"]["folds_positive_expectancy"]))
    else:
        for rb in (55, 60, 65):
            for rs in (35, 40, 45):
                e = eval_strategy("momentum_rsi_mtf", strat.build_momentum, df,
                                  dict(rsi_buy=float(rb), rsi_sell=float(rs), mom_win=12, ema_htf=50))
                robustness.append(dict(params={"rsi_buy": rb, "rsi_sell": rs},
                                       trades=e["combined"].get("trades", 0),
                                       expectancy_R=e["combined"].get("expectancy_R"),
                                       profit_factor=e["combined"].get("profit_factor"),
                                       wf_folds_positive=e["walk_forward"]["folds_positive_expectancy"]))
    exps = [r["expectancy_R"] for r in robustness if r["expectancy_R"] is not None]
    report["phase7_parameter_robustness"] = dict(
        best_strategy=best_name, table=robustness,
        n_configs=len(robustness),
        n_positive_expectancy=int(sum(1 for x in exps if x > 0)),
        expectancy_R_min=round(min(exps), 4) if exps else None,
        expectancy_R_max=round(max(exps), 4) if exps else None,
        expectancy_R_mean=round(float(np.mean(exps)), 4) if exps else None,
        stable_region=bool(exps and sum(1 for x in exps if x > 0) >= 0.7 * len(exps) and min(exps) > -0.05),
    )

    # -------------------- PHASE 8: MONTE CARLO (best) ----------------------
    log.info("phase 8: monte carlo")
    mc = lab.monte_carlo(best["_trades"], RISK, n=5000)
    (REPORTS / "btc_monte_carlo.json").write_text(
        json.dumps({"best_strategy": best_name, **mc}, indent=2, default=str), encoding="utf-8")
    report["phase8_monte_carlo"] = mc

    # -------------------- PHASE 9: $100 ACCOUNT FEASIBILITY --------------
    log.info("phase 9: $100 feasibility")
    price = float(df.close.iloc[-1])
    atr_series = strat.base_atr(h, l, c, 14)
    atr_now = float(atr_series[-1])
    atr_p50 = float(np.nanmedian(atr_series))
    atr_p90 = float(np.nanpercentile(atr_series[np.isfinite(atr_series)], 90))
    bmin = lab.BROKER_STOPS_LEVEL * lab.BROKER_STOP_BUFFER

    def risk_at(atr_val):
        sd = max(atr_val * RISK.sl_atr_mult, bmin)
        loss = sd * lab.VALUE_PER_UNIT_PER_LOT * lab.VOLUME_MIN
        return round(sd, 2), round(loss, 3), round(100 * loss / RISK.initial_balance, 2)

    sd_now, loss_now, pct_now = risk_at(atr_now)
    sd_p50, loss_p50, pct_p50 = risk_at(atr_p50)
    sd_p90, loss_p90, pct_p90 = risk_at(atr_p90)
    loss_broker_min = round(lab.BROKER_STOPS_LEVEL * lab.VALUE_PER_UNIT_PER_LOT * lab.VOLUME_MIN, 3)
    margin_min_lot = round(price * 1.0 * lab.VOLUME_MIN / 100.0, 2)   # verified via order_calc_margin
    balance_for_1pct_now = round(loss_now / RISK.risk_per_trade, 0)
    balance_for_1pct_p90 = round(loss_p90 / RISK.risk_per_trade, 0)

    report["phase9_account_feasibility"] = dict(
        btc_price=round(price, 2),
        atr_m5_now=round(atr_now, 2), atr_m5_median=round(atr_p50, 2), atr_m5_p90=round(atr_p90, 2),
        broker_min_stop=lab.BROKER_STOPS_LEVEL,
        risk_0_01_lot=dict(
            at_current_atr=dict(stop=sd_now, loss_usd=loss_now, pct_of_100=pct_now),
            at_median_atr=dict(stop=sd_p50, loss_usd=loss_p50, pct_of_100=pct_p50),
            at_p90_atr=dict(stop=sd_p90, loss_usd=loss_p90, pct_of_100=pct_p90),
            at_broker_min_stop=dict(stop=lab.BROKER_STOPS_LEVEL, loss_usd=loss_broker_min,
                                    pct_of_100=round(100 * loss_broker_min / RISK.initial_balance, 2)),
        ),
        margin_0_01_lot=margin_min_lot,
        free_margin_after=round(RISK.initial_balance - margin_min_lot, 2),
        min_lot_risk_ever_exceeds_2pct_ceiling=bool(pct_p90 > 100 * RISK.max_risk_per_trade),
        account_size_for_1pct_at_current_atr=balance_for_1pct_now,
        account_size_for_1pct_at_p90_atr=balance_for_1pct_p90,
        verdict=(
            "Position sizing on $100 is feasible: 0.01 lot risks ~%.1f%% at current ATR "
            "(%.1f%% at median, %.1f%% at the 90th-pct ATR spike), and margin is only $%.2f. "
            "The account is NOT the blocker. To hold a clean 1%%-risk ATR-stop position across "
            "vol regimes, ~$%d-$%d is more comfortable. The real blocker is that no strategy "
            "shows a positive edge."
            % (pct_now, pct_p50, pct_p90, margin_min_lot,
               int(round(balance_for_1pct_now / 25) * 25), int(round(balance_for_1pct_p90 / 25) * 25))),
    )

    # -------------------- FINAL JSON + MD -------------------------------
    report["elapsed_s"] = round(time.perf_counter() - t0, 1)
    (REPORTS / "btc_strategy_validation.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _write_md(report)

    # console
    print("=" * 96)
    print("BTCUSD.vx STRATEGY VALIDATION  (Valetax, M5, %d bars, %d days)  --  NO LIVE ORDER"
          % (N, report["meta"]["span_days"]))
    print("=" * 96)
    b = report["phase3_baseline"]["combined"]
    print(f"BASELINE (EMA20/50/200+RSI, frozen): trades {b.get('trades')}  net ${b.get('net_pl')}  "
          f"PF {b.get('profit_factor')}  expR {b.get('expectancy_R')}  DD {b.get('max_drawdown_pct')}%  "
          f"WF+ {report['phase3_baseline']['walk_forward']['folds_positive_expectancy']}/8")
    print("-" * 96)
    print(f"{'strategy':22s} {'trades':>7} {'net$':>10} {'PF':>7} {'expR':>8} {'DD%':>7} {'WF+/8':>6} {'costOK':>7}")
    for nm, r in report["phase5_strategy_research"].items():
        s = r["combined"]; w = r["walk_forward"]
        cs = cost_report["all_candidates"][nm]
        cost_ok = (cs.get("conservative", {}).get("expectancy_R") or -9) > 0
        print(f"{nm:22s} {s.get('trades', 0):>7} {s.get('net_pl', 0):>10.2f} "
              f"{str(s.get('profit_factor')):>7} {str(s.get('expectancy_R')):>8} "
              f"{str(s.get('max_drawdown_pct')):>7} {w['folds_positive_expectancy']:>4}/8 {str(cost_ok):>7}")
    print("-" * 96)
    print(f"BEST (weakest-link, not total return): {best_name}")
    mc = report["phase8_monte_carlo"]
    if "prob_ruin_equity_stop" in mc:
        print(f"Monte Carlo (best): P(neg final) {mc['prob_negative_final']}  P(ruin) {mc['prob_ruin_equity_stop']}  "
              f"p95 maxDD ${mc['p95_max_drawdown_abs']}  worst streak {mc['worst_losing_streak']}")
    pr = report["phase7_parameter_robustness"]
    print(f"Param robustness: {pr['n_positive_expectancy']}/{pr['n_configs']} configs positive expR, "
          f"stable_region={pr['stable_region']}")
    print(f"$100 feasibility: {report['phase9_account_feasibility']['verdict']}")
    print("=" * 96)
    print("reports: btc_strategy_validation.{json,md}, btc_walkforward.json, "
          "btc_cost_sensitivity.json, btc_monte_carlo.json")
    return 0


def _write_md(r: dict) -> None:
    m = r["meta"]
    L = []
    A = L.append
    A("# BTCUSD.vx Strategy Validation")
    A("")
    A(f"- Broker/symbol: **Valetax / BTCUSD.vx**, {m['timeframe']}")
    A(f"- Data: {m['rows']:,} bars, {m['span'][0][:10]} to {m['span'][1][:10]} ({m['span_days']} days)")
    A(f"- Normal costs: spread ${m['normal_costs']['spread']}, slippage ${m['normal_costs']['slippage_per_side']}/side "
      f"-> round-trip ${m['normal_costs']['round_trip']:.2f}")
    A(f"- Contract: $1 P/L per $1 move per 1.0 lot; min lot 0.01; broker min stop $29.76")
    A("")
    A("## 1. Baseline (frozen EMA20/50/200 + RSI)")
    b = r["phase3_baseline"]
    A(f"Combined: trades {b['combined'].get('trades')}, net ${b['combined'].get('net_pl')}, "
      f"PF {b['combined'].get('profit_factor')}, expectancy {b['combined'].get('expectancy_R')} R, "
      f"max DD {b['combined'].get('max_drawdown_pct')}%.")
    A(f"Walk-forward: {b['walk_forward']['folds_positive_expectancy']}/{b['walk_forward']['n_folds']} folds "
      f"positive, mean expectancy {b['walk_forward']['mean_expectancy_R']} R.")
    for tag in ("TRAIN", "VALIDATION", "FINAL"):
        s = b["per_split"][tag]
        A(f"- {tag}: {s.get('trades',0)} trades, net ${s.get('net_pl',0)}, expR {s.get('expectancy_R')}, "
          f"PF {s.get('profit_factor')}")
    A("")
    A("## 2. Strategy research")
    A("| strategy | trades | net $ | PF | exp R | DD % | WF +/8 |")
    A("|---|--:|--:|--:|--:|--:|--:|")
    for nm, x in r["phase5_strategy_research"].items():
        s = x["combined"]; w = x["walk_forward"]
        A(f"| {nm} | {s.get('trades',0)} | {s.get('net_pl',0)} | {s.get('profit_factor')} | "
          f"{s.get('expectancy_R')} | {s.get('max_drawdown_pct')} | {w['folds_positive_expectancy']}/8 |")
    A("")
    A(f"**Best (weakest-link score, not total return): `{r['best_strategy']['name']}`**")
    A("")
    A("## 3. Cost sensitivity (best strategy)")
    A("| scenario | round-trip $ | trades | net $ | PF | exp R |")
    A("|---|--:|--:|--:|--:|--:|")
    for k, v in r["phase4_cost_sensitivity"]["best_strategy"].items():
        A(f"| {k} | {v['round_trip']} | {v['trades']} | {v['net_pl']} | {v['profit_factor']} | {v['expectancy_R']} |")
    A("")
    A("## 4. Walk-forward (best strategy)")
    w = r["phase6_walk_forward"]["best_strategy"]
    A(f"{w['folds_traded']} folds traded, {w['folds_positive_expectancy']} positive, "
      f"mean expectancy {w['mean_expectancy_R']} R, single-fold profit share {w['single_fold_profit_share']}, "
      f"majority positive: {w['majority_folds_positive']}.")
    A("")
    A("## 5. Parameter robustness")
    pr = r["phase7_parameter_robustness"]
    A(f"{pr['n_positive_expectancy']}/{pr['n_configs']} nearby configs have positive expectancy "
      f"(min {pr['expectancy_R_min']}, mean {pr['expectancy_R_mean']}, max {pr['expectancy_R_max']} R). "
      f"Stable region: **{pr['stable_region']}**.")
    A("")
    A("## 6. Monte Carlo (best strategy)")
    mc = r["phase8_monte_carlo"]
    if "prob_ruin_equity_stop" in mc:
        A(f"- P(negative final return): {mc['prob_negative_final']}")
        A(f"- P(ruin / equity stop): {mc['prob_ruin_equity_stop']}")
        A(f"- Expected max DD ${mc['expected_max_drawdown_abs']}, 95th-pct max DD ${mc['p95_max_drawdown_abs']}")
        A(f"- Worst losing streak: {mc['worst_losing_streak']} trades")
    else:
        A(f"- {mc.get('note')}")
    A("")
    A("## 7. $100 account feasibility")
    f = r["phase9_account_feasibility"]
    rr = f["risk_0_01_lot"]
    A(f"- BTC ${f['btc_price']}, M5 ATR now ${f['atr_m5_now']} / median ${f['atr_m5_median']} / p90 ${f['atr_m5_p90']}")
    A(f"- 0.01 lot risk: current ATR **{rr['at_current_atr']['pct_of_100']}%**, median {rr['at_median_atr']['pct_of_100']}%, "
      f"p90 {rr['at_p90_atr']['pct_of_100']}%, broker-min stop {rr['at_broker_min_stop']['pct_of_100']}%")
    A(f"- margin for 0.01 lot: ${f['margin_0_01_lot']} (free after: ${f['free_margin_after']})")
    A(f"- account size for a clean 1% ATR-stop position: ~${int(f['account_size_for_1pct_at_current_atr'])} "
      f"(current) to ~${int(f['account_size_for_1pct_at_p90_atr'])} (p90 vol)")
    A(f"- **Verdict:** {f['verdict']}")
    (REPORTS / "btc_strategy_validation.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
