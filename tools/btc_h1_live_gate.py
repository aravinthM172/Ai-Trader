"""
STEP 10 -- BTCUSD.vx H1 LIVE-TRADING readiness gate.  Read-only.  NEVER sends an order.

Extends the M5 gate framework (tools/btc_live_gate.py) for the H1 artefacts.
Sets LIVE_CANDIDATE=true ONLY if every research condition passes.  Even at
pass-all, LIVE_TRADING stays false because live execution / reconciliation is
not implemented.

    python -m tools.btc_h1_live_gate  ->  reports/btc_h1_live_gate.json
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPORTS = ROOT / "reports"


def _load(name):
    p = REPORTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> int:
    val = _load("btc_h1_validation.json")
    wf = _load("btc_h1_walkforward.json")
    cost = _load("btc_h1_cost_sensitivity.json")
    mc = _load("btc_h1_monte_carlo.json")
    ds = _load("btc_h1_dataset.json")

    checks: list[dict] = []

    def chk(name, ok, detail):
        checks.append({"condition": name, "pass": bool(ok), "detail": detail})

    if not val:
        chk("H1 validation artefacts present", False, "run: python -m backtest.btc_h1_validation")
    else:
        best = val["best_strategy"]["name"]
        b = val["step3_strategies"][best]
        bc, bw = b["combined"], b["walk_forward"]
        ea = val.get("edge_authenticity", {})
        f9 = val.get("step9_account_feasibility", {})
        cs_best = (val.get("step6_cost_sensitivity", {}).get("best_strategy", {}) or {})

        oos = b["per_split"]["FINAL"].get("expectancy_R")
        chk("1. positive out-of-sample (FINAL split) expectancy",
            (oos is not None and oos > 0), f"FINAL expectancy_R = {oos}")

        chk("2. positive walk-forward evidence (>= 6/8 folds, weakest > 0)",
            bw.get("folds_positive_expectancy", 0) >= 6 and (bw.get("min_expectancy_R") or -9) > 0,
            f"{bw.get('folds_positive_expectancy')}/8 folds, weakest {bw.get('min_expectancy_R')} R")

        pf = bc.get("profit_factor")
        chk("3. profit factor > 1 after realistic costs", (pf and pf > 1.0),
            f"combined PF (realistic costs) = {pf}")

        dd = bc.get("max_drawdown_pct")
        chk("4. acceptable drawdown (< 35%)", (dd is not None and dd < 35.0),
            f"combined max DD = {dd}%")

        conv = cs_best.get("conservative", {})
        strs = cs_best.get("stressed", {})
        chk("5. survives conservative AND stressed cost assumptions",
            (conv.get("expectancy_R") or -9) > 0 and (conv.get("profit_factor") or 0) > 1.0
            and (strs.get("expectancy_R") or -9) > 0,
            f"conservative expR {conv.get('expectancy_R')} PF {conv.get('profit_factor')}; "
            f"stressed expR {strs.get('expectancy_R')}")

        pr = val.get("step7_parameter_robustness", {})
        chk("6. parameter robustness (stable region, not a single combo)",
            pr.get("stable_region") and not pr.get("single_combination_only"),
            f"{pr.get('n_positive_expectancy')}/{pr.get('n_configs')} nearby configs positive, "
            f"stable_region={pr.get('stable_region')}")

        m = mc or {}
        chk("7. Monte Carlo survival (P(neg final) < 25%, P(ruin) < 5%, P(DD>50%) < 10%)",
            (m.get("prob_negative_final") is not None and m["prob_negative_final"] < 0.25
             and m.get("prob_ruin_equity_stop", 1) < 0.05
             and m.get("prob_drawdown_over_50pct", 1) < 0.10),
            f"P(neg) {m.get('prob_negative_final')}, P(ruin) {m.get('prob_ruin_equity_stop')}, "
            f"P(DD>50%) {m.get('prob_drawdown_over_50pct')}")

        n_tr = bc.get("trades", 0)
        chk("8. adequate sample size (>= 300 trades, >= 6 folds)",
            n_tr >= 300 and bw.get("n_folds", 0) >= 6, f"{n_tr} trades over {bw.get('n_folds')} folds")

        p90 = (f9.get("risk_0_01_lot", {}).get("p90", {}) or {}).get("pct_of_100")
        chk("9. $100-account risk acceptable at 0.01 lot (p90 vol <= 2%)",
            (p90 is not None and p90 <= 100 * 0.02),
            f"0.01 lot risk at p90 H1 ATR = {p90}% of $100 "
            f"(median { (f9.get('risk_0_01_lot',{}).get('median',{}) or {}).get('pct_of_100') }%)")

        paper = _load("btc_h1_paper_status.json") or _load("btc_paper_status.json")
        paper_ok = bool(paper and paper.get("closed_trades", 0) >= 20 and (paper.get("net_pl_usd") or -1) > 0)
        chk("10. paper-trading confirmation (>= 20 H1 trades, net > 0)", paper_ok,
            f"paper closed = {paper.get('closed_trades') if paper else 'none'} "
            f"(no H1 forward record yet -> run  python run_btc.py --paper  with H1 config)")

        edge = (ea.get("look_ahead_probe") == "no look-ahead detected"
                and ea.get("both_directions_profitable")
                and ea.get("random_direction_null", {}).get("direction_carries_information")
                and ea.get("fixed_direction_both_lose")
                and ea.get("all_years_positive"))
        chk("11. genuine economic edge (not curve-fit / not regime drift)", edge,
            f"look-ahead={ea.get('look_ahead_probe')}, both-dir-profitable={ea.get('both_directions_profitable')}, "
            f"beats-random-dir={ea.get('random_direction_null',{}).get('direction_carries_information')}, "
            f"always-L/S-both-lose={ea.get('fixed_direction_both_lose')}, all-years+={ea.get('all_years_positive')}")

    passed = sum(1 for c in checks if c["pass"])
    total = len(checks)
    live_candidate = bool(total and passed == total)

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": "BTCUSD.vx", "broker": "Valetax", "timeframe": "H1",
        "LIVE_CANDIDATE": live_candidate,
        "checks_passed": passed, "checks_total": total,
        "checks": checks,
        "failing": [c["condition"] for c in checks if not c["pass"]],
        "LIVE_TRADING_env": "false (unchanged)",
        "order_send_implemented": False,
        "hard_rule": "Even at 11/11, LIVE_TRADING stays false: live order execution + "
                     "post-fill reconciliation is not implemented. This gate is for MANUAL review only.",
        "action": ("H1 shows a research-grade edge. Next: (a) build the H1 paper-trading config and "
                   "run  python run_btc.py --paper --loop 3600  for >= 4 weeks; (b) size for a "
                   ">= $1500 account; (c) implement + review order_send; (d) re-run this gate on "
                   "fresh out-of-sample data."
                   if passed >= max(1, total - 2) else
                   "Address the failing conditions; keep researching. LIVE_TRADING stays false."),
    }
    (REPORTS / "btc_h1_live_gate.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")

    print("=" * 96)
    print("BTC H1 LIVE-TRADING READINESS GATE  --  read-only, no order sent")
    print("=" * 96)
    for c in checks:
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['condition']}")
        print(f"         {c['detail']}")
    print("-" * 96)
    print(f"  {passed}/{total} conditions pass")
    print(f"  LIVE_CANDIDATE = {live_candidate}")
    print(f"  LIVE_TRADING   = false (unchanged)   order_send implemented = False")
    print(f"  {out['hard_rule']}")
    print(f"  NEXT: {out['action']}")
    print("=" * 96)
    return 0 if live_candidate else 1


if __name__ == "__main__":
    raise SystemExit(main())
