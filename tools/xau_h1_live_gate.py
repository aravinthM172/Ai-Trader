"""
XAUUSD.vx H1 LIVE-TRADING readiness gate.  Read-only.  NEVER sends an order.

Same 11 conditions as tools/btc_h1_live_gate.py, keyed to the XAU report layout.
LIVE_CANDIDATE=true only if ALL pass.  Even at 11/11 LIVE_TRADING stays false
(live execution / reconciliation not implemented).

    python -m tools.xau_h1_live_gate  ->  reports/xau_h1_live_gate.json
"""
from __future__ import annotations

import json
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
    val = _load("xau_h1_validation.json")
    mc = _load("xau_h1_monte_carlo.json")
    checks: list[dict] = []

    def chk(name, ok, detail):
        checks.append({"condition": name, "pass": bool(ok), "detail": detail})

    if not val:
        chk("XAU validation artefacts present", False, "run: python -m backtest.xau_h1_validation")
    else:
        best = val["best_strategy"]["name"]
        b = val["strategies"][best]
        bc, bw = b["combined"], b["walk_forward"]
        ea = val.get("edge_authenticity", {})
        pr = val.get("parameter_robustness", {})
        cs = val.get("cost_sensitivity", {}) or {}
        f9 = val.get("account_feasibility", {})
        v = val.get("verdict", {})
        rb = val.get("regime_breakdown", {})

        oos = b["per_split"]["FINAL"].get("expectancy_R")
        chk("1. positive out-of-sample (FINAL split) expectancy", (oos is not None and oos > 0),
            f"FINAL expectancy_R = {oos}")
        chk("2. positive walk-forward evidence (>= 6/8 folds, weakest > 0)",
            bw.get("folds_positive_expectancy", 0) >= 6 and (bw.get("min_expectancy_R") or -9) > 0,
            f"{bw.get('folds_positive_expectancy')}/8 folds, weakest {bw.get('min_expectancy_R')} R")
        pf = bc.get("profit_factor")
        chk("3. profit factor > 1.15 after realistic costs", (pf and pf > 1.15),
            f"combined PF = {pf}")
        dd = bc.get("max_drawdown_pct")
        chk("4. acceptable drawdown (< 35%)", (dd is not None and dd < 35.0),
            f"combined max DD = {dd}%")
        conv = cs.get("conservative", {}); strs = cs.get("stressed", {})
        chk("5. survives conservative AND stressed cost assumptions",
            (conv.get("expectancy_R") or -9) > 0 and (conv.get("profit_factor") or 0) > 1.0
            and (strs.get("expectancy_R") or -9) > 0,
            f"conservative expR {conv.get('expectancy_R')}; stressed expR {strs.get('expectancy_R')}")
        chk("6. parameter robustness (stable region, not a single combo)",
            pr.get("stable_region") and not pr.get("single_combination_only"),
            f"{pr.get('n_positive_expectancy')}/{pr.get('n_configs')} nearby configs positive, "
            f"stable_region={pr.get('stable_region')}")
        m = mc or {}
        chk("7. Monte Carlo survival (P(neg) < 25%, P(ruin) < 5%, P(DD>50%) < 10%)",
            (m.get("prob_negative_final") is not None and m["prob_negative_final"] < 0.25
             and m.get("prob_ruin_equity_stop", 1) < 0.05
             and m.get("prob_drawdown_over_50pct", 1) < 0.10),
            f"P(neg) {m.get('prob_negative_final')}, P(ruin) {m.get('prob_ruin_equity_stop')}, "
            f"P(DD>50%) {m.get('prob_drawdown_over_50pct')}")
        n_tr = bc.get("trades", 0)
        chk("8. adequate sample size (>= 500 trades)", n_tr >= 500, f"{n_tr} trades over 8 folds")
        p90 = ((f9.get("risk_0_01_lot", {}) or {}).get("p90", {}) or {}).get("pct")
        chk("9. $100-account risk acceptable at 0.01 lot (p90 vol <= 2%)",
            (p90 is not None and p90 <= 100 * 0.02),
            f"0.01 lot risk at p90 H1 ATR = {p90}% of $100")
        paper = _load("xau_h1_paper_status.json")
        paper_ok = bool(paper and paper.get("forward_out_of_sample", {}).get("trades", 0) >= 20
                        and (paper.get("forward_out_of_sample", {}).get("net_pl_usd") or -1) > 0)
        chk("10. paper-trading confirmation (>= 20 forward XAU trades, net > 0)", paper_ok,
            "no XAU forward record yet -> run the replay with the XAU export")
        edge = bool(v.get("edge_found"))
        chk("11. genuine cross-regime edge (not curve-fit / not a single-trend rider)", edge,
            f"look-ahead={ea.get('look_ahead')}, both-dir-profitable={ea.get('both_directions_profitable')}, "
            f"beats-random-dir={ea.get('random_direction_null',{}).get('direction_carries_information')}, "
            f"all-regimes-positive={v.get('all_regimes_positive')}, years+={v.get('years_positive')}, "
            f"regimes={ {k: rb[k]['expectancy_R'] for k in rb} }")

    passed = sum(1 for c in checks if c["pass"])
    total = len(checks)
    live_candidate = bool(total and passed == total)
    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": "XAUUSD.vx", "broker": "Valetax", "timeframe": "H1",
        "LIVE_CANDIDATE": live_candidate,
        "checks_passed": passed, "checks_total": total,
        "checks": checks, "failing": [c["condition"] for c in checks if not c["pass"]],
        "LIVE_TRADING_env": "false (unchanged)", "order_send_implemented": False,
        "hard_rule": "Even at 11/11, LIVE_TRADING stays false: live execution + reconciliation "
                     "is not implemented. Manual review only.",
    }
    (REPORTS / "xau_h1_live_gate.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")

    print("=" * 96)
    print("XAU H1 LIVE-TRADING READINESS GATE  --  read-only, no order sent")
    print("=" * 96)
    for c in checks:
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['condition']}")
        print(f"         {c['detail']}")
    print("-" * 96)
    print(f"  {passed}/{total} pass   LIVE_CANDIDATE = {live_candidate}   LIVE_TRADING = false   order_send = not implemented")
    print("=" * 96)
    return 0 if live_candidate else 1


if __name__ == "__main__":
    raise SystemExit(main())
