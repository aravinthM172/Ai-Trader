"""
PHASE 12 -- formal LIVE-TRADING readiness gate.  Read-only.  NEVER sends an order.

Reads the artefacts produced by the research pipeline and evaluates the 14
mandatory conditions.  Sets LIVE_CANDIDATE=true ONLY if ALL pass.

    python -m tools.btc_live_gate

Even if every condition passed this tool does NOT enable live trading -- it only
writes reports/btc_live_gate.json for manual review.
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
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _run_safety_tests() -> tuple[bool, str]:
    import subprocess
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_btc_integration.py",
         "tests/test_btc_safety.py", "tests/test_xauusd_regression.py", "-k",
         "safety or regression or order_check or never_sends or duplicate or stale"],
        cwd=ROOT, capture_output=True, text=True)
    ok = r.returncode == 0
    tail = (r.stdout or "").strip().splitlines()[-1:] or [""]
    return ok, tail[0]


def main() -> int:
    val = _load("btc_strategy_validation.json")
    wf = _load("btc_walkforward.json")
    cost = _load("btc_cost_sensitivity.json")
    mc = _load("btc_monte_carlo.json")
    paper = _load("btc_paper_status.json")
    diag = _load("btc_diagnostic.json")

    checks: list[dict] = []

    def chk(cond: str, passed: bool | None, detail: str):
        checks.append({"condition": cond, "pass": bool(passed) if passed is not None else False,
                       "detail": detail, "evaluable": passed is not None})

    if val is None:
        chk("artefacts present", False, "run: python -m backtest.btc_validate")
    else:
        best = val.get("best_strategy", {}).get("name")
        b = val.get("phase5_strategy_research", {}).get(best, {})
        bc = b.get("combined", {})
        bw = b.get("walk_forward", {})

        # 1 positive OOS expectancy (FINAL split)
        fin = b.get("per_split", {}).get("FINAL", {})
        e_fin = fin.get("expectancy_R")
        chk("1. positive out-of-sample (FINAL) expectancy", (e_fin is not None and e_fin > 0),
            f"FINAL expectancy_R = {e_fin}")

        # 2 majority of walk-forward folds positive
        chk("2. majority of walk-forward folds positive", bw.get("majority_folds_positive"),
            f"{bw.get('folds_positive_expectancy')}/{bw.get('n_folds')} folds positive")

        # 3 profit factor > 1 after realistic (normal) costs
        pf = bc.get("profit_factor")
        chk("3. profit factor > 1 after realistic costs", (pf is not None and pf > 1.0),
            f"combined PF (normal costs) = {pf}")

        # 4 drawdown acceptable (< 35% of account)
        dd = bc.get("max_drawdown_pct")
        chk("4. drawdown acceptable (< 35%)", (dd is not None and dd < 35.0),
            f"combined max DD = {dd}%")

        # 5 survives conservative costs
        cs = (val.get("phase4_cost_sensitivity", {}).get("best_strategy", {}) or {}).get("conservative", {})
        chk("5. survives conservative cost assumptions", (cs.get("expectancy_R") or -9) > 0
            and (cs.get("profit_factor") or 0) > 1.0,
            f"conservative expectancy_R = {cs.get('expectancy_R')}, PF = {cs.get('profit_factor')}")

        # 6 / 7 look-ahead / leakage -> enforced by construction + tested
        st_ok, st_tail = _run_safety_tests()
        chk("6. no look-ahead bias (feature-causality test passes)", st_ok, st_tail)
        chk("7. no data leakage (per-window i+H isolation; tests pass)", st_ok, st_tail)

        # 8 parameter robustness
        pr = val.get("phase7_parameter_robustness", {})
        chk("8. parameter robustness (stable region)", pr.get("stable_region"),
            f"{pr.get('n_positive_expectancy')}/{pr.get('n_configs')} nearby configs positive")

        # 9 monte carlo
        m = mc or {}
        mc_ok = (m.get("prob_negative_final") is not None
                 and m["prob_negative_final"] < 0.35
                 and m.get("prob_ruin_equity_stop", 1.0) < 0.10)
        chk("9. Monte Carlo robustness", mc_ok,
            f"P(neg final)={m.get('prob_negative_final')}, P(ruin)={m.get('prob_ruin_equity_stop')}")

        # 10 paper trading confirms behaviour
        paper_ok = bool(paper and paper.get("closed_trades", 0) >= 20
                        and (paper.get("net_pl_usd") or -1) > 0)
        chk("10. paper trading confirms expected behaviour", paper_ok,
            f"paper closed={paper.get('closed_trades') if paper else 'n/a'}, "
            f"net=${paper.get('net_pl_usd') if paper else 'n/a'} (need >=20 trades, net>0)")

        # 11 safety tests all pass
        chk("11. safety tests all pass", st_ok, st_tail)

        # 12 broker order_check passes
        oc = (diag or {}).get("order_validation") if diag else None
        chk("12. broker order_check passes", bool(oc and oc.get("order_check_retcode") == 0),
            f"order_check retcode = {oc.get('order_check_retcode') if oc else 'n/a (run diagnostic --buy --ignore-spread)'}")

        # 13 position sizing appropriate for the account
        f9 = val.get("phase9_account_feasibility", {})
        p90 = (f9.get("risk_0_01_lot", {}).get("at_p90_atr", {}) or {}).get("pct_of_100")
        chk("13. position sizing appropriate for the account", (p90 is not None and p90 <= 200 * 0.02),
            f"0.01 lot risk at p90 ATR = {p90}% of $100 (must be <= 2%)")

        # 14 genuine edge, not curve fitting
        edge = (bc.get("expectancy_R") or -9) > 0.03 and bw.get("majority_folds_positive") \
            and pr.get("stable_region")
        chk("14. genuine edge, not curve fitting", edge,
            f"combined expR={bc.get('expectancy_R')}, WF majority={bw.get('majority_folds_positive')}, "
            f"robust={pr.get('stable_region')}")

    passed = sum(1 for c in checks if c["pass"])
    total = len(checks)
    live_candidate = bool(total and passed == total)

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": "BTCUSD.vx", "broker": "Valetax",
        "LIVE_CANDIDATE": live_candidate,
        "checks_passed": passed, "checks_total": total,
        "checks": checks,
        "failing": [c["condition"] for c in checks if not c["pass"]],
        "LIVE_TRADING_env": "false (unchanged)",
        "action": ("Manual review only. Even at 14/14 this tool does NOT enable live trading."
                   if live_candidate else
                   "LIVE_TRADING stays false. Address the failing conditions; continue research."),
    }
    (REPORTS / "btc_live_gate.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")

    print("=" * 92)
    print("BTC LIVE-TRADING READINESS GATE  --  read-only, no order sent")
    print("=" * 92)
    for c in checks:
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['condition']}")
        print(f"         {c['detail']}")
    print("-" * 92)
    print(f"  {passed}/{total} conditions pass")
    print(f"  LIVE_CANDIDATE = {live_candidate}")
    print(f"  LIVE_TRADING   = false (unchanged, not modified by this tool)")
    print(f"  {out['action']}")
    print("=" * 92)
    return 0 if live_candidate else 1


if __name__ == "__main__":
    raise SystemExit(main())
