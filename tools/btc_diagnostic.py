"""
Complete BTC readiness check.  NEVER sends a live order.

    python -m tools.btc_diagnostic            (BUY-side evaluation)
    python -m tools.btc_diagnostic --sell
    python -m tools.btc_diagnostic --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.logging_setup import get_logger
from execution.pipeline import run_pipeline

log = get_logger("btc.diagnostic")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sell", action="store_true", help="evaluate the SELL side")
    ap.add_argument("--buy", action="store_true", help="force the BUY side")
    ap.add_argument("--bars", type=int, default=3000)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--ignore-spread", action="store_true",
                    help="TEST ONLY: bypass the spread filter to exercise the order_check path")
    args = ap.parse_args()

    force = "SELL" if args.sell else ("BUY" if args.buy else None)

    print("=" * 84)
    print("BTC READINESS DIAGNOSTIC  (Valetax BTCUSD.vx)   --   NO LIVE ORDER IS SENT")
    print("=" * 84)

    res = run_pipeline("BTC", history_bars=args.bars, force_direction=force,
                       ignore_spread_filter=args.ignore_spread)

    out_path = Path(args.json) if args.json else (ROOT / "reports" / "btc_diagnostic.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")

    print(f"\ntime           : {res['timestamp_utc']}")
    print(f"symbol         : {res['symbol']}  tf {res['timeframe']}")
    print(f"LIVE_TRADING   : {res['live_trading_enabled']}   kill_switch: {res['kill_switch_active']}")

    acc = res.get("account", {})
    if acc:
        print(f"account        : {acc.get('server')}  bal ${acc.get('balance'):.2f}  "
              f"eq ${acc.get('equity'):.2f}  free ${acc.get('margin_free'):.2f}  "
              f"lev 1:{acc.get('leverage')}  trade_allowed {acc.get('trade_allowed')}")
        print(f"terminal       : trade_allowed {res.get('terminal', {}).get('trade_allowed')} "
              f"(AutoTrading button)")

    tick = res.get("tick")
    if tick:
        print(f"tick           : bid {tick['bid']}  ask {tick['ask']}  spread ${tick['spread']:.2f}  "
              f"age {tick.get('age_seconds', 0):.0f}s")

    dq = res.get("data_quality", {})
    if dq:
        print(f"history        : {dq.get('n_clean')} clean / {dq.get('n_rejected')} rejected  "
              f"{dq.get('earliest')} -> {dq.get('latest')}  ok={dq.get('ok')}")
        if dq.get("issues"):
            print(f"  data issues  : {dq['issues']}")

    if res.get("atr") is not None:
        spec = res.get("symbol_spec", {})
        print(f"ATR            : ${res['atr']:.2f}   natr {res.get('features', {}).get('natr')}")
        print(f"broker min stop: {spec.get('stops_level_points')} pts = "
              f"${spec.get('stops_level_price', 0):.2f}")

    sig = res.get("signal", {})
    if sig:
        print(f"signal         : {sig.get('decision')}  conf {sig.get('confidence')}  ({sig.get('reason')})")

    sf = res.get("spread_filter")
    if sf:
        print(f"spread filter  : spread/ATR {sf['spread_to_atr']} (max {sf['max_spread_atr_ratio']})  ok={sf['ok']}")

    plan = res.get("stop_plan")
    if plan:
        print(f"stop plan      : entry {plan['entry']}  SL {plan['sl']}  TP {plan['tp']}  "
              f"R/R {plan['reward_risk']}  accepted={plan['accepted']}  {plan['reasons']}")

    sz = res.get("sizing")
    if sz:
        print(f"sizing         : vol {sz['volume']} (raw {sz['raw_volume']:.4f})  "
              f"risk ${sz['risk_dollars']:.2f}  est loss ${sz['est_loss_at_sl']:.2f} "
              f"({sz['est_loss_fraction']:.1%})  margin ${sz['required_margin']}  "
              f"accepted={sz['accepted']}  {sz['reasons']}")

    safe = res.get("safety")
    if safe:
        print(f"safety         : allow_validation={safe['allow_validation']}  "
              f"allow_live_send={safe['allow_live_send']}  {safe['reasons']}")

    print()
    if res.get("order_validation_render"):
        print(res["order_validation_render"])

    print("\n" + "=" * 84)
    print("PIPELINE STAGES")
    print("=" * 84)
    for s in res["stages"]:
        print("  " + s)

    decision = res.get("decision", "UNKNOWN")
    print("\n" + "=" * 84)
    print(f"FINAL READINESS : {'READY (dry-run)' if res.get('ready') else 'NOT READY'}")
    print(f"TRADE DECISION  : {decision}")
    if res.get("decision_reason"):
        print(f"REASON          : {res['decision_reason']}")
    if res.get("error"):
        print(f"ERROR           : {res['error']}")
    print(f"report saved    : {out_path.relative_to(ROOT)}")
    print("NO LIVE ORDER WAS SENT.")
    print("=" * 84)

    return 0 if res.get("ready") else 1


if __name__ == "__main__":
    raise SystemExit(main())
