"""
BTC bot -- SAFE / DRY-RUN + PAPER entry point.

order_send() is never called.  LIVE_TRADING must be explicitly true AND the
terminal AutoTrading enabled AND the kill switch clear before any live path
would even be considered (not implemented in this phase).

    python run_btc.py                                  one dry-run pass (M5)
    python run_btc.py --loop 300                       repeat every 300s
    python run_btc.py --paper                          paper forward-test, M5 (placeholder rule)
    python run_btc.py --paper --timeframe H1 --loop 300   VALIDATED momentum_rsi_mtf on H1
    python run_btc.py --sell                           evaluate the SELL side

--timeframe {M5,M15,H1,H4}.  H1/H4 automatically select the validated
`momentum_rsi_mtf` strategy (frozen parameters); M5 keeps the placeholder rule.
"""
from __future__ import annotations

import argparse
import time

from common.logging_setup import get_logger
from config.assets import get_asset_config
from execution import safety
from execution.pipeline import run_pipeline

log = get_logger("btc.run", filename="btc_run.log")

_VALID_TF = ("M5", "M15", "H1", "H4")


def _banner(mode: str, tf: str, asset: str = "BTC") -> None:
    cfg = get_asset_config(asset, timeframe=tf)
    print("=" * 88)
    print(f" {asset} BOT  --  {mode}  (no order_send)")
    print("=" * 88)
    print(f" TIMEFRAME : {cfg.timeframe}")
    print(f" STRATEGY  : {cfg.strategy}")
    print(f" LIVE_TRADING={str(safety.live_trading_enabled()).lower()}   "
          f"KILL_SWITCH={'ACTIVE' if safety.kill_switch_active() else 'clear'}")
    if safety.kill_switch_active():
        print(" *** KILL SWITCH ACTIVE -> all trading blocked ***")


def one_pass(force_direction, bars, tf) -> dict:
    _banner("SAFE / DRY-RUN MODE", tf)
    res = run_pipeline("BTC", timeframe=tf, history_bars=bars,
                       force_direction=force_direction, record_state=True)
    print()
    print(f" data: {res.get('timeframe')}  {res.get('history_bars_returned')} bars  "
          f"signal_bar {res.get('signal_bar_utc')}")
    for s in res["stages"]:
        print("  " + s)
    print()
    if res.get("order_validation_render"):
        print(res["order_validation_render"])
    print(f"\n DECISION: {res.get('decision')}  |  {res.get('decision_reason', '')}")
    if res.get("error"):
        print(f" ERROR: {res['error']}")
    print(" NO LIVE ORDER WAS SENT.")
    log.info("dry-run pass tf=%s strategy=%s decision=%s", res.get("timeframe"),
             res.get("strategy"), res.get("decision"))
    return res


def replay_pass(csv_path, sim_balance, asset="BTC") -> dict:
    """MT5-FREE replay from an exported candle CSV (for Oracle Cloud / any Linux box)."""
    from execution.paper_replay import run_from_csv
    _banner("PAPER REPLAY (MT5-free, from CSV export)", "H1", asset)
    if sim_balance:
        print(f" paper sizing balance: ${sim_balance:.0f}  (real account untouched; paper only)")
    st = run_from_csv(csv_path, asset=asset, sim_balance=sim_balance)
    if st.get("error"):
        print(f" ERROR: {st['error']}")
        return st
    fs = st["forward_out_of_sample"]
    isamp = st["in_sample"]
    print(f" export    : {st.get('export_utc')}   candles {st['candles']}  {st['candle_span'][0][:16]} -> {st['candle_span'][1][:16]}")
    print(f" cutoff    : {st['validation_cutoff_utc']}  (trades after this = FORWARD / out-of-sample)")
    print(f" spread    : {st['spread_source']}")
    print(f" in-sample : {isamp.get('trades')} trades  net ${isamp.get('net_pl_usd')}  "
          f"PF {isamp.get('profit_factor')}  expR {isamp.get('expectancy_R')}")
    print(f" FORWARD   : {fs.get('trades')} trades  net ${fs.get('net_pl_usd')}  "
          f"PF {fs.get('profit_factor')}  expR {fs.get('expectancy_R')}  "
          f"maxDD ${fs.get('max_drawdown_usd')}  maxConsecL {fs.get('max_consecutive_losses')}")
    print(f" reports   : {st.get('asset','BTC').lower()}_h1_paper_status.json  +  {st.get('asset','BTC').lower()}_paper_replay_trades.csv  (under reports/)")
    print(" NO LIVE ORDER (MetaTrader5 is not even imported in replay).")
    log.info("replay pass forward_trades=%s forward_expR=%s", fs.get("trades"), fs.get("expectancy_R"))
    return st


def paper_pass(ignore_spread, bars, force, tf, sim_balance) -> dict:
    from execution.paper import run_once
    _banner("PAPER / FORWARD-TEST MODE", tf)
    if ignore_spread:
        print(" *** --paper-ignore-spread: MECHANICS TEST ONLY, not the real forward test ***")
    if sim_balance:
        print(f" paper sizing balance: ${sim_balance:.0f}  (real account stays $100; paper only)")
    st = run_once(timeframe=tf, ignore_spread=ignore_spread, bars=bars, force_direction=force,
                  sim_balance=sim_balance)
    if st.get("error"):
        print(f" ERROR: {st['error']}")
        return st
    lp = st.get("last_pass", {})
    print(f" data      : {lp.get('timeframe')}  {lp.get('history_bars_returned')} bars returned")
    print(f" signal    : {lp.get('signal')} (conf {lp.get('signal_confidence')})  "
          f"bar {lp.get('signal_bar_utc')}  ATR ${lp.get('atr')}")
    print(f"             {lp.get('signal_reason')}")
    print(f" decision  : {lp.get('decision')}   spread_ok={lp.get('spread_filter_ok')} "
          f"(s/ATR {lp.get('spread_to_atr')})")
    if lp.get("opened"):
        print(f" >>> PAPER OPEN : {lp['opened']}")
    if lp.get("closed"):
        print(f" >>> PAPER CLOSE: {lp['closed']}")
    print(f" positions : {len(st.get('open_positions', []))} open   {st.get('closed_trades')} closed   "
          f"net ${st.get('net_pl_usd')}  expR {st.get('expectancy_R')}   "
          f"daily {st.get('daily')}  weekly {st.get('weekly')}")
    print(f" report    : reports/btc_paper_status_{lp.get('timeframe')}.json   |   NO LIVE ORDER WAS SENT.")
    log.info("paper pass tf=%s strategy=%s decision=%s opened=%s closed=%s",
             lp.get("timeframe"), lp.get("strategy"), lp.get("decision"),
             bool(lp.get("opened")), bool(lp.get("closed")))
    return st


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", type=int, default=0, help="seconds between passes (0 = single pass)")
    ap.add_argument("--timeframe", "--tf", dest="timeframe", default="M5", choices=_VALID_TF,
                    help="candle timeframe; H1/H4 use the validated momentum_rsi_mtf strategy")
    ap.add_argument("--sell", action="store_true")
    ap.add_argument("--buy", action="store_true")
    ap.add_argument("--bars", type=int, default=0, help="history bars (0 = auto per timeframe)")
    ap.add_argument("--paper", action="store_true", help="paper / forward-test mode (persists hypothetical trades)")
    ap.add_argument("--paper-ignore-spread", action="store_true",
                    help="paper mode only: test strategy mechanics with the spread filter off (NOT the real forward test)")
    ap.add_argument("--paper-balance", type=float, default=0.0,
                    help="paper mode only: hypothetical sizing balance (e.g. 1500 for a realistic H1 test); "
                         "the real $100 account and every live gate are untouched")
    ap.add_argument("--from-csv", type=str, default=None,
                    help="paper mode: MT5-free replay from a tools/export_h1.py CSV (for Oracle Cloud / Linux)")
    ap.add_argument("--asset", choices=["BTC", "XAU"], default="BTC",
                    help="replay asset (--from-csv only): BTCUSD.vx or XAUUSD.vx -- both run the validated momentum_rsi_mtf H1")
    args = ap.parse_args()
    force = "SELL" if args.sell else ("BUY" if args.buy else None)
    tf = args.timeframe.upper()
    bars = args.bars or None
    sim_bal = args.paper_balance or None

    if args.paper and args.from_csv:
        runner = lambda: replay_pass(args.from_csv, sim_bal, args.asset)
    elif args.paper:
        runner = lambda: paper_pass(args.paper_ignore_spread, bars, force, tf, sim_bal)
    else:
        runner = lambda: one_pass(force, bars, tf)

    if args.loop <= 0:
        res = runner()
        return 0 if (args.paper or res.get("ready")) else 1

    while True:
        try:
            runner()
        except KeyboardInterrupt:
            print("\nstopped.")
            return 0
        except Exception as e:
            log.exception("pass failed: %s", e)
        time.sleep(max(30, args.loop))


if __name__ == "__main__":
    raise SystemExit(main())
