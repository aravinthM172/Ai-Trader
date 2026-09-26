"""
BTCUSD.vx H1 LIVE trader (momentum_rsi_mtf, frozen).  See execution/live.py.

    python run_live.py              one pass
    python run_live.py --loop 30    production loop (every 30 s)

Sends real orders ONLY when LIVE_TRADING=true in .env, the terminal's Algo Trading
button is ON and state/KILL_SWITCH does not exist.  Otherwise it is a dry run that
logs what it WOULD send.  Emergency stop: create state/KILL_SWITCH (any content).
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from common.logging_setup import get_logger
from execution import live, safety

log = get_logger("btc.live", filename="btc_live.log")


def _print(st: dict) -> None:
    a = st.get("account", {})
    e = st.get("last_entry_check", {})
    print(f"[{st['generated_utc'][:19]}] {st['mode']}  kill={st['kill_switch_active']}  "
          f"bal ${a.get('balance')}  peak ${a.get('peak_balance')}  floor ${a.get('drawdown_floor')}  "
          f"open {len(a.get('open_positions') or [])}  closed {st['closed_trades']} "
          f"net ${st['net_pl_usd']}  expR {st['expectancy_R']}")
    print("   entry: " + json.dumps(e, default=str)[:300])


def _keep_awake() -> None:
    """Windows: stop idle sleep while this process runs (released automatically on exit).
    The display may still turn off; MT5 and the bot keep running."""
    if sys.platform != "win32":
        return
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    if not ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED):
        log.warning("could not disable idle sleep -- set Windows sleep to 'Never'")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", type=int, default=0, help="seconds between passes (0 = one pass)")
    args = ap.parse_args()
    if args.loop > 0:
        _keep_awake()
    print(f"BTC H1 LIVE TRADER  LIVE_TRADING={safety.live_trading_enabled()}  "
          f"KILL_SWITCH={'ACTIVE' if safety.kill_switch_active() else 'clear'}  magic {live.MAGIC}")
    while True:
        try:
            _print(live.run_once())
        except KeyboardInterrupt:
            print("\nstopped.")
            return 0
        except Exception as e:
            log.exception("live pass failed: %s", e)
        if args.loop <= 0:
            return 0
        try:
            time.sleep(max(10, args.loop))
        except KeyboardInterrupt:
            print("\nstopped.")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
