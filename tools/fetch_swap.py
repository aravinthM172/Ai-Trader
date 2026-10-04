"""
Read the broker's real swap (overnight financing) settings from MT5 and translate
them into an annual rate comparable with backtest/swap_sensitivity.py.

    python -m tools.fetch_swap            # needs the MT5 terminal running + logged in

Writes reports/swap_rates.json.  Read-only: no orders.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mt5.gateway import MT5Gateway

SYMBOLS = ("BTCUSD.vx", "XAUUSD.vx")
# MT5 SYMBOL_SWAP_MODE values
SWAP_MODES = {0: "disabled", 1: "points", 2: "base_currency", 3: "margin_currency",
              4: "deposit_currency", 5: "interest_current_price", 6: "interest_open_price",
              7: "reopen_current", 8: "reopen_bid"}
WEEKDAYS = ("sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday")


def annualise(info, side_value: float) -> float | None:
    """Approximate annual rate (fraction of notional) of one side's swap, per unit, per night."""
    mode = int(info.swap_mode)
    price = float(info.bid or info.last or 0.0)
    if price <= 0:
        return None
    if mode in (5, 6):                                      # already an annual interest %
        return side_value / 100.0
    if mode == 1:                                           # points per lot per night
        per_unit_money = side_value * float(info.point) * float(info.trade_tick_value) / float(info.trade_tick_size) \
            / float(info.trade_contract_size or 1.0)
        return per_unit_money * 365.0 / price
    if mode == 4:                                           # deposit currency (USD) per lot per night
        return side_value / float(info.trade_contract_size or 1.0) * 365.0 / price
    return None                                             # other modes: record raw values only


def main() -> int:
    gw = MT5Gateway()
    if not gw.connect():
        print("MT5 connection failed -- start the terminal and log in, then retry")
        return 1
    try:
        mt5 = gw.raw()
        out = {"fetched_utc": datetime.now(timezone.utc).isoformat(), "symbols": {}}
        for s in SYMBOLS:
            i = mt5.symbol_info(s)
            if i is None:
                out["symbols"][s] = {"error": "symbol not found"}
                continue
            triple = getattr(i, "swap_rollover3days", None)
            row = dict(swap_mode=SWAP_MODES.get(int(i.swap_mode), int(i.swap_mode)),
                       swap_long=float(i.swap_long), swap_short=float(i.swap_short),
                       triple_swap_day=WEEKDAYS[triple] if isinstance(triple, int) and 0 <= triple < 7 else triple,
                       price=float(i.bid),
                       annual_rate_long=annualise(i, float(i.swap_long)),
                       annual_rate_short=annualise(i, float(i.swap_short)))
            out["symbols"][s] = row
            print(f"{s}: mode={row['swap_mode']} long={row['swap_long']} short={row['swap_short']} "
                  f"triple={row['triple_swap_day']}  ~annual long {row['annual_rate_long']} short {row['annual_rate_short']}")
        (ROOT / "reports" / "swap_rates.json").write_text(json.dumps(out, indent=2))
        print("compare with breakeven_annual_rate in reports/swap_sensitivity.json")
        return 0
    finally:
        gw.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
