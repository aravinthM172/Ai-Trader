"""
Run this on the WINDOWS PC that has MT5 (Valetax) connected.

It exports everything the MT5-free replay on Oracle Cloud needs:
    data/export/<sym>_H1_export.csv     -- real H1 candles (time, OHLC, spread pts, volume)
    data/export/<sym>_H1_export.json    -- broker spec + account snapshot + export time

where <sym> is  btcusd_vx  or  xauusd_vx.

Then copy BOTH files to the cloud box, e.g.:
    scp data/export/btcusd_vx_H1_export.*  ubuntu@<oci-ip>:~/gold-ai-trader/data/export/

Historical spread is read from MT5's own column -- never fabricated.  No order is sent.

    python -m tools.export_h1                       # BTC, ~6000 H1 bars
    python -m tools.export_h1 --asset XAU --bars 24000
    python -m tools.export_h1 --asset BTC --bars 24000
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
from config.assets import get_asset_config
from mt5.gateway import MT5Gateway

log = get_logger("h1.export")
OUT = ROOT / "data" / "export"
OUT.mkdir(parents=True, exist_ok=True)

_ASSETS = {
    "BTC": ("BTCUSD.vx", "btcusd_vx"),
    "XAU": ("XAUUSD.vx", "xauusd_vx"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", choices=["BTC", "XAU"], default="BTC")
    ap.add_argument("--symbol", default=None, help="override the MT5 symbol (default: broker default for --asset)")
    ap.add_argument("--bars", type=int, default=6000)
    args = ap.parse_args()

    default_sym, slug = _ASSETS[args.asset]
    try:
        symbol = args.symbol or get_asset_config(args.asset).symbol
    except Exception:
        symbol = args.symbol or default_sym

    gw = MT5Gateway()
    if not gw.connect():
        print("MT5 connection failed -- is the Valetax terminal running and logged in?")
        return 1
    try:
        spec = gw.get_spec(symbol)
        acc = gw.account_info()
        tick = gw.get_tick(symbol)
        df = gw.get_rates(symbol, "H1", args.bars)
        if df is None or df.empty:
            print(f"no H1 data returned for {symbol}")
            return 1

        csv_path = OUT / f"{slug}_H1_export.csv"
        df.to_csv(csv_path, index=False)

        meta = {
            "export_utc": datetime.now(timezone.utc).isoformat(),
            "server_utc_offset_hours": gw.server_utc_offset_seconds // 3600,
            "asset": args.asset,
            "symbol": symbol,
            "bars": int(len(df)),
            "first_bar_utc": str(df["time"].iloc[0]),
            "last_bar_utc": str(df["time"].iloc[-1]),
            "spec": {
                "point": spec.point, "tick_size": spec.tick_size, "tick_value": spec.tick_value,
                "contract_size": spec.contract_size,
                "value_per_unit_per_lot": spec.value_per_price_unit_per_lot,
                "volume_min": spec.volume_min, "volume_step": spec.volume_step,
                "volume_max": spec.volume_max, "stops_level_points": spec.stops_level_points,
                "stops_level_price": spec.stops_level_price,
                "filling_mode": spec.filling_mode, "trade_exemode": spec.trade_exemode,
                "trade_mode": spec.trade_mode,
                "spread_points_now": spec.spread_points,
            },
            "account_snapshot": {
                "server": acc.get("server"), "currency": acc.get("currency"),
                "balance": acc.get("balance"), "equity": acc.get("equity"),
                "margin_free": acc.get("margin_free"), "leverage": acc.get("leverage"),
                "trade_allowed": acc.get("trade_allowed"),
            },
            "tick_snapshot": {"bid": tick["bid"], "ask": tick["ask"], "spread": tick["spread"],
                              "time_utc": tick.get("time_utc")} if tick else None,
        }
        json_path = OUT / f"{slug}_H1_export.json"
        json_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

        replay_asset = "" if args.asset == "BTC" else " --asset XAU"
        print("=" * 80)
        print(f"{symbol} H1 EXPORT (for Oracle Cloud replay)  --  NO ORDER SENT")
        print("=" * 80)
        print(f"bars      : {len(df):,}   {df['time'].iloc[0]} -> {df['time'].iloc[-1]}")
        print(f"spread    : median ${df['spread'].median() * spec.point:.2f}  "
              f"max ${df['spread'].max() * spec.point:.2f}  (broker column, not fabricated)")
        print(f"account   : {acc.get('server')}  ${acc.get('balance'):.2f}  "
              f"free ${acc.get('margin_free'):.2f}")
        print(f"files     : {csv_path.relative_to(ROOT)}")
        print(f"            {json_path.relative_to(ROOT)}")
        print()
        print("copy BOTH to the cloud box under  data/export/  then run there:")
        print(f"  python run_btc.py --paper --timeframe H1{replay_asset} "
              f"--from-csv data/export/{slug}_H1_export.csv --paper-balance 1500")
        print("=" * 80)
        log.info("exported %d H1 bars for %s -> %s", len(df), symbol, csv_path.name)
        return 0
    finally:
        gw.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
