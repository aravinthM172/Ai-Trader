from pathlib import Path
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

SYMBOL = "BTC"
TIMEFRAME = mt5.TIMEFRAME_M5
BARS = 100000

out = Path(r"data\btc_m5_5y.csv")
out.parent.mkdir(exist_ok=True)

if not mt5.initialize():
    raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

if not mt5.symbol_select(SYMBOL, True):
    mt5.shutdown()
    raise RuntimeError(f"Could not select {SYMBOL}: {mt5.last_error()}")

rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, BARS)

if rates is None or len(rates) == 0:
    err = mt5.last_error()
    mt5.shutdown()
    raise RuntimeError(f"History download failed: {err}")

df = pd.DataFrame(rates)
df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

df.to_csv(out, index=False)

tick = mt5.symbol_info_tick(SYMBOL)
info = mt5.symbol_info(SYMBOL)

print("=" * 80)
print("BTC M5 HISTORY VALIDATION")
print("=" * 80)
print(f"Rows              : {len(df):,}")
print(f"From              : {df.time.min()}")
print(f"To                : {df.time.max()}")
print(f"File              : {out.resolve()}")

print("\nOHLC:")
print(f"NaN rows          : {df[['open','high','low','close']].isna().any(axis=1).sum()}")
print(f"Duplicate times   : {df.time.duplicated().sum()}")

bad = (
    (df.high < df.low) |
    (df.open < df.low) | (df.open > df.high) |
    (df.close < df.low) | (df.close > df.high)
)
print(f"Bad OHLC rows     : {bad.sum()}")

if "spread" in df.columns:
    sp = df["spread"].astype(float) * info.point
    print("\nSPREAD:")
    print(sp.quantile([0,.25,.50,.75,.90,.95,.99,1]).to_string())
    print(f"Current Bid       : {tick.bid}")
    print(f"Current Ask       : {tick.ask}")
    print(f"Current spread    : {tick.ask - tick.bid:.6f}")

print("\nCONTRACT:")
print(f"Tick size         : {info.trade_tick_size}")
print(f"Tick value        : {info.trade_tick_value}")
print(f"Contract size     : {info.trade_contract_size}")
print(f"Min lot           : {info.volume_min}")
print(f"Lot step          : {info.volume_step}")
print(f"Max lot           : {info.volume_max}")
print(f"Stops level       : {info.trade_stops_level}")

print("\nSTATUS:")
print("BTC history saved and validated.")
print("No live orders placed.")

mt5.shutdown()
