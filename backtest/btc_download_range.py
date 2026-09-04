from pathlib import Path
from datetime import datetime, timedelta, timezone
import MetaTrader5 as mt5
import pandas as pd

SYMBOL = "BTC"
TIMEFRAME = mt5.TIMEFRAME_M5

OUT = Path(r"data\btc_m5_history.csv")
OUT.parent.mkdir(exist_ok=True)

if not mt5.initialize():
    raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

if not mt5.symbol_select(SYMBOL, True):
    mt5.shutdown()
    raise RuntimeError(f"Cannot select {SYMBOL}: {mt5.last_error()}")

# Start from a known recent date and walk backward in 30-day chunks.
end = datetime.now(timezone.utc)
start = end - timedelta(days=30)

parts = []
total = 0

for i in range(12):
    rates = mt5.copy_rates_range(
        SYMBOL,
        TIMEFRAME,
        start,
        end,
    )

    if rates is None:
        print(f"Chunk {i+1}: {mt5.last_error()}")
    elif len(rates):
        part = pd.DataFrame(rates)
        parts.append(part)
        total += len(part)
        print(
            f"Chunk {i+1}: {len(part):,} bars | "
            f"{start.date()} -> {end.date()} | total {total:,}"
        )
    else:
        print(f"Chunk {i+1}: empty")

    end = start
    start = end - timedelta(days=30)

    if not parts:
        continue

df = pd.concat(parts, ignore_index=True)
df = df.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

df.to_csv(OUT, index=False)

info = mt5.symbol_info(SYMBOL)
tick = mt5.symbol_info_tick(SYMBOL)

print("\n" + "=" * 80)
print("BTC HISTORY RESULT")
print("=" * 80)
print(f"Rows           : {len(df):,}")
print(f"From           : {df.time.min()}")
print(f"To             : {df.time.max()}")
print(f"Saved          : {OUT.resolve()}")
print(f"Bid            : {tick.bid}")
print(f"Ask            : {tick.ask}")
print(f"Spread         : {tick.ask - tick.bid:.6f}")
print(f"Tick size      : {info.trade_tick_size}")
print(f"Tick value     : {info.trade_tick_value}")
print(f"Contract size  : {info.trade_contract_size}")
print("=" * 80)

mt5.shutdown()
