from pathlib import Path
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timezone, timedelta

SYMBOL = "BTC"
TIMEFRAME = mt5.TIMEFRAME_M5
OUTPUT = Path(r"data\btc_m5_history_5y.csv")

# ~5 years
END = datetime.now(timezone.utc)
START = END - timedelta(days=365 * 5)

CHUNK_DAYS = 30

print("=" * 80)
print("BTC 5-YEAR M5 DOWNLOAD")
print("=" * 80)
print(f"Symbol : {SYMBOL}")
print(f"From   : {START}")
print(f"To     : {END}")
print()

if not mt5.initialize():
    raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

info = mt5.symbol_info(SYMBOL)
if info is None:
    mt5.shutdown()
    raise RuntimeError(f"Symbol not found: {SYMBOL}")

if not info.visible:
    mt5.symbol_select(SYMBOL, True)

all_data = []
cur_end = END
chunk = 0

while cur_end > START:
    cur_start = max(START, cur_end - timedelta(days=CHUNK_DAYS))

    rates = mt5.copy_rates_range(
        SYMBOL,
        TIMEFRAME,
        cur_start,
        cur_end
    )

    chunk += 1

    if rates is None:
        print(f"Chunk {chunk}: FAILED {mt5.last_error()}")
        cur_end = cur_start
        continue

    if len(rates) > 0:
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        all_data.append(df)

        print(
            f"Chunk {chunk:03d}: "
            f"{len(df):5d} bars | "
            f"{df['time'].iloc[0]} -> {df['time'].iloc[-1]}"
        )
    else:
        print(
            f"Chunk {chunk:03d}: 0 bars | "
            f"{cur_start} -> {cur_end}"
        )

    cur_end = cur_start

if not all_data:
    mt5.shutdown()
    raise RuntimeError("No BTC history downloaded.")

df = pd.concat(all_data, ignore_index=True)

df = (
    df.drop_duplicates(subset="time")
      .sort_values("time")
      .reset_index(drop=True)
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

tick = mt5.symbol_info_tick(SYMBOL)
info = mt5.symbol_info(SYMBOL)

print()
print("=" * 80)
print("BTC 5-YEAR HISTORY RESULT")
print("=" * 80)
print(f"Rows          : {len(df):,}")
print(f"From          : {df['time'].iloc[0]}")
print(f"To            : {df['time'].iloc[-1]}")
print(f"Saved         : {OUTPUT.resolve()}")

if tick:
    print(f"Bid           : {tick.bid}")
    print(f"Ask           : {tick.ask}")
    print(f"Spread        : {tick.ask - tick.bid:.6f}")

print(f"Tick size     : {info.trade_tick_size}")
print(f"Tick value    : {info.trade_tick_value}")
print(f"Contract size : {info.trade_contract_size}")
print(f"Min lot       : {info.volume_min}")
print(f"Lot step      : {info.volume_step}")
print(f"Max lot       : {info.volume_max}")

mt5.shutdown()
