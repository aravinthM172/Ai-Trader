import MetaTrader5 as mt5
import pandas as pd
import os
import time
from datetime import datetime, timedelta, timezone


SYMBOL = "XAUUSD"

TIMEFRAME = mt5.TIMEFRAME_M5

YEARS = 5

CHUNK_DAYS = 30

OUTPUT = "data/xauusd_m5_5y.csv"


def download():

    print("=" * 70)
    print("XAUUSD HISTORICAL DATA DOWNLOADER")
    print("=" * 70)

    if not mt5.initialize():

        print(
            "MT5 initialization failed:",
            mt5.last_error()
        )

        return False

    info = mt5.symbol_info(SYMBOL)

    if info is None:

        print(
            f"{SYMBOL} does not exist in MT5."
        )

        mt5.shutdown()

        return False

    if not info.visible:

        print(
            f"Selecting {SYMBOL}..."
        )

        if not mt5.symbol_select(
            SYMBOL,
            True
        ):

            print(
                "Could not select symbol."
            )

            mt5.shutdown()

            return False

    print()
    print("Symbol:", SYMBOL)
    print("Description:", info.description)

    # UTC is important for MT5 historical data.
    end = datetime.now(timezone.utc)

    start = end - timedelta(
        days=365 * YEARS
    )

    print()
    print("Requested period:")
    print("START:", start)
    print("END  :", end)
    print()
    print(
        f"Downloading in {CHUNK_DAYS}-day chunks..."
    )

    all_data = []

    current = start

    chunk_number = 0

    while current < end:

        chunk_number += 1

        chunk_end = min(
            current + timedelta(
                days=CHUNK_DAYS
            ),
            end
        )

        print(
            f"[{chunk_number}] "
            f"{current.date()} ? "
            f"{chunk_end.date()}",
            end=" ... ",
            flush=True
        )

        rates = mt5.copy_rates_range(
            SYMBOL,
            TIMEFRAME,
            current,
            chunk_end
        )

        if rates is None:

            print(
                "FAILED:",
                mt5.last_error()
            )

        else:

            df = pd.DataFrame(rates)

            if not df.empty:

                all_data.append(df)

                print(
                    f"{len(df):,} candles"
                )

            else:

                print("0 candles")

        current = chunk_end

        # Small pause so the terminal isn't hammered.
        time.sleep(0.1)

    mt5.shutdown()

    if not all_data:

        print()
        print(
            "No historical data was returned."
        )

        print(
            "Check that XAUUSD has history loaded "
            "inside MT5."
        )

        return False

    print()
    print("Combining data...")

    data = pd.concat(
        all_data,
        ignore_index=True
    )

    # Remove duplicate timestamps.
    data = data.drop_duplicates(
        subset=["time"]
    )

    # Sort chronologically.
    data = data.sort_values(
        "time"
    ).reset_index(
        drop=True
    )

    data["time"] = pd.to_datetime(
        data["time"],
        unit="s",
        utc=True
    )

    os.makedirs(
        "data",
        exist_ok=True
    )

    data.to_csv(
        OUTPUT,
        index=False
    )

    print()
    print("=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)

    print(
        "Candles:",
        f"{len(data):,}"
    )

    print(
        "Start:",
        data["time"].iloc[0]
    )

    print(
        "End:",
        data["time"].iloc[-1]
    )

    print(
        "File:",
        OUTPUT
    )

    print(
        "Size:",
        f"{os.path.getsize(OUTPUT) / 1024 / 1024:.2f} MB"
    )

    return True


if __name__ == "__main__":

    download()
