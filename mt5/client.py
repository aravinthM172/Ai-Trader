import os
import MetaTrader5 as mt5
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

class MT5Client:

    def __init__(self):
        self.symbol = os.getenv("MT5_SYMBOL", "XAUUSD")
        self.timeframe_name = os.getenv("MT5_TIMEFRAME", "M5")

        self.timeframes = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4
        }

        self.timeframe = self.timeframes.get(
            self.timeframe_name,
            mt5.TIMEFRAME_M5
        )

    def connect(self):

        if not mt5.initialize():
            print("MT5 initialize error:", mt5.last_error())
            return False

        info = mt5.symbol_info(self.symbol)

        if info is None:
            print(f"Symbol '{self.symbol}' not found in MT5.")
            print("Check the exact Gold symbol in Market Watch.")
            return False

        if not info.visible:
            if not mt5.symbol_select(self.symbol, True):
                print("Could not select symbol.")
                return False

        return True

    def get_price(self):

        tick = mt5.symbol_info_tick(self.symbol)

        if tick is None:
            return None

        return {
            "bid": tick.bid,
            "ask": tick.ask,
            "last": tick.last,
            "time": tick.time
        }

    def get_candles(self, count=1000):

        rates = mt5.copy_rates_from_pos(
            self.symbol,
            self.timeframe,
            0,
            count
        )

        if rates is None:
            print("Candle error:", mt5.last_error())
            return None

        df = pd.DataFrame(rates)

        if df.empty:
            return df

        df["time"] = pd.to_datetime(
            df["time"],
            unit="s"
        )

        return df

    def get_positions(self):

        positions = mt5.positions_get(
            symbol=self.symbol
        )

        if positions is None:
            return []

        return list(positions)

    def shutdown(self):

        mt5.shutdown()
