import MetaTrader5 as mt5
import pandas as pd


class MT5MultiAsset:

    def __init__(self):

        self.assets = {
            "XAUUSD": "XAUUSD",
            "BTC": "BTC"
        }

    def connect(self):

        if not mt5.initialize():

            print(
                "MT5 initialization failed:",
                mt5.last_error()
            )

            return False

        print("MT5 connected.")

        return True

    def symbol_exists(self, symbol):

        info = mt5.symbol_info(symbol)

        if info is None:

            print(
                f"{symbol} not found."
            )

            return False

        if not info.visible:

            print(
                f"Selecting {symbol}..."
            )

            if not mt5.symbol_select(
                symbol,
                True
            ):

                print(
                    f"Unable to select {symbol}"
                )

                return False

        return True

    def get_tick(self, symbol):

        if not self.symbol_exists(symbol):

            return None

        tick = mt5.symbol_info_tick(
            symbol
        )

        if tick is None:

            print(
                f"No tick available for {symbol}"
            )

            return None

        return {
            "symbol": symbol,
            "bid": float(tick.bid),
            "ask": float(tick.ask),
            "last": float(tick.last),
            "time": tick.time
        }

    def get_candles(
        self,
        symbol,
        timeframe=mt5.TIMEFRAME_M5,
        count=5000
    ):

        if not self.symbol_exists(symbol):

            return None

        rates = mt5.copy_rates_from_pos(
            symbol,
            timeframe,
            0,
            count
        )

        if rates is None:

            print(
                f"Historical data error for {symbol}:",
                mt5.last_error()
            )

            return None

        df = pd.DataFrame(rates)

        if df.empty:

            return df

        df["time"] = pd.to_datetime(
            df["time"],
            unit="s"
        )

        return df

    def positions(self, symbol=None):

        if symbol:

            positions = mt5.positions_get(
                symbol=symbol
            )

        else:

            positions = mt5.positions_get()

        if positions is None:

            return []

        return list(positions)

    def account(self):

        info = mt5.account_info()

        if info is None:

            return None

        return {
            "login": info.login,
            "balance": float(info.balance),
            "equity": float(info.equity),
            "margin": float(info.margin),
            "free_margin": float(
                info.margin_free
            )
        }

    def shutdown(self):

        mt5.shutdown()
