import pandas as pd
import ta

class GoldStrategy:

    def analyze(self, df):

        data = df.copy()

        data["ema20"] = ta.trend.ema_indicator(
            data["close"],
            window=20
        )

        data["ema50"] = ta.trend.ema_indicator(
            data["close"],
            window=50
        )

        data["ema200"] = ta.trend.ema_indicator(
            data["close"],
            window=200
        )

        data["rsi"] = ta.momentum.rsi(
            data["close"],
            window=14
        )

        data["atr"] = ta.volatility.average_true_range(
            data["high"],
            data["low"],
            data["close"],
            window=14
        )

        data["adx"] = ta.trend.adx(
            data["high"],
            data["low"],
            data["close"],
            window=14
        )

        data["macd"] = ta.trend.macd(
            data["close"]
        )

        last = data.iloc[-1]

        if (
            last["ema20"] > last["ema50"]
            and last["ema50"] > last["ema200"]
            and last["rsi"] > 50
            and last["macd"] > 0
        ):
            trend = "BULLISH"

        elif (
            last["ema20"] < last["ema50"]
            and last["ema50"] < last["ema200"]
            and last["rsi"] < 50
            and last["macd"] < 0
        ):
            trend = "BEARISH"

        else:
            trend = "NEUTRAL"

        return {
            "symbol": "XAUUSD",
            "price": float(last["close"]),
            "trend": trend,
            "ema20": float(last["ema20"]),
            "ema50": float(last["ema50"]),
            "ema200": float(last["ema200"]),
            "rsi": float(last["rsi"]),
            "atr": float(last["atr"]),
            "adx": float(last["adx"]),
            "macd": float(last["macd"])
        }
