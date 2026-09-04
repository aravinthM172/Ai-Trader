import numpy as np
import pandas as pd
import ta


class FeatureEngine:

    @staticmethod
    def calculate(df):

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

        data["bb_high"] = (
            ta.volatility.BollingerBands(
                data["close"],
                window=20
            ).bollinger_hband()
        )

        data["bb_low"] = (
            ta.volatility.BollingerBands(
                data["close"],
                window=20
            ).bollinger_lband()
        )

        data["returns"] = (
            data["close"].pct_change()
        )

        data["volatility"] = (
            data["returns"]
            .rolling(20)
            .std()
        )

        return data.dropna().reset_index(
            drop=True
        )


    @staticmethod
    def latest(df):

        row = df.iloc[-1]

        return {
            "price": float(row["close"]),
            "ema20": float(row["ema20"]),
            "ema50": float(row["ema50"]),
            "ema200": float(row["ema200"]),
            "rsi": float(row["rsi"]),
            "atr": float(row["atr"]),
            "adx": float(row["adx"]),
            "macd": float(row["macd"]),
            "bb_high": float(row["bb_high"]),
            "bb_low": float(row["bb_low"]),
            "volatility": float(row["volatility"])
        }
