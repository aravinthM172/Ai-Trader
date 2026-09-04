import pandas as pd
import numpy as np

class Backtester:

    def __init__(
        self,
        initial_balance=10000,
        risk_per_trade=0.005
    ):

        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade

    def run(self, df):

        if df.empty:
            return None

        balance = self.initial_balance

        trades = []

        for i in range(1, len(df)):

            previous = df.iloc[i - 1]
            current = df.iloc[i]

            if previous["close"] > previous["ema20"]:
                signal = "BUY"

            elif previous["close"] < previous["ema20"]:
                signal = "SELL"

            else:
                signal = "HOLD"

            if signal == "HOLD":
                continue

            entry = current["open"]

            if signal == "BUY":
                exit_price = current["close"]
            else:
                exit_price = current["close"]

            pnl = (
                exit_price - entry
                if signal == "BUY"
                else entry - exit_price
            )

            trades.append(pnl)

        if not trades:
            return {
                "trades": 0,
                "win_rate": 0,
                "net_result": 0
            }

        wins = [
            x for x in trades
            if x > 0
        ]

        return {
            "trades": len(trades),
            "winning_trades": len(wins),
            "win_rate": len(wins) / len(trades),
            "net_result": sum(trades)
        }
