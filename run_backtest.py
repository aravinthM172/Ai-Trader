import os
import pandas as pd

from strategy.gold import GoldStrategy
from backtest.engine import Backtester

def main():

    path = "data/xauusd_m5_5y.csv"

    if not os.path.exists(path):
        print("Historical data not found.")
        print("Run:")
        print("python download_history.py")
        return

    df = pd.read_csv(path)

    strategy = GoldStrategy()

    analyzed = strategy.analyze

    # Calculate indicators for backtest
    import ta

    df["ema20"] = ta.trend.ema_indicator(
        df["close"],
        window=20
    )

    df["ema50"] = ta.trend.ema_indicator(
        df["close"],
        window=50
    )

    df["ema200"] = ta.trend.ema_indicator(
        df["close"],
        window=200
    )

    df = df.dropna().reset_index(drop=True)

    tester = Backtester()

    result = tester.run(df)

    print()
    print("=" * 50)
    print("XAUUSD BACKTEST")
    print("=" * 50)

    for key, value in result.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
