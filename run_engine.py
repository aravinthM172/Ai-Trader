import os
import MetaTrader5 as mt5
from dotenv import load_dotenv

from mt5.multi_asset import MT5MultiAsset
from strategy.features import FeatureEngine
from strategy.multi_asset import MultiAssetStrategy
from news.filter import NewsFilter
from risk.engine import RiskEngine

load_dotenv()


def main():

    print("=" * 70)
    print(" GOLD + BTC AI TRADING ENGINE")
    print("=" * 70)

    gold = os.getenv(
        "MT5_GOLD_SYMBOL",
        "XAUUSD"
    )

    btc = os.getenv(
        "MT5_BTC_SYMBOL",
        "BTC"
    )

    symbols = [
        ("GOLD", gold),
        ("BITCOIN", btc)
    ]

    client = MT5MultiAsset()

    if not client.connect():

        return

    feature_engine = FeatureEngine()

    strategy = MultiAssetStrategy()

    news = NewsFilter()

    risk = RiskEngine()

    account = client.account()

    if account:

        print()
        print("ACCOUNT")
        print("-" * 40)
        print("Balance :", account["balance"])
        print("Equity  :", account["equity"])
        print(
            "Free margin :",
            account["free_margin"]
        )

    for asset_name, symbol in symbols:

        print()
        print("=" * 70)
        print(
            f"{asset_name} | {symbol}"
        )
        print("=" * 70)

        if not client.symbol_exists(symbol):

            print(
                f"{symbol} unavailable."
            )

            continue

        info = mt5.symbol_info(symbol)

        print(
            "Description:",
            info.description
        )

        print(
            "Digits:",
            info.digits
        )

        print(
            "Point:",
            info.point
        )

        tick = client.get_tick(symbol)

        if tick:

            print()
            print("LIVE TICK")

            print(
                "Bid :",
                tick["bid"]
            )

            print(
                "Ask :",
                tick["ask"]
            )

            print(
                "Last:",
                tick["last"]
            )

            if (
                tick["bid"] == 0
                and tick["ask"] == 0
            ):

                print()
                print(
                    "WARNING: MT5 has no valid live tick."
                )

                print(
                    "Historical candles are available,"
                    " but live market data is not."
                )

        candles = client.get_candles(
            symbol,
            mt5.TIMEFRAME_M5,
            5000
        )

        if (
            candles is None
            or candles.empty
        ):

            print(
                "No candle data."
            )

            continue

        print()
        print(
            f"Historical candles: "
            f"{len(candles):,}"
        )

        data = feature_engine.calculate(
            candles
        )

        features = feature_engine.latest(
            data
        )

        signal = strategy.analyze(
            symbol,
            features
        )

        print()
        print("TECHNICAL SIGNAL")
        print("-" * 40)

        print(
            "Decision:",
            signal["decision"]
        )

        print(
            "Score:",
            signal["technical_score"]
        )

        print(
            "Price:",
            signal["price"]
        )

        print(
            "RSI:",
            signal["rsi"]
        )

        print(
            "ADX:",
            signal["adx"]
        )

        blocked = news.is_blocked(
            symbol
        )

        positions = client.positions(
            symbol
        )

        approved, reason = risk.approve(
            signal,
            positions,
            blocked
        )

        print()
        print("RISK ENGINE")
        print("-" * 40)

        print(
            "Status:",
            "APPROVED"
            if approved
            else "REJECTED"
        )

        print(
            "Reason:",
            reason
        )

    client.shutdown()

    print()
    print("=" * 70)
    print("ENGINE FINISHED")
    print("=" * 70)


if __name__ == "__main__":

    main()
