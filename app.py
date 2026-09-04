from mt5.client import MT5Client
from strategy.gold import GoldStrategy
from claude.analyzer import ClaudeAnalyzer
from risk.manager import RiskManager

def main():
    print("=" * 60)
    print(" GOLD AI TRADER")
    print(" XAUUSD + MT5 + CLAUDE")
    print("=" * 60)

    mt5 = MT5Client()

    if not mt5.connect():
        print("MT5 connection failed.")
        return

    print("MT5 connected.")

    price = mt5.get_price()

    if price:
        print(f"XAUUSD Bid : {price['bid']}")
        print(f"XAUUSD Ask : {price['ask']}")

    candles = mt5.get_candles()

    if candles is None or candles.empty:
        print("No candle data received.")
        mt5.shutdown()
        return

    print(f"Received {len(candles)} candles.")

    strategy = GoldStrategy()
    analysis = strategy.analyze(candles)

    print("\nTECHNICAL ANALYSIS")
    print("-" * 40)

    for key, value in analysis.items():
        print(f"{key}: {value}")

    risk = RiskManager()

    if not risk.can_trade():
        print("\nRisk engine rejected trading.")
        mt5.shutdown()
        return

    claude = ClaudeAnalyzer()

    signal = claude.analyze(analysis)

    print("\nCLAUDE SIGNAL")
    print("-" * 40)
    print(signal)

    print("\nDRY RUN MODE - NO REAL ORDER SENT")

    mt5.shutdown()


if __name__ == "__main__":
    main()
