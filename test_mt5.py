from mt5.client import MT5Client

client = MT5Client()

if client.connect():

    print("================================")
    print("MT5 CONNECTION SUCCESSFUL")
    print("================================")

    price = client.get_price()

    print("Symbol:", client.symbol)

    if price:
        print("Bid :", price["bid"])
        print("Ask :", price["ask"])

    candles = client.get_candles(10)

    if candles is not None:
        print()
        print("Latest candles:")
        print(
            candles[
                ["time", "open", "high", "low", "close"]
            ].tail()
        )

    client.shutdown()

else:

    print("MT5 connection FAILED.")
