class MultiAssetStrategy:

    def analyze(
        self,
        symbol,
        features
    ):

        bullish = 0
        bearish = 0

        if features["ema20"] > features["ema50"]:
            bullish += 1
        else:
            bearish += 1

        if features["ema50"] > features["ema200"]:
            bullish += 1
        else:
            bearish += 1

        if features["rsi"] > 55:
            bullish += 1

        elif features["rsi"] < 45:
            bearish += 1

        if features["macd"] > 0:
            bullish += 1
        else:
            bearish += 1

        if features["adx"] >= 20:

            if bullish >= 3:

                decision = "BUY"

            elif bearish >= 3:

                decision = "SELL"

            else:

                decision = "HOLD"

        else:

            decision = "HOLD"

        score = max(
            bullish,
            bearish
        ) / 4

        return {
            "symbol": symbol,
            "decision": decision,
            "technical_score": round(
                score,
                4
            ),
            **features
        }
