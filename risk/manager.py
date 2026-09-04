import os
from dotenv import load_dotenv

load_dotenv()

class RiskManager:

    def __init__(self):

        self.max_risk = float(
            os.getenv(
                "MAX_RISK_PERCENT",
                "0.5"
            )
        )

        self.max_daily_loss = float(
            os.getenv(
                "MAX_DAILY_LOSS_PERCENT",
                "2.0"
            )
        )

        self.max_positions = int(
            os.getenv(
                "MAX_OPEN_POSITIONS",
                "1"
            )
        )

    def can_trade(self):

        # Initial safety gate.
        # Live account/equity/position checks
        # will be added before execution is enabled.

        return True

    def validate_signal(self, signal):

        if signal.get("decision") not in [
            "BUY",
            "SELL",
            "HOLD"
        ]:
            return False

        confidence = float(
            signal.get("confidence", 0)
        )

        if confidence < 0.70:
            return False

        if signal.get("risk_level") == "HIGH":
            return False

        return True
