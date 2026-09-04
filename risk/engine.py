class RiskEngine:

    def __init__(
        self,
        max_risk_percent=0.5,
        max_daily_loss_percent=2.0,
        max_positions=2
    ):

        self.max_risk_percent = (
            max_risk_percent
        )

        self.max_daily_loss_percent = (
            max_daily_loss_percent
        )

        self.max_positions = max_positions


    def approve(
        self,
        signal,
        open_positions,
        news_blocked=False
    ):

        if news_blocked:

            return False, "NEWS_BLOCK"

        if signal["decision"] == "HOLD":

            return False, "HOLD"

        if (
            signal["technical_score"]
            < 0.70
        ):

            return False, "LOW_SCORE"

        if (
            len(open_positions)
            >= self.max_positions
        ):

            return False, "POSITION_LIMIT"

        return True, "APPROVED"
