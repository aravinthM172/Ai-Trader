from datetime import datetime


class NewsFilter:

    def __init__(
        self,
        enabled=True,
        before_minutes=15,
        after_minutes=30
    ):

        self.enabled = enabled
        self.before_minutes = before_minutes
        self.after_minutes = after_minutes


    def get_events(self):

        # Provider integration will be added here.
        #
        # Expected event format:
        #
        # {
        #   "time": datetime,
        #   "currency": "USD",
        #   "impact": "HIGH",
        #   "event": "CPI"
        # }

        return []


    def is_blocked(
        self,
        asset
    ):

        if not self.enabled:

            return False

        events = self.get_events()

        now = datetime.utcnow()

        for event in events:

            if event["impact"] != "HIGH":
                continue

            difference = (
                event["time"] - now
            ).total_seconds() / 60

            if (
                -self.after_minutes
                <= difference
                <= self.before_minutes
            ):

                return True

        return False
