import numpy as np


class Metrics:

    @staticmethod
    def calculate(trades):

        if not trades:

            return {
                "trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "net_profit": 0,
                "max_drawdown": 0
            }

        profits = [
            x for x in trades
            if x > 0
        ]

        losses = [
            x for x in trades
            if x < 0
        ]

        gross_profit = sum(profits)

        gross_loss = abs(
            sum(losses)
        )

        equity = 0
        peak = 0
        max_drawdown = 0

        for pnl in trades:

            equity += pnl

            peak = max(
                peak,
                equity
            )

            drawdown = peak - equity

            max_drawdown = max(
                max_drawdown,
                drawdown
            )

        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0
            else float("inf")
        )

        return {
            "trades": len(trades),

            "wins": len(profits),

            "losses": len(losses),

            "win_rate": (
                len(profits)
                / len(trades)
            ),

            "profit_factor": profit_factor,

            "net_profit": sum(trades),

            "max_drawdown": max_drawdown
        }
