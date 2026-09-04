
import numpy as np
import pandas as pd
from pathlib import Path

INPUT = "reports/xauusd_walkforward.csv"
OUTPUT = "reports/xauusd_monte_carlo.csv"

SIMULATIONS = 10000
INITIAL_BALANCE = 100.0

def main():

    print("=" * 70)
    print("XAUUSD MONTE CARLO STRESS TEST")
    print("=" * 70)

    if not Path(INPUT).exists():
        print(f"ERROR: {INPUT} not found")
        return

    df = pd.read_csv(INPUT)

    # Select the strongest walk-forward candidate.
    summary = (
        df.groupby("candidate")
        .agg(
            profitable_folds=("profit", lambda x: (x > 0).sum()),
            total_profit=("profit", "sum"),
            avg_pf=("profit_factor", "mean"),
            worst_dd=("max_drawdown", "max"),
            total_trades=("trades", "sum")
        )
        .sort_values(
            ["profitable_folds", "avg_pf", "total_profit"],
            ascending=False
        )
    )

    candidate = int(summary.index[0])

    selected = df[
        df["candidate"] == candidate
    ].copy()

    print()
    print(f"Selected candidate: {candidate}")
    print(f"Walk-forward rows: {len(selected)}")

    # Approximate trade distribution from each fold.
    profits = selected["profit"].values
    trades = selected["trades"].values

    if len(profits) == 0:
        print("ERROR: No candidate data.")
        return

    # Convert fold-level returns into synthetic trade-level returns.
    trade_returns = []

    for profit, n in zip(profits, trades):

        if n <= 0:
            continue

        # Preserve the fold's total result while creating
        # a synthetic distribution for stress testing.
        avg = profit / n

        noise = np.random.normal(
            1.0,
            0.35,
            int(n)
        )

        r = avg * noise

        correction = (
            profit / r.sum()
            if r.sum() != 0
            else 1.0
        )

        r = r * correction

        trade_returns.extend(r)

    trade_returns = np.array(
        trade_returns,
        dtype=np.float64
    )

    if len(trade_returns) < 20:
        print("ERROR: Not enough trades for Monte Carlo.")
        return

    rng = np.random.default_rng(42)

    final_balances = np.empty(SIMULATIONS)
    max_drawdowns = np.empty(SIMULATIONS)

    for s in range(SIMULATIONS):

        shuffled = rng.permutation(
            trade_returns
        )

        balance = INITIAL_BALANCE
        peak = balance
        max_dd = 0.0

        for r in shuffled:

            balance += r

            if balance > peak:
                peak = balance

            dd = peak - balance

            if dd > max_dd:
                max_dd = dd

        final_balances[s] = balance
        max_drawdowns[s] = max_dd

    results = pd.DataFrame({
        "final_balance": final_balances,
        "max_drawdown": max_drawdowns
    })

    results.to_csv(
        OUTPUT,
        index=False
    )

    profitable = (
        results["final_balance"] > INITIAL_BALANCE
    ).mean()

    ruin = (
        results["final_balance"] <= 0
    ).mean()

    p5_balance = results[
        "final_balance"
    ].quantile(0.05)

    p50_balance = results[
        "final_balance"
    ].quantile(0.50)

    p95_balance = results[
        "final_balance"
    ].quantile(0.95)

    p95_dd = results[
        "max_drawdown"
    ].quantile(0.95)

    print()
    print("=" * 70)
    print("MONTE CARLO RESULTS")
    print("=" * 70)

    print(
        f"Simulations:              {SIMULATIONS:,}"
    )

    print(
        f"Probability profitable:   {profitable*100:.2f}%"
    )

    print(
        f"Probability of ruin:      {ruin*100:.2f}%"
    )

    print(
        f"5th percentile balance:   ${p5_balance:.2f}"
    )

    print(
        f"Median balance:           ${p50_balance:.2f}"
    )

    print(
        f"95th percentile balance:  ${p95_balance:.2f}"
    )

    print(
        f"95th percentile drawdown: ${p95_dd:.2f}"
    )

    print()

    if profitable >= 0.80 and ruin < 0.01:
        verdict = "PROMISING ? proceed to further validation"

    elif profitable >= 0.60 and ruin < 0.05:
        verdict = "WEAK/MODERATE ? requires additional testing"

    else:
        verdict = "NOT ROBUST ? do not deploy"

    print(
        f"VERDICT: {verdict}"
    )

    print()
    print(
        f"Saved: {OUTPUT}"
    )


if __name__ == "__main__":
    main()
