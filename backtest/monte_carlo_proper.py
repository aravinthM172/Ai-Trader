
import numpy as np
import pandas as pd
from pathlib import Path

DATA = "data/xauusd_m5_5y.csv"
VALIDATION = "reports/xauusd_validation.csv"
TRADES_FILE = "reports/xauusd_trade_log.csv"
RESULTS_FILE = "reports/xauusd_monte_carlo_proper.csv"

INITIAL_BALANCE = 100.0
SIMULATIONS = 10000

# Best candidate from the current validation.
FAST = 10
SLOW = 80
TREND = 200
RSI_PERIOD = 10
ATR_PERIOD = 10
SL_MULT = 2.0
TP_MULT = 3.0

SPREAD = 0.35
SLIPPAGE = 0.05


def ema(x, p):
    return pd.Series(x).ewm(
        span=p,
        adjust=False
    ).mean().values


def rsi(x, p):
    d = np.diff(x, prepend=x[0])

    gain = np.maximum(d, 0)
    loss = np.maximum(-d, 0)

    ag = pd.Series(gain).ewm(
        alpha=1/p,
        adjust=False
    ).mean()

    al = pd.Series(loss).ewm(
        alpha=1/p,
        adjust=False
    ).mean()

    rs = ag / al.replace(0, np.nan)

    return (
        100 - 100/(1+rs)
    ).fillna(50).values


def atr(high, low, close, p):

    prev = np.roll(close, 1)
    prev[0] = close[0]

    tr = np.maximum(
        high-low,
        np.maximum(
            np.abs(high-prev),
            np.abs(low-prev)
        )
    )

    return pd.Series(tr).ewm(
        alpha=1/p,
        adjust=False
    ).mean().values


def generate_trades(df):

    close = df.close.values
    high = df.high.values
    low = df.low.values
    times = df.time.values

    ef = ema(close, FAST)
    es = ema(close, SLOW)
    et = ema(close, TREND)

    rv = rsi(
        close,
        RSI_PERIOD
    )

    av = atr(
        high,
        low,
        close,
        ATR_PERIOD
    )

    trades = []

    position = 0
    entry = 0
    entry_time = None
    sl = 0
    tp = 0

    start = max(
        TREND,
        RSI_PERIOD,
        ATR_PERIOD
    )

    for i in range(start, len(df)):

        # Manage LONG.
        if position == 1:

            if low[i] <= sl:

                exit_price = sl - SLIPPAGE

                pnl = (
                    exit_price-entry
                )

                trades.append({
                    "entry_time": entry_time,
                    "exit_time": times[i],
                    "direction": "BUY",
                    "entry": entry,
                    "exit": exit_price,
                    "pnl": pnl,
                    "result": "WIN" if pnl > 0 else "LOSS"
                })

                position = 0

            elif high[i] >= tp:

                exit_price = tp - SLIPPAGE

                pnl = (
                    exit_price-entry
                )

                trades.append({
                    "entry_time": entry_time,
                    "exit_time": times[i],
                    "direction": "BUY",
                    "entry": entry,
                    "exit": exit_price,
                    "pnl": pnl,
                    "result": "WIN"
                    if pnl > 0 else "LOSS"
                })

                position = 0

        # Manage SHORT.
        elif position == -1:

            if high[i] >= sl:

                exit_price = sl + SLIPPAGE

                pnl = (
                    entry-exit_price
                )

                trades.append({
                    "entry_time": entry_time,
                    "exit_time": times[i],
                    "direction": "SELL",
                    "entry": entry,
                    "exit": exit_price,
                    "pnl": pnl,
                    "result": "WIN"
                    if pnl > 0 else "LOSS"
                })

                position = 0

            elif low[i] <= tp:

                exit_price = tp + SLIPPAGE

                pnl = (
                    entry-exit_price
                )

                trades.append({
                    "entry_time": entry_time,
                    "exit_time": times[i],
                    "direction": "SELL",
                    "entry": entry,
                    "exit": exit_price,
                    "pnl": pnl,
                    "result": "WIN"
                    if pnl > 0 else "LOSS"
                })

                position = 0

        if position != 0:
            continue

        # BUY signal.
        buy = (
            ef[i-1] > es[i-1]
            and es[i-1] > et[i-1]
            and rv[i-1] > 55
        )

        # SELL signal.
        sell = (
            ef[i-1] < es[i-1]
            and es[i-1] < et[i-1]
            and rv[i-1] < 45
        )

        if av[i-1] <= 0:
            continue

        if buy:

            entry = (
                close[i]
                + SPREAD/2
                + SLIPPAGE
            )

            sl = (
                entry
                - av[i-1]*SL_MULT
            )

            tp = (
                entry
                + av[i-1]*TP_MULT
            )

            entry_time = times[i]

            position = 1

        elif sell:

            entry = (
                close[i]
                - SPREAD/2
                - SLIPPAGE
            )

            sl = (
                entry
                + av[i-1]*SL_MULT
            )

            tp = (
                entry
                - av[i-1]*TP_MULT
            )

            entry_time = times[i]

            position = -1

    return pd.DataFrame(trades)


def main():

    print("="*70)
    print("PROPER XAUUSD TRADE-LEVEL MONTE CARLO")
    print("="*70)

    if not Path(DATA).exists():
        print("ERROR: Historical data not found.")
        return

    df = pd.read_csv(DATA)

    df["time"] = pd.to_datetime(
        df["time"],
        utc=True
    )

    # Use the unseen 30% only.
    split = int(len(df)*0.70)

    test = (
        df.iloc[split:]
        .reset_index(drop=True)
    )

    print()
    print(
        f"Test candles: {len(test):,}"
    )

    print(
        "Generating individual historical trades..."
    )

    trades = generate_trades(test)

    trades.to_csv(
        TRADES_FILE,
        index=False
    )

    print(
        f"Trades generated: {len(trades):,}"
    )

    if len(trades) < 100:
        print(
            "ERROR: Not enough trades."
        )
        return

    pnl = trades["pnl"].values.astype(
        np.float64
    )

    rng = np.random.default_rng(42)

    final_balances = np.zeros(
        SIMULATIONS
    )

    drawdowns = np.zeros(
        SIMULATIONS
    )

    ruin = np.zeros(
        SIMULATIONS
    )

    for s in range(SIMULATIONS):

        sequence = rng.permutation(
            pnl
        )

        balance = INITIAL_BALANCE
        peak = balance
        max_dd = 0

        for p in sequence:

            balance += p

            if balance > peak:
                peak = balance

            dd = peak - balance

            if dd > max_dd:
                max_dd = dd

            if balance <= 0:
                break

        final_balances[s] = balance
        drawdowns[s] = max_dd

        if balance <= 0:
            ruin[s] = 1

    result = pd.DataFrame({
        "final_balance": final_balances,
        "max_drawdown": drawdowns,
        "ruin": ruin
    })

    result.to_csv(
        RESULTS_FILE,
        index=False
    )

    print()
    print("="*70)
    print("MONTE CARLO RESULTS")
    print("="*70)

    print(
        f"Simulations: {SIMULATIONS:,}"
    )

    print(
        f"Probability profitable: "
        f"{(result.final_balance > INITIAL_BALANCE).mean()*100:.2f}%"
    )

    print(
        f"Probability of ruin: "
        f"{result.ruin.mean()*100:.2f}%"
    )

    print(
        f"5% worst balance: "
        f"${result.final_balance.quantile(.05):.2f}"
    )

    print(
        f"Median balance: "
        f"${result.final_balance.quantile(.50):.2f}"
    )

    print(
        f"95% balance: "
        f"${result.final_balance.quantile(.95):.2f}"
    )

    print(
        f"95% max drawdown: "
        f"${result.max_drawdown.quantile(.95):.2f}"
    )

    print()
    print(
        f"Trade log: {TRADES_FILE}"
    )

    print(
        f"Results:   {RESULTS_FILE}"
    )


if __name__ == "__main__":
    main()
