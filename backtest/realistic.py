import os
import math
import pandas as pd
import numpy as np
import ta


DATA_FILE = "data/xauusd_m5_5y.csv"
REPORT_DIR = "reports"

INITIAL_BALANCE = 100.0

RISK_PER_TRADE = 0.005

ATR_SL_MULTIPLIER = 1.5
ATR_TP_MULTIPLIER = 2.5

MAX_DAILY_LOSS = 0.02

SPREAD = 0.35

SLIPPAGE = 0.05

COMMISSION = 0.0

MAX_BARS_IN_TRADE = 60


def load_data():

    if not os.path.exists(DATA_FILE):

        raise FileNotFoundError(
            f"Missing {DATA_FILE}"
        )

    df = pd.read_csv(
        DATA_FILE
    )

    df["time"] = pd.to_datetime(
        df["time"],
        utc=True
    )

    df = df.sort_values(
        "time"
    )

    df = df.drop_duplicates(
        "time"
    )

    df = df.reset_index(
        drop=True
    )

    return df


def add_features(df):

    data = df.copy()

    data["ema20"] = ta.trend.ema_indicator(
        data["close"],
        window=20
    )

    data["ema50"] = ta.trend.ema_indicator(
        data["close"],
        window=50
    )

    data["ema200"] = ta.trend.ema_indicator(
        data["close"],
        window=200
    )

    data["rsi"] = ta.momentum.rsi(
        data["close"],
        window=14
    )

    data["atr"] = ta.volatility.average_true_range(
        data["high"],
        data["low"],
        data["close"],
        window=14
    )

    data["adx"] = ta.trend.adx(
        data["high"],
        data["low"],
        data["close"],
        window=14
    )

    data["macd"] = ta.trend.macd(
        data["close"]
    )

    data["volume_ma"] = (
        data["tick_volume"]
        .rolling(20)
        .mean()
    )

    data["body"] = (
        data["close"]
        - data["open"]
    )

    data["range"] = (
        data["high"]
        - data["low"]
    )

    data["momentum"] = (
        data["close"]
        .pct_change(5)
    )

    return data.dropna().reset_index(
        drop=True
    )


def signal(row):

    bullish = 0
    bearish = 0

    if row["ema20"] > row["ema50"]:
        bullish += 1
    else:
        bearish += 1

    if row["ema50"] > row["ema200"]:
        bullish += 1
    else:
        bearish += 1

    if row["rsi"] >= 55:
        bullish += 1

    elif row["rsi"] <= 45:
        bearish += 1

    if row["macd"] > 0:
        bullish += 1
    else:
        bearish += 1

    if row["adx"] >= 20:

        if bullish >= 3:

            return "BUY"

        if bearish >= 3:

            return "SELL"

    return "HOLD"


def position_size(
    balance,
    entry,
    stop
):

    risk_money = (
        balance
        * RISK_PER_TRADE
    )

    stop_distance = abs(
        entry - stop
    )

    if stop_distance <= 0:

        return 0

    # Simplified XAUUSD sizing.
    # Broker-specific contract sizing
    # will be added to the live engine.

    lots = (
        risk_money
        / stop_distance
    )

    return max(
        lots,
        0
    )


def run_backtest(df):

    balance = INITIAL_BALANCE

    equity_curve = []

    trades = []

    position = None

    day = None

    daily_start_balance = balance

    for i in range(
        1,
        len(df)
    ):

        row = df.iloc[i]

        current_day = row["time"].date()

        if day != current_day:

            day = current_day

            daily_start_balance = balance

        # -------------------------------------------------
        # Manage existing position
        # -------------------------------------------------

        if position is not None:

            position["bars"] += 1

            exit_price = None

            exit_reason = None

            high = row["high"]
            low = row["low"]

            if position["side"] == "BUY":

                if low <= position["sl"]:

                    exit_price = (
                        position["sl"]
                        - SLIPPAGE
                    )

                    exit_reason = "STOP"

                elif high >= position["tp"]:

                    exit_price = (
                        position["tp"]
                        - SLIPPAGE
                    )

                    exit_reason = "TAKE_PROFIT"

            else:

                if high >= position["sl"]:

                    exit_price = (
                        position["sl"]
                        + SLIPPAGE
                    )

                    exit_reason = "STOP"

                elif low <= position["tp"]:

                    exit_price = (
                        position["tp"]
                        + SLIPPAGE
                    )

                    exit_reason = "TAKE_PROFIT"

            if (
                exit_price is None
                and position["bars"]
                >= MAX_BARS_IN_TRADE
            ):

                exit_price = row["close"]

                exit_reason = "TIME_EXIT"

            if exit_price is not None:

                if position["side"] == "BUY":

                    price_change = (
                        exit_price
                        - position["entry"]
                    )

                else:

                    price_change = (
                        position["entry"]
                        - exit_price
                    )

                pnl = (
                    price_change
                    * position["size"]
                )

                pnl -= COMMISSION

                balance += pnl

                trades.append({
                    "entry_time":
                        position["entry_time"],

                    "exit_time":
                        row["time"],

                    "side":
                        position["side"],

                    "entry":
                        position["entry"],

                    "exit":
                        exit_price,

                    "sl":
                        position["sl"],

                    "tp":
                        position["tp"],

                    "size":
                        position["size"],

                    "pnl":
                        pnl,

                    "reason":
                        exit_reason
                })

                position = None

        # -------------------------------------------------
        # Daily loss protection
        # -------------------------------------------------

        daily_loss = (
            daily_start_balance
            - balance
        )

        if (
            daily_loss
            >= daily_start_balance
            * MAX_DAILY_LOSS
        ):

            equity_curve.append({
                "time": row["time"],
                "balance": balance
            })

            continue

        # -------------------------------------------------
        # New signal
        # -------------------------------------------------

        if position is None:

            previous = df.iloc[i - 1]

            decision = signal(
                previous
            )

            if decision == "BUY":

                entry = (
                    row["open"]
                    + SPREAD / 2
                    + SLIPPAGE
                )

                atr = previous["atr"]

                sl = (
                    entry
                    - ATR_SL_MULTIPLIER
                    * atr
                )

                tp = (
                    entry
                    + ATR_TP_MULTIPLIER
                    * atr
                )

                size = position_size(
                    balance,
                    entry,
                    sl
                )

                if size > 0:

                    position = {
                        "side": "BUY",
                        "entry": entry,
                        "sl": sl,
                        "tp": tp,
                        "size": size,
                        "entry_time":
                            row["time"],
                        "bars": 0
                    }

            elif decision == "SELL":

                entry = (
                    row["open"]
                    - SPREAD / 2
                    - SLIPPAGE
                )

                atr = previous["atr"]

                sl = (
                    entry
                    + ATR_SL_MULTIPLIER
                    * atr
                )

                tp = (
                    entry
                    - ATR_TP_MULTIPLIER
                    * atr
                )

                size = position_size(
                    balance,
                    entry,
                    sl
                )

                if size > 0:

                    position = {
                        "side": "SELL",
                        "entry": entry,
                        "sl": sl,
                        "tp": tp,
                        "size": size,
                        "entry_time":
                            row["time"],
                        "bars": 0
                    }

        equity_curve.append({
            "time": row["time"],
            "balance": balance
        })

    trades_df = pd.DataFrame(
        trades
    )

    equity_df = pd.DataFrame(
        equity_curve
    )

    return (
        trades_df,
        equity_df,
        balance
    )


def calculate_metrics(
    trades,
    equity,
    final_balance
):

    if trades.empty:

        return {
            "initial_balance":
                INITIAL_BALANCE,

            "final_balance":
                final_balance,

            "net_profit":
                final_balance
                - INITIAL_BALANCE,

            "trades":
                0
        }

    wins = trades[
        trades["pnl"] > 0
    ]

    losses = trades[
        trades["pnl"] < 0
    ]

    win_rate = (
        len(wins)
        / len(trades)
    )

    gross_profit = (
        wins["pnl"].sum()
    )

    gross_loss = abs(
        losses["pnl"].sum()
    )

    if gross_loss > 0:

        profit_factor = (
            gross_profit
            / gross_loss
        )

    else:

        profit_factor = np.inf

    equity_values = (
        equity["balance"]
        .astype(float)
    )

    running_max = (
        equity_values
        .cummax()
    )

    drawdown = (
        running_max
        - equity_values
    )

    max_drawdown = (
        drawdown.max()
    )

    returns = (
        equity_values
        .pct_change()
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
    )

    if len(returns) > 1:

        sharpe = (
            returns.mean()
            / returns.std()
            * np.sqrt(252)
        )

    else:

        sharpe = 0

    expectancy = (
        trades["pnl"].mean()
    )

    return {
        "initial_balance":
            INITIAL_BALANCE,

        "final_balance":
            final_balance,

        "net_profit":
            final_balance
            - INITIAL_BALANCE,

        "return_percent":
            (
                final_balance
                / INITIAL_BALANCE
                - 1
            )
            * 100,

        "trades":
            len(trades),

        "wins":
            len(wins),

        "losses":
            len(losses),

        "win_rate_percent":
            win_rate * 100,

        "profit_factor":
            profit_factor,

        "expectancy":
            expectancy,

        "max_drawdown":
            max_drawdown,

        "max_drawdown_percent":
            (
                max_drawdown
                / INITIAL_BALANCE
            )
            * 100,

        "sharpe":
            sharpe
    }


def main():

    os.makedirs(
        REPORT_DIR,
        exist_ok=True
    )

    print("=" * 70)
    print("XAUUSD REALISTIC BACKTEST")
    print("=" * 70)

    df = load_data()

    print(
        f"Loaded {len(df):,} candles."
    )

    print(
        "Start:",
        df["time"].iloc[0]
    )

    print(
        "End:",
        df["time"].iloc[-1]
    )

    df = add_features(df)

    print(
        f"Usable candles: {len(df):,}"
    )

    # 70/30 chronological split.
    split = int(
        len(df) * 0.70
    )

    train = df.iloc[
        :split
    ].copy()

    test = df.iloc[
        split:
    ].copy()

    print()
    print("TRAIN")
    print(
        train["time"].iloc[0],
        "?",
        train["time"].iloc[-1]
    )

    print()
    print("OUT-OF-SAMPLE TEST")
    print(
        test["time"].iloc[0],
        "?",
        test["time"].iloc[-1]
    )

    print()
    print("Running OUT-OF-SAMPLE test...")

    trades, equity, final_balance = (
        run_backtest(test)
    )

    metrics = calculate_metrics(
        trades,
        equity,
        final_balance
    )

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    for key, value in metrics.items():

        if isinstance(
            value,
            float
        ):

            print(
                f"{key:<25} "
                f"{value:.4f}"
            )

        else:

            print(
                f"{key:<25} "
                f"{value}"
            )

    trades.to_csv(
        f"{REPORT_DIR}/xauusd_trades.csv",
        index=False
    )

    equity.to_csv(
        f"{REPORT_DIR}/xauusd_equity.csv",
        index=False
    )

    pd.DataFrame(
        [metrics]
    ).to_csv(
        f"{REPORT_DIR}/xauusd_metrics.csv",
        index=False
    )

    print()
    print("Reports created:")
    print(
        "reports/xauusd_trades.csv"
    )
    print(
        "reports/xauusd_equity.csv"
    )
    print(
        "reports/xauusd_metrics.csv"
    )


if __name__ == "__main__":

    main()
