import os
import itertools
import pandas as pd
import numpy as np
import ta

DATA_FILE = "data/xauusd_m5_5y.csv"
OUTPUT = "reports/optimization_results.csv"

INITIAL_BALANCE = 100.0
RISK_PER_TRADE = 0.005

SPREAD = 0.35
SLIPPAGE = 0.05
COMMISSION = 0.0


def load():

    df = pd.read_csv(DATA_FILE)

    df["time"] = pd.to_datetime(
        df["time"],
        utc=True
    )

    df = df.sort_values("time")
    df = df.drop_duplicates("time")
    df = df.reset_index(drop=True)

    return df


def features(df, ema_fast, ema_slow, ema_trend, rsi_period, atr_period):

    x = df.copy()

    x["ema_fast"] = ta.trend.ema_indicator(
        x["close"],
        window=ema_fast
    )

    x["ema_slow"] = ta.trend.ema_indicator(
        x["close"],
        window=ema_slow
    )

    x["ema_trend"] = ta.trend.ema_indicator(
        x["close"],
        window=ema_trend
    )

    x["rsi"] = ta.momentum.rsi(
        x["close"],
        window=rsi_period
    )

    x["atr"] = ta.volatility.average_true_range(
        x["high"],
        x["low"],
        x["close"],
        window=atr_period
    )

    x["adx"] = ta.trend.adx(
        x["high"],
        x["low"],
        x["close"],
        window=14
    )

    x["macd"] = ta.trend.macd(
        x["close"]
    )

    return x.dropna().reset_index(drop=True)


def get_signal(row):

    bull = 0
    bear = 0

    if row.ema_fast > row.ema_slow:
        bull += 1
    else:
        bear += 1

    if row.ema_slow > row.ema_trend:
        bull += 1
    else:
        bear += 1

    if row.rsi > 55:
        bull += 1

    elif row.rsi < 45:
        bear += 1

    if row.macd > 0:
        bull += 1
    else:
        bear += 1

    if row.adx < 20:
        return "HOLD"

    if bull >= 3:
        return "BUY"

    if bear >= 3:
        return "SELL"

    return "HOLD"


def backtest(
    df,
    sl_multiplier,
    tp_multiplier
):

    balance = INITIAL_BALANCE

    trades = []

    position = None

    for i in range(1, len(df)):

        prev = df.iloc[i - 1]
        row = df.iloc[i]

        if position:

            exit_price = None

            reason = None

            if position["side"] == "BUY":

                if row.high <= position["sl"]:
                    exit_price = position["sl"]
                    reason = "SL"

                elif row.high >= position["tp"]:
                    exit_price = position["tp"]
                    reason = "TP"

            else:

                if row.high >= position["sl"]:
                    exit_price = position["sl"]
                    reason = "SL"

                elif row.low <= position["tp"]:
                    exit_price = position["tp"]
                    reason = "TP"

            if exit_price is not None:

                if position["side"] == "BUY":

                    pnl = (
                        exit_price
                        - position["entry"]
                    ) * position["size"]

                else:

                    pnl = (
                        position["entry"]
                        - exit_price
                    ) * position["size"]

                pnl -= COMMISSION

                balance += pnl

                trades.append(pnl)

                position = None

        if position is None:

            signal = get_signal(prev)

            atr = prev.atr

            if signal == "BUY":

                entry = (
                    row.open
                    + SPREAD / 2
                    + SLIPPAGE
                )

                sl = (
                    entry
                    - atr * sl_multiplier
                )

                tp = (
                    entry
                    + atr * tp_multiplier
                )

            elif signal == "SELL":

                entry = (
                    row.open
                    - SPREAD / 2
                    - SLIPPAGE
                )

                sl = (
                    entry
                    + atr * sl_multiplier
                )

                tp = (
                    entry
                    - atr * tp_multiplier
                )

            else:
                continue

            distance = abs(
                entry - sl
            )

            if distance <= 0:
                continue

            risk_money = (
                balance
                * RISK_PER_TRADE
            )

            size = (
                risk_money
                / distance
            )

            position = {
                "side": signal,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "size": size
            }

    if not trades:

        return {
            "trades": 0,
            "profit": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "max_drawdown": 0
        }

    wins = [
        x for x in trades
        if x > 0
    ]

    losses = [
        x for x in trades
        if x < 0
    ]

    gross_profit = sum(wins)

    gross_loss = abs(sum(losses))

    equity = INITIAL_BALANCE

    peak = equity

    max_dd = 0

    for pnl in trades:

        equity += pnl

        peak = max(
            peak,
            equity
        )

        dd = peak - equity

        max_dd = max(
            max_dd,
            dd
        )

    return {
        "trades": len(trades),

        "profit": balance - INITIAL_BALANCE,

        "win_rate":
            len(wins)
            / len(trades),

        "profit_factor":
            (
                gross_profit
                / gross_loss
                if gross_loss > 0
                else 999
            ),

        "max_drawdown": max_dd
    }


def main():

    os.makedirs(
        "reports",
        exist_ok=True
    )

    print("=" * 70)
    print("XAUUSD PARAMETER OPTIMIZATION")
    print("=" * 70)

    df = load()

    # Chronological split.
    split = int(
        len(df) * 0.70
    )

    train = df.iloc[:split].copy()

    test = df.iloc[split:].copy()

    print(
        "Training candles:",
        len(train)
    )

    print(
        "Testing candles:",
        len(test)
    )

    # Deliberately modest grid first.
    ema_fast_values = [
        10,
        20,
        30
    ]

    ema_slow_values = [
        40,
        50,
        80
    ]

    ema_trend_values = [
        100,
        150,
        200
    ]

    rsi_values = [
        10,
        14,
        20
    ]

    atr_values = [
        10,
        14,
        20
    ]

    sl_values = [
        1.0,
        1.5,
        2.0
    ]

    tp_values = [
        1.5,
        2.0,
        2.5,
        3.0
    ]

    combinations = list(
        itertools.product(
            ema_fast_values,
            ema_slow_values,
            ema_trend_values,
            rsi_values,
            atr_values,
            sl_values,
            tp_values
        )
    )

    print(
        "Parameter combinations:",
        len(combinations)
    )

    results = []

    for n, params in enumerate(
        combinations,
        1
    ):

        (
            ema_fast,
            ema_slow,
            ema_trend,
            rsi_period,
            atr_period,
            sl,
            tp
        ) = params

        if not (
            ema_fast
            < ema_slow
            < ema_trend
        ):
            continue

        train_features = features(
            train,
            ema_fast,
            ema_slow,
            ema_trend,
            rsi_period,
            atr_period
        )

        result = backtest(
            train_features,
            sl,
            tp
        )

        result.update({
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "ema_trend": ema_trend,
            "rsi_period": rsi_period,
            "atr_period": atr_period,
            "sl_multiplier": sl,
            "tp_multiplier": tp
        })

        results.append(result)

        if n % 100 == 0:

            print(
                f"Processed {n:,}/"
                f"{len(combinations):,}"
            )

    results_df = pd.DataFrame(results)

    results_df = results_df[
        results_df["trades"] >= 30
    ]

    results_df = results_df.sort_values(
        [
            "profit_factor",
            "profit",
            "win_rate"
        ],
        ascending=False
    )

    results_df.to_csv(
        OUTPUT,
        index=False
    )

    print()
    print("=" * 70)
    print("TOP TRAINING RESULTS")
    print("=" * 70)

    print(
        results_df.head(10).to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # Out-of-sample validation of top candidates.
    # -----------------------------------------------------

    top = results_df.head(10)

    validation = []

    for _, candidate in top.iterrows():

        test_features = features(
            test,
            int(candidate.ema_fast),
            int(candidate.ema_slow),
            int(candidate.ema_trend),
            int(candidate.rsi_period),
            int(candidate.atr_period)
        )

        result = backtest(
            test_features,
            candidate.sl_multiplier,
            candidate.tp_multiplier
        )

        result.update({
            "ema_fast":
                candidate.ema_fast,

            "ema_slow":
                candidate.ema_slow,

            "ema_trend":
                candidate.ema_trend,

            "rsi_period":
                candidate.rsi_period,

            "atr_period":
                candidate.atr_period,

            "sl_multiplier":
                candidate.sl_multiplier,

            "tp_multiplier":
                candidate.tp_multiplier
        })

        validation.append(result)

    validation_df = pd.DataFrame(
        validation
    )

    validation_df = validation_df.sort_values(
        "profit_factor",
        ascending=False
    )

    validation_df.to_csv(
        "reports/xauusd_validation.csv",
        index=False
    )

    print()
    print("=" * 70)
    print("OUT-OF-SAMPLE VALIDATION")
    print("=" * 70)

    print(
        validation_df.to_string(
            index=False
        )
    )

    print()
    print("Created:")
    print(
        "reports/optimization_results.csv"
    )
    print(
        "reports/xauusd_validation.csv"
    )


if __name__ == "__main__":
    main()
