
import os
import time
import itertools
import pandas as pd
import numpy as np
from numba import njit

DATA_FILE = "data/xauusd_m5_5y.csv"
CHECKPOINT = "reports/xauusd_checkpoint.csv"
FINAL_FILE = "reports/xauusd_optimization.csv"
VALIDATION_FILE = "reports/xauusd_validation.csv"

INITIAL_BALANCE = 100.0
RISK_PER_TRADE = 0.005
SPREAD = 0.35
SLIPPAGE = 0.05


def ema(x, period):
    return pd.Series(x).ewm(
        span=period,
        adjust=False
    ).mean().values


def rsi(x, period):
    d = np.diff(x, prepend=x[0])

    gain = np.maximum(d, 0)
    loss = np.maximum(-d, 0)

    ag = pd.Series(gain).ewm(
        alpha=1 / period,
        adjust=False
    ).mean().values

    al = pd.Series(loss).ewm(
        alpha=1 / period,
        adjust=False
    ).mean().values

    rs = ag / np.where(al == 0, np.nan, al)

    return 100 - 100 / (1 + rs)


def atr(high, low, close, period):

    prev = np.roll(close, 1)
    prev[0] = close[0]

    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - prev),
            np.abs(low - prev)
        )
    )

    return pd.Series(tr).ewm(
        alpha=1 / period,
        adjust=False
    ).mean().values


def adx(high, low, close, period=14):

    up = np.diff(high, prepend=high[0])
    down = -np.diff(low, prepend=low[0])

    plus = np.where(
        (up > down) & (up > 0),
        up,
        0
    )

    minus = np.where(
        (down > up) & (down > 0),
        down,
        0
    )

    prev = np.roll(close, 1)
    prev[0] = close[0]

    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - prev),
            np.abs(low - prev)
        )
    )

    atr_v = pd.Series(tr).ewm(
        alpha=1 / period,
        adjust=False
    ).mean().values

    pdi = (
        100
        * pd.Series(plus).ewm(
            alpha=1 / period,
            adjust=False
        ).mean().values
        / atr_v
    )

    mdi = (
        100
        * pd.Series(minus).ewm(
            alpha=1 / period,
            adjust=False
        ).mean().values
        / atr_v
    )

    dx = (
        100
        * np.abs(pdi - mdi)
        / np.where(
            pdi + mdi == 0,
            np.nan,
            pdi + mdi
        )
    )

    return pd.Series(dx).ewm(
        alpha=1 / period,
        adjust=False
    ).mean().values


def prepare(df):

    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)

    data = {
        "open": df["open"].values.astype(np.float64),
        "high": high,
        "low": low,
        "close": close
    }

    for p in [10,20,30,40,50,80,100,150,200]:
        data[f"ema{p}"] = ema(close, p)

    for p in [10,14,20]:
        data[f"rsi{p}"] = rsi(close, p)
        data[f"atr{p}"] = atr(
            high,
            low,
            close,
            p
        )

    data["adx"] = adx(
        high,
        low,
        close
    )

    e12 = ema(close, 12)
    e26 = ema(close, 26)

    data["macd"] = e12 - e26

    return data


@njit(cache=True)
def backtest(
    opens,
    highs,
    lows,
    ema_fast,
    ema_slow,
    ema_trend,
    rsi_values,
    atr_values,
    adx_values,
    macd_values,
    sl_mult,
    tp_mult
):

    balance = INITIAL_BALANCE

    position = 0
    entry = 0.0
    sl = 0.0
    tp = 0.0
    size = 0.0

    trades = 0
    wins = 0
    gross_profit = 0.0
    gross_loss = 0.0

    peak = balance
    max_dd = 0.0

    n = len(opens)

    for i in range(1, n):

        # Manage existing position.
        if position == 1:

            if lows[i] <= sl:

                exit_price = sl - SLIPPAGE

                pnl = (
                    exit_price - entry
                ) * size

                balance += pnl

                trades += 1

                if pnl > 0:
                    wins += 1
                    gross_profit += pnl
                else:
                    gross_loss += -pnl

                position = 0

            elif highs[i] >= tp:

                exit_price = tp - SLIPPAGE

                pnl = (
                    exit_price - entry
                ) * size

                balance += pnl

                trades += 1

                if pnl > 0:
                    wins += 1
                    gross_profit += pnl
                else:
                    gross_loss += -pnl

                position = 0

        elif position == -1:

            if highs[i] >= sl:

                exit_price = sl + SLIPPAGE

                pnl = (
                    entry - exit_price
                ) * size

                balance += pnl

                trades += 1

                if pnl > 0:
                    wins += 1
                    gross_profit += pnl
                else:
                    gross_loss += -pnl

                position = 0

            elif lows[i] <= tp:

                exit_price = tp + SLIPPAGE

                pnl = (
                    entry - exit_price
                ) * size

                balance += pnl

                trades += 1

                if pnl > 0:
                    wins += 1
                    gross_profit += pnl
                else:
                    gross_loss += -pnl

                position = 0

        # Generate signal from previous candle.
        bull = 0
        bear = 0

        if ema_fast[i-1] > ema_slow[i-1]:
            bull += 1
        else:
            bear += 1

        if ema_slow[i-1] > ema_trend[i-1]:
            bull += 1
        else:
            bear += 1

        if rsi_values[i-1] > 55:
            bull += 1
        elif rsi_values[i-1] < 45:
            bear += 1

        if macd_values[i-1] > 0:
            bull += 1
        else:
            bear += 1

        signal = 0

        if adx_values[i-1] >= 20:

            if bull >= 3:
                signal = 1

            elif bear >= 3:
                signal = -1

        # Open new trade.
        if position == 0 and signal != 0:

            a = atr_values[i-1]

            if a <= 0:
                continue

            if signal == 1:

                entry = (
                    opens[i]
                    + SPREAD / 2
                    + SLIPPAGE
                )

                sl = (
                    entry
                    - a * sl_mult
                )

                tp = (
                    entry
                    + a * tp_mult
                )

            else:

                entry = (
                    opens[i]
                    - SPREAD / 2
                    - SLIPPAGE
                )

                sl = (
                    entry
                    + a * sl_mult
                )

                tp = (
                    entry
                    - a * tp_mult
                )

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
                risk_money / distance
            )

            position = signal

        if balance > peak:
            peak = balance

        dd = peak - balance

        if dd > max_dd:
            max_dd = dd

    if gross_loss > 0:
        pf = gross_profit / gross_loss
    else:
        pf = 999.0

    if trades > 0:
        win_rate = wins / trades
    else:
        win_rate = 0.0

    return (
        trades,
        balance - INITIAL_BALANCE,
        win_rate,
        pf,
        max_dd
    )


def main():

    print("=" * 70)
    print("XAUUSD NUMBA FAST OPTIMIZER")
    print("=" * 70)

    df = pd.read_csv(DATA_FILE)

    df["time"] = pd.to_datetime(
        df["time"],
        utc=True
    )

    df = (
        df.sort_values("time")
        .drop_duplicates("time")
        .reset_index(drop=True)
    )

    split = int(
        len(df) * 0.70
    )

    train_df = df.iloc[
        :split
    ].reset_index(drop=True)

    test_df = df.iloc[
        split:
    ].reset_index(drop=True)

    print(
        f"Training candles: {len(train_df):,}"
    )

    print(
        f"Testing candles: {len(test_df):,}"
    )

    print()
    print("Preparing indicators ONCE...")

    train = prepare(train_df)

    print("Indicators ready.")

    combinations = []

    for fast in [10,20,30]:

        for slow in [40,50,80]:

            for trend in [100,150,200]:

                if not (
                    fast < slow < trend
                ):
                    continue

                for rsi_p in [10,14,20]:

                    for atr_p in [10,14,20]:

                        for sl in [1.0,1.5,2.0]:

                            for tp in [1.5,2.0,2.5,3.0]:

                                combinations.append(
                                    (
                                        fast,
                                        slow,
                                        trend,
                                        rsi_p,
                                        atr_p,
                                        sl,
                                        tp
                                    )
                                )

    total = len(combinations)

    print(
        f"Configurations: {total:,}"
    )

    # Resume support.
    completed = 0
    results = []

    if os.path.exists(CHECKPOINT):

        old = pd.read_csv(
            CHECKPOINT
        )

        if len(old) > 0:

            results = old.to_dict(
                "records"
            )

            completed = len(results)

            print()
            print(
                f"Checkpoint found: "
                f"{completed:,}/{total:,}"
            )

            print(
                "Resuming..."
            )

    else:

        print()
        print(
            "No checkpoint found."
        )

    start = time.time()

    for n in range(
        completed,
        total
    ):

        p = combinations[n]

        fast, slow, trend, rsi_p, atr_p, sl, tp = p

        result = backtest(
            train["open"],
            train["high"],
            train["low"],
            train[f"ema{fast}"],
            train[f"ema{slow}"],
            train[f"ema{trend}"],
            train[f"rsi{rsi_p}"],
            train[f"atr{atr_p}"],
            train["adx"],
            train["macd"],
            sl,
            tp
        )

        trades, profit, win_rate, pf, max_dd = result

        results.append({
            "ema_fast": fast,
            "ema_slow": slow,
            "ema_trend": trend,
            "rsi_period": rsi_p,
            "atr_period": atr_p,
            "sl": sl,
            "tp": tp,
            "trades": trades,
            "profit": profit,
            "win_rate": win_rate,
            "profit_factor": pf,
            "max_drawdown": max_dd
        })

        done = n + 1

        # Save checkpoint every 25.
        if done % 25 == 0:

            checkpoint_df = pd.DataFrame(
                results
            )

            checkpoint_df.to_csv(
                CHECKPOINT,
                index=False
            )

            elapsed = (
                time.time() - start
            )

            rate = (
                (done - completed) / elapsed
                if elapsed > 0
                else 0
            )

            remaining_count = (
                total - done
            )

            eta = (
                remaining_count / rate
                if rate > 0
                else 0
            )

            pct = (
                done / total * 100
            )

            print(
                f"Progress: {done:,}/{total:,} "
                f"({pct:.1f}%) | "
                f"Speed: {rate:.2f}/sec | "
                f"Elapsed: {int(elapsed)//60}m "
                f"{int(elapsed)%60}s | "
                f"ETA: {int(eta)//60}m "
                f"{int(eta)%60}s",
                flush=True
            )

    results_df = pd.DataFrame(
        results
    )

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
        FINAL_FILE,
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

    # Prepare test data.
    print()
    print(
        "Preparing out-of-sample data..."
    )

    test = prepare(test_df)

    validation = []

    for _, c in results_df.head(
        10
    ).iterrows():

        result = backtest(
            test["open"],
            test["high"],
            test["low"],
            test[f"ema{int(c.ema_fast)}"],
            test[f"ema{int(c.ema_slow)}"],
            test[f"ema{int(c.ema_trend)}"],
            test[f"rsi{int(c.rsi_period)}"],
            test[f"atr{int(c.atr_period)}"],
            test["adx"],
            test["macd"],
            float(c.sl),
            float(c.tp)
        )

        validation.append({
            "ema_fast": c.ema_fast,
            "ema_slow": c.ema_slow,
            "ema_trend": c.ema_trend,
            "rsi_period": c.rsi_period,
            "atr_period": c.atr_period,
            "sl": c.sl,
            "tp": c.tp,
            "trades": result[0],
            "profit": result[1],
            "win_rate": result[2],
            "profit_factor": result[3],
            "max_drawdown": result[4]
        })

    validation_df = pd.DataFrame(
        validation
    )

    validation_df = validation_df.sort_values(
        "profit_factor",
        ascending=False
    )

    validation_df.to_csv(
        VALIDATION_FILE,
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
    print("Files:")
    print(FINAL_FILE)
    print(VALIDATION_FILE)
    print(CHECKPOINT)


if __name__ == "__main__":
    main()
