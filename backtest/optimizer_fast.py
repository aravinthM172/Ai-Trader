import os
import time
import itertools
import pandas as pd
import numpy as np

DATA_FILE = "data/xauusd_m5_5y.csv"
OUTPUT = "reports/xauusd_optimization.csv"
VALIDATION_OUTPUT = "reports/xauusd_validation.csv"

INITIAL_BALANCE = 100.0
RISK_PER_TRADE = 0.005
SPREAD = 0.35
SLIPPAGE = 0.05


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    delta = s.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / n,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / n,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def atr(df, n=14):

    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ],
        axis=1
    ).max(axis=1)

    return tr.ewm(
        alpha=1 / n,
        adjust=False
    ).mean()


def adx(df, n=14):

    high = df["high"]
    low = df["low"]

    up = high.diff()
    down = -low.diff()

    plus_dm = np.where(
        (up > down) & (up > 0),
        up,
        0
    )

    minus_dm = np.where(
        (down > up) & (down > 0),
        down,
        0
    )

    tr = pd.concat(
        [
            high - low,
            (high - df["close"].shift()).abs(),
            (low - df["close"].shift()).abs()
        ],
        axis=1
    ).max(axis=1)

    atr_value = tr.ewm(
        alpha=1 / n,
        adjust=False
    ).mean()

    plus_di = (
        100
        * pd.Series(plus_dm, index=df.index)
        .ewm(alpha=1 / n, adjust=False)
        .mean()
        / atr_value
    )

    minus_di = (
        100
        * pd.Series(minus_dm, index=df.index)
        .ewm(alpha=1 / n, adjust=False)
        .mean()
        / atr_value
    )

    dx = (
        100
        * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(
            0,
            np.nan
        )
    )

    return dx.ewm(
        alpha=1 / n,
        adjust=False
    ).mean()


def load():

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

    return df


def prepare(df):

    print("Preparing indicators...")

    x = df.copy()

    ema_periods = [
        10, 20, 30,
        40, 50, 80,
        100, 150, 200
    ]

    for n in ema_periods:

        x[f"ema{n}"] = ema(
            x["close"],
            n
        )

    for n in [10, 14, 20]:

        x[f"rsi{n}"] = rsi(
            x["close"],
            n
        )

        x[f"atr{n}"] = atr(
            x,
            n
        )

    x["adx"] = adx(x)

    # MACD
    fast = ema(
        x["close"],
        12
    )

    slow = ema(
        x["close"],
        26
    )

    x["macd"] = fast - slow

    return x.dropna().reset_index(
        drop=True
    )


def make_signal(
    df,
    fast,
    slow,
    trend,
    rsi_period
):

    bull = (
        (df[f"ema{fast}"] > df[f"ema{slow}"]).astype(int)
        +
        (df[f"ema{slow}"] > df[f"ema{trend}"]).astype(int)
        +
        (df[f"rsi{rsi_period}"] > 55).astype(int)
        +
        (df["macd"] > 0).astype(int)
    )

    bear = (
        (df[f"ema{fast}"] < df[f"ema{slow}"]).astype(int)
        +
        (df[f"ema{slow}"] < df[f"ema{trend}"]).astype(int)
        +
        (df[f"rsi{rsi_period}"] < 45).astype(int)
        +
        (df["macd"] < 0).astype(int)
    )

    buy = (
        (bull >= 3)
        & (df["adx"] >= 20)
    )

    sell = (
        (bear >= 3)
        & (df["adx"] >= 20)
    )

    signal = np.zeros(
        len(df),
        dtype=np.int8
    )

    signal[buy] = 1
    signal[sell] = -1

    return signal


def fast_backtest(
    df,
    signal,
    atr_period,
    sl_mult,
    tp_mult
):

    entry_signal = np.roll(
        signal,
        1
    )

    entry_signal[0] = 0

    entry_price = (
        df["open"].values
        + np.where(
            entry_signal == 1,
            SPREAD / 2 + SLIPPAGE,
            np.where(
                entry_signal == -1,
                -SPREAD / 2 - SLIPPAGE,
                0
            )
        )
    )

    atr_values = df[
        f"atr{atr_period}"
    ].values

    long_sl = (
        entry_price
        - atr_values * sl_mult
    )

    long_tp = (
        entry_price
        + atr_values * tp_mult
    )

    short_sl = (
        entry_price
        + atr_values * sl_mult
    )

    short_tp = (
        entry_price
        - atr_values * tp_mult
    )

    balance = INITIAL_BALANCE

    trades = []

    position = 0

    entry = 0
    sl = 0
    tp = 0
    size = 0

    for i in range(
        1,
        len(df)
    ):

        high = df["high"].iloc[i]
        low = df["low"].iloc[i]

        # Position management
        if position == 1:

            if low <= sl:

                exit_price = sl - SLIPPAGE

                pnl = (
                    exit_price - entry
                ) * size

                balance += pnl

                trades.append(pnl)

                position = 0

            elif high >= tp:

                exit_price = tp - SLIPPAGE

                pnl = (
                    exit_price - entry
                ) * size

                balance += pnl

                trades.append(pnl)

                position = 0

        elif position == -1:

            if high >= sl:

                exit_price = sl + SLIPPAGE

                pnl = (
                    entry - exit_price
                ) * size

                balance += pnl

                trades.append(pnl)

                position = 0

            elif low <= tp:

                exit_price = tp + SLIPPAGE

                pnl = (
                    entry - exit_price
                ) * size

                balance += pnl

                trades.append(pnl)

                position = 0

        # New position
        if position == 0:

            direction = entry_signal[i]

            if direction == 0:
                continue

            atr_value = atr_values[i - 1]

            if not np.isfinite(
                atr_value
            ):
                continue

            if direction == 1:

                entry = (
                    df["open"].iloc[i]
                    + SPREAD / 2
                    + SLIPPAGE
                )

                sl = (
                    entry
                    - atr_value * sl_mult
                )

                tp = (
                    entry
                    + atr_value * tp_mult
                )

            else:

                entry = (
                    df["open"].iloc[i]
                    - SPREAD / 2
                    - SLIPPAGE
                )

                sl = (
                    entry
                    + atr_value * sl_mult
                )

                tp = (
                    entry
                    - atr_value * tp_mult
                )

            risk_money = (
                balance
                * RISK_PER_TRADE
            )

            distance = abs(
                entry - sl
            )

            if distance <= 0:
                continue

            size = (
                risk_money
                / distance
            )

            position = direction

    if not trades:

        return {
            "trades": 0,
            "profit": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "max_drawdown": 0
        }

    trades = np.array(
        trades
    )

    wins = trades[
        trades > 0
    ]

    losses = trades[
        trades < 0
    ]

    gross_profit = wins.sum()

    gross_loss = abs(
        losses.sum()
    )

    equity = (
        INITIAL_BALANCE
        + np.cumsum(trades)
    )

    peak = np.maximum.accumulate(
        equity
    )

    drawdown = (
        peak - equity
    )

    return {
        "trades": len(trades),

        "profit":
            balance - INITIAL_BALANCE,

        "win_rate":
            len(wins) / len(trades),

        "profit_factor":
            (
                gross_profit / gross_loss
                if gross_loss > 0
                else 999
            ),

        "max_drawdown":
            drawdown.max()
    }


def main():

    os.makedirs(
        "reports",
        exist_ok=True
    )

    print("=" * 70)
    print("XAUUSD FAST OPTIMIZER")
    print("=" * 70)

    df = load()

    df = prepare(df)

    split = int(
        len(df) * 0.70
    )

    train = df.iloc[
        :split
    ].reset_index(drop=True)

    test = df.iloc[
        split:
    ].reset_index(drop=True)

    print()
    print(
        "Training:",
        f"{len(train):,}"
    )

    print(
        "Testing:",
        f"{len(test):,}"
    )

    combinations = []

    for fast in [10, 20, 30]:

        for slow in [40, 50, 80]:

            for trend in [100, 150, 200]:

                if not (
                    fast < slow < trend
                ):
                    continue

                for rsi_period in [
                    10, 14, 20
                ]:

                    for atr_period in [
                        10, 14, 20
                    ]:

                        for sl in [
                            1.0,
                            1.5,
                            2.0
                        ]:

                            for tp in [
                                1.5,
                                2.0,
                                2.5,
                                3.0
                            ]:

                                combinations.append(
                                    (
                                        fast,
                                        slow,
                                        trend,
                                        rsi_period,
                                        atr_period,
                                        sl,
                                        tp
                                    )
                                )

    print(
        "Configurations:",
        len(combinations)
    )

    results = []

    start_time = time.time()

    print()
    print("Optimization started...")
    print("Live progress every 25 configurations.")
    print()

    for n, p in enumerate(
        combinations,
        1
    ):

        fast, slow, trend, rsi_period, atr_period, sl, tp = p

        signal = make_signal(
            train,
            fast,
            slow,
            trend,
            rsi_period
        )

        result = fast_backtest(
            train,
            signal,
            atr_period,
            sl,
            tp
        )

        result.update({
            "ema_fast": fast,
            "ema_slow": slow,
            "ema_trend": trend,
            "rsi_period": rsi_period,
            "atr_period": atr_period,
            "sl": sl,
            "tp": tp
        })

        results.append(result)

        if n % 25 == 0 or n == 1:

            elapsed = time.time() - start_time

            rate = (
                n / elapsed
                if elapsed > 0
                else 0
            )

            remaining = (
                (len(combinations) - n) / rate
                if rate > 0
                else 0
            )

            percent = (
                n / len(combinations) * 100
            )

            minutes = int(
                remaining // 60
            )

            seconds = int(
                remaining % 60
            )

            print(
                f"Progress: {n:,}/{len(combinations):,} "
                f"({percent:.1f}%) | "
                f"Speed: {rate:.2f}/sec | "
                f"Elapsed: {int(elapsed)//60}m "
                f"{int(elapsed)%60}s | "
                f"ETA: {minutes}m {seconds}s",
                flush=True
            )

    results = pd.DataFrame(
        results
    )

    results = results[
        results["trades"] >= 30
    ]

    results = results.sort_values(
        [
            "profit_factor",
            "profit",
            "win_rate"
        ],
        ascending=False
    )

    results.to_csv(
        OUTPUT,
        index=False
    )

    print()
    print("=" * 70)
    print("TOP TRAINING RESULTS")
    print("=" * 70)

    print(
        results.head(10).to_string(
            index=False
        )
    )

    # OOS validation
    validation = []

    for _, c in results.head(10).iterrows():

        signal = make_signal(
            test,
            int(c.ema_fast),
            int(c.ema_slow),
            int(c.ema_trend),
            int(c.rsi_period)
        )

        result = fast_backtest(
            test,
            signal,
            int(c.atr_period),
            float(c.sl),
            float(c.tp)
        )

        result.update({
            "ema_fast": c.ema_fast,
            "ema_slow": c.ema_slow,
            "ema_trend": c.ema_trend,
            "rsi_period": c.rsi_period,
            "atr_period": c.atr_period,
            "sl": c.sl,
            "tp": c.tp
        })

        validation.append(
            result
        )

    validation = pd.DataFrame(
        validation
    )

    validation = validation.sort_values(
        "profit_factor",
        ascending=False
    )

    validation.to_csv(
        VALIDATION_OUTPUT,
        index=False
    )

    print()
    print("=" * 70)
    print("OUT-OF-SAMPLE VALIDATION")
    print("=" * 70)

    print(
        validation.to_string(
            index=False
        )
    )

    print()
    print("Saved:")
    print(OUTPUT)
    print(VALIDATION_OUTPUT)


if __name__ == "__main__":

    main()

