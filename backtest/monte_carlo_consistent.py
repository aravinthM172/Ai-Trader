
import numpy as np
import pandas as pd
from pathlib import Path

DATA = "data/xauusd_m5_5y.csv"

TRADE_LOG = "reports/xauusd_trade_log_consistent.csv"
MC_FILE = "reports/xauusd_monte_carlo_consistent.csv"

INITIAL_BALANCE = 100.0

# Strategy from the out-of-sample winner
EMA_FAST = 10
EMA_SLOW = 80
EMA_TREND = 200

RSI_PERIOD = 10
ATR_PERIOD = 10

SL_ATR = 2.0
TP_ATR = 3.0

# Strategy filter
ADX_PERIOD = 14
ADX_MIN = 20.0

# Risk controls
RISK_PER_TRADE = 0.01
MAX_RISK_DOLLARS = 2.0

# Execution assumptions
SPREAD = 0.35
SLIPPAGE = 0.05

SIMULATIONS = 10000

# XAUUSD approximation.
# We deliberately use percentage risk rather than raw price movement.
# Broker-specific tick value will be added later from MT5.
PRICE_VALUE_PER_UNIT = 1.0


def ema(series, period):
    return series.ewm(
        span=period,
        adjust=False
    ).mean()


def rsi(series, period):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1/period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1/period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    value = 100 - (100 / (1 + rs))

    return value.fillna(50)


def atr(df, period):

    previous_close = df["close"].shift(1)

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    return tr.ewm(
        alpha=1/period,
        adjust=False
    ).mean()


def adx(df, period):

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

    previous_close = df["close"].shift(1)

    tr = pd.concat(
        [
            high-low,
            (high-previous_close).abs(),
            (low-previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    atr_value = tr.ewm(
        alpha=1/period,
        adjust=False
    ).mean()

    plus = (
        pd.Series(plus_dm, index=df.index)
        .ewm(alpha=1/period, adjust=False)
        .mean()
    )

    minus = (
        pd.Series(minus_dm, index=df.index)
        .ewm(alpha=1/period, adjust=False)
        .mean()
    )

    plus_di = 100 * plus / atr_value.replace(0, np.nan)
    minus_di = 100 * minus / atr_value.replace(0, np.nan)

    dx = (
        100 *
        (plus_di-minus_di).abs() /
        (plus_di+minus_di).replace(0, np.nan)
    )

    return dx.ewm(
        alpha=1/period,
        adjust=False
    ).mean().fillna(0)


def generate_trades(df):

    df = df.copy()

    df["ema_fast"] = ema(
        df["close"],
        EMA_FAST
    )

    df["ema_slow"] = ema(
        df["close"],
        EMA_SLOW
    )

    df["ema_trend"] = ema(
        df["close"],
        EMA_TREND
    )

    df["rsi"] = rsi(
        df["close"],
        RSI_PERIOD
    )

    df["atr"] = atr(
        df,
        ATR_PERIOD
    )

    df["adx"] = adx(
        df,
        ADX_PERIOD
    )

    start = max(
        EMA_TREND,
        RSI_PERIOD,
        ATR_PERIOD,
        ADX_PERIOD
    )

    balance = INITIAL_BALANCE

    position = None

    trades = []

    for i in range(start + 1, len(df)):

        row = df.iloc[i]
        previous = df.iloc[i-1]

        # --------------------------------------------------
        # MANAGE OPEN POSITION
        # --------------------------------------------------

        if position is not None:

            direction = position["direction"]
            sl = position["sl"]
            tp = position["tp"]

            exit_price = None
            exit_reason = None

            if direction == "BUY":

                if row["low"] <= sl:

                    exit_price = sl - SLIPPAGE
                    exit_reason = "SL"

                elif row["high"] >= tp:

                    exit_price = tp - SLIPPAGE
                    exit_reason = "TP"

                if exit_price is not None:

                    price_move = (
                        exit_price -
                        position["entry"]
                    )

                    pnl = (
                        price_move *
                        position["units"]
                    )

            else:

                if row["high"] >= sl:

                    exit_price = sl + SLIPPAGE
                    exit_reason = "SL"

                elif row["low"] <= tp:

                    exit_price = tp + SLIPPAGE
                    exit_reason = "TP"

                if exit_price is not None:

                    price_move = (
                        position["entry"] -
                        exit_price
                    )

                    pnl = (
                        price_move *
                        position["units"]
                    )

            if exit_price is not None:

                balance += pnl

                trades.append({
                    "entry_time": position["entry_time"],
                    "exit_time": row["time"],
                    "direction": direction,
                    "entry": position["entry"],
                    "exit": exit_price,
                    "sl": sl,
                    "tp": tp,
                    "atr": position["atr"],
                    "adx": position["adx"],
                    "units": position["units"],
                    "pnl": pnl,
                    "balance": balance,
                    "result": "WIN" if pnl > 0 else "LOSS",
                    "exit_reason": exit_reason
                })

                position = None

                continue

        # --------------------------------------------------
        # SIGNAL
        # --------------------------------------------------

        if position is not None:
            continue

        bullish = (
            previous["ema_fast"] >
            previous["ema_slow"]
            and
            previous["ema_slow"] >
            previous["ema_trend"]
            and
            previous["rsi"] > 55
            and
            previous["adx"] >= ADX_MIN
        )

        bearish = (
            previous["ema_fast"] <
            previous["ema_slow"]
            and
            previous["ema_slow"] <
            previous["ema_trend"]
            and
            previous["rsi"] < 45
            and
            previous["adx"] >= ADX_MIN
        )

        atr_value = previous["atr"]

        if not np.isfinite(atr_value):
            continue

        if atr_value <= 0:
            continue

        # --------------------------------------------------
        # POSITION SIZING
        # --------------------------------------------------

        risk_dollars = min(
            balance * RISK_PER_TRADE,
            MAX_RISK_DOLLARS
        )

        stop_distance = (
            atr_value * SL_ATR
        )

        if stop_distance <= 0:
            continue

        units = (
            risk_dollars /
            (
                stop_distance *
                PRICE_VALUE_PER_UNIT
            )
        )

        if units <= 0:
            continue

        # --------------------------------------------------
        # BUY
        # --------------------------------------------------

        if bullish:

            entry = (
                row["open"] +
                SPREAD / 2 +
                SLIPPAGE
            )

            sl = (
                entry -
                stop_distance
            )

            tp = (
                entry +
                atr_value * TP_ATR
            )

            position = {
                "direction": "BUY",
                "entry_time": row["time"],
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "atr": atr_value,
                "adx": previous["adx"],
                "units": units
            }

        # --------------------------------------------------
        # SELL
        # --------------------------------------------------

        elif bearish:

            entry = (
                row["open"] -
                SPREAD / 2 -
                SLIPPAGE
            )

            sl = (
                entry +
                stop_distance
            )

            tp = (
                entry -
                atr_value * TP_ATR
            )

            position = {
                "direction": "SELL",
                "entry_time": row["time"],
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "atr": atr_value,
                "adx": previous["adx"],
                "units": units
            }

    return pd.DataFrame(trades)


def monte_carlo(pnl):

    rng = np.random.default_rng(42)

    n = len(pnl)

    final_balance = np.zeros(
        SIMULATIONS
    )

    max_drawdown = np.zeros(
        SIMULATIONS
    )

    max_loss_streak = np.zeros(
        SIMULATIONS
    )

    ruin = np.zeros(
        SIMULATIONS
    )

    for simulation in range(SIMULATIONS):

        sequence = rng.permutation(pnl)

        balance = INITIAL_BALANCE
        peak = balance
        worst_dd = 0.0

        current_loss_streak = 0
        worst_loss_streak = 0

        for value in sequence:

            balance += value

            if value < 0:
                current_loss_streak += 1
            else:
                current_loss_streak = 0

            worst_loss_streak = max(
                worst_loss_streak,
                current_loss_streak
            )

            peak = max(
                peak,
                balance
            )

            dd = peak - balance

            worst_dd = max(
                worst_dd,
                dd
            )

            if balance <= 0:

                ruin[simulation] = 1

                break

        final_balance[simulation] = balance
        max_drawdown[simulation] = worst_dd
        max_loss_streak[simulation] = worst_loss_streak

    return pd.DataFrame({
        "final_balance": final_balance,
        "max_drawdown": max_drawdown,
        "max_loss_streak": max_loss_streak,
        "ruin": ruin
    })


def main():

    print("=" * 70)
    print("CONSISTENT XAUUSD BACKTEST + MONTE CARLO")
    print("=" * 70)

    if not Path(DATA).exists():

        print(
            f"ERROR: {DATA} not found"
        )

        return

    df = pd.read_csv(DATA)

    df["time"] = pd.to_datetime(
        df["time"],
        utc=True
    )

    df = df.sort_values(
        "time"
    ).reset_index(drop=True)

    # IMPORTANT:
    # Only unseen 30%.
    split = int(
        len(df) * 0.70
    )

    test = df.iloc[
        split:
    ].reset_index(drop=True)

    print()
    print(
        f"Total candles : {len(df):,}"
    )

    print(
        f"Test candles  : {len(test):,}"
    )

    print()
    print("Generating trades...")

    trades = generate_trades(test)

    if len(trades) == 0:

        print(
            "ERROR: No trades generated."
        )

        return

    trades.to_csv(
        TRADE_LOG,
        index=False
    )

    pnl = trades[
        "pnl"
    ].values.astype(
        np.float64
    )

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    gross_profit = (
        wins.sum()
        if len(wins)
        else 0
    )

    gross_loss = (
        abs(losses.sum())
        if len(losses)
        else 0
    )

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else 999
    )

    win_rate = (
        len(wins) / len(pnl)
    )

    balance_curve = (
        INITIAL_BALANCE +
        np.cumsum(pnl)
    )

    running_peak = np.maximum.accumulate(
        balance_curve
    )

    drawdown = (
        running_peak -
        balance_curve
    )

    max_dd = drawdown.max()

    print()
    print("=" * 70)
    print("HISTORICAL TEST")
    print("=" * 70)

    print(
        f"Trades         : {len(pnl):,}"
    )

    print(
        f"Profit         : ${pnl.sum():.2f}"
    )

    print(
        f"Final balance  : ${balance_curve[-1]:.2f}"
    )

    print(
        f"Win rate       : {win_rate*100:.2f}%"
    )

    print(
        f"Profit factor  : {profit_factor:.3f}"
    )

    print(
        f"Max drawdown   : ${max_dd:.2f}"
    )

    print()
    print("Running 10,000 Monte Carlo simulations...")

    mc = monte_carlo(pnl)

    mc.to_csv(
        MC_FILE,
        index=False
    )

    profitable = (
        mc["final_balance"] >
        INITIAL_BALANCE
    ).mean()

    ruin_probability = (
        mc["ruin"].mean()
    )

    print()
    print("=" * 70)
    print("MONTE CARLO RESULTS")
    print("=" * 70)

    print(
        f"Simulations          : {SIMULATIONS:,}"
    )

    print(
        f"Probability profitable: "
        f"{profitable*100:.2f}%"
    )

    print(
        f"Probability of ruin   : "
        f"{ruin_probability*100:.2f}%"
    )

    print(
        f"5% final balance     : "
        f"${mc.final_balance.quantile(.05):.2f}"
    )

    print(
        f"Median final balance : "
        f"${mc.final_balance.quantile(.50):.2f}"
    )

    print(
        f"95% final balance    : "
        f"${mc.final_balance.quantile(.95):.2f}"
    )

    print(
        f"95% max drawdown     : "
        f"${mc.max_drawdown.quantile(.95):.2f}"
    )

    print(
        f"95% loss streak      : "
        f"{mc.max_loss_streak.quantile(.95):.0f}"
    )

    print()
    print("Files:")
    print(TRADE_LOG)
    print(MC_FILE)


if __name__ == "__main__":
    main()
