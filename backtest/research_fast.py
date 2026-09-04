
import time
import numpy as np
import pandas as pd
from numba import njit, prange

DATA = "data/xauusd_m5_5y.csv"
OUT = "reports/xauusd_strategy_research.csv"

INITIAL_BALANCE = 100.0
RISK = 0.01

@njit
def ema(x, period):
    n = len(x)
    out = np.empty(n)
    out[0] = x[0]

    a = 2.0 / (period + 1.0)

    for i in range(1, n):
        out[i] = a*x[i] + (1-a)*out[i-1]

    return out


@njit
def rsi(x, period):

    n = len(x)

    out = np.full(n, 50.0)

    gain = 0.0
    loss = 0.0

    for i in range(1, n):

        d = x[i] - x[i-1]

        g = max(d, 0.0)
        l = max(-d, 0.0)

        gain = (gain*(period-1)+g)/period
        loss = (loss*(period-1)+l)/period

        if loss > 0:

            rs = gain/loss

            out[i] = 100.0 - 100.0/(1.0+rs)

    return out


@njit
def atr(high, low, close, period):

    n = len(close)

    out = np.zeros(n)

    tr_prev = 0.0

    for i in range(1, n):

        tr = max(
            high[i]-low[i],
            abs(high[i]-close[i-1]),
            abs(low[i]-close[i-1])
        )

        tr_prev = (
            (tr_prev*(period-1)+tr)
            /period
        )

        out[i] = tr_prev

    return out


@njit
def backtest(
    close,
    high,
    low,
    fast,
    slow,
    trend,
    rsi_period,
    rsi_buy,
    rsi_sell,
    atr_period,
    sl_mult,
    tp_mult
):

    n = len(close)

    ef = ema(close, fast)
    es = ema(close, slow)
    et = ema(close, trend)

    rv = rsi(
        close,
        rsi_period
    )

    av = atr(
        high,
        low,
        close,
        atr_period
    )

    balance = INITIAL_BALANCE
    peak = balance
    max_dd = 0.0

    position = 0
    entry = 0.0
    sl = 0.0
    tp = 0.0

    trades = 0
    wins = 0
    gross_profit = 0.0
    gross_loss = 0.0

    start = max(
        fast,
        slow,
        trend,
        rsi_period,
        atr_period
    ) + 2

    for i in range(start, n):

        # Manage position.
        if position == 1:

            if low[i] <= sl:

                pnl = sl-entry

                balance += pnl
                trades += 1

                if pnl > 0:
                    wins += 1
                    gross_profit += pnl
                else:
                    gross_loss -= pnl

                position = 0

            elif high[i] >= tp:

                pnl = tp-entry

                balance += pnl
                trades += 1
                wins += 1
                gross_profit += pnl

                position = 0

        elif position == -1:

            if high[i] >= sl:

                pnl = entry-sl

                balance += pnl
                trades += 1

                if pnl > 0:
                    wins += 1
                    gross_profit += pnl
                else:
                    gross_loss -= pnl

                position = 0

            elif low[i] <= tp:

                pnl = entry-tp

                balance += pnl
                trades += 1
                wins += 1
                gross_profit += pnl

                position = 0

        # Drawdown.
        if balance > peak:
            peak = balance

        dd = peak-balance

        if dd > max_dd:
            max_dd = dd

        # Stop if account is dead.
        if balance <= 1.0:
            break

        if position != 0:
            continue

        a = av[i-1]

        if a <= 0:
            continue

        buy = (
            ef[i-1] > es[i-1]
            and es[i-1] > et[i-1]
            and rv[i-1] >= rsi_buy
        )

        sell = (
            ef[i-1] < es[i-1]
            and es[i-1] < et[i-1]
            and rv[i-1] <= rsi_sell
        )

        # Risk based sizing.
        risk_dollars = balance * RISK

        stop_distance = a*sl_mult

        units = (
            risk_dollars /
            stop_distance
        )

        if buy:

            entry = close[i]

            sl = (
                entry -
                stop_distance
            )

            tp = (
                entry +
                a*tp_mult
            )

            position = 1

        elif sell:

            entry = close[i]

            sl = (
                entry +
                stop_distance
            )

            tp = (
                entry -
                a*tp_mult
            )

            position = -1

    win_rate = (
        wins/trades
        if trades > 0
        else 0
    )

    pf = (
        gross_profit/gross_loss
        if gross_loss > 0
        else 0
    )

    return (
        trades,
        balance-INITIAL_BALANCE,
        win_rate,
        pf,
        max_dd
    )


def main():

    print("="*70)
    print("FAST XAUUSD STRATEGY RESEARCH")
    print("="*70)

    df = pd.read_csv(DATA)

    close = df["close"].values.astype(
        np.float64
    )

    high = df["high"].values.astype(
        np.float64
    )

    low = df["low"].values.astype(
        np.float64
    )

    # 70/15/15 split.
    n = len(df)

    train_end = int(n*0.70)
    validation_end = int(n*0.85)

    train_close = close[:train_end]
    train_high = high[:train_end]
    train_low = low[:train_end]

    val_close = close[
        train_end:validation_end
    ]

    val_high = high[
        train_end:validation_end
    ]

    val_low = low[
        train_end:validation_end
    ]

    test_close = close[
        validation_end:
    ]

    test_high = high[
        validation_end:
    ]

    test_low = low[
        validation_end:
    ]

    fasts = [5,10,15,20,30]
    slows = [30,40,50,60,80]
    trends = [100,150,200]
    rsi_periods = [7,10,14,20]
    rsi_buy = [50,55,60]
    rsi_sell = [50,45,40]
    atr_periods = [10,14,20]
    sls = [1.5,2.0,2.5]
    tps = [2.0,2.5,3.0,3.5]

    configs = []

    for f in fasts:

        for s in slows:

            if f >= s:
                continue

            for t in trends:

                if s >= t:
                    continue

                for rp in rsi_periods:

                    for rb, rs in zip(
                        rsi_buy,
                        rsi_sell
                    ):

                        for ap in atr_periods:

                            for sl in sls:

                                for tp in tps:

                                    configs.append(
                                        (
                                            f,s,t,
                                            rp,rb,rs,
                                            ap,sl,tp
                                        )
                                    )

    print()
    print(
        f"Configurations: {len(configs):,}"
    )

    # Warm up Numba.
    print("Compiling Numba engine...")

    backtest(
        train_close,
        train_high,
        train_low,
        10,50,200,
        14,55,45,
        14,2.0,3.0
    )

    print("Compilation complete.")
    print()

    results = []

    start_time = time.time()

    total = len(configs)

    for idx,p in enumerate(configs,1):

        f,s,t,rp,rb,rs,ap,sl,tp = p

        tr = backtest(
            train_close,
            train_high,
            train_low,
            f,s,t,rp,rb,rs,ap,sl,tp
        )

        # Only carry promising training candidates
        # to validation.
        if (
            tr[3] < 1.05
            or tr[0] < 100
            or tr[1] <= 0
        ):
            continue

        va = backtest(
            val_close,
            val_high,
            val_low,
            f,s,t,rp,rb,rs,ap,sl,tp
        )

        if va[0] < 50:
            continue

        results.append({

            "fast": f,
            "slow": s,
            "trend": t,
            "rsi_period": rp,
            "rsi_buy": rb,
            "rsi_sell": rs,
            "atr_period": ap,
            "sl": sl,
            "tp": tp,

            "train_trades": tr[0],
            "train_profit": tr[1],
            "train_winrate": tr[2],
            "train_pf": tr[3],
            "train_dd": tr[4],

            "val_trades": va[0],
            "val_profit": va[1],
            "val_winrate": va[2],
            "val_pf": va[3],
            "val_dd": va[4]
        })

        if idx % 250 == 0 or idx == total:

            elapsed = time.time()-start_time

            speed = idx/elapsed

            eta = (
                total-idx
            )/speed if speed else 0

            print(
                f"Progress: {idx:,}/{total:,} "
                f"({idx/total*100:.1f}%) | "
                f"Speed: {speed:.1f}/sec | "
                f"ETA: {eta:.0f}s | "
                f"Valid: {len(results):,}",
                flush=True
            )

    if not results:

        print()
        print("No promising strategies found.")
        return

    out = pd.DataFrame(results)

    # Rank primarily on validation PF,
    # then validation profit and drawdown.
    out = out.sort_values(
        [
            "val_pf",
            "val_profit"
        ],
        ascending=False
    )

    out.to_csv(
        OUT,
        index=False
    )

    print()
    print("="*70)
    print("TOP VALIDATION STRATEGIES")
    print("="*70)

    print(
        out.head(20).to_string(
            index=False
        )
    )

    # Final untouched test only for top 20.
    print()
    print("Running untouched final test...")

    final_rows = []

    for _,r in out.head(20).iterrows():

        te = backtest(
            test_close,
            test_high,
            test_low,
            int(r.fast),
            int(r.slow),
            int(r.trend),
            int(r.rsi_period),
            int(r.rsi_buy),
            int(r.rsi_sell),
            int(r.atr_period),
            float(r.sl),
            float(r.tp)
        )

        final_rows.append({
            **r.to_dict(),
            "test_trades": te[0],
            "test_profit": te[1],
            "test_winrate": te[2],
            "test_pf": te[3],
            "test_dd": te[4]
        })

    final = pd.DataFrame(
        final_rows
    )

    final.to_csv(
        "reports/xauusd_final_candidates.csv",
        index=False
    )

    print()
    print("="*70)
    print("FINAL UNTOUCHED TEST")
    print("="*70)

    print(
        final[
            [
                "fast",
                "slow",
                "trend",
                "rsi_period",
                "rsi_buy",
                "rsi_sell",
                "atr_period",
                "sl",
                "tp",
                "test_trades",
                "test_profit",
                "test_winrate",
                "test_pf",
                "test_dd"
            ]
        ].to_string(index=False)
    )

    print()
    print("Saved:")
    print(OUT)
    print("reports/xauusd_final_candidates.csv")


if __name__ == "__main__":
    main()
