
import time
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit

DATA = "data/xauusd_m5_5y.csv"
VALIDATION = "reports/xauusd_validation.csv"
OUTPUT = "reports/xauusd_walkforward.csv"

@njit(cache=True)
def simulate(close, high, low, fast, slow, trend, rsi_p, atr_p, sl_mult, tp_mult):

    n = len(close)

    balance = 100.0
    peak = balance
    max_dd = 0.0

    position = 0
    entry = 0.0
    sl = 0.0
    tp = 0.0

    trades = 0
    wins = 0
    gp = 0.0
    gl = 0.0

    # Lightweight EMA arrays.
    ef = np.empty(n)
    es = np.empty(n)
    et = np.empty(n)

    ef[0] = close[0]
    es[0] = close[0]
    et[0] = close[0]

    af = 2.0 / (fast + 1.0)
    ass = 2.0 / (slow + 1.0)
    at = 2.0 / (trend + 1.0)

    for i in range(1, n):
        ef[i] = af * close[i] + (1-af) * ef[i-1]
        es[i] = ass * close[i] + (1-ass) * es[i-1]
        et[i] = at * close[i] + (1-at) * et[i-1]

    gains = np.zeros(n)
    losses = np.zeros(n)

    for i in range(1, n):
        d = close[i] - close[i-1]

        if d > 0:
            gains[i] = d
        else:
            losses[i] = -d

    alpha = 1.0 / rsi_p

    ag = np.zeros(n)
    al = np.zeros(n)

    for i in range(1, n):
        ag[i] = alpha * gains[i] + (1-alpha) * ag[i-1]
        al[i] = alpha * losses[i] + (1-alpha) * al[i-1]

    rsi = np.zeros(n)

    for i in range(n):
        if al[i] > 0:
            rs = ag[i] / al[i]
            rsi[i] = 100.0 - 100.0/(1.0+rs)
        else:
            rsi[i] = 50.0

    atr = np.zeros(n)

    for i in range(1, n):
        tr = max(
            high[i]-low[i],
            abs(high[i]-close[i-1]),
            abs(low[i]-close[i-1])
        )

        atr[i] = (
            (atr[i-1] * (atr_p-1)) + tr
        ) / atr_p

    for i in range(max(trend, rsi_p, atr_p), n):

        if position == 1:

            if low[i] <= sl:
                pnl = sl-entry
                balance += pnl
                trades += 1

                if pnl > 0:
                    wins += 1
                    gp += pnl
                else:
                    gl -= pnl

                position = 0

            elif high[i] >= tp:
                pnl = tp-entry
                balance += pnl
                trades += 1
                wins += 1
                gp += pnl
                position = 0

        elif position == -1:

            if high[i] >= sl:
                pnl = entry-sl
                balance += pnl
                trades += 1

                if pnl > 0:
                    wins += 1
                    gp += pnl
                else:
                    gl -= pnl

                position = 0

            elif low[i] <= tp:
                pnl = entry-tp
                balance += pnl
                trades += 1
                wins += 1
                gp += pnl
                position = 0

        if position == 0:

            buy = (
                ef[i-1] > es[i-1]
                and es[i-1] > et[i-1]
                and rsi[i-1] > 55
            )

            sell = (
                ef[i-1] < es[i-1]
                and es[i-1] < et[i-1]
                and rsi[i-1] < 45
            )

            if buy and atr[i-1] > 0:
                entry = close[i]
                sl = entry - atr[i-1] * sl_mult
                tp = entry + atr[i-1] * tp_mult
                position = 1

            elif sell and atr[i-1] > 0:
                entry = close[i]
                sl = entry + atr[i-1] * sl_mult
                tp = entry - atr[i-1] * tp_mult
                position = -1

        if balance > peak:
            peak = balance

        dd = peak-balance

        if dd > max_dd:
            max_dd = dd

    pf = gp/gl if gl > 0 else 999.0
    wr = wins/trades if trades > 0 else 0.0

    return trades, balance-100.0, wr, pf, max_dd


def main():

    print("="*70)
    print("XAUUSD FAST WALK-FORWARD TEST")
    print("="*70)

    df = pd.read_csv(DATA)

    df["time"] = pd.to_datetime(df["time"], utc=True)

    df = df.sort_values("time").reset_index(drop=True)

    candidates = pd.read_csv(VALIDATION).head(10)

    folds = [
        (0.50, 0.60),
        (0.55, 0.65),
        (0.60, 0.70),
        (0.65, 0.75),
        (0.70, 0.80),
        (0.75, 0.85),
        (0.80, 0.90),
        (0.85, 1.00),
    ]

    results = []

    total = len(candidates) * len(folds)
    done = 0
    start = time.time()

    print(f"Candidates: {len(candidates)}")
    print(f"Walk-forward folds: {len(folds)}")
    print(f"Total tests: {total}")
    print()

    for ci, c in candidates.iterrows():

        for fold, (a,b) in enumerate(folds,1):

            start_i = int(len(df)*a)
            end_i = int(len(df)*b)

            d = df.iloc[start_i:end_i]

            result = simulate(
                d["close"].values.astype(np.float64),
                d["high"].values.astype(np.float64),
                d["low"].values.astype(np.float64),
                int(c.ema_fast),
                int(c.ema_slow),
                int(c.ema_trend),
                int(c.rsi_period),
                int(c.atr_period),
                float(c.sl),
                float(c.tp)
            )

            results.append({
                "candidate": ci+1,
                "fold": fold,
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

            done += 1

            elapsed = time.time()-start
            speed = done/elapsed if elapsed else 0
            eta = (total-done)/speed if speed else 0

            print(
                f"Progress: {done}/{total} "
                f"({done/total*100:.1f}%) | "
                f"Speed: {speed:.2f}/sec | "
                f"ETA: {eta:.0f}s",
                flush=True
            )

    out = pd.DataFrame(results)

    out.to_csv(
        OUTPUT,
        index=False
    )

    print()
    print("="*70)
    print("WALK-FORWARD COMPLETE")
    print("="*70)

    summary = (
        out.groupby("candidate")
        .agg(
            folds=("fold","count"),
            profitable_folds=("profit", lambda x: (x>0).sum()),
            total_profit=("profit","sum"),
            avg_pf=("profit_factor","mean"),
            worst_dd=("max_drawdown","max"),
            total_trades=("trades","sum")
        )
        .sort_values(
            ["profitable_folds","avg_pf","total_profit"],
            ascending=False
        )
    )

    print(summary)

    print()
    print("Saved:")
    print(OUTPUT)


if __name__ == "__main__":
    main()
