from pathlib import Path

p = Path("backtest/optimizer_realistic_numba.py")

code = r'''
import os
import time
import math
import itertools
import numpy as np
import pandas as pd

try:
    from numba import njit, prange
    NUMBA = True
except ImportError:
    NUMBA = False

DATA = "data/xauusd_m5_5y.csv"
OUT = "reports/xauusd_realistic_optimization.csv"

INITIAL_BALANCE = 100.0
RISK_PCT = 0.01

TICK_VALUE = 0.10
TICK_SIZE = 0.01
VALUE_PER_PRICE_PER_LOT = TICK_VALUE / TICK_SIZE

MIN_LOT = 0.01
LOT_STEP = 0.01
MAX_LOT = 100.0

DEFAULT_SPREAD = 0.35
SLIPPAGE = 0.02

TRAIN_END = 70000
VAL_END = 85000
TOTAL = 100000

FAST = [5, 10, 15, 20, 30]
SLOW = [30, 40, 50, 60, 80]
TREND = [100, 150, 200]
RSI_PERIOD = [7, 10, 14, 20]
RSI_BUY = [55, 60]
RSI_SELL = [40, 45]
ATR_PERIOD = [10, 14, 20]
SL = [1.0, 1.5, 2.0, 2.5]
TP = [1.5, 2.0, 2.5, 3.0, 3.5]


def ema(x, period):
    out = np.empty(len(x), dtype=np.float64)
    out[:] = np.nan

    if len(x) < period:
        return out

    out[period - 1] = np.mean(x[:period])
    alpha = 2.0 / (period + 1.0)

    for i in range(period, len(x)):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]

    return out


def rsi(close, period):
    out = np.empty(len(close), dtype=np.float64)
    out[:] = np.nan

    if len(close) <= period:
        return out

    delta = np.diff(close)

    gains = np.maximum(delta, 0.0)
    losses = np.maximum(-delta, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        out[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        out[period] = 100.0 - 100.0 / (1.0 + rs)

    for i in range(period + 1, len(close)):
        gain = gains[i - 1]
        loss = losses[i - 1]

        avg_gain = ((avg_gain * (period - 1)) + gain) / period
        avg_loss = ((avg_loss * (period - 1)) + loss) / period

        if avg_loss == 0:
            out[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i] = 100.0 - 100.0 / (1.0 + rs)

    return out


def atr(high, low, close, period):
    tr = np.empty(len(close), dtype=np.float64)
    tr[0] = high[0] - low[0]

    for i in range(1, len(close)):
        a = high[i] - low[i]
        b = abs(high[i] - close[i - 1])
        c = abs(low[i] - close[i - 1])
        tr[i] = max(a, b, c)

    out = np.empty(len(close), dtype=np.float64)
    out[:] = np.nan

    if len(close) <= period:
        return out

    value = np.mean(tr[:period])
    out[period - 1] = value

    for i in range(period, len(close)):
        value = ((value * (period - 1)) + tr[i]) / period
        out[i] = value

    return out


def load_data():
    df = pd.read_csv(DATA)

    df.columns = [c.lower().strip() for c in df.columns]

    required = ["open", "high", "low", "close"]

    for c in required:
        if c not in df.columns:
            raise RuntimeError(f"Missing column: {c}")

    spread = (
        df["spread"].to_numpy(np.float64)
        if "spread" in df.columns
        else np.full(len(df), DEFAULT_SPREAD)
    )

    return (
        df["open"].to_numpy(np.float64),
        df["high"].to_numpy(np.float64),
        df["low"].to_numpy(np.float64),
        df["close"].to_numpy(np.float64),
        spread,
    )


def prepare_indicators(close, high, low):
    print("Preparing shared indicators...")

    ema_cache = {}
    for p in set(FAST + SLOW + TREND):
        ema_cache[p] = ema(close, p)

    rsi_cache = {}
    for p in RSI_PERIOD:
        rsi_cache[p] = rsi(close, p)

    atr_cache = {}
    for p in ATR_PERIOD:
        atr_cache[p] = atr(high, low, close, p)

    return ema_cache, rsi_cache, atr_cache


def floor_lot(raw):
    if raw < MIN_LOT:
        return 0.0

    lots = math.floor((raw / LOT_STEP) + 1e-9) * LOT_STEP

    if lots < MIN_LOT:
        return 0.0

    return min(lots, MAX_LOT)


def backtest(
    op,
    hi,
    lo,
    cl,
    spread,
    ema_fast,
    ema_slow,
    ema_trend,
    rsi_arr,
    atr_arr,
    sl_mult,
    tp_mult,
    start,
    end,
):
    balance = INITIAL_BALANCE
    peak = balance
    max_dd = 0.0

    trades = 0
    wins = 0
    gross_profit = 0.0
    gross_loss = 0.0

    position = 0
    entry = 0.0
    sl_price = 0.0
    tp_price = 0.0
    lots = 0.0

    warmup = max(ema_slow, ema_trend) + 5

    first = max(start + warmup, warmup)

    for i in range(first, end - 1):

        # Manage existing position using current completed candle.
        if position != 0:

            exit_price = 0.0
            result = 0.0

            if position == 1:

                stop_hit = lo[i] <= sl_price
                target_hit = hi[i] >= tp_price

                if stop_hit:
                    # Gap-aware stop.
                    if op[i] < sl_price:
                        exit_price = op[i] - SLIPPAGE
                    else:
                        exit_price = sl_price - SLIPPAGE

                    result = (
                        exit_price - entry
                    ) * VALUE_PER_PRICE_PER_LOT * lots

                elif target_hit:
                    exit_price = tp_price
                    result = (
                        exit_price - entry
                    ) * VALUE_PER_PRICE_PER_LOT * lots

            else:

                stop_hit = hi[i] >= sl_price
                target_hit = lo[i] <= tp_price

                if stop_hit:
                    if op[i] > sl_price:
                        exit_price = op[i] + SLIPPAGE
                    else:
                        exit_price = sl_price + SLIPPAGE

                    result = (
                        entry - exit_price
                    ) * VALUE_PER_PRICE_PER_LOT * lots

                elif target_hit:
                    exit_price = tp_price
                    result = (
                        entry - exit_price
                    ) * VALUE_PER_PRICE_PER_LOT * lots

            if exit_price != 0.0:

                balance += result
                trades += 1

                if result > 0:
                    wins += 1
                    gross_profit += result
                else:
                    gross_loss += -result

                position = 0
                entry = 0.0
                lots = 0.0

        # Mark-to-market equity.
        if position == 1:
            equity = balance + (
                cl[i] - entry
            ) * VALUE_PER_PRICE_PER_LOT * lots

        elif position == -1:
            equity = balance + (
                entry - cl[i]
            ) * VALUE_PER_PRICE_PER_LOT * lots

        else:
            equity = balance

        if equity > peak:
            peak = equity

        dd = peak - equity

        if dd > max_dd:
            max_dd = dd

        # Do not enter while position is open.
        if position != 0:
            continue

        ef = ema_fast[i]
        es = ema_slow[i]
        et = ema_trend[i]
        rv = rsi_arr[i]
        av = atr_arr[i]

        if (
            np.isnan(ef)
            or np.isnan(es)
            or np.isnan(et)
            or np.isnan(rv)
            or np.isnan(av)
            or av <= 0
        ):
            continue

        signal = 0

        if ef > es and cl[i] > et and rv >= RSI_BUY[0]:
            signal = 1

        elif ef < es and cl[i] < et and rv <= RSI_SELL[0]:
            signal = -1

        if signal == 0:
            continue

        stop_distance = av * sl_mult

        if stop_distance <= 0:
            continue

        # Correct $ risk calculation.
        risk_dollars = balance * RISK_PCT

        dollars_per_lot = (
            stop_distance * VALUE_PER_PRICE_PER_LOT
        )

        if dollars_per_lot <= 0:
            continue

        raw_lots = risk_dollars / dollars_per_lot

        lots_new = floor_lot(raw_lots)

        # Never force minimum lot.
        if lots_new <= 0:
            continue

        # Entry happens at NEXT BAR OPEN.
        next_open = op[i + 1]

        next_spread = spread[i + 1]

        if signal == 1:
            entry_new = next_open + next_spread / 2.0
        else:
            entry_new = next_open - next_spread / 2.0

        if signal == 1:
            position = 1
            entry = entry_new
            sl_price = entry - stop_distance
            tp_price = entry + av * tp_mult

        else:
            position = -1
            entry = entry_new
            sl_price = entry + stop_distance
            tp_price = entry - av * tp_mult

        lots = lots_new

    # Close remaining position at final close.
    if position == 1:
        result = (
            cl[end - 1] - entry
        ) * VALUE_PER_PRICE_PER_LOT * lots
        balance += result

        trades += 1

        if result > 0:
            wins += 1
            gross_profit += result
        else:
            gross_loss += -result

    elif position == -1:
        result = (
            entry - cl[end - 1]
        ) * VALUE_PER_PRICE_PER_LOT * lots
        balance += result

        trades += 1

        if result > 0:
            wins += 1
            gross_profit += result
        else:
            gross_loss += -result

    win_rate = wins / trades if trades else 0.0
    pf = (
        gross_profit / gross_loss
        if gross_loss > 0
        else (999.0 if gross_profit > 0 else 0.0)
    )

    return (
        trades,
        wins,
        balance - INITIAL_BALANCE,
        win_rate,
        pf,
        max_dd,
    )


def main():

    print("=" * 70)
    print("REALISTIC XAUUSD NUMBA RESEARCH OPTIMIZER")
    print("=" * 70)

    if not os.path.exists(DATA):
        raise RuntimeError(f"Missing {DATA}")

    op, hi, lo, cl, spread = load_data()

    n = len(cl)

    print(f"Candles: {n:,}")
    print(f"Training: {TRAIN_END:,}")
    print(f"Validation: {VAL_END - TRAIN_END:,}")
    print(f"Final test: {n - VAL_END:,}")

    ema_cache, rsi_cache, atr_cache = prepare_indicators(
        cl, hi, lo
    )

    configs = list(
        itertools.product(
            FAST,
            SLOW,
            TREND,
            RSI_PERIOD,
            ATR_PERIOD,
            SL,
            TP,
        )
    )

    configs = [
        x for x in configs
        if x[0] < x[1] and x[1] < x[2]
    ]

    print(f"Configurations: {len(configs):,}")

    results = []

    start_time = time.time()

    for idx, (
        ef,
        es,
        et,
        rp,
        ap,
        slm,
        tpm,
    ) in enumerate(configs, 1):

        rsi_arr = rsi_cache[rp]
        atr_arr = atr_cache[ap]

        tr = backtest(
            op, hi, lo, cl, spread,
            ema_cache[ef],
            ema_cache[es],
            ema_cache[et],
            rsi_arr,
            atr_arr,
            slm,
            tpm,
            0,
            TRAIN_END,
        )

        if tr[0] < 30:
            continue

        if tr[4] < 1.05:
            continue

        val = backtest(
            op, hi, lo, cl, spread,
            ema_cache[ef],
            ema_cache[es],
            ema_cache[et],
            rsi_arr,
            atr_arr,
            slm,
            tpm,
            TRAIN_END,
            VAL_END,
        )

        if val[0] < 20:
            continue

        if val[2] <= 0:
            continue

        if val[4] < 1.05:
            continue

        results.append(
            (
                ef, es, et, rp, ap, slm, tpm,
                tr[0], tr[2], tr[3], tr[4], tr[5],
                val[0], val[2], val[3], val[4], val[5],
            )
        )

        if idx % 500 == 0 or idx == len(configs):

            elapsed = time.time() - start_time
            speed = idx / elapsed if elapsed else 0
            eta = (
                (len(configs) - idx) / speed
                if speed
                else 0
            )

            print(
                f"Progress: {idx:,}/{len(configs):,} "
                f"({idx/len(configs)*100:.1f}%) | "
                f"Speed: {speed:.1f}/sec | "
                f"ETA: {eta:.0f}s | "
                f"Valid: {len(results):,}",
                flush=True,
            )

    columns = [
        "ema_fast",
        "ema_slow",
        "ema_trend",
        "rsi_period",
        "atr_period",
        "sl",
        "tp",
        "train_trades",
        "train_profit",
        "train_winrate",
        "train_pf",
        "train_dd",
        "val_trades",
        "val_profit",
        "val_winrate",
        "val_pf",
        "val_dd",
    ]

    df = pd.DataFrame(results, columns=columns)

    if len(df) == 0:
        print("NO VALID STRATEGIES FOUND.")
        return

    df = df.sort_values(
        ["val_pf", "val_profit"],
        ascending=False,
    )

    print()
    print("=" * 70)
    print("VALIDATION CANDIDATES")
    print("=" * 70)
    print(df.head(20).to_string(index=False))

    Path = __import__("pathlib").Path
    Path("reports").mkdir(exist_ok=True)

    df.to_csv(OUT, index=False)

    # Untouched final test ONLY after validation selection.
    top = df.head(10)

    final_rows = []

    for _, r in top.iterrows():

        ef = int(r.ema_fast)
        es = int(r.ema_slow)
        et = int(r.ema_trend)
        rp = int(r.rsi_period)
        ap = int(r.atr_period)

        result = backtest(
            op, hi, lo, cl, spread,
            ema_cache[ef],
            ema_cache[es],
            ema_cache[et],
            rsi_cache[rp],
            atr_cache[ap],
            float(r.sl),
            float(r.tp),
            VAL_END,
            n,
        )

        final_rows.append(
            [
                ef, es, et, rp, ap,
                r.sl, r.tp,
                result[0],
                result[2],
                result[3],
                result[4],
                result[5],
            ]
        )

    final_df = pd.DataFrame(
        final_rows,
        columns=[
            "ema_fast",
            "ema_slow",
            "ema_trend",
            "rsi_period",
            "atr_period",
            "sl",
            "tp",
            "test_trades",
            "test_profit",
            "test_winrate",
            "test_pf",
            "test_dd",
        ],
    )

    final_df.to_csv(
        "reports/xauusd_realistic_final_test.csv",
        index=False,
    )

    print()
    print("=" * 70)
    print("UNTOUCHED FINAL TEST")
    print("=" * 70)
    print(final_df.to_string(index=False))

    print()
    print("Saved:")
    print(OUT)
    print("reports/xauusd_realistic_final_test.csv")


if __name__ == "__main__":
    main()
'''

p.write_text(code, encoding="utf-8")

print("Created:", p)
print("Size:", p.stat().st_size, "bytes")
