from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest import engine_core as ec
from backtest import optimizer_realistic_numba as opt

DATA = ROOT / "data" / "xauusd_m5_5y.csv"
OUT = ROOT / "reports" / "xauusd_true_walk_forward.csv"

WINDOWS = [
    ("WF1", 0, 40000, 40000, 50000),
    ("WF2", 0, 50000, 50000, 60000),
    ("WF3", 0, 60000, 60000, 70000),
    ("WF4", 0, 70000, 70000, 85000),
]

MIN_TRADES = 30
MIN_PF = 1.05
TOP_N = 10

df = pd.read_csv(DATA)

o = np.ascontiguousarray(df["open"].to_numpy(np.float64))
h = np.ascontiguousarray(df["high"].to_numpy(np.float64))
l = np.ascontiguousarray(df["low"].to_numpy(np.float64))
c = np.ascontiguousarray(df["close"].to_numpy(np.float64))
spread = ec.prepare_spread(df)

configs = opt.build_configs()

# Determine tuple layout from build_configs().
# Existing optimizer uses:
# fast, slow, trend, rsi_period, atr_period, sl, tp
def unpack_config(p):
    return (
        int(p[0]), int(p[1]), int(p[2]),
        int(p[3]), int(p[4]),
        float(p[5]), float(p[6])
    )

print("=" * 80)
print("TRUE WALK-FORWARD OPTIMIZATION")
print("=" * 80)
print(f"Configurations/window : {len(configs):,}")
print(f"Windows                : {len(WINDOWS)}")
print("Final 85k:100k         : LOCKED / UNUSED")
print("=" * 80)

all_rows = []

for name, tr0, tr1, va0, va1 in WINDOWS:
    print(f"\n{name}: TRAIN {tr0}:{tr1} -> VALIDATION {va0}:{va1}")

    train_rows = []

    for j, raw_p in enumerate(configs):
        (
            ema_fast, ema_slow, ema_trend,
            rsi_period, atr_period, sl, tp
        ) = unpack_config(raw_p)

        ef, es, et, rv, av, warmup = ec.compute_indicators(
            o, h, l, c,
            ema_fast, ema_slow, ema_trend,
            rsi_period, atr_period,
        )

        r = ec.run_window_fast(
            o, h, l, c,
            np.ascontiguousarray(ef),
            np.ascontiguousarray(es),
            np.ascontiguousarray(et),
            np.ascontiguousarray(rv),
            np.ascontiguousarray(av),
            spread,
            tr0, tr1, warmup,
            60.0, 40.0,
            sl, tp,
            ec.SLIPPAGE,
            ec.COMMISSION_PER_LOT,
        )

        train_rows.append({
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "ema_trend": ema_trend,
            "rsi_period": rsi_period,
            "atr_period": atr_period,
            "sl": sl,
            "tp": tp,
            "train_trades": int(r[0]),
            "train_winrate": float(r[3]),
            "train_profit": float(r[2]),
            "train_pf": float(r[4]),
            "train_dd": float(r[5]),
        })

        if (j + 1) % 2000 == 0:
            print(f"  optimized {j+1:,}/{len(configs):,}", flush=True)

    train = pd.DataFrame(train_rows)

    eligible = train[
        (train.train_trades >= MIN_TRADES) &
        (train.train_pf >= MIN_PF) &
        (train.train_profit > 0)
    ].copy()

    if len(eligible) == 0:
        eligible = train[train.train_trades >= MIN_TRADES].copy()

    eligible = eligible.sort_values(
        ["train_pf", "train_profit"],
        ascending=False
    ).head(TOP_N)

    print(f"  Training survivors: {len(eligible)}")

    for _, p in eligible.iterrows():
        ef, es, et, rv, av, warmup = ec.compute_indicators(
            o, h, l, c,
            int(p.ema_fast),
            int(p.ema_slow),
            int(p.ema_trend),
            int(p.rsi_period),
            int(p.atr_period),
        )

        r = ec.run_window_fast(
            o, h, l, c,
            np.ascontiguousarray(ef),
            np.ascontiguousarray(es),
            np.ascontiguousarray(et),
            np.ascontiguousarray(rv),
            np.ascontiguousarray(av),
            spread,
            va0, va1, warmup,
            60.0, 40.0,
            float(p.sl), float(p.tp),
            ec.SLIPPAGE,
            ec.COMMISSION_PER_LOT,
        )

        all_rows.append({
            "window": name,
            "ema_fast": int(p.ema_fast),
            "ema_slow": int(p.ema_slow),
            "ema_trend": int(p.ema_trend),
            "rsi_period": int(p.rsi_period),
            "atr_period": int(p.atr_period),
            "sl": float(p.sl),
            "tp": float(p.tp),

            "train_trades": int(p.train_trades),
            "train_profit": float(p.train_profit),
            "train_pf": float(p.train_pf),
            "train_dd": float(p.train_dd),

            "val_trades": int(r[0]),
            "val_winrate": float(r[3]),
            "val_profit": float(r[2]),
            "val_pf": float(r[4]),
            "val_dd": float(r[5]),
        })

result = pd.DataFrame(all_rows)
result.to_csv(OUT, index=False)

print("\n" + "=" * 80)
print("TRUE WALK-FORWARD RESULTS")
print("=" * 80)
print(result.to_string(index=False))

print("\n" + "=" * 80)
print("BEST VALIDATION RESULT PER WINDOW")
print("=" * 80)

winners = (
    result.sort_values(
        ["window", "val_pf", "val_profit"],
        ascending=[True, False, False]
    )
    .groupby("window", sort=False)
    .head(1)
)

print(
    winners[
        ["window","ema_fast","ema_slow","ema_trend",
         "rsi_period","atr_period","sl","tp",
         "train_profit","train_pf",
         "val_trades","val_profit","val_pf","val_dd"]
    ].to_string(index=False)
)

print("\n" + "=" * 80)
print("PARAMETER STABILITY")
print("=" * 80)

print(
    winners[
        ["window","ema_fast","ema_slow","ema_trend",
         "rsi_period","atr_period","sl","tp"]
    ].to_string(index=False)
)

print(f"\nSaved: {OUT}")
print("Final 85k:100k was NOT used.")
