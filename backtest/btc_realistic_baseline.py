from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT))

from backtest import btc_engine_core as ec
from backtest import btc_realistic_config as bc

DATA = ROOT / "data" / "btc_m5_history.csv"
OUT = ROOT / "reports" / "btc_realistic_baseline.csv"

df = pd.read_csv(DATA)

o = np.ascontiguousarray(df.open.to_numpy(np.float64))
h = np.ascontiguousarray(df.high.to_numpy(np.float64))
l = np.ascontiguousarray(df.low.to_numpy(np.float64))
c = np.ascontiguousarray(df.close.to_numpy(np.float64))

# BTC history spread is already in MT5 points.
# Use the actual historical spread, with BTC's own floor.
ec.DEFAULT_SPREAD = bc.DEFAULT_SPREAD
spread = ec.prepare_spread(df)

# Baseline strategy
FAST = 20
SLOW = 60
TREND = 150
RSI = 20
ATR = 20
SL = 2.0
TP = 3.5

TRAIN_END = int(len(c) * 0.70)
VAL_END = int(len(c) * 0.85)
TOTAL = len(c)

ef, es, et, rv, av, warmup = ec.compute_indicators(
    o, h, l, c, FAST, SLOW, TREND, RSI, ATR
)

def run(i0, i1):
    r = ec.run_window_fast(
        o, h, l, c,
        np.ascontiguousarray(ef),
        np.ascontiguousarray(es),
        np.ascontiguousarray(et),
        np.ascontiguousarray(rv),
        np.ascontiguousarray(av),
        spread,
        i0, i1, warmup,
        bc.RSI_BUY_TH,
        bc.RSI_SELL_TH,
        SL, TP,
        bc.SLIPPAGE,
        0.0,
    )
    return {
        "trades": int(r[0]),
        "profit": float(r[2]),
        "winrate": float(r[3]),
        "pf": float(r[4]),
        "dd": float(r[5]),
    }

train = run(0, TRAIN_END)
val = run(TRAIN_END, VAL_END)
test = run(VAL_END, TOTAL)

rows = [
    {"split":"TRAIN", **train},
    {"split":"VALIDATION", **val},
    {"split":"FINAL", **test},
]

out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)

print("=" * 80)
print("BTC REALISTIC BASELINE")
print("=" * 80)
print(f"Data rows       : {TOTAL:,}")
print(f"Train           : {TRAIN_END:,}")
print(f"Validation      : {VAL_END-TRAIN_END:,}")
print(f"Final           : {TOTAL-VAL_END:,}")
print(f"Spread floor    : ${bc.DEFAULT_SPREAD:.2f}")
print(f"Slippage        : ${bc.SLIPPAGE:.2f}")
print(f"Tick size       : ${bc.TICK_SIZE:.2f}")
print(f"Tick value      : ${bc.TICK_VALUE:.4f}")
print(f"Contract size   : {bc.CONTRACT_SIZE}")
print(f"Strategy        : EMA {FAST}/{SLOW}/{TREND} | RSI {RSI} | ATR {ATR} | SL {SL} | TP {TP}")
print("=" * 80)
print(out.to_string(index=False))
print(f"\nSaved: {OUT}")
