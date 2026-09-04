from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT))

from backtest import engine_core as ec

DATA = ROOT / "data" / "xauusd_m5_5y.csv"
OUT = ROOT / "reports" / "true_walk_forward_final_holdout.csv"

df = pd.read_csv(DATA)

o = np.ascontiguousarray(df["open"].to_numpy(np.float64))
h = np.ascontiguousarray(df["high"].to_numpy(np.float64))
l = np.ascontiguousarray(df["low"].to_numpy(np.float64))
c = np.ascontiguousarray(df["close"].to_numpy(np.float64))
spread = ec.prepare_spread(df)

# Winners selected ONLY from WF validation data
WINNERS = [
    ("WF1", 30, 80, 200, 14, 14, 2.5, 3.5),
    ("WF2", 20, 80, 200, 7,  14, 2.5, 3.5),
    ("WF3", 30, 80, 150, 10, 20, 2.0, 3.5),
    ("WF4", 20, 60, 150, 20, 20, 2.0, 3.5),
]

FINAL_START = 85000
FINAL_END = len(c)

rows = []

for source, fast, slow, trend, rsi_p, atr_p, sl, tp in WINNERS:
    ef, es, et, rv, av, warmup = ec.compute_indicators(
        o, h, l, c, fast, slow, trend, rsi_p, atr_p
    )

    r = ec.run_window_fast(
        o, h, l, c,
        np.ascontiguousarray(ef),
        np.ascontiguousarray(es),
        np.ascontiguousarray(et),
        np.ascontiguousarray(rv),
        np.ascontiguousarray(av),
        spread,
        FINAL_START, FINAL_END, warmup,
        60.0, 40.0,
        sl, tp,
        ec.SLIPPAGE,
        ec.COMMISSION_PER_LOT,
    )

    rows.append({
        "selected_from": source,
        "ema_fast": fast,
        "ema_slow": slow,
        "ema_trend": trend,
        "rsi_period": rsi_p,
        "atr_period": atr_p,
        "sl": sl,
        "tp": tp,
        "test_trades": int(r[0]),
        "test_profit": float(r[2]),
        "test_winrate": float(r[3]),
        "test_pf": float(r[4]),
        "test_dd": float(r[5]),
    })

out = pd.DataFrame(rows).sort_values(
    ["test_pf", "test_profit"], ascending=False
)

out.to_csv(OUT, index=False)

print("=" * 80)
print("FINAL HOLDOUT TEST — 85,000:100,000")
print("=" * 80)
print(out.to_string(index=False))
print(f"\nSaved: {OUT}")
print("This final holdout was NOT used for parameter selection.")
