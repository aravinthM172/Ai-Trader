from pathlib import Path
import sys
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Walk-forward robustness test
#
# IMPORTANT:
# - Does NOT modify Experiment #1 outputs.
# - Does NOT use the locked final 15,000 bars for parameter selection.
# - Tests the discovered parameter family/candidates across multiple
#   chronological windows.
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest import engine_core as ec


DATA = ROOT / "data" / "xauusd_m5_5y.csv"
OUT = ROOT / "reports" / "xauusd_walk_forward_results.csv"


# ---------------------------------------------------------------------------
# Candidate set from Experiment #1
#
# We intentionally test a small, pre-registered set rather than searching
# thousands of combinations again on the same historical data.
# ---------------------------------------------------------------------------

CANDIDATES = [
    # Main candidate
    dict(name="C01_15_80_200", ema_fast=15, ema_slow=80, ema_trend=200,
         rsi_period=20, atr_period=20, sl=2.5, tp=3.5),

    # Same family / neighboring fast EMA
    dict(name="C02_5_80_200", ema_fast=5, ema_slow=80, ema_trend=200,
         rsi_period=20, atr_period=20, sl=2.5, tp=3.5),

    dict(name="C03_20_80_200", ema_fast=20, ema_slow=80, ema_trend=200,
         rsi_period=20, atr_period=20, sl=2.5, tp=3.5),

    # Other strong validation survivors
    dict(name="C04_15_50_150", ema_fast=15, ema_slow=50, ema_trend=150,
         rsi_period=20, atr_period=20, sl=2.0, tp=3.5),

    dict(name="C05_30_60_150", ema_fast=30, ema_slow=60, ema_trend=150,
         rsi_period=20, atr_period=20, sl=2.0, tp=3.5),

    dict(name="C06_30_50_100", ema_fast=30, ema_slow=50, ema_trend=100,
         rsi_period=20, atr_period=20, sl=2.0, tp=3.5),
]


# ---------------------------------------------------------------------------
# Chronological windows
#
# Each validation period is strictly after its training period.
# The final 15,000 bars remain untouched and are NOT used below.
#
# The last row is reserved only as an OOS confirmation observation later.
# ---------------------------------------------------------------------------

WINDOWS = [
    ("WF1", 0, 40000, 40000, 50000),
    ("WF2", 0, 50000, 50000, 60000),
    ("WF3", 0, 60000, 60000, 70000),
    ("WF4", 0, 70000, 70000, 85000),
]


def load_data():
    if not DATA.exists():
        raise RuntimeError(f"Missing data file: {DATA}")

    df = pd.read_csv(DATA)

    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns: {sorted(missing)}")

    # Support either datetime or the existing raw CSV format.
    if "time" in df.columns:
        try:
            ts = pd.to_datetime(df["time"], utc=True)
        except Exception:
            ts = None
    else:
        ts = None

    o = np.ascontiguousarray(df["open"].to_numpy(np.float64))
    h = np.ascontiguousarray(df["high"].to_numpy(np.float64))
    l = np.ascontiguousarray(df["low"].to_numpy(np.float64))
    c = np.ascontiguousarray(df["close"].to_numpy(np.float64))
    spread = ec.prepare_spread(df)

    return df, ts, o, h, l, c, spread


def compute_full_indicators(o, h, l, c, p):
    ef, es, et, rv, av, warmup = ec.compute_indicators(
        o, h, l, c,
        p["ema_fast"],
        p["ema_slow"],
        p["ema_trend"],
        p["rsi_period"],
        p["atr_period"],
    )
    return (
        np.ascontiguousarray(ef),
        np.ascontiguousarray(es),
        np.ascontiguousarray(et),
        np.ascontiguousarray(rv),
        np.ascontiguousarray(av),
        warmup,
    )


def normalize_result(r):
    """
    run_window_fast returns a tuple/array containing the same core metrics
    used by the realistic optimizer:
        0 trades
        2 net profit
        3 win rate
        4 profit factor
        5 max drawdown
    """
    return {
        "trades": int(r[0]),
        "profit": float(r[2]),
        "winrate": float(r[3]),
        "pf": float(r[4]),
        "dd": float(r[5]),
    }


def run_one(ef, es, et, rv, av, warmup, o, h, l, c, spread, p, i0, i1):
    r = ec.run_window_fast(
        o, h, l, c,
        ef, es, et, rv, av, spread,
        int(i0), int(i1), int(warmup),
        60.0, 40.0,
        float(p["sl"]), float(p["tp"]),
        ec.SLIPPAGE,
        ec.COMMISSION_PER_LOT,
    )
    return normalize_result(r)


def main():
    print("=" * 80)
    print("XAUUSD WALK-FORWARD ROBUSTNESS EXPERIMENT")
    print("=" * 80)
    print(f"Data:             {DATA}")
    print(f"Spread floor:     ${ec.DEFAULT_SPREAD:.2f}")
    print(f"Slippage:         ${ec.SLIPPAGE:.2f}")
    print(f"Candidates:       {len(CANDIDATES)}")
    print(f"Reserved final:   last 15,000 bars remain untouched")
    print("=" * 80)

    df, ts, o, h, l, c, spread = load_data()
    n = len(c)

    if n < 100000:
        raise RuntimeError(f"Expected approximately 100,000 bars, found {n}")

    # Preserve the original final-test block.
    FINAL_START = n - 15000
    if FINAL_START < 85000:
        raise RuntimeError("Dataset is too short for the intended final block.")

    # Compile / warm Numba once.
    p0 = CANDIDATES[0]
    ef0, es0, et0, rv0, av0, wu0 = compute_full_indicators(o, h, l, c, p0)

    _ = run_one(
        ef0, es0, et0, rv0, av0, wu0,
        o, h, l, c, spread, p0,
        0, 1000,
    )

    rows = []

    for p in CANDIDATES:
        ef, es, et, rv, av, warmup = compute_full_indicators(o, h, l, c, p)

        for window_name, tr0, tr1, va0, va1 in WINDOWS:
            train = run_one(
                ef, es, et, rv, av, warmup,
                o, h, l, c, spread, p,
                tr0, tr1,
            )

            validation = run_one(
                ef, es, et, rv, av, warmup,
                o, h, l, c, spread, p,
                va0, va1,
            )

            rows.append({
                "candidate": p["name"],
                "ema_fast": p["ema_fast"],
                "ema_slow": p["ema_slow"],
                "ema_trend": p["ema_trend"],
                "rsi_period": p["rsi_period"],
                "atr_period": p["atr_period"],
                "sl": p["sl"],
                "tp": p["tp"],
                "window": window_name,

                "train_start": tr0,
                "train_end": tr1,
                "val_start": va0,
                "val_end": va1,

                "train_trades": train["trades"],
                "train_profit": train["profit"],
                "train_winrate": train["winrate"],
                "train_pf": train["pf"],
                "train_dd": train["dd"],

                "val_trades": validation["trades"],
                "val_profit": validation["profit"],
                "val_winrate": validation["winrate"],
                "val_pf": validation["pf"],
                "val_dd": validation["dd"],
            })

    result = pd.DataFrame(rows)

    Path(OUT.parent).mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False)

    print()
    print("=" * 80)
    print("WALK-FORWARD RESULTS")
    print("=" * 80)
    print(
        result[
            [
                "candidate", "window",
                "train_trades", "train_profit", "train_pf", "train_dd",
                "val_trades", "val_profit", "val_pf", "val_dd"
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 80)
    print("CANDIDATE ROBUSTNESS SUMMARY")
    print("=" * 80)

    summary_rows = []

    for name, g in result.groupby("candidate", sort=False):
        positive_windows = int((g["val_profit"] > 0).sum())
        pf_windows = int((g["val_pf"] > 1.0).sum())

        summary_rows.append({
            "candidate": name,
            "positive_validation_windows": positive_windows,
            "pf_gt_1_windows": pf_windows,
            "validation_windows": len(g),
            "positive_rate": positive_windows / len(g),
            "median_val_pf": g["val_pf"].median(),
            "median_val_profit": g["val_profit"].median(),
            "worst_val_profit": g["val_profit"].min(),
            "worst_val_pf": g["val_pf"].min(),
            "worst_val_dd": g["val_dd"].max(),
            "total_val_profit": g["val_profit"].sum(),
        })

    summary = pd.DataFrame(summary_rows).sort_values(
        [
            "positive_validation_windows",
            "pf_gt_1_windows",
            "median_val_pf",
            "total_val_profit",
        ],
        ascending=False,
    )

    print(summary.to_string(index=False))

    print()
    print("=" * 80)
    print("DECISION GUIDE")
    print("=" * 80)
    print(
        """
Prefer candidates that:
  - remain profitable across most walk-forward validation windows;
  - keep PF > 1 across multiple independent periods;
  - do not rely on a single unusually strong window;
  - avoid extreme drawdown;
  - remain competitive with neighboring parameter choices.

Do NOT modify the locked Experiment #1 final-test result based on this output.
This experiment is separate and should be treated as new evidence.
"""
    )

    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
