import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine_core import (
    load_ohlc,
    compute_indicators,
    prepare_spread,
    run_window_fast,
)

DATA = "data/xauusd.csv"

CANDIDATES = [
    (10, 90, 180),
    (15, 70, 220),
    (20, 90, 180),
    (20, 90, 200),
    (15, 80, 200),
]

RSI_PERIOD = 20
RSI_BUY = 60.0
RSI_SELL = 40.0
ATR_PERIOD = 20
SL_MULT = 2.5
TP_MULT = 3.5

# chronological walk-forward windows
# Each test window is evaluated using indicators calculated from
# the complete causal series, but parameters are NEVER changed
# using the test result.
WINDOWS = [
    ("W1", 0.70, 0.75),
    ("W2", 0.75, 0.80),
    ("W3", 0.80, 0.85),
    ("W4", 0.85, 0.90),
    ("W5", 0.90, 0.95),
    ("W6", 0.95, 1.00),
]

print("=" * 78)
print("XAUUSD EMA WALK-FORWARD ROBUSTNESS TEST")
print("=" * 78)

df = load_ohlc(DATA)

o = df["open"].to_numpy(np.float64)
h = df["high"].to_numpy(np.float64)
l = df["low"].to_numpy(np.float64)
c = df["close"].to_numpy(np.float64)

n = len(c)
sp = prepare_spread(df)

print(f"Candles: {n:,}")
print("Candidates:", len(CANDIDATES))
print("Windows:", len(WINDOWS))

rows = []

for ci, (fast, slow, trend) in enumerate(CANDIDATES, 1):

    print()
    print("=" * 78)
    print(f"[{ci}/{len(CANDIDATES)}] EMA {fast}/{slow}/{trend}")
    print("=" * 78)

    ef, es, et, rv, av, warmup = compute_indicators(
        o, h, l, c,
        fast,
        slow,
        trend,
        RSI_PERIOD,
        ATR_PERIOD,
    )

    for wi, (window, p0, p1) in enumerate(WINDOWS, 1):

        i0 = int(n * p0)
        i1 = int(n * p1)

        res = run_window_fast(
            o, h, l, c,
            ef, es, et, rv, av, sp,
            i0, i1,
            warmup,
            RSI_BUY,
            RSI_SELL,
            SL_MULT,
            TP_MULT,
        )

        (
            trades,
            balance,
            profit,
            wr,
            pf,
            dd_abs,
            dd_pct,
            wins,
            losses,
            comm,
            _gp,
            _gl,
            _nr,
        ) = res

        initial = 100.0

        rows.append({
            "candidate": f"{fast}/{slow}/{trend}",
            "fast": fast,
            "slow": slow,
            "trend": trend,
            "window": window,
            "bar_start": i0,
            "bar_end": i1,
            "trades": int(round(trades)),
            "final_balance": round(balance, 4),
            "return_pct": round((balance / initial - 1) * 100, 4),
            "win_rate": round(wr * 100, 4),
            "profit_factor": (
                round(pf, 4) if np.isfinite(pf) else np.inf
            ),
            "max_drawdown_pct": round(dd_pct * 100, 4),
        })

        print(
            f"{window}: "
            f"return={rows[-1]['return_pct']:7.2f}% | "
            f"PF={rows[-1]['profit_factor']:6.3f} | "
            f"WR={rows[-1]['win_rate']:6.2f}% | "
            f"trades={rows[-1]['trades']:4d} | "
            f"DD={rows[-1]['max_drawdown_pct']:6.2f}%"
        )

os.makedirs("reports", exist_ok=True)

out = pd.DataFrame(rows)
out.to_csv(
    "reports/walk_forward_results.csv",
    index=False,
)

# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

summary = (
    out.groupby(["fast", "slow", "trend"])
    .agg(
        windows=("window", "count"),
        positive_windows=("return_pct", lambda x: int((x > 0).sum())),
        avg_return_pct=("return_pct", "mean"),
        median_return_pct=("return_pct", "median"),
        worst_return_pct=("return_pct", "min"),
        avg_profit_factor=("profit_factor", "mean"),
        worst_profit_factor=("profit_factor", "min"),
        avg_win_rate=("win_rate", "mean"),
        max_drawdown_pct=("max_drawdown_pct", "max"),
        total_trades=("trades", "sum"),
    )
    .reset_index()
)

summary["candidate"] = (
    summary["fast"].astype(str)
    + "/"
    + summary["slow"].astype(str)
    + "/"
    + summary["trend"].astype(str)
)

# Robustness score:
# prioritize consistency rather than highest single return.
summary["robust_score"] = (
    summary["positive_windows"] * 10
    + summary["worst_profit_factor"].clip(lower=0) * 10
    + summary["avg_profit_factor"]
    + summary["avg_return_pct"] * 0.25
    - summary["max_drawdown_pct"] * 0.10
)

summary = summary.sort_values(
    ["robust_score", "positive_windows", "avg_profit_factor"],
    ascending=False,
)

summary.to_csv(
    "reports/walk_forward_summary.csv",
    index=False,
)

print()
print("=" * 100)
print("WALK-FORWARD SUMMARY")
print("=" * 100)

print(
    summary[
        [
            "candidate",
            "positive_windows",
            "avg_return_pct",
            "median_return_pct",
            "worst_return_pct",
            "avg_profit_factor",
            "worst_profit_factor",
            "avg_win_rate",
            "max_drawdown_pct",
            "total_trades",
        ]
    ].to_string(index=False)
)

print()
print("=" * 100)
print("BEST ROBUSTNESS CANDIDATE")
print("=" * 100)

best = summary.iloc[0]

print(f"EMA:                 {best['candidate']}")
print(f"Positive windows:    {int(best['positive_windows'])}/{int(best['windows'])}")
print(f"Average return:      {best['avg_return_pct']:.2f}%")
print(f"Median return:       {best['median_return_pct']:.2f}%")
print(f"Worst return:        {best['worst_return_pct']:.2f}%")
print(f"Average PF:          {best['avg_profit_factor']:.3f}")
print(f"Worst PF:            {best['worst_profit_factor']:.3f}")
print(f"Average win rate:    {best['avg_win_rate']:.2f}%")
print(f"Maximum DD:          {best['max_drawdown_pct']:.2f}%")
print(f"Total trades:        {int(best['total_trades'])}")

print()
print("Saved:")
print("reports/walk_forward_results.csv")
print("reports/walk_forward_summary.csv")
