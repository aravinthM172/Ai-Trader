import os
import sys
import subprocess
import itertools
import pandas as pd

PYTHON = sys.executable
SCRIPT = os.path.join("backtest", "realistic_xauusd.py")
REPORT = os.path.join("reports", "parameter_stability.csv")

FAST_VALUES = [10, 15, 20]
SLOW_VALUES = [70, 80, 90]
TREND_VALUES = [180, 200, 220]

os.makedirs("reports", exist_ok=True)

results = []

total = len(FAST_VALUES) * len(SLOW_VALUES) * len(TREND_VALUES)
count = 0

print("=" * 80)
print("GOLD AI TRADER — EMA PARAMETER ROBUSTNESS TEST")
print("=" * 80)
print(f"Testing {total} EMA combinations")
print()

for fast, slow, trend in itertools.product(
    FAST_VALUES, SLOW_VALUES, TREND_VALUES
):
    count += 1

    print()
    print("=" * 80)
    print(f"[{count}/{total}] EMA {fast}/{slow}/{trend}")
    print("=" * 80)

    cmd = [
        PYTHON,
        SCRIPT,
        "--fast", str(fast),
        "--slow", str(slow),
        "--trend", str(trend),
        "--rsi-period", "20",
        "--rsi-buy", "60",
        "--rsi-sell", "40",
        "--atr-period", "20",
        "--sl", "2.5",
        "--tp", "3.5",
    ]

    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    output = completed.stdout + "\n" + completed.stderr

    if completed.returncode != 0:
        print(output)
        print("FAILED")
        results.append({
            "fast": fast,
            "slow": slow,
            "trend": trend,
            "status": "FAILED"
        })
        continue

    # Find the FINAL TEST section.
    marker = "FINAL TEST"
    idx = output.rfind(marker)

    section = output[idx:] if idx >= 0 else output

    def value_after(label):
        for line in section.splitlines():
            if label in line:
                try:
                    return float(
                        line.split(":")[1]
                        .strip()
                        .replace("$", "")
                        .replace("%", "")
                        .replace(",", "")
                    )
                except Exception:
                    return None
        return None

    trades = value_after("Trades")
    final_balance = value_after("Final balance")
    net_profit = value_after("Net profit")
    win_rate = value_after("Win rate")
    profit_factor = value_after("Profit factor")
    max_dd = value_after("Max drawdown")

    # Return is stored in Net profit line as "$x (x%)".
    return_pct = None
    for line in section.splitlines():
        if "Net profit" in line and "(" in line:
            try:
                inside = line.split("(")[1].split("%")[0]
                return_pct = float(inside)
            except Exception:
                pass

    results.append({
        "fast": fast,
        "slow": slow,
        "trend": trend,
        "trades": trades,
        "final_balance": final_balance,
        "net_profit": net_profit,
        "return_pct": return_pct,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown": max_dd,
        "status": "OK"
    })

df = pd.DataFrame(results)

if not df.empty:
    df.to_csv(REPORT, index=False)

    ok = df[df["status"] == "OK"].copy()

    if not ok.empty:
        ok["score"] = (
            ok["return_pct"].fillna(-999)
            * ok["profit_factor"].fillna(0)
        )

        ok = ok.sort_values(
            ["return_pct", "profit_factor"],
            ascending=False
        )

        print()
        print("=" * 100)
        print("ROBUSTNESS RESULTS — FINAL TEST")
        print("=" * 100)

        cols = [
            "fast", "slow", "trend",
            "trades", "return_pct",
            "win_rate", "profit_factor",
            "max_drawdown"
        ]

        print(
            ok[cols].to_string(index=False)
        )

        print()
        print("=" * 100)
        print("TOP 10")
        print("=" * 100)
        print(ok[cols].head(10).to_string(index=False))

        positive = ok[
            (ok["return_pct"] > 0) &
            (ok["profit_factor"] > 1)
        ]

        print()
        print(f"Profitable + PF>1 combinations: {len(positive)}/{len(ok)}")

        if len(positive) > 0:
            print()
            print("BEST CURRENT ROBUST CANDIDATES:")
            print(
                positive[cols].head(10).to_string(index=False)
            )

print()
print(f"Saved: {REPORT}")
