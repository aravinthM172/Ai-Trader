import os
import subprocess
import sys

print("=" * 70)
print("FAST REALISTIC XAUUSD OPTIMIZATION")
print("=" * 70)
print("Using Python:", sys.executable)
print("CPU cores:", os.cpu_count())
print()

candidates = [
    "backtest/optimizer_realistic_numba.py",
    "backtest/realistic_optimizer_numba.py",
    "backtest/optimizer_numba.py",
]

for f in candidates:
    if os.path.exists(f):
        print("FOUND:", f)
        print("Starting optimizer...")
        subprocess.run([sys.executable, f], check=False)
        break
else:
    print("NO REALISTIC NUMBA OPTIMIZER FOUND")
    print("Create optimizer_realistic_numba.py before continuing.")
