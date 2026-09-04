from pathlib import Path

src = Path("backtest/realistic_xauusd.py")

if not src.exists():
    raise SystemExit("ERROR: realistic_xauusd.py not found")

text = src.read_text(encoding="utf-8")

print("=" * 70)
print("REALISTIC ENGINE CHECK")
print("=" * 70)

checks = {
    "tick value": "TICK_VALUE" in text or "TICK_VALUE =" in text,
    "tick size": "TICK_SIZE" in text or "TICK_SIZE =" in text,
    "contract size": "CONTRACT_SIZE" in text,
    "spread": "SPREAD" in text.upper(),
    "slippage": "SLIPPAGE" in text.upper(),
    "minimum lot": "MIN_LOT" in text,
    "bid/ask": "ask" in text.lower() and "bid" in text.lower(),
    "ATR": "ATR" in text,
    "RSI": "RSI" in text,
    "equity drawdown": "equity" in text.lower() and "drawdown" in text.lower(),
}

for name, ok in checks.items():
    print(f"{'OK' if ok else 'MISSING':8} {name}")

print()
print("Creating realistic optimizer specification...")

Path("backtest/REALISTIC_OPTIMIZER_REQUIRED.txt").write_text("""
REALISTIC XAUUSD NUMBA OPTIMIZER

MUST USE:
- XAUUSD M5
- Correct MT5 tick value
- tick_value / tick_size for $/price/lot
- bid/ask execution
- spread cost
- slippage
- ATR
- SMA-seeded EMA
- SMA-seeded RSI
- no look-ahead
- signal from completed candle
- entry on next candle open
- SL/TP intrabar execution
- gap-aware stop
- minimum lot rejection
- lot-step floor
- maximum lot
- margin constraint
- equity-based drawdown
- train/validation/final chronological split

NEVER:
- use close[i] as an entry after deciding on candle i
- use contract_size as the direct $/price multiplier
- force 0.01 lot
- optimize using final test
- select parameters based on final test
- claim profitability from training alone

TARGET:
31,104+ configurations
Numba compiled
parallel execution where safe
checkpointing
progress every 500 configurations
training ranking
validation filtering
untouched final test
CSV output

RECOMMENDED FILTER:
training PF >= 1.05
validation PF >= 1.05
validation profit > 0
final PF >= 1.05
final profit > 0
controlled drawdown
minimum trade count
""", encoding="utf-8")

print("Created:")
print("backtest/REALISTIC_OPTIMIZER_REQUIRED.txt")
print()
print("STOPPING OLD OPTIMIZER.")
print("Do NOT run backtest/optimizer_numba.py for strategy selection.")
