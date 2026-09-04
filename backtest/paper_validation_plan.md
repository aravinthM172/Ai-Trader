# XAUUSD Paper Validation ? Locked Research Protocol

## Current status

Backtests are complete.
No live deployment.
No further parameter optimization on the locked final holdout.

## Strategy family under observation

Primary candidate:
EMA 20 / 60 / 150
RSI 20
ATR 20
SL 2.0 ATR
TP 3.5 ATR

Reference candidate:
EMA 15 / 80 / 200
RSI 20
ATR 20
SL 2.5 ATR
TP 3.5 ATR

These are OBSERVATION candidates, not proven winners.

## Paper validation requirements

Record every signal and simulated order using the broker's actual:

- Bid
- Ask
- Spread
- Tick size
- Tick value
- Contract size
- Volume step
- Minimum volume
- Stop-level restrictions
- Slippage/fill information

Do NOT replace broker spread with an artificial $0.35 floor during paper validation.

## Minimum observation target

Collect at least:

- 100 trades, preferably 200+
- Multiple market sessions
- Multiple volatility regimes
- Both winning and losing periods

## Metrics

Track:

- Total trades
- Win rate
- Gross profit
- Gross loss
- Profit factor
- Net profit
- Maximum drawdown
- Average win
- Average loss
- Expectancy/trade
- Average spread
- Average slippage
- Rejected/invalid orders
- Signal-to-fill delay

## Decision rule

Do not change parameters during the observation period.

If the strategy fails, record the failure.

If the strategy passes, continue validation before risking real capital.

## Important

Existing files remain immutable research records:

- reports/xauusd_realistic_optimization.csv
- reports/xauusd_realistic_final_test.csv
- reports/xauusd_true_walk_forward.csv
- reports/true_walk_forward_final_holdout.csv
- reports/final_holdout_cost_sensitivity.csv

The original final holdout remains locked.
