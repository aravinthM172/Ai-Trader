# Portfolio simulation -- frozen momentum on all passing symbols, one account

Symbols (4): DAX40.vx, XAUUSD.vx, XAUEUR.vx, BTCUSD.vx

Start $10,000, risk 0.5% of equity per trade, swap included.

| guard config | trades/mo | total return | CAGR | max DD | longest underwater | ret/DD | months + | worst month | prop pass (all 3) | adopt? |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|
| caps_only | 43.1 | 83834.5% | 49.2% | -15.1% | 414 d | 5565.3 | 72.8% | -6.9% | 89% | baseline |
| caps+stoploss_guard | 38.4 | 75999.3% | 48.4% | -12.6% | 367 d | 6025.13 | 70.8% | -6.1% | 97% | no |
| caps+daily4%+dd8% | 42.4 | 82246.7% | 49.1% | -17.1% | 730 d | 4797.12 | 72.1% | -7.5% | 90% | no |
| all_protections | 37.9 | 68292.4% | 47.4% | -13.9% | 419 d | 4914.61 | 70.3% | -8.0% | 91% | no |
