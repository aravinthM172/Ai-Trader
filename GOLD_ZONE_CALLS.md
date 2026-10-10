# Gold: zone calls (H4 zone, small-chart confirmation, TP1 / TP2 / TP3)

N for the deflated Sharpe: 2108. Look-ahead check ok: True.

| chart | variant | years | trades/yr | win % | exits (%) | mean R | t | R/yr | buys R | sells R | first 70 % | last 30 % | last 2 y | slices | 2x cost | max DD (R) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M15 | bias_tp2 | 4.2 | 44.4 | 39.9 | {'stop': 58.5, 'target': 37.8, 'time': 3.7} | 0.1496 | 1.42 | 6.6 | -0.0123 | 0.3374 | 0.1729 | 0.0985 | 0.1409 | 6/8 | 0.1401 | 9.3 | - |
| M15 | bias_tp123 | 4.2 | 44.9 | 52.1 | {'stop': 47.4, 'breakeven': 31.1, 'target': 21.1, 'time': 0.5} | 0.0638 | 0.75 | 2.9 | -0.032 | 0.1702 | 0.1338 | -0.0881 | -0.0124 | 5/8 | 0.0542 | 13.9 | - |
| M15 | nobias_tp2 | 4.2 | 64.5 | 37.0 | {'stop': 61.9, 'target': 35.2, 'time': 2.9} | 0.0662 | 0.77 | 4.3 | -0.0173 | 0.1389 | 0.0841 | 0.0214 | 0.0505 | 7/8 | 0.0561 | 10.4 | - |
| H1 | bias_tp2 | 21.5 | 38.1 | 40.5 | {'stop': 54.8, 'target': 31.0, 'time': 14.2} | 0.1011 | 2.14 | 3.9 | 0.1269 | 0.0689 | 0.0876 | 0.1305 | 0.1551 | 6/8 | 0.0939 | 45.3 | weak |
| H1 | bias_tp123 | 21.5 | 38.0 | 53.1 | {'stop': 44.8, 'breakeven': 25.9, 'target': 25.5, 'time': 3.8} | 0.0906 | 2.27 | 3.4 | 0.1203 | 0.0525 | 0.0948 | 0.0811 | 0.0138 | 7/8 | 0.0834 | 25.5 | weak |
| H1 | nobias_tp2 | 21.5 | 55.5 | 37.9 | {'stop': 58.0, 'target': 29.8, 'time': 12.2} | 0.0429 | 1.1 | 2.4 | 0.0532 | 0.0326 | 0.0618 | 0.0013 | 0.1303 | 4/8 | 0.0351 | 70.9 | - |

## Follow-up (post-hoc)

- Second data source, FundingPips H1 2010-, bias_tp2: 664 trades, **+0.013R, t 0.2** (Dukascopy 2005-: +0.101R, t 2.1).
  From 2018: Dukascopy +0.04R, FundingPips -0.035R.  The Dukascopy result leans on 2005-2014.
- M15 (FundingPips 2022-): +0.15R on 188 trades, t 1.4 -- not significant; 2025 was negative.
- Stop per 0.01 lot over the last 12 months: median $43 (M15) / $62 (H1), never under $12.50.
- TP1 / TP2 / TP3 with stop to entry lifts the "win rate" from 40 % to 52 % and lowers the profit per trade.

## Sources

TradingView gold ideas and scripts (H4 bias, supply / demand zone, M15 change of character, TP1-TP3, stop to entry
after TP1); reviews of Telegram gold channels (selective reporting, no verified records).
