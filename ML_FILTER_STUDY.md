# Machine-learning signal filter -- study of 2026-10-10

Signals: 14860 (2005-09-23 to 2026-10-07), markets BTCUSD, ETHUSD, XAUUSD, NDX100, USDJPY, GER40. Out-of-sample years 2016-2026: 11457 signals. Model: ridge on 9 features (strategy/ml_filter.py). Costs include standard commission. Selection is post-hoc (a skipped trade does not free the slot). PAPER ONLY: execution/paper_ml_filter.py scores the live trades.

## Out-of-sample result

| set | trades | avg R | total R | t | win % | years + |
|---|---|---|---|---|---|---|
| all signals | 11457 | 0.08 | 914.8 | 4.89 | 30.5 | 11/11 |
| live rule | 7352 | 0.116 | 852.4 | 5.65 | 31.9 | 11/11 |
| live rule + filter: taken | 4393 | 0.173 | 758.7 | 6.37 | 33.1 | 11/11 |
| live rule + filter: skipped | 2959 | 0.032 | 93.6 | 1.01 | 30.2 | 7/11 |
| all signals + filter: taken | 5633 | 0.152 | 854.3 | 6.35 | 32.4 | 11/11 |
| all signals + filter: skipped | 5824 | 0.01 | 60.5 | 0.47 | 28.8 | 5/11 |

## Last 3 years, inside the live rule

| set | trades | avg R | total R | t | win % | years + |
|---|---|---|---|---|---|---|
| live rule | 2238 | 0.067 | 150.4 | 1.81 | 30.3 | 3/3 |
| live rule + filter: taken | 1305 | 0.137 | 179.0 | 2.77 | 32.0 | 3/3 |
| live rule + filter: skipped | 933 | -0.031 | -28.6 | -0.55 | 28.1 | 1/3 |

## Per market, inside the live rule

| market | trades | avg R | taken | taken avg R | skipped avg R |
|---|---|---|---|---|---|
| BTCUSD | 2769 | 0.14 | 1340 | 0.211 | 0.073 |
| ETHUSD | 2027 | 0.137 | 979 | 0.238 | 0.043 |
| XAUUSD | 531 | 0.093 | 472 | 0.167 | -0.506 |
| NDX100 | 425 | 0.105 | 379 | 0.126 | -0.064 |
| USDJPY | 1050 | 0.087 | 794 | 0.147 | -0.098 |
| GER40 | 550 | 0.005 | 429 | 0.001 | 0.017 |

## Score fifths (1 = lowest score), all signals

| score fifth | trades | avg R | total R | t | win % | years + |
|---|---|---|---|---|---|---|
| 1 | 2292 | -0.022 | -50.1 | -0.63 | 28.3 | 5/11 |
| 2 | 2291 | 0.003 | 6.5 | 0.08 | 28.4 | 6/11 |
| 3 | 2291 | 0.093 | 213.4 | 2.54 | 30.9 | 8/11 |
| 4 | 2291 | 0.084 | 191.4 | 2.26 | 30.4 | 10/11 |
| 5 | 2292 | 0.242 | 553.6 | 6.31 | 34.8 | 10/11 |

## Is it luck?

| scope | taken avg R | random same size | p | taken - skipped | 95 % low | 95 % high |
|---|---|---|---|---|---|---|
| all signals | 0.152 | 0.082 | 0.0005 | 0.141 | 0.078 | 0.205 |
| inside the live rule | 0.173 | 0.113 | 0.001 | 0.141 | 0.061 | 0.225 |

## Final model (all signals), standardised coefficients

vol_rel -0.060, rsi_s +0.057, trend_ok +0.046, mom_atr +0.040, is_buy +0.033, hour_cos -0.029, ema_dist -0.025, hour_sin +0.018, trend_dist -0.001
; threshold +0.0828
