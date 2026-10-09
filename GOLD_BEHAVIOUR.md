# How XAUUSD moves -- FundingPips H1 2010-today

Discovery 2010-2017 vs confirmation 2018-today; **real** = same sign and |t| >= 2 in both.  Descriptive only -- nothing here is a tested strategy.

## 1. By session (UTC)

| session | vol_bp | mean_ret_bp | disc_t | conf_t | real |
|---|---|---|---|---|---|
| Asia 00-07 | 10.3 | 0.33 | 3.87 | 1.33 | no |
| London 07-12 | 12.5 | -0.31 | -4.78 | 1.22 | no |
| London+NY 12-16 | 21 | -0.2 | -0.62 | -0.56 | no |
| NY 16-21 | 13.1 | 0.04 | 0.42 | -0.05 | no |
| Rollover 21-24 | 7.8 | 0.3 | 1.46 | 0.83 | no |

By hour:

| hour | vol_bp | share_of_daily_range | mean_ret_bp | disc_t | conf_t | real |
|---|---|---|---|---|---|---|
| 0 | 10 | nan | 0.71 | 1.35 | 2.43 | no |
| 1 | 12.8 | nan | 0.01 | 0.26 | -0.15 | no |
| 2 | 12.4 | nan | 0.54 | 2.09 | 0.63 | no |
| 3 | 8.9 | nan | 0.18 | 1.47 | -0.23 | no |
| 4 | 8 | nan | 0.04 | 0.47 | -0.17 | no |
| 5 | 9.2 | nan | 0 | 1.95 | -1.44 | no |
| 6 | 11 | nan | 0.85 | 2.66 | 2.05 | **yes** |
| 7 | 13.1 | nan | -0.77 | -3.63 | -0.26 | no |
| 8 | 14.4 | nan | -0.12 | -1.32 | 0.82 | no |
| 9 | 11.9 | nan | -0.56 | -3.03 | 0.03 | no |
| 10 | 11.3 | nan | -0.39 | -2.98 | 0.67 | no |
| 11 | 12 | nan | 0.28 | -0.07 | 1.42 | no |
| 12 | 15.3 | nan | -0.09 | 0.38 | -0.67 | no |
| 13 | 24 | nan | -0.69 | -1.22 | -0.66 | no |
| 14 | 23 | nan | -0.61 | -1.45 | -0.32 | no |
| 15 | 21.6 | nan | 0.6 | 1.34 | 0.45 | no |
| 16 | 16.1 | nan | 0.32 | 1.08 | 0.18 | no |
| 17 | 13.2 | nan | -0.27 | -0.39 | -0.91 | no |
| 18 | 13.1 | nan | -0.03 | -0.46 | 0.34 | no |
| 19 | 12.4 | nan | 0.08 | -0.64 | 0.98 | no |
| 20 | 10.6 | nan | 0.09 | 1.49 | -1.05 | no |
| 21 | 6.5 | nan | 0.2 | 1.46 | 0.22 | no |
| 22 | nan | nan | nan | 0 | 0 | no |
| 23 | 16.3 | nan | 0.99 | 0 | 0.87 | no |

## 2. Day of week

| day | n | mean_ret_bp | vol_bp | disc_t | conf_t | real |
|---|---|---|---|---|---|---|
| Mon | 863 | -3.3 | 66.2 | -1.28 | -0.11 | no |
| Tue | 869 | 0.9 | 71.8 | -0.38 | 0.66 | no |
| Wed | 869 | -1.8 | 71.7 | -1.53 | 0.71 | no |
| Thu | 869 | -1.2 | 74.6 | -0.67 | 0.16 | no |
| Fri | 853 | 8.2 | 75.5 | 2.54 | 0.68 | no |
| Sun | 97 | -5.6 | 22 | 0 | -1.79 | no |

## 3. Trend or reversal?  (lag-1 autocorrelation; > 0 = moves continue)

| horizon | autocorr_disc | t_disc | autocorr_conf | t_conf | real |
|---|---|---|---|---|---|
| 1 hour | 0.007 | 0.75 | -0.016 | -1.44 | no |
| 4 hours | -0.014 | -0.81 | -0.001 | -0.04 | no |
| 1 day | -0.006 | -0.25 | -0.002 | -0.06 | no |
| 1 week | 0.048 | 0.94 | -0.08 | -1.25 | no |

Variance ratios (> 1 = trending, < 1 = mean-reverting): VR(4h) 2010-17 1.031, VR(4h) 2018-26 0.973, VR(24h) 2010-17 1.082, VR(24h) 2018-26 1.021, VR(120h) 2010-17 1.061, VR(120h) 2018-26 0.874

## 4. Breakouts

| setup | days | close_beyond_pct | mean_followthrough_atr | disc_t | conf_t | real |
|---|---|---|---|---|---|---|
| break of yesterday's HIGH | 2474 | 52.1 | 0.051 | 2.26 | 2.84 | **yes** |
| break of yesterday's LOW | 2256 | 48.7 | 0.021 | 1.31 | 0.54 | no |
| Asian range broken in London (first side) | 3748 | 51 | 0.094 | 2.72 | 1.91 | no | follow-through in units of the Asian range |

## 5. News hours (volatility vs a normal same hour)

| event | events | vol_multiple_event_hour | vol_multiple_next_hour | mean_ret_event_hour_bp | t | continuation_next_hour_pct |
|---|---|---|---|---|---|---|
| FOMC | 132 | 3.13 | 3.26 | 5.2 | 0.95 | 55.3 |
| NFP | 194 | 1.55 | 1.77 | -5.8 | -1.79 | 53.6 |

## 6. Macro drivers (daily)

Same-day correlation with gold: dxy 2010-17 -0.329, dxy 2018-26 -0.388, ry 2010-17 -0.24, ry 2018-26 -0.289, y10 2010-17 -0.185, y10 2018-26 -0.243, silver 2010-17 0.812, silver 2018-26 0.773, vix 2010-17 0.018, vix 2018-26 -0.05

Does yesterday's move predict today's gold?

| driver | disc_bp | disc_t | conf_bp | conf_t | real | meaning |
|---|---|---|---|---|---|---|
| dxy | -2.62 | -1.12 | 0.8 | 0.34 | no | gold return today x sign of yesterday's dxy change |
| ry | -9.58 | -4.27 | -3.76 | -1.67 | no | gold return today x sign of yesterday's ry change |
| y10 | -7.27 | -3.19 | -1.75 | -0.77 | no | gold return today x sign of yesterday's y10 change |
| silver | -4.95 | -2.12 | -4.13 | -1.75 | no | gold return today x sign of yesterday's silver change |
| vix | -0.64 | -0.27 | -1.43 | -0.6 | no | gold return today x sign of yesterday's vix change |
| gold itself | -3.07 | -1.31 | -4.39 | -1.88 | no | 1-day momentum |

COT (managed money) -> next week's gold return (bp):

| rule | disc_mean | disc_t | conf_mean | conf_t | real |
|---|---|---|---|---|---|
| follow managed-money change | -7.5 | -0.66 | -21.3 | -2.25 | no |
| fade managed-money extremes (|z|>1.5) | 24.1 | 1.09 | -7.9 | -0.46 | no |

## 7. Live momentum trades by context -- exit tp3 (all: 4898 trades, 0.002 R)

| context | trades | mean_R | disc_R | conf_R | disc_t | conf_t |
|---|---|---|---|---|---|---|
| entry Asia 00-07 | 820 | 0.038 | 0.064 | 0.02 | 0.95 | 0.36 |
| entry London 07-12 | 919 | 0.012 | 0.046 | -0.017 | 0.75 | -0.32 |
| entry London+NY 12-16 | 1598 | -0.021 | -0.059 | 0.015 | -1.35 | 0.34 |
| entry NY 16-21 | 1332 | -0.005 | -0.035 | 0.024 | -0.73 | 0.51 |
| entry Rollover 21-24 | 229 | 0.034 | 0.208 | -0.072 | 1.53 | -0.71 |
| volatility low third | 1155 | 0.029 | 0.053 | -0.007 | 1.1 | -0.12 |
| volatility mid third | 1505 | -0.067 | -0.109 | -0.03 | -2.4 | -0.68 |
| volatility high third | 2174 | 0.037 | 0.041 | 0.034 | 0.96 | 1.02 |
| with DXY (trade in gold-friendly direction) | 2537 | 0.026 | 0.051 | 0.004 | 1.4 | 0.11 |
| against DXY | 2361 | -0.024 | -0.065 | 0.011 | -1.77 | 0.33 |
| with real yield (trade in gold-friendly direction) | 2594 | 0.018 | -0.001 | 0.035 | -0.01 | 1.03 |
| against real yield | 2304 | -0.016 | -0.009 | -0.023 | -0.24 | -0.65 |
| with gold (trade in gold-friendly direction) | 2675 | 0.037 | 0.016 | 0.055 | 0.46 | 1.68 |
| against gold | 2223 | -0.041 | -0.029 | -0.051 | -0.76 | -1.44 |
| longs | 2574 | 0.018 | -0.039 | 0.064 | -1.05 | 1.93 |
| shorts | 2324 | -0.016 | 0.029 | -0.061 | 0.8 | -1.73 |

## 7. Live momentum trades by context -- exit tp6 (all: 2981 trades, 0.039 R)

| context | trades | mean_R | disc_R | conf_R | disc_t | conf_t |
|---|---|---|---|---|---|---|
| entry Asia 00-07 | 522 | 0.094 | 0.07 | 0.113 | 0.59 | 1.1 |
| entry London 07-12 | 561 | 0.028 | 0.015 | 0.04 | 0.14 | 0.4 |
| entry London+NY 12-16 | 917 | 0.107 | 0.1 | 0.112 | 1.18 | 1.4 |
| entry NY 16-21 | 844 | -0.056 | -0.07 | -0.043 | -0.85 | -0.55 |
| entry Rollover 21-24 | 137 | -0.004 | -0.033 | 0.021 | -0.15 | 0.1 |
| volatility low third | 696 | 0.075 | 0.135 | -0.008 | 1.48 | -0.08 |
| volatility mid third | 870 | 0.001 | -0.076 | 0.071 | -0.93 | 0.87 |
| volatility high third | 1372 | 0.045 | 0.017 | 0.062 | 0.23 | 1.05 |
| with DXY (trade in gold-friendly direction) | 1518 | 0.083 | 0.089 | 0.078 | 1.35 | 1.26 |
| against DXY | 1463 | -0.008 | -0.046 | 0.025 | -0.72 | 0.41 |
| with real yield (trade in gold-friendly direction) | 1572 | 0.044 | 0.004 | 0.079 | 0.06 | 1.31 |
| against real yield | 1409 | 0.033 | 0.046 | 0.022 | 0.68 | 0.35 |
| with gold (trade in gold-friendly direction) | 1600 | 0.115 | 0.064 | 0.158 | 0.98 | 2.6 |
| against gold | 1381 | -0.05 | -0.021 | -0.076 | -0.33 | -1.25 |
| longs | 1579 | 0.056 | -0.042 | 0.134 | -0.66 | 2.24 |
| shorts | 1402 | 0.019 | 0.091 | -0.05 | 1.37 | -0.81 |
