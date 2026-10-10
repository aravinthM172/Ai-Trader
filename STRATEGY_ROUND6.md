# Strategy round 6 -- trailing stops, rotation, IBS, lead-lag, outside filters, funding rate

Deflated-Sharpe N = 696.  FundingPips costs and swaps.  Rules pre-registered in `backtest/strategy_round6.py`.
Simulator checks: momentum baseline = live simulation **True**, breakout baseline = round 5 **True**.

## A. Trailing stops vs the untrailed baseline (R per year at fixed risk)

| test | market | base R/yr | trailed R/yr | base mean R | trailed mean R | 1st half | 2nd half | last 2 y | base DD | trailed DD | better everywhere |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|
| A1_mom_trail3_tp6 | BTCUSD | 43.4 | 28.4 | 0.1709 | 0.0837 | 394.0 -> 264.4 | 158.6 -> 97.3 | 42.4 -> 14.8 | 41.0 | 34.8 | no |
| A1_mom_trail3_notp | BTCUSD | 43.4 | 32.0 | 0.1709 | 0.1051 | 394.0 -> 297.9 | 158.6 -> 108.8 | 42.4 -> 17.9 | 41.0 | 33.1 | no |
| A2_vbo_daytrail | BTCUSD | 24.2 | 45.0 | 0.1137 | 0.3687 | 244.9 -> 469.8 | 63.1 -> 102.4 | 15.5 -> 68.0 | 18.9 | 35.0 | **yes** |
| A1_mom_trail3_tp6 | ETHUSD | 42.4 | 39.5 | 0.1693 | 0.1181 | 291.8 -> 243.2 | 74.0 -> 97.7 | 26.8 -> 61.2 | 35.9 | 29.9 | no |
| A1_mom_trail3_notp | ETHUSD | 42.4 | 41.6 | 0.1693 | 0.139 | 291.8 -> 247.9 | 74.0 -> 111.1 | 26.8 -> 73.2 | 35.9 | 26.9 | no |
| A2_vbo_daytrail | ETHUSD | 20.3 | 33.3 | 0.1006 | 0.2787 | 105.4 -> 183.4 | 71.1 -> 105.9 | 38.3 -> 78.7 | 18.8 | 39.2 | **yes** |
| A1_mom_trail3_tp6 | XAUUSD | 14.4 | 14.1 | 0.1523 | 0.1128 | 180.3 -> 206.9 | 121.8 -> 89.4 | 60.7 -> 65.5 | 45.3 | 49.0 | no |
| A1_mom_trail3_notp | XAUUSD | 14.4 | 15.3 | 0.1523 | 0.1434 | 180.3 -> 224.1 | 121.8 -> 97.7 | 60.7 -> 68.1 | 45.3 | 45.8 | no |
| A2_vbo_daytrail | XAUUSD | 11.6 | 10.9 | 0.074 | 0.1093 | 195.4 -> 147.2 | 56.8 -> 89.3 | 61.9 -> 67.3 | 52.6 | 53.4 | no |
| A1_mom_trail3_tp6 | USDJPY | 14.9 | 14.8 | 0.2183 | 0.1717 | 80.5 -> 108.3 | 137.3 -> 107.5 | 6.1 -> -0.2 | 20.2 | 15.5 | no |
| A1_mom_trail3_notp | USDJPY | 14.9 | 15.6 | 0.2183 | 0.2194 | 80.5 -> 108.2 | 137.3 -> 119.6 | 6.1 -> 0.0 | 20.2 | 19.5 | no |
| A2_vbo_daytrail | USDJPY | 8.7 | 11.4 | 0.0517 | 0.103 | 16.8 -> 60.8 | 120.0 -> 118.3 | 43.4 -> 14.9 | 31.3 | 45.0 | no |
| A1_mom_trail3_tp6 | NDX100 | 8.3 | 7.6 | 0.1125 | 0.0818 | 49.9 -> 45.2 | 56.6 -> 51.3 | -0.8 -> -9.0 | 29.4 | 27.9 | no |
| A1_mom_trail3_notp | NDX100 | 8.3 | 6.3 | 0.1125 | 0.0813 | 49.9 -> 32.5 | 56.6 -> 48.2 | -0.8 -> -9.8 | 29.4 | 34.7 | no |
| A2_vbo_daytrail | NDX100 | 6.5 | -4.6 | 0.0457 | -0.0512 | -1.9 -> -65.7 | 88.1 -> 4.2 | 21.4 -> -31.4 | 47.1 | 78.0 | no |

## Stand-alone tests

| test | market | trades | /yr | mean R | t | win % | PF | first 70 % | last 30 % | last 2 y | WF+ | 2x cost | 2nd src | DSR | PASS | lead | failed checks |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|:-:|---|
| A1_mom_trail3_tp6 | BTCUSD | 4322 | 339.4 | 0.0837 | 3.94 | 36 | 1.164 | 0.0979 | 0.0539 | 0.0202 | 8/8 | 0.0587 | None | 0.794 | no | **lead** | dsr_95 |
| A1_mom_trail3_notp | BTCUSD | 3870 | 303.9 | 0.1051 | 4.07 | 36 | 1.206 | 0.12 | 0.0742 | 0.0271 | 8/8 | 0.0798 | None | 0.841 | no | **lead** | dsr_95 |
| A2_vbo_daytrail | BTCUSD | 1552 | 121.9 | 0.3687 | 4.45 | 32 | 1.631 | 0.4704 | 0.1596 | 0.2596 | 7/8 | 0.3458 | None | 0.969 | **yes** | **lead** |  |
| A1_mom_trail3_tp6 | ETHUSD | 2886 | 329.4 | 0.1181 | 4.51 | 38 | 1.239 | 0.1223 | 0.1068 | 0.1053 | 8/8 | 0.1028 | None | 0.923 | no | **lead** | years_10, dsr_95 |
| A1_mom_trail3_notp | ETHUSD | 2582 | 294.7 | 0.139 | 4.31 | 37 | 1.278 | 0.1415 | 0.1323 | 0.1396 | 8/8 | 0.1235 | None | 0.901 | no | **lead** | years_10, dsr_95 |
| A2_vbo_daytrail | ETHUSD | 1038 | 118.5 | 0.2787 | 3.47 | 31 | 1.483 | 0.2585 | 0.3323 | 0.3819 | 6/8 | 0.2659 | None | 0.65 | no | **lead** | years_10, dsr_95 |
| A1_mom_trail3_tp6 | XAUUSD | 2626 | 122.3 | 0.1128 | 4.1 | 37 | 1.225 | 0.1282 | 0.0777 | 0.275 | 6/8 | 0.1024 | None | 0.84 | no | **lead** | dsr_95 |
| A1_mom_trail3_notp | XAUUSD | 2244 | 104.5 | 0.1434 | 4.29 | 38 | 1.286 | 0.1613 | 0.1025 | 0.3583 | 6/8 | 0.133 | None | 0.893 | no | **lead** | dsr_95 |
| A2_vbo_daytrail | XAUUSD | 2164 | 100.8 | 0.1093 | 2.34 | 33 | 1.19 | 0.116 | 0.0938 | 0.3485 | 5/8 | 0.0994 | None | 0.177 | no |  | wf_6of8, dsr_95 |
| A1_mom_trail3_tp6 | USDJPY | 1257 | 79.8 | 0.1717 | 4.14 | 37 | 1.347 | 0.2061 | 0.1 | -0.0009 | 7/8 | 0.1678 | None | 0.854 | no | **lead** | dsr_95 |
| A1_mom_trail3_notp | USDJPY | 1038 | 65.9 | 0.2194 | 4.13 | 38 | 1.444 | 0.2544 | 0.1469 | 0.0001 | 7/8 | 0.2155 | None | 0.868 | no | **lead** | dsr_95 |
| A2_vbo_daytrail | USDJPY | 1739 | 110.3 | 0.103 | 2.08 | 32 | 1.185 | 0.0675 | 0.1939 | 0.0759 | 5/8 | 0.099 | None | 0.12 | no |  | wf_6of8, dsr_95 |
| A1_mom_trail3_tp6 | NDX100 | 1179 | 99.0 | 0.0818 | 1.96 | 38 | 1.155 | 0.1053 | 0.031 | -0.0489 | 6/8 | 0.0657 | None | 0.11 | no |  | dsr_95 |
| A1_mom_trail3_notp | NDX100 | 993 | 83.4 | 0.0813 | 1.58 | 36 | 1.151 | 0.0988 | 0.0428 | -0.0617 | 6/8 | 0.0656 | None | 0.05 | no |  | dsr_95 |
| A2_vbo_daytrail | NDX100 | 1201 | 100.8 | -0.0512 | -1.11 | 32 | 0.913 | -0.0445 | -0.0699 | -0.1771 | 3/8 | -0.0644 | None | 0.0 | no |  | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95 |
| B_rotation_60d | all | 6690 | 311.1 | -0.0126 | -1.28 | 48 | 0.959 | 0.0011 | -0.0446 | -0.0221 | 4/8 | -0.0184 | None | 0.0 | no |  | full_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95 |
| B_rotation_250d | all | 6462 | 311.0 | 0.0053 | 0.53 | 50 | 1.018 | 0.0086 | -0.0025 | 0.0044 | 4/8 | -0.0 | None | 0.004 | no |  | last30_pos, wf_6of8, cost2x_pos, dsr_95 |
| C_ibs | NDX100 | 565 | 26.0 | 0.0156 | 1.02 | 63 | 1.126 | 0.0078 | 0.0311 | 0.0374 | 6/8 | 0.0137 | 0.0599 | 0.02 | no |  | dsr_95 |
| C_ibs | SPX500 | 535 | 24.6 | 0.0308 | 2.23 | 64 | 1.302 | 0.032 | 0.0285 | 0.0196 | 7/8 | 0.0253 | 0.0319 | 0.194 | no |  | dsr_95 |
| C_ibs | DJI30 | 534 | 24.5 | 0.0325 | 2.3 | 62 | 1.314 | 0.0282 | 0.0407 | 0.0686 | 7/8 | 0.0315 | 0.0068 | 0.211 | no |  | dsr_95 |
| C_ibs | GER40 | 510 | 23.4 | -0.0119 | -0.77 | 59 | 0.909 | -0.0027 | -0.0296 | -0.051 | 4/8 | -0.0151 | -0.0151 | 0.0 | no |  | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| C_ibs | JP225 | 489 | 22.5 | 0.001 | 0.06 | 58 | 1.007 | 0.0131 | -0.0226 | 0.0298 | 4/8 | -0.0045 | 0.1252 | 0.001 | no |  | last30_pos, wf_6of8, cost2x_pos, dsr_95 |
| C_ibs | FTSE100 | 523 | 24.0 | -0.0186 | -1.26 | 58 | 0.859 | -0.0141 | -0.0277 | -0.0306 | 3/8 | -0.022 | 0.0715 | 0.0 | no |  | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95 |
| C_ibs | pooled | 3156 | 150.6 | 0.0087 | 1.4 | 61 | 1.071 | 0.0107 | 0.0046 | 0.009 | 6/8 | 0.0053 | None | 0.043 | no |  | dsr_95 |
| D_btc_leads_eth | ETHUSD | 375 | 41.1 | 0.0075 | 0.2 | 47 | 1.024 | 0.0087 | 0.005 | -0.0318 | 5/8 | -0.012 | None | 0.002 | no |  | years_10, wf_6of8, cost2x_pos, dsr_95 |
| D_eth_leads_btc | BTCUSD | 515 | 56.7 | -0.0386 | -1.2 | 44 | 0.883 | -0.0023 | -0.1292 | -0.1249 | 2/8 | -0.0697 | None | 0.0 | no |  | years_10, full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95 |
| D_ndx_leads_btc | BTCUSD | 1306 | 143.9 | -0.0365 | -1.42 | 49 | 0.9 | -0.0329 | -0.0469 | -0.0553 | 3/8 | -0.0663 | None | 0.0 | no |  | years_10, full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95 |
| F1_funding_fade | BTCUSD | 230 | 35.3 | -0.0199 | -0.22 | 37 | 0.969 | -0.0416 | 0.0223 | 0.0223 | 3/8 | -0.0451 | None | 0.0 | no |  | years_10, full_pos, first70_pos, wf_6of8, cost2x_pos, dsr_95 |
| F1_funding_fade | ETHUSD | 245 | 39.5 | -0.0633 | -0.73 | 34 | 0.905 | -0.0128 | -0.1343 | -0.1389 | 3/8 | -0.0785 | None | 0.0 | no |  | years_10, full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95 |

## E / F2. Filters on the live trades

| filter | trades | kept | all R | kept R | removed R | kept R last 30 % | removed R last 30 % | folds helped | Welch t | PASS | failed checks |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|---|
| E1_vix_ndx100 | 946 | 645 | 0.1125 | 0.0841 | 0.1733 | 0.125 | 0.0649 | 3/8 | -0.7 | no | kept_gt_removed_full, kept_gt_removed_first70, folds_6of8, welch_t_2 |
| E2_real_yield_gold | 1984 | 1256 | 0.1523 | 0.0994 | 0.2434 | 0.1231 | 0.171 | 3/8 | -1.73 | no | kept_gt_removed_full, kept_gt_removed_first70, kept_gt_removed_last30, folds_6of8, welch_t_2 |
| F2_funding_filter|BTCUSD | 1777 | 1624 | 0.0942 | 0.102 | 0.0118 | 0.0905 | -0.1497 | 5/8 | 0.62 | no | folds_6of8, welch_t_2 |
| F2_funding_filter|ETHUSD | 1575 | 1420 | 0.1255 | 0.1493 | -0.0926 | 0.0822 | -0.3481 | 7/8 | 1.76 | no | welch_t_2 |

Lead-lag clock check (correlation of hourly returns; the largest must be lag 0):
- D_btc_leads_eth: {'lag -1': -0.02, 'lag +0': 0.804, 'lag +1': -0.03}
- D_eth_leads_btc: {'lag -1': -0.03, 'lag +0': 0.804, 'lag +1': -0.02}
- D_ndx_leads_btc: {'lag -1': 0.004, 'lag +0': 0.247, 'lag +1': 0.02}

## What IBS would add on the $5k account (2018+, 0.25 % per trade)

```
                                   trades/yr  $ / yr  max DD $   pass  fail  median days
live now                               749.0  1111.0     537.0  0.922   0.0        194.0
live + IBS on 6 indices at 0.25 %      907.0  1108.0     542.0  0.931   0.0        187.0
```

**Passing (1):** A2_vbo_daytrail|BTCUSD

**Leads (10):** A1_mom_trail3_tp6|BTCUSD, A1_mom_trail3_notp|BTCUSD, A2_vbo_daytrail|BTCUSD, A1_mom_trail3_tp6|ETHUSD, A1_mom_trail3_notp|ETHUSD, A2_vbo_daytrail|ETHUSD, A1_mom_trail3_tp6|XAUUSD, A1_mom_trail3_notp|XAUUSD, A1_mom_trail3_tp6|USDJPY, A1_mom_trail3_notp|USDJPY

**Trailing stop better everywhere (2):** A2_vbo_daytrail|BTCUSD, A2_vbo_daytrail|ETHUSD
