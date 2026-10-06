# Strategy round 3 -- classic book strategies on the DAILY chart, 2005-today

49 symbols, 245 (strategy, symbol) pairs, deflated-Sharpe N = 276. Broker spread + slippage + swap. Pass rule pre-registered in `backtest/strategy_round3_daily.py`.

## Pooled per strategy (all eligible symbols, 1 R per trade)

| strategy | symbols | trades/yr | mean R | win % | PF | first 70 % | last 30 % | WF+ | 2x cost | broker D1 | yrs + | DSR | corr live | PASS |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|
| turtle55 | 45 | 242.2 | -0.0586 | 29 | 0.907 | -0.0155 | -0.1522 | 2/8 | -0.0858 | -0.1456 | 8/22 | 0.0 | 0.023 | no |
| clenow_trend | 45 | 153.0 | -0.0322 | 36 | 0.935 | -0.0071 | -0.091 | 2/8 | -0.0496 | -0.0783 | 9/22 | 0.0 | -0.053 | no |
| tsmom12 | 45 | 90.8 | 0.046 | 24 | 1.06 | 0.1499 | -0.2484 | 3/8 | 0.0282 | -0.3299 | 8/21 | 0.002 | -0.032 | no |
| connors_rsi2 | 10 | 36.2 | 0.0635 | 71 | 1.374 | 0.0446 | 0.1033 | 8/8 | 0.0451 | 0.068 | 17/22 | 0.71 | -0.071 | no |
| williams_vbo | 45 | 7390.0 | -0.2863 | 24 | 0.284 | -0.2488 | -0.3684 | 0/8 | -0.3501 | -0.0199 | 2/22 | 0.0 | 0.104 | no |
| crabel_nr7 | 45 | 1664.3 | -0.2439 | 25 | 0.449 | -0.2109 | -0.3148 | 1/8 | -0.3418 | -0.0893 | 4/22 | 0.0 | 0.094 | no |
| turn_of_month | 10 | 118.3 | 0.019 | 54 | 1.095 | 0.0086 | 0.043 | 5/8 | 0.0016 | 0.0246 | 12/22 | 0.149 | 0.022 | no |

## Summary per strategy

| strategy | symbols tested | mean R (avg over symbols) | symbols with mean R > 0 | PASS |
|---|--:|--:|--:|--:|
| turtle55 | 45 | -0.0246 | 15 | 0 |
| clenow_trend | 45 | -0.0192 | 18 | 0 |
| tsmom12 | 45 | -0.2943 | 8 | 0 |
| connors_rsi2 | 10 | 0.0598 | 9 | 0 |
| williams_vbo | 45 | -0.2777 | 10 | 2 |
| crabel_nr7 | 45 | -0.2350 | 11 | 1 |
| turn_of_month | 10 | 0.0186 | 7 | 0 |

## Top 30 pairs (by deflated Sharpe, then mean R)

| strategy | symbol | years | trades/yr | mean R | win % | PF | first 70 % | last 30 % | WF+ | 2x cost | broker D1 | yrs + | DSR | corr live | PASS |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|
| crabel_nr7 | BTCUSD.vx | 12.1 | 55.3 | 0.4597 | 56 | 2.477 | 0.5714 | 0.1881 | 8/8 | 0.4201 | 0.0154 | 12/13 | 1.0 | 0.086 | **yes** |
| williams_vbo | BTCUSD.vx | 12.1 | 222.0 | 0.221 | 56 | 2.191 | 0.2874 | 0.0706 | 8/8 | 0.1995 | -0.0246 | 13/13 | 1.0 | 0.045 | no |
| crabel_nr7 | US30.vx | 21.8 | 37.8 | 0.2007 | 57 | 1.713 | 0.2414 | 0.1155 | 8/8 | 0.1895 | -0.0011 | 18/22 | 1.0 | 0.121 | no |
| williams_vbo | JPN225.vx | 21.8 | 160.5 | 0.088 | 54 | 1.491 | 0.1016 | 0.0571 | 8/8 | 0.067 | 0.0228 | 21/22 | 1.0 | 0.045 | **yes** |
| williams_vbo | US30.vx | 21.8 | 170.3 | 0.0726 | 55 | 1.39 | 0.1022 | 0.0052 | 7/8 | 0.0657 | 0.4687 | 18/22 | 1.0 | 0.031 | **yes** |
| williams_vbo | ETHUSD.vx | 8.9 | 229.6 | 0.0975 | 49 | 1.446 | 0.1333 | 0.0134 | 6/8 | 0.0261 | -0.1205 | 8/10 | 0.997 | 0.034 | no |
| williams_vbo | XAUUSD.vx | 21.8 | 160.9 | 0.0452 | 50 | 1.22 | 0.0546 | 0.024 | 7/8 | 0.0364 | 0.3325 | 17/22 | 0.901 | 0.5 | no |
| crabel_nr7 | JPN225.vx | 21.8 | 38.5 | 0.112 | 52 | 1.416 | 0.1489 | 0.0282 | 7/8 | 0.0859 | 0.0473 | 18/22 | 0.839 | 0.016 | no |
| clenow_trend | BTCUSD.vx | 12.1 | 4.8 | 1.0561 | 50 | 3.959 | 1.2353 | 0.624 | 6/8 | 1.0508 | 0.6027 | 10/12 | 0.669 | None | no |
| williams_vbo | NAS100.vx | 21.8 | 164.7 | 0.0346 | 53 | 1.176 | 0.0266 | 0.0532 | 7/8 | 0.0204 | 0.0494 | 17/22 | 0.743 | 0.026 | no |
| crabel_nr7 | ETHUSD.vx | 8.9 | 53.0 | 0.2003 | 48 | 1.54 | 0.3022 | -0.0438 | 6/8 | 0.0699 | 0.0182 | 8/10 | 0.63 | 0.069 | no |
| crabel_nr7 | AUS200.vx | 21.8 | 36.7 | 0.1163 | 53 | 1.319 | 0.1423 | 0.0582 | 7/8 | -0.1503 | -0.2488 | 15/22 | 0.591 | 0.002 | no |
| crabel_nr7 | SP500.vx | 21.8 | 38.9 | 0.0984 | 53 | 1.304 | 0.1301 | 0.03 | 4/8 | -0.0403 | -0.0715 | 13/22 | 0.572 | 0.061 | no |
| williams_vbo | UK100.vx | 21.8 | 176.3 | 0.0298 | 49 | 1.148 | 0.0417 | 0.002 | 6/8 | -0.032 | -0.0837 | 13/22 | 0.578 | 0.079 | no |
| connors_rsi2 | SP500.vx | 21.8 | 4.1 | 0.1449 | 78 | 2.206 | 0.0755 | 0.2649 | 8/8 | 0.1199 | 0.1617 | 14/20 | 0.544 | -0.103 | no |
| williams_vbo | AUS200.vx | 21.8 | 181.2 | 0.0297 | 47 | 1.141 | 0.0285 | 0.0324 | 7/8 | -0.1289 | -0.1228 | 15/22 | 0.547 | 0.07 | no |
| williams_vbo | DAX40.vx | 21.8 | 166.1 | 0.029 | 50 | 1.142 | 0.0358 | 0.0134 | 7/8 | 0.0115 | 0.001 | 17/22 | 0.494 | 0.075 | no |
| crabel_nr7 | XAUUSD.vx | 21.8 | 37.6 | 0.1025 | 48 | 1.288 | 0.1085 | 0.0888 | 6/8 | 0.0882 | 0.1778 | 15/22 | 0.427 | 0.567 | no |
| connors_rsi2 | DAX40.vx | 21.8 | 3.5 | 0.135 | 77 | 2.068 | 0.0704 | 0.2696 | 6/8 | 0.1301 | 0.0788 | 12/20 | 0.336 | 0.043 | no |
| turtle55 | BTCUSD.vx | 12.1 | 7.0 | 2.1906 | 40 | 5.006 | 3.562 | 0.1738 | 7/8 | 2.1813 | 0.9937 | 8/13 | 0.04 | -0.047 | no |
| crabel_nr7 | UK100.vx | 21.8 | 38.6 | 0.0784 | 49 | 1.225 | 0.0836 | 0.0677 | 7/8 | -0.0243 | -0.1941 | 14/22 | 0.245 | 0.036 | no |
| turn_of_month | NAS100.vx | 21.8 | 12.0 | 0.0613 | 57 | 1.364 | 0.0593 | 0.0658 | 5/8 | 0.0574 | 0.0525 | 14/22 | 0.191 | 0.01 | no |
| clenow_trend | HK50.vx | 21.8 | 3.5 | 0.2696 | 47 | 1.771 | 0.2095 | 0.4409 | 6/8 | 0.2434 | 0.1756 | 13/22 | 0.129 | None | no |
| turtle55 | NAS100.vx | 21.8 | 5.2 | 0.3104 | 39 | 1.582 | 0.3254 | 0.2756 | 8/8 | 0.3049 | 0.3392 | 13/22 | 0.117 | -0.011 | no |
| clenow_trend | NAS100.vx | 21.8 | 3.7 | 0.2883 | 46 | 1.688 | 0.1635 | 0.6176 | 6/8 | 0.2846 | 0.3521 | 13/22 | 0.116 | None | no |
| connors_rsi2 | US30.vx | 21.8 | 4.6 | 0.083 | 73 | 1.514 | 0.0941 | 0.0592 | 7/8 | 0.0808 | 0.0776 | 13/20 | 0.134 | -0.211 | no |
| turtle55 | XAUUSD.vx | 21.8 | 4.6 | 0.5577 | 37 | 2.041 | 0.3776 | 1.1053 | 4/8 | 0.5536 | 0.5352 | 12/22 | 0.086 | 0.335 | no |
| tsmom12 | USDJPY.vx | 21.8 | 1.9 | 1.1657 | 42 | 3.494 | 1.1974 | 1.0975 | 4/8 | 1.1581 | 1.0078 | 9/14 | 0.006 | None | no |
| clenow_trend | JPN225.vx | 21.8 | 4.0 | 0.2699 | 40 | 1.641 | 0.1331 | 0.6149 | 5/8 | 0.265 | 0.2166 | 11/22 | 0.087 | -0.113 | no |
| turtle55 | ETHUSD.vx | 8.9 | 7.0 | 0.9194 | 44 | 2.772 | 1.066 | 0.5609 | 7/8 | 0.8879 | 1.1361 | 7/9 | 0.011 | None | no |

**Passing pairs (3):** williams_vbo|JPN225.vx, williams_vbo|US30.vx, crabel_nr7|BTCUSD.vx

Failed exactly one check:

- williams_vbo / BTCUSD.vx: broker_pos
- crabel_nr7 / US30.vx: broker_pos
- williams_vbo / XAUUSD.vx: dsr_95
- crabel_nr7 / JPN225.vx: dsr_95
- williams_vbo / NAS100.vx: dsr_95
- williams_vbo / DAX40.vx: dsr_95
- crabel_nr7 / XAUUSD.vx: dsr_95
- turtle55 / NAS100.vx: dsr_95
- connors_rsi2 / US30.vx: dsr_95
- turtle55 / JPN225.vx: dsr_95
- crabel_nr7 / XAUEUR.vx: dsr_95
- turn_of_month / DAX40.vx: dsr_95
- turtle55 / SP500.vx: dsr_95

## Post-hoc checks (run after seeing the results; NOT part of the pass rule)

**By market group** (mean R per trade after costs and swap; 2005-today, and 2019+ only):

| strategy | Crypto | Metals | Indexes | Energies | FX Majors | FX Crosses |
|---|--:|--:|--:|--:|--:|--:|
| turtle55 | +1.65 (2019+ +1.44) | +0.28 (+0.67) | +0.03 (0.00) | -0.43 | -0.12 | -0.17 |
| clenow_trend | +0.76 (+0.46) | +0.19 (+0.39) | +0.04 (+0.06) | -0.33 | -0.06 | -0.10 |
| connors_rsi2 | - | - | +0.06 (+0.07) | - | - | - |
| turn_of_month | - | - | +0.02 (+0.03) | - | - | - |
| williams_vbo | +0.17 (+0.11) | +0.01 | +0.02 (-0.01) | -0.23 | -0.43 | -0.44 |
| crabel_nr7 | +0.35 (+0.22) | +0.08 | +0.05 (+0.01) | -0.35 | -0.36 | -0.40 |

Trend works on crypto and gold, is flat on indices and loses on FX and energies -- the same split the
H1 multi-symbol scan found.  FX loses even with zero costs.

**Broker H1 replay of the 3 passing pairs** (real order of events inside the day, broker prices, ~7-8 years):

| pair | trades | mean R | t-stat | verdict |
|---|--:|--:|--:|---|
| williams_vbo / US30 | 480 | +0.053 | 2.00 | marginal -- watch only |
| williams_vbo / JPN225 | 514 | +0.017 | 0.66 | not confirmed |
| crabel_nr7 / BTCUSD | 137 | -0.044 | -0.52 | not confirmed (the Yahoo-day result does not survive broker prices) |

Conclusion: no round-3 candidate is strong enough to trade.  Forward-test (dry-run) candidates:
connors_rsi2 on indices (9/10 indices positive, 8/8 folds, broker D1 positive, corr with live -0.07, but DSR 0.71)
and turtle55 / clenow_trend on gold + BTC (large R per trade, too few trades to prove).

## Unseen-data test (2026-10-06) -- connors_rsi2 on 1985-2004

Round 3 used only 2005+.  Pre-registered: pooled mean R > 0 and t > 2 on trades entered 1985-2004, same code and costs.

| index | years | trades | mean R | win % |
|---|---|--:|--:|--:|
| SP500 | 1985-2004 | 95 | +0.183 | 80 |
| US30 | 1993-2004 | 48 | +0.182 | 81 |
| FRA40 | 1991-2004 | 51 | +0.160 | 71 |
| DAX40 | 1995-2004 | 34 | +0.115 | 76 |
| HK50 | 1990-2004 | 79 | +0.088 | 68 |
| AUS200 | 2001-2004 | 14 | +0.059 | 64 |
| NAS100 | 1986-2004 | 74 | +0.047 | 74 |
| UK100 | 1985-2004 | 76 | +0.034 | 64 |
| JPN225 | 1989-2004 | 34 | -0.088 | 59 |
| **pooled** | | **505** | **+0.097** | **72** (t = 4.60) -> **PASS** |

Caveat: Connors published in 2008, so 1985-2004 may overlap the data he designed on; the stronger evidence is that
2009-2026 (after publication) is also positive.  Historic spreads were wider than today's, so 1985-2004 costs are optimistic.
Gold trend on GC=F 2000-2004 (informative, 20 trades): turtle55 +0.35 R, clenow_trend +0.85 R.

**Combined with the live H1 momentum (2019-today, backtest, guard as live: 1 per group; RSI2 limited to 1 per region US/EU/Asia):**

| portfolio | trades/mo | CAGR | max DD | months + | prop pass (all 3) |
|---|--:|--:|--:|--:|--:|
| live only | 46.9 | 69.3 % | -11.0 % | 77 % | 98.8 % |
| live + RSI2 | 48.3 | 73.7 % | -10.4 % | 80 % | 100 % |

Daily correlation with live: -0.02.  A small but real diversifier: slightly more return, slightly less drawdown.
