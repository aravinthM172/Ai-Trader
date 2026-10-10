# Every strategy, market by market (2026-10-10)

H1 data, FundingPips costs and swaps, rules as in `backtest/strategy_round5.py` / `strategy_round6.py`.  R = one unit of risk (the stop distance).  Verdict rule in `backtest/strategy_by_market.py`.

## Verdict grid

| strategy | BTCUSD | ETHUSD | NDX100 | XAUUSD | GER40 | EURUSD | GBPUSD |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Live momentum (2 ATR stop, 6 ATR target) | **works** | **works** | - | **works** | - | - | - |
| Live momentum + daily EMA200 filter | **works** | **works** | - | **works** | - | - | - |
| Live momentum, 3 ATR trailing stop, no target | **works** | **works** | - | **works** | - | - | - |
| Volatility breakout, next-day exit | **works** | **works** | weak | **works** | - | - | - |
| Volatility breakout, daily trailing stop | **works** | **works** | - | weak | - | - | - |
| Breakout only after a quiet (NR7) day | weak | weak | weak | weak | - | - | - |
| Ichimoku cloud | - | - | - | **works** | - | - | - |
| Volume profile breakout | weak | - | - | weak | - | - | - |
| Volume profile fade (daily) | - | - | - | - | - | - | - |
| Volume profile fade (weekly) | - | - | - | - | - | - | - |
| Asian range breakout | - | - | - | - | - | - | - |
| Liquidity sweep reversal | - | **works** | - | - | - | - | - |
| Fair value gap retrace | - | - | - | - | - | - | - |
| Bollinger band fade | - | - | - | - | - | - | - |
| VWAP fade | - | - | - | - | - | - | - |
| Pivot point bounce | - | - | - | - | - | - | - |

## Average R per trade

| strategy | BTCUSD | ETHUSD | NDX100 | XAUUSD | GER40 | EURUSD | GBPUSD |
|---|--:|--:|--:|--:|--:|--:|--:|
| Live momentum (2 ATR stop, 6 ATR target) | 0.1709 | 0.1693 | -0.0008 | 0.0815 | 0.0025 | -0.0023 | -0.0152 |
| Live momentum + daily EMA200 filter | 0.2296 | 0.1707 | 0.0685 | 0.1523 | 0.008 | 0.0374 | 0.0181 |
| Live momentum, 3 ATR trailing stop, no target | 0.1051 | 0.139 | -0.0051 | 0.0778 | 0.0511 | 0.0025 | -0.0116 |
| Volatility breakout, next-day exit | 0.1137 | 0.1006 | 0.0457 | 0.074 | -0.0152 | -0.0306 | 0.0081 |
| Volatility breakout, daily trailing stop | 0.3687 | 0.2787 | -0.0512 | 0.1093 | -0.1338 | -0.0195 | 0.0579 |
| Breakout only after a quiet (NR7) day | 0.2367 | 0.1695 | 0.1436 | 0.1959 | -0.0476 | -0.1005 | 0.0401 |
| Ichimoku cloud | 0.0463 | 0.0219 | -0.0359 | 0.0981 | -0.0302 | -0.0557 | -0.0155 |
| Volume profile breakout | 0.0797 | -0.0185 | -0.0029 | 0.0621 | -0.0118 | -0.0172 | 0.019 |
| Volume profile fade (daily) | -0.0669 | -0.0201 | -0.0234 | -0.0495 | -0.0517 | -0.0135 | -0.0598 |
| Volume profile fade (weekly) | -0.0414 | 0.0938 | 0.0294 | -0.0406 | -0.0893 | -0.0315 | 0.0256 |
| Asian range breakout | -0.0039 | 0.0007 | -0.0715 | -0.0039 | 0.026 | 0.0225 | -0.0138 |
| Liquidity sweep reversal | -0.0211 | 0.094 | -0.0651 | -0.0076 | -0.0389 | 0.0026 | -0.0334 |
| Fair value gap retrace | -0.0626 | -0.0379 | -0.0891 | -0.0086 | -0.0319 | -0.0327 | -0.0543 |
| Bollinger band fade | -0.1482 | -0.1051 | -0.0497 | -0.0887 | -0.1065 | -0.0388 | -0.035 |
| VWAP fade | -0.1359 | -0.0999 | -0.0837 | -0.0623 | -0.0796 | -0.0327 | -0.0154 |
| Pivot point bounce | -0.0521 | -0.015 | 0.0141 | -0.007 | -0.0669 | -0.0051 | -0.0465 |

## BTCUSD

| strategy | trades/yr | mean R | t | R per year | max DD R | first 70 % | last 30 % | last 2 y | WF+ | 2x cost | verdict |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|
| Volatility breakout, daily trailing stop | 121.9 | 0.3687 | 4.45 | 44.9 | 35.0 | 0.4704 | 0.1596 | 0.2596 | 7/8 | 0.3458 | works |
| Live momentum (2 ATR stop, 6 ATR target) | 254.0 | 0.1709 | 5.52 | 43.4 | 41.0 | 0.2057 | 0.1014 | 0.0765 | 8/8 | 0.1457 | works |
| Live momentum, 3 ATR trailing stop, no target | 303.9 | 0.1051 | 4.07 | 31.9 | 33.1 | 0.12 | 0.0742 | 0.0271 | 8/8 | 0.0798 | works |
| Live momentum + daily EMA200 filter | 132.3 | 0.2296 | 5.27 | 30.4 | 36.5 | 0.3014 | 0.0773 | 0.1332 | 7/8 | 0.2038 | works |
| Volatility breakout, next-day exit | 212.8 | 0.1137 | 4.53 | 24.2 | 18.9 | 0.1398 | 0.0567 | 0.0333 | 8/8 | 0.0923 | works |
| Volume profile breakout | 194.4 | 0.0797 | 2.31 | 15.5 | 74.4 | 0.1414 | -0.0429 | -0.0092 | 5/8 | 0.0541 | weak |
| Breakout only after a quiet (NR7) day | 49.5 | 0.2367 | 3.78 | 11.7 | 22.7 | 0.3474 | -0.038 | -0.0407 | 6/8 | 0.2054 | weak |
| Ichimoku cloud | 180.3 | 0.0463 | 1.31 | 8.3 | 44.3 | 0.0734 | -0.0083 | -0.0134 | 5/8 | 0.0207 | no |
| Asian range breakout | 273.7 | -0.0039 | -0.29 | -1.1 | 96.8 | 0.0101 | -0.0343 | -0.0286 | 4/8 | -0.0267 | no |
| Volume profile fade (weekly) | 76.9 | -0.0414 | -0.79 | -3.2 | 82.2 | -0.0583 | -0.0107 | 0.1423 | 2/8 | -0.0663 | no |
| Liquidity sweep reversal | 260.6 | -0.0211 | -0.87 | -5.5 | 115.4 | -0.0408 | 0.0212 | 0.021 | 5/8 | -0.0695 | no |
| Pivot point bounce | 257.9 | -0.0521 | -2.74 | -13.4 | 174.2 | -0.0489 | -0.0589 | -0.0581 | 1/8 | -0.088 | no |
| Volume profile fade (daily) | 234.6 | -0.0669 | -3.95 | -15.7 | 200.2 | -0.0573 | -0.0876 | -0.103 | 2/8 | -0.0938 | no |
| Fair value gap retrace | 311.1 | -0.0626 | -2.79 | -19.5 | 282.2 | -0.0322 | -0.1215 | -0.1384 | 1/8 | -0.1249 | no |
| Bollinger band fade | 162.7 | -0.1482 | -7.21 | -24.1 | 306.1 | -0.1548 | -0.1355 | -0.1074 | 0/8 | -0.1795 | no |
| VWAP fade | 225.7 | -0.1359 | -7.99 | -30.7 | 390.0 | -0.1691 | -0.0855 | -0.0576 | 0/8 | -0.1609 | no |

## ETHUSD

| strategy | trades/yr | mean R | t | R per year | max DD R | first 70 % | last 30 % | last 2 y | WF+ | 2x cost | verdict |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|
| Live momentum (2 ATR stop, 6 ATR target) | 246.5 | 0.1693 | 4.42 | 41.7 | 35.9 | 0.2027 | 0.0829 | 0.0588 | 8/8 | 0.1541 | works |
| Live momentum, 3 ATR trailing stop, no target | 294.7 | 0.139 | 4.31 | 41.0 | 26.9 | 0.1415 | 0.1323 | 0.1396 | 8/8 | 0.1235 | works |
| Volatility breakout, daily trailing stop | 118.5 | 0.2787 | 3.47 | 33.0 | 39.2 | 0.2585 | 0.3323 | 0.3819 | 6/8 | 0.2659 | works |
| Liquidity sweep reversal | 242.3 | 0.094 | 3.02 | 22.8 | 37.7 | 0.0882 | 0.1096 | 0.0738 | 6/8 | 0.0649 | works |
| Live momentum + daily EMA200 filter | 118.8 | 0.1707 | 3.08 | 20.3 | 32.9 | 0.2129 | 0.0555 | 0.0167 | 7/8 | 0.1554 | works |
| Volatility breakout, next-day exit | 200.3 | 0.1006 | 3.51 | 20.1 | 18.8 | 0.0991 | 0.1045 | 0.1122 | 8/8 | 0.0886 | works |
| Breakout only after a quiet (NR7) day | 44.7 | 0.1695 | 2.11 | 7.6 | 15.0 | 0.1853 | 0.1199 | 0.1647 | 6/8 | 0.152 | weak |
| Volume profile fade (weekly) | 73.7 | 0.0938 | 1.35 | 6.9 | 27.7 | 0.1449 | -0.0155 | -0.06 | 5/8 | 0.0787 | no |
| Ichimoku cloud | 181.9 | 0.0219 | 0.51 | 4.0 | 71.9 | 0.0568 | -0.0619 | -0.1121 | 6/8 | 0.0061 | no |
| Asian range breakout | 257.0 | 0.0007 | 0.04 | 0.2 | 60.2 | 0.0055 | -0.0121 | 0.015 | 4/8 | -0.0121 | no |
| Volume profile breakout | 186.2 | -0.0185 | -0.44 | -3.4 | 111.8 | -0.0278 | 0.0063 | -0.0735 | 5/8 | -0.0341 | no |
| Pivot point bounce | 241.9 | -0.015 | -0.61 | -3.6 | 62.5 | 0.0167 | -0.0998 | -0.0782 | 4/8 | -0.0376 | no |
| Volume profile fade (daily) | 231.1 | -0.0201 | -0.94 | -4.6 | 87.0 | -0.019 | -0.023 | -0.0221 | 2/8 | -0.036 | no |
| Fair value gap retrace | 309.5 | -0.0379 | -1.28 | -11.7 | 120.7 | -0.0364 | -0.0421 | -0.0441 | 1/8 | -0.0728 | no |
| Bollinger band fade | 151.8 | -0.1051 | -4.05 | -16.0 | 139.9 | -0.1187 | -0.0669 | -0.0871 | 0/8 | -0.1228 | no |
| VWAP fade | 237.5 | -0.0999 | -4.94 | -23.7 | 222.9 | -0.1318 | -0.0212 | -0.002 | 1/8 | -0.1146 | no |

## NDX100

| strategy | trades/yr | mean R | t | R per year | max DD R | first 70 % | last 30 % | last 2 y | WF+ | 2x cost | verdict |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|
| Volatility breakout, next-day exit | 158.2 | 0.0457 | 1.76 | 7.2 | 47.1 | 0.0311 | 0.0868 | 0.083 | 6/8 | 0.0328 | weak |
| Live momentum + daily EMA200 filter | 90.5 | 0.0685 | 1.28 | 6.2 | 47.7 | 0.0711 | 0.0627 | -0.0233 | 6/8 | 0.0537 | no |
| Breakout only after a quiet (NR7) day | 26.4 | 0.1436 | 1.7 | 3.8 | 13.0 | 0.1338 | 0.1736 | 0.1402 | 6/8 | 0.1235 | weak |
| Pivot point bounce | 175.7 | 0.0141 | 0.55 | 2.5 | 47.8 | 0.0397 | -0.0534 | -0.0785 | 5/8 | -0.0061 | no |
| Volume profile fade (weekly) | 75.2 | 0.0294 | 0.57 | 2.2 | 38.2 | 0.0432 | -0.005 | -0.1323 | 5/8 | 0.0151 | no |
| Live momentum (2 ATR stop, 6 ATR target) | 175.1 | -0.0008 | -0.02 | -0.1 | 49.6 | -0.0063 | 0.0125 | -0.0043 | 3/8 | -0.0149 | no |
| Volume profile breakout | 128.9 | -0.0029 | -0.06 | -0.4 | 59.5 | -0.0094 | 0.0146 | -0.0262 | 5/8 | -0.0171 | no |
| Live momentum, 3 ATR trailing stop, no target | 202.7 | -0.0051 | -0.17 | -1.0 | 53.4 | -0.0018 | -0.0132 | -0.0529 | 4/8 | -0.0191 | no |
| Volume profile fade (daily) | 142.5 | -0.0234 | -0.95 | -3.3 | 48.9 | -0.0276 | -0.0133 | -0.0221 | 2/8 | -0.0386 | no |
| Ichimoku cloud | 130.3 | -0.0359 | -0.83 | -4.7 | 90.3 | -0.0552 | 0.0148 | -0.0329 | 3/8 | -0.0503 | no |
| Volatility breakout, daily trailing stop | 100.8 | -0.0512 | -1.11 | -5.2 | 78.0 | -0.0445 | -0.0699 | -0.1771 | 3/8 | -0.0644 | no |
| Bollinger band fade | 114.8 | -0.0497 | -1.8 | -5.7 | 106.0 | -0.027 | -0.105 | -0.0793 | 2/8 | -0.0651 | no |
| Asian range breakout | 169.6 | -0.0715 | -3.13 | -12.1 | 147.1 | -0.0693 | -0.0757 | -0.0501 | 1/8 | -0.0893 | no |
| Liquidity sweep reversal | 202.2 | -0.0651 | -2.3 | -13.2 | 177.8 | -0.0335 | -0.1437 | -0.189 | 3/8 | -0.0918 | no |
| VWAP fade | 183.6 | -0.0837 | -4.03 | -15.4 | 195.1 | -0.0976 | -0.0553 | -0.0616 | 0/8 | -0.0974 | no |
| Fair value gap retrace | 229.2 | -0.0891 | -3.23 | -20.4 | 288.8 | -0.1014 | -0.0564 | 0.0077 | 1/8 | -0.1254 | no |

## XAUUSD

| strategy | trades/yr | mean R | t | R per year | max DD R | first 70 % | last 30 % | last 2 y | WF+ | 2x cost | verdict |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|
| Live momentum, 3 ATR trailing stop, no target | 205.8 | 0.0778 | 3.39 | 16.0 | 46.4 | 0.0846 | 0.0625 | 0.1527 | 6/8 | 0.0676 | works |
| Live momentum (2 ATR stop, 6 ATR target) | 175.2 | 0.0815 | 2.91 | 14.3 | 58.7 | 0.0892 | 0.0643 | 0.1233 | 8/8 | 0.0714 | works |
| Live momentum + daily EMA200 filter | 92.4 | 0.1523 | 3.83 | 14.1 | 45.3 | 0.1563 | 0.1431 | 0.3299 | 7/8 | 0.142 | works |
| Ichimoku cloud | 133.2 | 0.0981 | 3.03 | 13.1 | 48.1 | 0.1147 | 0.061 | 0.2252 | 6/8 | 0.0878 | works |
| Volatility breakout, next-day exit | 158.7 | 0.074 | 3.58 | 11.7 | 52.6 | 0.0952 | 0.0276 | 0.1841 | 6/8 | 0.0646 | works |
| Volatility breakout, daily trailing stop | 100.8 | 0.1093 | 2.34 | 11.0 | 53.4 | 0.116 | 0.0938 | 0.3485 | 5/8 | 0.0994 | weak |
| Volume profile breakout | 129.9 | 0.0621 | 1.91 | 8.1 | 64.5 | 0.0536 | 0.0802 | 0.2257 | 6/8 | 0.0517 | weak |
| Breakout only after a quiet (NR7) day | 27.5 | 0.1959 | 2.81 | 5.4 | 30.4 | 0.1721 | 0.245 | 0.5883 | 5/8 | 0.182 | weak |
| Asian range breakout | 214.1 | -0.0039 | -0.29 | -0.8 | 80.1 | -0.0073 | 0.005 | 0.0143 | 5/8 | -0.0133 | no |
| Pivot point bounce | 179.5 | -0.007 | -0.39 | -1.3 | 80.8 | -0.0109 | 0.0013 | 0.0215 | 3/8 | -0.021 | no |
| Liquidity sweep reversal | 200.6 | -0.0076 | -0.36 | -1.5 | 73.7 | -0.0045 | -0.0146 | -0.0776 | 5/8 | -0.0273 | no |
| Fair value gap retrace | 219.1 | -0.0086 | -0.41 | -1.9 | 151.6 | -0.0105 | -0.0046 | 0.024 | 3/8 | -0.0342 | no |
| Volume profile fade (weekly) | 65.4 | -0.0406 | -1.02 | -2.7 | 95.3 | -0.0265 | -0.0727 | -0.0157 | 3/8 | -0.0509 | no |
| Volume profile fade (daily) | 155.6 | -0.0495 | -2.89 | -7.7 | 200.1 | -0.0466 | -0.0561 | 0.032 | 2/8 | -0.0603 | no |
| Bollinger band fade | 120.9 | -0.0887 | -4.82 | -10.7 | 239.4 | -0.0942 | -0.0763 | -0.115 | 1/8 | -0.0997 | no |
| VWAP fade | 191.1 | -0.0623 | -4.64 | -11.9 | 265.0 | -0.076 | -0.0279 | -0.0221 | 1/8 | -0.0718 | no |

## GER40

| strategy | trades/yr | mean R | t | R per year | max DD R | first 70 % | last 30 % | last 2 y | WF+ | 2x cost | verdict |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|
| Live momentum, 3 ATR trailing stop, no target | 212.8 | 0.0511 | 1.3 | 10.9 | 40.1 | -0.0032 | 0.0966 | 0.0365 | 3/8 | 0.0279 | no |
| Asian range breakout | 203.6 | 0.026 | 1.03 | 5.3 | 29.0 | 0.018 | 0.0315 | 0.0685 | 3/8 | 0.0027 | no |
| Live momentum + daily EMA200 filter | 87.3 | 0.008 | 0.12 | 0.7 | 52.3 | -0.116 | 0.0962 | 0.0406 | 2/8 | -0.0157 | no |
| Live momentum (2 ATR stop, 6 ATR target) | 183.5 | 0.0025 | 0.05 | 0.5 | 73.1 | -0.0468 | 0.0441 | -0.0004 | 2/8 | -0.0208 | no |
| Breakout only after a quiet (NR7) day | 27.1 | -0.0476 | -0.53 | -1.3 | 32.0 | -0.2089 | 0.0992 | 0.2446 | 2/8 | -0.0772 | no |
| Volume profile breakout | 145.5 | -0.0118 | -0.22 | -1.7 | 52.1 | -0.0344 | 0.0051 | -0.0456 | 1/8 | -0.0355 | no |
| Volatility breakout, next-day exit | 165.9 | -0.0152 | -0.52 | -2.5 | 46.1 | -0.0264 | -0.006 | 0.0531 | 1/8 | -0.0349 | no |
| Ichimoku cloud | 132.9 | -0.0302 | -0.54 | -4.0 | 79.4 | -0.029 | -0.0312 | -0.0804 | 2/8 | -0.0537 | no |
| Volume profile fade (weekly) | 81.3 | -0.0893 | -1.43 | -7.3 | 73.4 | -0.1118 | -0.0728 | -0.1687 | 1/8 | -0.1131 | no |
| Fair value gap retrace | 244.5 | -0.0319 | -0.91 | -7.8 | 70.5 | -0.0598 | -0.0122 | -0.0352 | 1/8 | -0.0926 | no |
| Liquidity sweep reversal | 212.8 | -0.0389 | -1.07 | -8.3 | 124.8 | 0.0007 | -0.067 | 0.0416 | 2/8 | -0.085 | no |
| Volume profile fade (daily) | 162.5 | -0.0517 | -1.75 | -8.4 | 66.7 | -0.0177 | -0.0749 | -0.0244 | 0/8 | -0.0768 | no |
| VWAP fade | 138.4 | -0.0796 | -2.88 | -11.0 | 88.4 | -0.0581 | -0.0982 | -0.0907 | 0/8 | -0.1024 | no |
| Bollinger band fade | 123.1 | -0.1065 | -3.13 | -13.1 | 100.3 | -0.0505 | -0.15 | -0.1166 | 1/8 | -0.1319 | no |
| Pivot point bounce | 206.0 | -0.0669 | -2.23 | -13.8 | 105.1 | -0.0376 | -0.0897 | -0.1149 | 1/8 | -0.0996 | no |
| Volatility breakout, daily trailing stop | 107.1 | -0.1338 | -2.19 | -14.3 | 110.6 | -0.1906 | -0.0822 | -0.1116 | 1/8 | -0.1543 | no |

## EURUSD

| strategy | trades/yr | mean R | t | R per year | max DD R | first 70 % | last 30 % | last 2 y | WF+ | 2x cost | verdict |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|
| Asian range breakout | 243.3 | 0.0225 | 1.41 | 5.5 | 25.7 | 0.0176 | 0.0329 | -0.0181 | 5/8 | 0.0179 | no |
| Live momentum + daily EMA200 filter | 91.1 | 0.0374 | 0.74 | 3.4 | 52.6 | 0.0397 | 0.0325 | 0.1169 | 4/8 | 0.0326 | no |
| Liquidity sweep reversal | 225.3 | 0.0026 | 0.1 | 0.6 | 86.8 | -0.0029 | 0.0147 | -0.1411 | 4/8 | -0.0068 | no |
| Live momentum, 3 ATR trailing stop, no target | 214.2 | 0.0025 | 0.09 | 0.5 | 57.1 | 0.0114 | -0.0159 | 0.0466 | 5/8 | -0.0024 | no |
| Live momentum (2 ATR stop, 6 ATR target) | 186.2 | -0.0023 | -0.07 | -0.4 | 86.2 | -0.0076 | 0.009 | 0.1021 | 4/8 | -0.0071 | no |
| Pivot point bounce | 193.3 | -0.0051 | -0.22 | -1.0 | 62.3 | -0.0033 | -0.0092 | 0.0846 | 4/8 | -0.0116 | no |
| Volatility breakout, daily trailing stop | 106.7 | -0.0195 | -0.36 | -2.1 | 93.0 | -0.018 | -0.023 | -0.0913 | 4/8 | -0.024 | no |
| Volume profile fade (daily) | 165.3 | -0.0135 | -0.6 | -2.2 | 90.2 | -0.018 | -0.004 | 0.0796 | 3/8 | -0.0186 | no |
| Breakout only after a quiet (NR7) day | 23.5 | -0.1005 | -1.31 | -2.4 | 40.0 | -0.1619 | 0.0279 | -0.6602 | 2/8 | -0.1071 | no |
| Volume profile breakout | 139.5 | -0.0172 | -0.43 | -2.4 | 59.6 | 0.0063 | -0.0699 | -0.2723 | 4/8 | -0.0221 | no |
| Volume profile fade (weekly) | 76.7 | -0.0315 | -0.64 | -2.4 | 71.6 | -0.0072 | -0.0767 | -0.2191 | 5/8 | -0.0363 | no |
| Volatility breakout, next-day exit | 167.5 | -0.0306 | -1.42 | -5.1 | 87.0 | -0.0406 | -0.0079 | -0.0703 | 2/8 | -0.0348 | no |
| Bollinger band fade | 130.6 | -0.0388 | -1.65 | -5.1 | 74.2 | -0.0338 | -0.0485 | 0.0102 | 1/8 | -0.044 | no |
| VWAP fade | 208.2 | -0.0327 | -1.98 | -6.8 | 112.6 | -0.0519 | 0.0063 | 0.1563 | 1/8 | -0.0373 | no |
| Fair value gap retrace | 216.9 | -0.0327 | -1.15 | -7.1 | 183.7 | -0.0402 | -0.0171 | 0.2727 | 3/8 | -0.0446 | no |
| Ichimoku cloud | 144.9 | -0.0557 | -1.43 | -8.1 | 119.7 | -0.0485 | -0.0706 | 0.0753 | 2/8 | -0.0606 | no |

## GBPUSD

| strategy | trades/yr | mean R | t | R per year | max DD R | first 70 % | last 30 % | last 2 y | WF+ | 2x cost | verdict |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|
| Volatility breakout, daily trailing stop | 112.6 | 0.0579 | 1.12 | 6.5 | 42.6 | 0.043 | 0.0926 | 0.2142 | 5/8 | 0.0495 | no |
| Volume profile breakout | 131.2 | 0.019 | 0.51 | 2.5 | 55.6 | 0.0298 | -0.0068 | -0.0634 | 4/8 | 0.0106 | no |
| Volume profile fade (weekly) | 75.4 | 0.0256 | 0.57 | 1.9 | 64.2 | 0.0945 | -0.1209 | 0.0514 | 4/8 | 0.0174 | no |
| Live momentum + daily EMA200 filter | 92.1 | 0.0181 | 0.4 | 1.7 | 47.9 | 0.0246 | 0.0025 | -0.0717 | 4/8 | 0.0098 | no |
| Volatility breakout, next-day exit | 164.8 | 0.0081 | 0.36 | 1.3 | 55.9 | 0.0044 | 0.0166 | 0.0882 | 4/8 | -0.0 | no |
| Breakout only after a quiet (NR7) day | 18.2 | 0.0401 | 0.37 | 0.7 | 25.5 | -0.0069 | 0.1227 | 0.0899 | 3/8 | 0.0244 | no |
| Ichimoku cloud | 140.9 | -0.0155 | -0.43 | -2.2 | 67.9 | 0.0013 | -0.0541 | -0.1488 | 5/8 | -0.0238 | no |
| Live momentum, 3 ATR trailing stop, no target | 219.8 | -0.0116 | -0.48 | -2.6 | 85.4 | 0.0037 | -0.046 | -0.0303 | 4/8 | -0.02 | no |
| Live momentum (2 ATR stop, 6 ATR target) | 188.4 | -0.0152 | -0.49 | -2.9 | 144.5 | -0.0067 | -0.0345 | -0.028 | 3/8 | -0.0234 | no |
| Asian range breakout | 245.0 | -0.0138 | -0.94 | -3.4 | 104.8 | -0.012 | -0.0181 | 0.0269 | 3/8 | -0.0222 | no |
| VWAP fade | 219.9 | -0.0154 | -1.08 | -3.4 | 101.5 | -0.0256 | 0.0089 | 0.0204 | 3/8 | -0.0234 | no |
| Bollinger band fade | 139.0 | -0.035 | -1.74 | -4.9 | 112.6 | -0.0672 | 0.0306 | 0.0418 | 3/8 | -0.0438 | no |
| Liquidity sweep reversal | 233.4 | -0.0334 | -1.44 | -7.8 | 194.8 | -0.0607 | 0.0268 | 0.054 | 3/8 | -0.0496 | no |
| Pivot point bounce | 183.1 | -0.0465 | -2.22 | -8.5 | 147.4 | -0.0411 | -0.0594 | -0.0987 | 2/8 | -0.0579 | no |
| Volume profile fade (daily) | 156.4 | -0.0598 | -2.88 | -9.4 | 151.4 | -0.045 | -0.093 | -0.141 | 1/8 | -0.0687 | no |
| Fair value gap retrace | 239.7 | -0.0543 | -2.33 | -13.0 | 218.1 | -0.0366 | -0.0948 | -0.0537 | 1/8 | -0.0752 | no |
