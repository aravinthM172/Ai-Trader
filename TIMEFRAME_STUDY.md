# Timeframe study: H1 vs M30 vs M15 (2026-10-09)

Common window: 2023-07-13 to 2026-10-09 (FundingPips MT5 candles). Live filters, costs and risk as on the VPS.
Check: generic simulator = live H1 simulation on BTCUSD: **yes**.

## Portfolio (5 live symbols, $5k account)

```
                                      trades/yr  avg R  $ per year  max DD $  prop pass  prop fail  median days
version                                                                                                        
H1 (live)                                 749.0  0.111       897.0     463.0      0.595      0.081        133.0
M30 A: same rules                        1456.0  0.067       937.0     609.0      0.676      0.216        141.0
M30 B: H1 rules checked every 30 min     1057.0  0.079       755.0     878.0      0.541      0.378        168.0
M15 A: same rules                        2825.0 -0.006      -136.0    1845.0      0.270      0.621         82.0
M15 B: H1 rules checked every 15 min     1129.0  0.085       930.0     703.0      0.622      0.270        137.0
```

## Per symbol, common window

```
                                                   from years trades trades/yr win %  avg R total R     t max DD R 1st half 2nd half last 2y years +
symbol version                                                                                                                                      
BTCUSD H1 (live)                             2023-07-13   3.2    762       236  30.6   0.07    53.5  1.09     47.5    0.148   -0.037  -0.017     3/4
ETHUSD H1 (live)                             2023-07-13   3.2    770       238  29.9  0.069    53.5  1.09     35.9    0.036     0.11   0.059     3/4
XAUUSD H1 (live)                             2023-07-18   3.2    334       104  36.5  0.337   112.4  3.35      9.5    0.394    0.275   0.287     4/4
USDJPY H1 (live)                             2023-07-17   3.2    245        76  29.8  0.083    20.2  0.76     16.0    0.081    0.085   0.043     3/4
NDX100 H1 (live)                             2023-07-13   3.2    310        96  31.9  0.095    29.4  0.96     19.9    0.159    0.017   0.035     2/4
BTCUSD M30 A: same rules                     2023-07-13   3.2   1511       467  29.6  0.026    40.0  0.58     63.5    0.015    0.043   0.002     2/4
ETHUSD M30 A: same rules                     2023-07-13   3.2   1492       461  30.3  0.077   114.3  1.67     43.0    0.034    0.137   0.066     4/4
XAUUSD M30 A: same rules                     2023-07-14   3.2    650       201  32.6  0.195   127.0  2.75     15.9    0.195    0.195   0.217     4/4
USDJPY M30 A: same rules                     2023-07-14   3.2    472       148  29.7  0.074    34.7  0.92     23.6    0.086    0.059   0.043     3/4
NDX100 M30 A: same rules                     2023-07-13   3.2    591       182  28.9  0.002     1.4  0.03     47.9     0.03   -0.032  -0.053     2/4
BTCUSD M30 B: H1 rules checked every 30 min  2023-07-13   3.2   1059       327  27.9  0.008     8.9  0.15     58.1    0.039   -0.032   -0.03     2/4
ETHUSD M30 B: H1 rules checked every 30 min  2023-07-13   3.2   1058       327  29.4  0.113   119.2  1.97     32.9    0.086    0.148   0.142     4/4
XAUUSD M30 B: H1 rules checked every 30 min  2023-07-18   3.2    519       161  30.6  0.177    92.1  2.19     26.2    0.072    0.299   0.281     4/4
USDJPY M30 B: H1 rules checked every 30 min  2023-07-17   3.2    344       107  27.3  0.078    27.0  0.82     20.4    0.131    0.016   0.021     3/4
NDX100 M30 B: H1 rules checked every 30 min  2023-07-14   3.2    444       138  27.9  0.049    21.6  0.58     25.0    0.112   -0.035  -0.007     2/4
BTCUSD M15 A: same rules                     2023-07-13   3.2   3003       927  28.7 -0.042  -126.6 -1.31    174.3   -0.062   -0.013  -0.029     1/4
ETHUSD M15 A: same rules                     2023-07-13   3.2   2980       920  28.2 -0.021   -63.1 -0.64    122.4   -0.063    0.036  -0.006     1/4
XAUUSD M15 A: same rules                     2023-07-14   3.2   1218       377  30.9  0.115   139.5  2.27     29.9    0.102    0.128   0.132     4/4
USDJPY M15 A: same rules                     2023-07-17   3.2    875       272  29.3  0.035    30.9  0.61     32.2    0.016     0.06   0.062     3/4
NDX100 M15 A: same rules                     2023-07-13   3.2   1075       332  29.1  -0.03   -32.0 -0.59     69.4    -0.04   -0.018   -0.04     2/4
BTCUSD M15 B: H1 rules checked every 15 min  2023-07-13   3.2   1144       354  28.2  0.012    13.5   0.2     54.7    0.008    0.017   -0.02     2/4
ETHUSD M15 B: H1 rules checked every 15 min  2023-07-13   3.2   1152       356  29.3  0.124   143.4  2.23     30.6    0.115    0.137   0.158     3/4
XAUUSD M15 B: H1 rules checked every 15 min  2023-07-18   3.2    549       171  30.6  0.173    95.2  2.19     26.8    0.054    0.305   0.259     3/4
USDJPY M15 B: H1 rules checked every 15 min  2023-07-17   3.2    346       109  24.9 -0.011    -3.8 -0.12     24.0   -0.028    0.015   0.052     2/4
NDX100 M15 B: H1 rules checked every 15 min  2023-07-13   3.2    467       144  30.0  0.134    62.6  1.58     20.1    0.171    0.074   0.125     4/4
```

## Per symbol, longer M30 window (5-8 years) vs H1 on the same years

```
                                                   from years trades trades/yr win %  avg R total R     t max DD R 1st half 2nd half last 2y years +
symbol version                                                                                                                                      
BTCUSD H1 (live)                             2020-09-05   6.1   1493       245  31.0  0.076   112.9  1.67     47.5    0.071    0.081  -0.017     6/7
ETHUSD H1 (live)                             2020-09-05   6.1   1502       247  31.2  0.112   167.5  2.45     35.9    0.135    0.086   0.059     7/7
XAUUSD H1 (live)                             2018-05-21   8.4    837       100  31.8  0.174   145.3   2.8     48.3     0.09    0.253   0.287     8/9
USDJPY H1 (live)                             2018-10-24   7.9    530        67  33.6  0.251   132.8  3.21     16.0    0.448    0.082   0.043     7/9
NDX100 H1 (live)                             2019-10-15   7.0    584        84  33.2  0.153    89.1  2.09     19.9    0.169    0.141   0.035     5/8
BTCUSD M30 A: same rules                     2020-09-05   6.1   3036       498  29.6  0.034   102.1  1.07     66.4    0.025    0.043   0.002     4/7
ETHUSD M30 A: same rules                     2020-09-04   6.1   3006       493  29.8  0.063   189.8  1.99     52.8    0.033    0.098   0.066     6/7
XAUUSD M30 A: same rules                     2018-12-19   7.8   1488       191  30.6  0.122   182.2  2.66     68.1    0.069    0.171   0.217     7/9
USDJPY M30 A: same rules                     2019-10-15   6.9    977       141  32.0  0.167   163.0  2.95     23.6    0.274    0.073   0.043     6/8
NDX100 M30 A: same rules                     2013-10-18  13.0   1296       100  30.5  0.063    82.1  1.32     47.9    0.044    0.068  -0.053   11/14
BTCUSD M30 B: H1 rules checked every 30 min  2020-09-05   6.1   2092       344  28.2  0.041    85.5  1.03     58.1    0.054    0.026   -0.03     6/7
ETHUSD M30 B: H1 rules checked every 30 min  2020-09-04   6.1   2120       348  29.8  0.135   285.6  3.38     38.9    0.143    0.125   0.142     7/7
XAUUSD M30 B: H1 rules checked every 30 min  2018-12-18   7.8   1167       150  28.6  0.104   121.8  1.97     73.8     0.05    0.154   0.281     7/9
USDJPY M30 B: H1 rules checked every 30 min  2019-10-23   7.0    718       103  31.1  0.236   169.7  3.43     20.4     0.41    0.081   0.021     6/8
NDX100 M30 B: H1 rules checked every 30 min  2013-10-18  13.0    974        75  29.9  0.117   114.2  2.02     25.0    0.134    0.114  -0.007    8/14
```
