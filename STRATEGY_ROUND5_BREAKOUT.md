# Round 5 follow-up: the volatility breakout (vbo_day)

```
=== 1. other data sources (same rules) ===
                          trades  per_yr  mean_R     t   win  last2y
BTCUSD Binance            1949.0   213.0  0.0876  3.23  45.0  0.0265
BTCUSD FundingPips 2018+  1758.0   201.0  0.1010  3.36  45.0  0.0724
BTCUSD main               2710.0   213.0  0.1137  4.53  44.0  0.0333
ETHUSD Binance            1954.0   214.0  0.0794  3.06  46.0  0.0903
ETHUSD main               1755.0   202.0  0.1006  3.51  47.0  0.1122
NDX100 FundingPips        1278.0   165.0  0.0580  2.16  50.0  0.0238
NDX100 main               1885.0   141.0  0.0457  1.76  47.0  0.0830
USDJPY main               2645.0   168.0  0.0517  2.05  44.0  0.1294
XAUUSD FundingPips        2632.0   157.0  0.0850  3.08  43.0  0.1212
XAUUSD main               3409.0   157.0  0.0740  3.58  45.0  0.1841

=== 2. day starts at 00:00 UTC instead of 21:00 UTC ===
        trades  per_yr  mean_R     t   win  last2y
BTCUSD  2670.0   210.0  0.1625  6.53  46.0  0.0589
ETHUSD  1791.0   206.0  0.1125  3.91  46.0  0.0484
XAUUSD  3832.0   176.0  0.0982  4.22  43.0  0.2302
USDJPY  2959.0   188.0  0.0527  2.14  43.0  0.0434
NDX100  2010.0   151.0  0.0688  2.24  45.0  0.0407

=== 3. per year (mean R, trades) ===
      BTCUSD        ETHUSD        XAUUSD       USDJPY        NDX100       
        mean  count   mean  count   mean count   mean  count   mean  count
entry                                                                     
2014   0.140  224.0    NaN    NaN  0.267   132 -0.058  169.0 -0.114  158.0
2015   0.149  202.0    NaN    NaN -0.092   152  0.057  172.0  0.104  165.0
2016   0.155  197.0    NaN    NaN  0.090   148  0.097  175.0 -0.017  150.0
2017   0.154  226.0    NaN    NaN -0.085   167  0.001  172.0  0.123  134.0
2018   0.185  199.0  0.195  185.0  0.031   155  0.038  181.0  0.048  125.0
2019   0.212  196.0  0.110  189.0  0.082   159 -0.040  161.0 -0.028  171.0
2020   0.241  206.0  0.131  214.0  0.094   162  0.119  169.0  0.136  179.0
2021   0.005  213.0  0.064  221.0 -0.054   151  0.152  179.0  0.081  142.0
2022  -0.009  210.0  0.011  229.0 -0.040   167  0.219  161.0  0.185   67.0
2023   0.118  202.0  0.064  204.0 -0.123   167 -0.011  169.0 -0.063  140.0
2024   0.123  231.0  0.095  214.0  0.002   172  0.069  161.0  0.275  130.0
2025   0.006  236.0  0.065  173.0  0.249   156  0.070  175.0  0.007  101.0
2026   0.002  168.0  0.240  126.0  0.142   135  0.090  121.0  0.125  123.0
2005     NaN    NaN    NaN    NaN  0.077   140    NaN    NaN    NaN    NaN
2006     NaN    NaN    NaN    NaN  0.039   152    NaN    NaN    NaN    NaN
2007     NaN    NaN    NaN    NaN  0.231   162    NaN    NaN    NaN    NaN
2008     NaN    NaN    NaN    NaN  0.339   167    NaN    NaN    NaN    NaN
2009     NaN    NaN    NaN    NaN  0.069   150    NaN    NaN    NaN    NaN
2010     NaN    NaN    NaN    NaN  0.192   150    NaN    NaN    NaN    NaN
2011     NaN    NaN    NaN    NaN -0.014   171 -0.124  154.0    NaN    NaN
2012     NaN    NaN    NaN    NaN  0.075   139  0.025  167.0    NaN    NaN
2013     NaN    NaN    NaN    NaN  0.116   155  0.121  159.0 -0.192  100.0

=== 4. long / short / exit reason ===
BTCUSD long 0.158 1364 | short 0.069 1346 | {'flat': {'mean': 0.463, 'count': 2074}, 'stop': {'mean': -1.027, 'count': 636}}
ETHUSD long 0.121 882 | short 0.08 873 | {'flat': {'mean': 0.436, 'count': 1348}, 'stop': {'mean': -1.015, 'count': 406}, 'time': {'mean': 1.145, 'count': 1}}
XAUUSD long 0.069 1832 | short 0.08 1577 | {'flat': {'mean': 0.496, 'count': 2455}, 'stop': {'mean': -1.013, 'count': 954}}
USDJPY long 0.13 1430 | short -0.04 1215 | {'flat': {'mean': 0.525, 'count': 1827}, 'stop': {'mean': -1.006, 'count': 818}}
NDX100 long 0.093 1003 | short -0.008 882 | {'flat': {'mean': 0.472, 'count': 1346}, 'stop': {'mean': -1.019, 'count': 539}}

=== 5. overlap with the live strategy (daily R correlation) ===
BTCUSD corr 0.412 | live mean R 0.171 trades/yr 254
ETHUSD corr 0.439 | live mean R 0.169 trades/yr 250
XAUUSD corr 0.206 | live mean R 0.152 trades/yr 94
USDJPY corr 0.187 | live mean R 0.218 trades/yr 68
NDX100 corr 0.151 | live mean R 0.113 trades/yr 74

=== 6. portfolio on a $5k account, 2018+ (risk in % of balance per trade) ===
                                            trades/yr  $ / yr  $ / yr last 2y  max DD $   pass   fail   open  median days
live now (5 markets)                            749.0  1111.0           733.0     537.0  0.922  0.000  0.078        194.0
live + breakout BTC/ETH/XAU at 0.125 %         1326.0  1409.0          1082.0     503.0  0.931  0.000  0.069        148.0
live + breakout BTC/ETH/XAU at 0.25 %          1326.0  1707.0          1431.0     619.0  0.882  0.049  0.069        107.0
breakout only BTC/ETH/XAU at 0.25 %             578.0   597.0           698.0     476.0  0.922  0.000  0.078        340.0
live + breakout BTC/ETH only at 0.125 %        1163.0  1368.0           882.0     510.0  0.931  0.000  0.069        154.0
live + breakout BTC/ETH only at 0.25 %         1163.0  1625.0          1031.0     553.0  0.931  0.000  0.069        117.0
live BTC+ETH only                               510.0   701.0           297.0     598.0  0.833  0.020  0.147        288.0
live BTC+ETH + breakout BTC/ETH/XAU 0.25 %     1088.0  1297.0           995.0     785.0  0.873  0.059  0.069        161.0

=== 7. typical stop distance of the breakout (to size the lot) ===
BTCUSD median stop distance last year: 1366.83 | price 84049.56 | contract 1.0
ETHUSD median stop distance last year: 62.06 | price 2605.08 | contract 1.0
XAUUSD median stop distance last year: 51.37 | price 4155.68 | contract 100.0
```
