# Multi-symbol scan -- frozen momentum_rsi_mtf on every Valetax symbol

55 symbols. Realistic broker spread, slippage 0.25 x spread, swap included. Pass rule pre-registered in `backtest/multi_symbol_scan.py`.

| symbol | group | years | trades/mo | exp R (after swap) | FINAL | WF+ | 2x spread | PF | DSR | PASS |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|:-:|
| BTCUSD.vx | Crypto | 6.1 | 18.0 | 0.2005 | 0.1725 | 8/8 | 0.1653 | 1.3803 | 1.0 | **yes** |
| XAUEUR.vx | Metals | 7.5 | 11.7 | 0.1977 | 0.1825 | 8/8 | 0.1698 | 1.3534 | 0.997 | **yes** |
| DAX40.vx | Indexes | 5.5 | 14.2 | 0.1674 | 0.0879 | 7/8 | 0.1309 | 1.2369 | 0.956 | **yes** |
| ETHUSD.vx | Crypto | 6.1 | 16.6 | 0.155 | -0.1053 | 7/8 | 0.0702 | 1.194 | 0.979 | no |
| XAUUSD.vx | Metals | 16.8 | 26.9 | 0.1425 | 0.223 | 8/8 | 0.1002 | 1.392 | 1.0 | **yes** |
| XAGUSD.vx | Metals | 7.4 | 10.9 | 0.1243 | 0.119 | 7/8 | 0.089 | 1.2263 | 0.783 | no |
| XAGEUR.vx | Metals | 8.1 | 9.3 | 0.0939 | 0.0927 | 5/8 | 0.0321 | 1.1549 | 0.414 | no |
| NAS100.vx | Indexes | 7.6 | 12.7 | 0.0895 | -0.033 | 6/8 | 0.0576 | 1.1077 | 0.509 | no |
| DOGEUSD.vx | Crypto | 4.7 | 19.0 | 0.0807 | -0.0983 | 6/8 | -0.014 | 1.0995 | 0.438 | no |
| USDJPY.vx | FX Majors | 7.4 | 11.7 | 0.0597 | 0.0122 | 4/8 | -0.0231 | 1.0781 | 0.219 | no |
| AUDJPY.vx | FX Crosses | 7.4 | 11.7 | 0.0583 | 0.0124 | 6/8 | 0.0195 | 1.0719 | 0.202 | no |
| EURAUD.vx | FX Crosses | 7.4 | 11.0 | 0.0577 | 0.0351 | 5/8 | -0.0042 | 1.088 | 0.196 | no |
| FRA40.vx | Indexes | 7.4 | 8.2 | 0.0576 | 0.0347 | 6/8 | -0.0292 | 1.0832 | 0.141 | no |
| US30.vx | Indexes | 7.6 | 12.9 | 0.0559 | 0.0242 | 4/8 | 0.0459 | 1.0654 | 0.2 | no |
| HK50.vx | Indexes | 7.6 | 9.1 | 0.0541 | 0.0193 | 7/8 | -0.0158 | 1.0827 | 0.145 | no |
| EURUSD.vx | FX Majors | 7.4 | 11.8 | 0.0537 | -0.0648 | 5/8 | 0.0033 | 1.0725 | 0.169 | no |
| JPN225.vx | Indexes | 7.4 | 11.9 | 0.0493 | 0.1428 | 6/8 | 0.033 | 1.0656 | 0.141 | no |
| CHFJPY.vx | FX Crosses | 7.4 | 11.0 | 0.0435 | 0.0121 | 5/8 | -0.036 | 1.0693 | 0.105 | no |
| EURJPY.vx | FX Crosses | 7.4 | 11.7 | 0.0416 | 0.0318 | 6/8 | 0.0161 | 1.0556 | 0.099 | no |
| CADJPY.vx | FX Crosses | 7.4 | 11.0 | 0.0407 | -0.0884 | 4/8 | -0.0506 | 1.0487 | 0.097 | no |
| GBPJPY.vx | FX Crosses | 7.4 | 11.8 | 0.0355 | 0.0203 | 5/8 | -0.0145 | 1.0413 | 0.077 | no |
| GBPAUD.vx | FX Crosses | 7.4 | 10.9 | 0.0326 | -0.0132 | 5/8 | -0.011 | 1.0394 | 0.067 | no |
| BTCEUR.vx | Crypto | 2.1 | 36.2 | 0.0244 | -0.1374 | 5/8 | -0.1294 | 1.0263 | 0.044 | no |
| XRPUSD.vx | Crypto | 6.0 | 7.6 | 0.0113 | -0.2281 | 3/8 | -0.1404 | 1.0058 | 0.018 | no |
| BCHUSD.vx | Crypto | 5.4 | 16.2 | -0.0135 | 0.3025 | 4/8 | -0.1635 | 0.9639 | 0.004 | no |
| GBPNZD.vx | FX Crosses | 7.4 | 10.9 | -0.0148 | -0.0602 | 3/8 | -0.0732 | 0.9679 | 0.004 | no |
| GBPCAD.vx | FX Crosses | 7.4 | 11.5 | -0.0242 | -0.0979 | 4/8 | -0.092 | 0.9479 | 0.002 | no |
| NZDJPY.vx | FX Crosses | 7.4 | 11.4 | -0.0254 | 0.0073 | 3/8 | -0.0674 | 0.9485 | 0.002 | no |
| XBRUSD.vx | Energies | 7.6 | 9.1 | -0.0363 | 0.1754 | 5/8 | -0.2468 | 0.9154 | 0.001 | no |
| EURCAD.vx | FX Crosses | 7.4 | 11.4 | -0.052 | -0.0748 | 3/8 | -0.107 | 0.9057 | 0.0 | no |
| GBPUSD.vx | FX Majors | 7.4 | 10.6 | -0.0533 | -0.0979 | 2/8 | -0.1713 | 0.9064 | 0.0 | no |
| XTIUSD.vx | Energies | 6.9 | 10.5 | -0.0538 | 0.0991 | 4/8 | -0.2416 | 0.8967 | 0.0 | no |
| EURNZD.vx | FX Crosses | 7.4 | 11.2 | -0.0564 | -0.1185 | 2/8 | -0.1128 | 0.9007 | 0.0 | no |
| EU50.vx | Indexes | 7.4 | 7.1 | -0.0662 | -0.2483 | 4/8 | -0.178 | 0.8955 | 0.0 | no |
| SP500.vx | Indexes | 7.5 | 11.2 | -0.0674 | -0.1733 | 1/8 | -0.1484 | 0.894 | 0.0 | no |
| AUDCHF.vx | FX Crosses | 7.4 | 10.8 | -0.0943 | -0.3197 | 3/8 | -0.1464 | 0.9025 | 0.0 | no |
| GBPCHF.vx | FX Crosses | 7.4 | 10.1 | -0.1281 | -0.0986 | 2/8 | -0.1693 | 0.8321 | 0.0 | no |
| USDCHF.vx | FX Majors | 7.4 | 10.3 | -0.1333 | -0.1378 | 1/8 | -0.2344 | 0.839 | 0.0 | no |
| NZDCHF.vx | FX Crosses | 7.4 | 9.7 | -0.134 | -0.5607 | 1/8 | -0.175 | 0.8406 | 0.0 | no |
| EURCHF.vx | FX Crosses | 7.4 | 8.4 | -0.1555 | None | 2/8 | -0.2257 | 0.8248 | 0.0 | no |
| CADCHF.vx | FX Crosses | 7.4 | 8.4 | -0.1567 | None | 1/8 | -0.224 | 0.7803 | 0.0 | no |
| AUDUSD.vx | FX Majors | 7.4 | 8.1 | -0.1698 | -1.0196 | 2/8 | -0.3142 | 0.7656 | 0.0 | no |
| NZDCAD.vx | FX Crosses | 7.4 | 7.7 | -0.1702 | None | 0/8 | -0.2715 | 0.728 | 0.0 | no |
| EURGBP.vx | FX Crosses | 7.4 | 7.5 | -0.1755 | None | 0/8 | -0.3034 | 0.7316 | 0.0 | no |
| XPTUSD.vx | Metals | 8.4 | 5.5 | -0.1828 | -0.1066 | 0/8 | -0.3603 | 0.6921 | 0.0 | no |
| NZDUSD.vx | FX Majors | 7.4 | 7.5 | -0.1829 | None | 1/8 | -0.3921 | 0.7481 | 0.0 | no |
| LTCUSD.vx | Crypto | 5.4 | 2.6 | -0.1863 | None | 0/8 | -1.0489 | 0.7263 | 0.0 | no |
| AUDCAD.vx | FX Crosses | 7.4 | 6.6 | -0.2015 | None | 0/8 | -0.2331 | 0.7085 | 0.0 | no |
| USDCAD.vx | FX Majors | 7.4 | 6.2 | -0.2164 | None | 2/8 | -0.3401 | 0.6972 | 0.0 | no |
| AUDNZD.vx | FX Crosses | 7.4 | 6.0 | -0.2199 | None | 1/8 | -0.2593 | 0.7223 | 0.0 | no |
| AUS200.vx | Indexes | 7.4 | 5.9 | -0.2228 | None | 0/8 | -0.2927 | 0.682 | 0.0 | no |
| UK100.vx | Indexes | 7.6 | 5.7 | -0.2244 | None | 2/8 | -0.3396 | 0.6668 | 0.0 | no |
| XPDUSD.vx | Metals | 3.6 | 4.8 | -0.3546 | -0.6053 | 2/8 | -0.5028 | 0.5657 | 0.0 | no |
| XNGUSD.vx | Energies | 3.6 | 1.4 | -0.3825 | None | 0/8 | None | 0.5013 | 0.0 | no |
| DXY.vx |  |  |  | too few trades |  | /8 |  |  |  | no |

**Passing symbols (4):** DAX40.vx, XAUUSD.vx, XAUEUR.vx, BTCUSD.vx
Combined: ~70.8 trades/month, average pairwise daily correlation 0.067, worst day -10.28 R at 1 R per trade.
