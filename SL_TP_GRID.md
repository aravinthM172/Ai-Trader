# Stop-loss x take-profit grid -- live H1 momentum, ~20 years of verified data

Gold 2005-01-02..2026-10-07 (choose 2005-2015, confirm 2016-), BTC 2014-01-01..2026-09-26 (choose 2014-2019, confirm 2020-).  Pre-registered in `backtest/sl_tp_grid.py`.

**Chosen on discovery years: stop / target = 1.5/8.0 ATR -- confirmation PASSED** (must beat the live 2 / 3 on both symbols and be > 0).


### XAUUSD: Discovery (choose) -- mean R

| stop \ target | 3 ATR | 4 ATR | 6 ATR | 8 ATR |
|---|--:|--:|--:|--:|
| 1.5 ATR | 0.0368 | 0.0628 | 0.0742 | **0.1287** |
| 2 ATR | 0.0325 | 0.0704 | 0.0815 | 0.143 |
| 2.5 ATR | 0.0311 | 0.0613 | 0.0794 | 0.1138 |
| 3 ATR | 0.0248 | 0.0471 | 0.0527 | 0.0962 |
| 4 ATR | 0.0183 | 0.0403 | 0.0834 | 0.0922 |

### BTCUSD: Discovery (choose) -- mean R

| stop \ target | 3 ATR | 4 ATR | 6 ATR | 8 ATR |
|---|--:|--:|--:|--:|
| 1.5 ATR | 0.0836 | 0.1452 | 0.2433 | **0.3307** |
| 2 ATR | 0.1096 | 0.1473 | 0.2466 | 0.3164 |
| 2.5 ATR | 0.1224 | 0.1581 | 0.2511 | 0.2647 |
| 3 ATR | 0.1253 | 0.1604 | 0.2226 | 0.2672 |
| 4 ATR | 0.0924 | 0.1465 | 0.1847 | 0.2278 |

### XAUUSD: Confirmation (unseen) -- mean R

| stop \ target | 3 ATR | 4 ATR | 6 ATR | 8 ATR |
|---|--:|--:|--:|--:|
| 1.5 ATR | 0.0065 | 0.0266 | 0.0318 | **0.0905** |
| 2 ATR | 0.0078 | 0.0268 | 0.0463 | 0.0808 |
| 2.5 ATR | 0.0001 | 0.0184 | 0.0301 | 0.0649 |
| 3 ATR | 0.0011 | 0.0207 | 0.0589 | 0.0752 |
| 4 ATR | 0.0144 | 0.0271 | 0.0493 | 0.0702 |

### BTCUSD: Confirmation (unseen) -- mean R

| stop \ target | 3 ATR | 4 ATR | 6 ATR | 8 ATR |
|---|--:|--:|--:|--:|
| 1.5 ATR | 0.0128 | 0.012 | 0.0793 | **0.0774** |
| 2 ATR | 0.0093 | 0.0169 | 0.091 | 0.09 |
| 2.5 ATR | 0.0001 | 0.0013 | 0.0557 | 0.0536 |
| 3 ATR | 0.0254 | 0.0214 | 0.0757 | 0.0837 |
| 4 ATR | 0.0089 | 0.0294 | 0.0442 | 0.022 |

### XAUUSD: Whole period -- R per year

| stop \ target | 3 ATR | 4 ATR | 6 ATR | 8 ATR |
|---|--:|--:|--:|--:|
| 1.5 ATR | 7.0 | 12.1 | 11.0 | **19.7** |
| 2 ATR | 5.6 | 11.0 | 11.0 | 16.8 |
| 2.5 ATR | 3.7 | 7.9 | 8.3 | 11.7 |
| 3 ATR | 2.8 | 6.0 | 7.6 | 10.2 |
| 4 ATR | 3.0 | 5.0 | 7.5 | 8.2 |

### BTCUSD: Whole period -- R per year

| stop \ target | 3 ATR | 4 ATR | 6 ATR | 8 ATR |
|---|--:|--:|--:|--:|
| 1.5 ATR | 21.0 | 28.1 | 46.7 | **51.5** |
| 2 ATR | 21.3 | 24.7 | 40.5 | 42.6 |
| 2.5 ATR | 18.4 | 20.2 | 31.5 | 28.6 |
| 3 ATR | 21.2 | 21.1 | 28.2 | 28.7 |
| 4 ATR | 12.3 | 17.5 | 18.3 | 17.2 |

### XAUUSD: Whole period -- max drawdown R

| stop \ target | 3 ATR | 4 ATR | 6 ATR | 8 ATR |
|---|--:|--:|--:|--:|
| 1.5 ATR | -89.1 | -76.3 | -97.0 | **-82.8** |
| 2 ATR | -67.6 | -58.3 | -65.4 | -59.5 |
| 2.5 ATR | -78.4 | -52.0 | -56.9 | -47.3 |
| 3 ATR | -81.3 | -55.0 | -41.4 | -42.2 |
| 4 ATR | -45.0 | -22.8 | -28.3 | -22.2 |

### BTCUSD: Whole period -- max drawdown R

| stop \ target | 3 ATR | 4 ATR | 6 ATR | 8 ATR |
|---|--:|--:|--:|--:|
| 1.5 ATR | -86.2 | -92.2 | -54.0 | **-75.5** |
| 2 ATR | -69.3 | -52.2 | -44.2 | -51.4 |
| 2.5 ATR | -59.3 | -49.1 | -54.9 | -45.0 |
| 3 ATR | -31.7 | -29.0 | -26.8 | -28.7 |
| 4 ATR | -24.5 | -19.6 | -20.7 | -30.7 |

### XAUUSD: Whole period -- worst losing streak

| stop \ target | 3 ATR | 4 ATR | 6 ATR | 8 ATR |
|---|--:|--:|--:|--:|
| 1.5 ATR | 16 | 17 | 32 | **33** |
| 2 ATR | 11 | 14 | 18 | 22 |
| 2.5 ATR | 9 | 12 | 17 | 17 |
| 3 ATR | 10 | 10 | 14 | 14 |
| 4 ATR | 8 | 9 | 9 | 11 |

### BTCUSD: Whole period -- worst losing streak

| stop \ target | 3 ATR | 4 ATR | 6 ATR | 8 ATR |
|---|--:|--:|--:|--:|
| 1.5 ATR | 16 | 18 | 21 | **24** |
| 2 ATR | 12 | 12 | 16 | 18 |
| 2.5 ATR | 12 | 14 | 13 | 17 |
| 3 ATR | 10 | 11 | 13 | 16 |
| 4 ATR | 9 | 10 | 9 | 10 |
