# Strategy round 1 -- research candidates vs live momentum_rsi_mtf

Realistic costs, swap NOT included (applies to all equally). Pass rule pre-registered in `backtest/strategy_round1.py`.

| strategy | BTC trades | BTC exp R | BTC FINAL | BTC WF+ | Bitstamp unseen exp R | t | yrs + | XAU exp R | XAU WF+ | corr w/ live | PASS | USEFUL |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|:-:|
| momentum_rsi_mtf | 1243 | 0.2214 | 0.188 | 8/8 | 0.2894 | 14.31 | 10/10 | 0.1634 | 8/8 | - | live | - |
| tsmom_multi | 1260 | 0.1041 | 0.0988 | 7/8 | 0.2229 | 10.7 | 10/10 | 0.1201 | 8/8 | 0.726 | yes | no |
| consensus_2of3 | 1014 | 0.2121 | 0.1959 | 8/8 | 0.2737 | 12.16 | 10/10 | 0.1869 | 8/8 | 0.864 | yes | no |
| momentum_vol_regime | 1093 | 0.1968 | 0.1409 | 8/8 | 0.2704 | 12.8 | 10/10 | 0.1301 | 8/8 | 0.9 | yes | no |
| momentum_no_shock | 1183 | 0.2032 | 0.1749 | 8/8 | 0.2805 | 13.52 | 10/10 | 0.1184 | 7/8 | 0.961 | yes | no |

Failed checks per candidate:

- **tsmom_multi**: none
- **consensus_2of3**: none
- **momentum_vol_regime**: none
- **momentum_no_shock**: none
