# Trading plan from verified FundingPips data

30 eligible symbols (>= 10 dense H1 years, clean); ledger N = 565.  Rules pre-registered in `backtest/fp_strategy_build.py`.

## Plan (passing): nothing passed


Turtle D1 pooled over all eligible symbols: 3179 trades, mean R -0.1888, last 30 % -0.3142, WF 1/8, DSR 0.0, PASS no

## All results (sorted by mean R)

| symbol | sleeve | exit | group | trades | mean R | first 70 % | last 30 % | WF+ | 2x cost | 2nd src | DSR | failed checks |
|---|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|---|
| XAUUSD | D1 | turtle55 | Metals | 101 | 0.4777 | 0.3157 | 0.9703 | 4/8 | 0.4757 | 0.4414 | 0.029 | wf_6of8, dsr_95 |
| USDJPY | H1 | tp8 | FX | 2592 | 0.0908 | 0.0579 | 0.1661 | 7/8 | 0.0872 | 0.1435 | 0.223 | dsr_95 |
| XAUUSD | H1 | tp8 | Metals | 2612 | 0.0799 | 0.0523 | 0.1408 | 7/8 | 0.0692 | None | 0.154 | dsr_95 |
| GBPJPY | H1 | tp8 | FX | 2471 | 0.0757 | 0.0979 | 0.0254 | 6/8 | 0.0546 | -0.0437 | 0.119 | dsr_95, broker_pos |
| GBPJPY | H1 | tp6 | FX | 2835 | 0.056 | 0.0831 | -0.004 | 6/8 | 0.0351 | -0.1094 | 0.083 | last30_pos, dsr_95, broker_pos |
| USDJPY | H1 | tp6 | FX | 3026 | 0.0542 | 0.02 | 0.1321 | 6/8 | 0.0506 | 0.09 | 0.082 | dsr_95 |
| XAUUSD | H1 | tp6 | Metals | 2981 | 0.0486 | 0.0405 | 0.0663 | 7/8 | 0.038 | None | 0.058 | dsr_95 |
| AUDNZD | D1 | turtle55 | FX | 101 | 0.0474 | 0.1525 | -0.1792 | 4/8 | 0.0366 | 0.003 | 0.002 | last30_pos, wf_6of8, dsr_95 |
| EURUSD | H1 | tp8 | FX | 2563 | 0.0402 | 0.0271 | 0.07 | 5/8 | 0.0349 | 0.1211 | 0.02 | wf_6of8, dsr_95 |
| EURAUD | H1 | tp8 | FX | 2459 | 0.0361 | 0.0884 | -0.0824 | 5/8 | 0.0207 | -0.029 | 0.015 | last30_pos, wf_6of8, dsr_95, broker_pos |
| EURAUD | H1 | tp6 | FX | 2831 | 0.0296 | 0.0753 | -0.0717 | 5/8 | 0.0144 | -0.0367 | 0.014 | last30_pos, wf_6of8, dsr_95, broker_pos |
| USDJPY | H1 | tp3 | FX | 4841 | 0.0239 | 0.0018 | 0.0744 | 5/8 | 0.0203 | 0.047 | 0.04 | wf_6of8, dsr_95 |
| AUDJPY | H1 | tp6 | FX | 2909 | 0.0236 | 0.0353 | -0.002 | 6/8 | 0.0029 | -0.0318 | 0.009 | last30_pos, dsr_95, broker_pos |
| AUDJPY | H1 | tp8 | FX | 2483 | 0.0163 | 0.0427 | -0.0435 | 5/8 | -0.0045 | -0.0152 | 0.004 | last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| CHFJPY | H1 | tp6 | FX | 2765 | 0.0161 | 0.0355 | -0.0279 | 4/8 | -0.0021 | -0.0767 | 0.005 | last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| EURJPY | H1 | tp8 | FX | 2545 | 0.0156 | 0.0399 | -0.0428 | 6/8 | 0.0016 | -0.0926 | 0.004 | last30_pos, dsr_95, broker_pos |
| EURUSD | H1 | tp6 | FX | 2966 | 0.0147 | 0.009 | 0.0282 | 6/8 | 0.0095 | 0.0535 | 0.004 | dsr_95 |
| XAUUSD | H1 | tp3 | Metals | 4898 | 0.0119 | 0.0089 | 0.0185 | 5/8 | 0.0011 | None | 0.008 | wf_6of8, dsr_95 |
| CHFJPY | H1 | tp8 | FX | 2423 | 0.0115 | 0.0325 | -0.0348 | 4/8 | -0.0068 | -0.0647 | 0.003 | last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| XAGUSD | D1 | turtle55 | Metals | 97 | 0.0094 | 0.1375 | -0.2391 | 4/8 | -0.0145 | 0.206 | 0.001 | trades_100, last30_pos, wf_6of8, cost2x_pos, dsr_95 |
| GBPJPY | H1 | tp3 | FX | 4675 | 0.0064 | 0.0127 | -0.0079 | 4/8 | -0.0146 | -0.0423 | 0.003 | last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| AUDJPY | H1 | tp3 | FX | 4735 | 0.0005 | 0.0078 | -0.0155 | 5/8 | -0.0203 | -0.0335 | 0.001 | last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| GBPJPY | D1 | turtle55 | FX | 122 | -0.0003 | 0.0847 | -0.2035 | 4/8 | -0.0042 | 0.0183 | 0.001 | full_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95 |
| EURJPY | H1 | tp3 | FX | 4819 | -0.0004 | 0.0023 | -0.0068 | 4/8 | -0.0143 | -0.0702 | 0.001 | full_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| NZDJPY | H1 | tp8 | FX | 2427 | -0.0016 | 0.0445 | -0.1018 | 4/8 | -0.0267 | -0.0781 | 0.001 | full_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| NZDUSD | H1 | tp8 | FX | 2354 | -0.0045 | 0.0111 | -0.04 | 4/8 | -0.0327 | -0.0523 | 0.001 | full_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| EURAUD | H1 | tp3 | FX | 4678 | -0.0048 | 0.0244 | -0.072 | 4/8 | -0.0202 | -0.034 | 0.0 | full_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| CHFJPY | H1 | tp3 | FX | 4477 | -0.0059 | 0.0 | -0.0194 | 3/8 | -0.0241 | -0.0513 | 0.0 | full_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| CADJPY | H1 | tp6 | FX | 2970 | -0.0061 | 0.0044 | -0.03 | 3/8 | -0.0335 | -0.0845 | 0.001 | full_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| EURJPY | H1 | tp6 | FX | 3036 | -0.0078 | 0.0055 | -0.0387 | 3/8 | -0.0216 | -0.023 | 0.0 | full_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| NZDJPY | H1 | tp6 | FX | 2848 | -0.0091 | 0.0205 | -0.0757 | 3/8 | -0.0341 | -0.105 | 0.0 | full_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| NZDUSD | H1 | tp6 | FX | 2806 | -0.0109 | 0.0014 | -0.0385 | 2/8 | -0.039 | -0.0476 | 0.0 | full_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| CADJPY | H1 | tp8 | FX | 2512 | -0.0131 | 0.0082 | -0.0627 | 3/8 | -0.0406 | -0.1805 | 0.0 | full_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| GBPUSD | H1 | tp3 | FX | 4773 | -0.0134 | -0.0012 | -0.0419 | 4/8 | -0.0217 | -0.0277 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| GBPUSD | H1 | tp6 | FX | 2969 | -0.0152 | -0.0067 | -0.0345 | 3/8 | -0.0234 | -0.058 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| AUDUSD | H1 | tp6 | FX | 2859 | -0.0168 | 0.0191 | -0.0993 | 2/8 | -0.0289 | -0.1297 | 0.0 | full_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| CADJPY | H1 | tp3 | FX | 4772 | -0.0188 | -0.0191 | -0.0181 | 2/8 | -0.0463 | -0.0844 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| EURUSD | H1 | tp3 | FX | 4764 | -0.0254 | -0.0213 | -0.0347 | 4/8 | -0.0306 | -0.083 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| GBPUSD | H1 | tp8 | FX | 2566 | -0.0291 | -0.0254 | -0.0376 | 4/8 | -0.0373 | -0.009 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| NZDUSD | H1 | tp3 | FX | 4666 | -0.0296 | -0.0148 | -0.0627 | 2/8 | -0.058 | -0.0309 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| NZDJPY | H1 | tp3 | FX | 4665 | -0.0357 | -0.0194 | -0.0718 | 2/8 | -0.0608 | -0.0802 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| AUDUSD | H1 | tp3 | FX | 4719 | -0.0375 | -0.0149 | -0.0909 | 2/8 | -0.0497 | -0.1182 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| USDCAD | H1 | tp3 | FX | 4674 | -0.0388 | -0.0476 | -0.0182 | 1/8 | -0.0564 | 0.034 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95 |
| USDCAD | H1 | tp8 | FX | 2551 | -0.0396 | -0.0378 | -0.0439 | 2/8 | -0.0573 | 0.0147 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95 |
| EURCAD | H1 | tp3 | FX | 4708 | -0.0433 | -0.023 | -0.0903 | 1/8 | -0.065 | -0.1315 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| AUDUSD | H1 | tp8 | FX | 2509 | -0.0482 | -0.0217 | -0.1086 | 2/8 | -0.0604 | -0.154 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| GBPAUD | H1 | tp6 | FX | 2775 | -0.0497 | -0.0174 | -0.1173 | 1/8 | -0.0645 | -0.088 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| USDCHF | H1 | tp6 | FX | 2925 | -0.0502 | -0.0518 | -0.0465 | 3/8 | -0.0827 | 0.0116 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95 |
| GBPAUD | H1 | tp3 | FX | 4516 | -0.0509 | -0.0308 | -0.0943 | 1/8 | -0.0656 | -0.0909 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| AUDCHF | H1 | tp6 | FX | 2823 | -0.0512 | 0.0108 | -0.1887 | 2/8 | -0.0876 | -0.2202 | 0.0 | full_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| USDCAD | H1 | tp6 | FX | 2905 | -0.0517 | -0.0563 | -0.0409 | 3/8 | -0.0693 | 0.0076 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95 |
| AUDCHF | H1 | tp3 | FX | 4677 | -0.0522 | -0.0151 | -0.1356 | 2/8 | -0.0889 | -0.1856 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| USDCHF | H1 | tp3 | FX | 4724 | -0.0546 | -0.0467 | -0.0726 | 1/8 | -0.0872 | -0.0583 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| GBPAUD | H1 | tp8 | FX | 2423 | -0.0611 | -0.0238 | -0.1411 | 2/8 | -0.0757 | -0.0836 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| EURJPY | D1 | turtle55 | FX | 115 | -0.0623 | -0.0134 | -0.1697 | 3/8 | -0.0649 | -0.0841 | 0.001 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| AUDCHF | H1 | tp8 | FX | 2497 | -0.0629 | 0.0047 | -0.2045 | 2/8 | -0.0995 | -0.2489 | 0.0 | full_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| EURCAD | H1 | tp6 | FX | 2938 | -0.0644 | -0.0627 | -0.0685 | 1/8 | -0.0861 | -0.1298 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| USDCHF | H1 | tp8 | FX | 2583 | -0.0673 | -0.053 | -0.0997 | 2/8 | -0.1003 | -0.0431 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| EURGBP | H1 | tp3 | FX | 4654 | -0.0726 | -0.036 | -0.157 | 1/8 | -0.1033 | -0.2139 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| EURNZD | H1 | tp3 | FX | 4512 | -0.0754 | -0.0436 | -0.1451 | 1/8 | -0.1401 | -0.1778 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| XAGUSD | H1 | tp8 | Metals | 2612 | -0.0768 | -0.1167 | 0.0173 | 1/8 | -0.1956 | 0.0162 | 0.0 | full_pos, first70_pos, wf_6of8, cost2x_pos, dsr_95 |
| EURGBP | H1 | tp6 | FX | 2938 | -0.0857 | -0.0258 | -0.2266 | 3/8 | -0.1165 | -0.2398 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| GBPCHF | H1 | tp3 | FX | 4596 | -0.0871 | -0.0487 | -0.1737 | 1/8 | -0.1253 | -0.1912 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| GBPNZD | H1 | tp3 | FX | 3972 | -0.0877 | -0.0614 | -0.1418 | 0/8 | -0.1011 | -0.095 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| EURGBP | H1 | tp8 | FX | 2566 | -0.0884 | -0.0389 | -0.1958 | 3/8 | -0.1196 | -0.2396 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| CADJPY | D1 | turtle55 | FX | 113 | -0.0931 | -0.0982 | -0.0817 | 4/8 | -0.0981 | 0.019 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95 |
| EURNZD | H1 | tp6 | FX | 2780 | -0.0933 | -0.054 | -0.178 | 3/8 | -0.1576 | -0.1328 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| AUDJPY | D1 | turtle55 | FX | 127 | -0.0935 | -0.1247 | -0.0113 | 3/8 | -0.0974 | -0.0256 | 0.001 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| EURCAD | H1 | tp8 | FX | 2551 | -0.0943 | -0.0722 | -0.1443 | 1/8 | -0.1162 | -0.216 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| CHFJPY | D1 | turtle55 | FX | 120 | -0.0968 | -0.1212 | -0.0399 | 4/8 | -0.1003 | -0.0708 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| XAGUSD | H1 | tp6 | Metals | 2962 | -0.0993 | -0.1305 | -0.029 | 1/8 | -0.2171 | -0.0202 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| XAGUSD | H1 | tp3 | Metals | 4727 | -0.0996 | -0.1208 | -0.0527 | 0/8 | -0.2172 | -0.0359 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| GBPCAD | H1 | tp3 | FX | 4632 | -0.1021 | -0.098 | -0.1117 | 0/8 | -0.1337 | -0.1218 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| GBPCHF | H1 | tp6 | FX | 2934 | -0.1026 | -0.0445 | -0.2315 | 2/8 | -0.141 | -0.299 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| GBPCHF | H1 | tp8 | FX | 2557 | -0.1056 | -0.022 | -0.2795 | 3/8 | -0.144 | -0.2834 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| AUDCAD | H1 | tp3 | FX | 4417 | -0.1087 | -0.087 | -0.1555 | 0/8 | -0.1346 | -0.135 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| EURNZD | H1 | tp8 | FX | 2421 | -0.115 | -0.0751 | -0.1993 | 2/8 | -0.1795 | -0.1696 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| NZDCAD | H1 | tp3 | FX | 4380 | -0.1151 | -0.0882 | -0.1712 | 0/8 | -0.1671 | -0.2162 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| AUDNZD | H1 | tp3 | FX | 4259 | -0.1218 | -0.1009 | -0.1648 | 1/8 | -0.1777 | -0.2212 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| CADCHF | H1 | tp3 | FX | 4672 | -0.1334 | -0.1002 | -0.2063 | 1/8 | -0.2005 | -0.2716 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| AUDNZD | H1 | tp6 | FX | 2646 | -0.1351 | -0.1055 | -0.1976 | 0/8 | -0.1902 | -0.2455 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| NZDCHF | H1 | tp3 | FX | 4604 | -0.1355 | -0.0935 | -0.2226 | 0/8 | -0.1967 | -0.2654 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| AUDCAD | H1 | tp8 | FX | 2422 | -0.1392 | -0.1063 | -0.2113 | 0/8 | -0.1652 | -0.2035 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| NZDUSD | D1 | turtle55 | FX | 127 | -0.1411 | 0.0641 | -0.479 | 3/8 | -0.1464 | -0.1369 | 0.0 | full_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| USDCAD | D1 | turtle55 | FX | 131 | -0.1432 | -0.0469 | -0.3624 | 2/8 | -0.1465 | -0.0791 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| GBPAUD | D1 | turtle55 | FX | 105 | -0.1452 | -0.0906 | -0.2755 | 2/8 | -0.1479 | -0.1095 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| EURCHF | H1 | tp3 | FX | 4346 | -0.1502 | -0.1726 | -0.1045 | 0/8 | -0.1887 | -0.1501 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| GBPCAD | H1 | tp6 | FX | 2948 | -0.1548 | -0.1526 | -0.1598 | 0/8 | -0.1864 | -0.1586 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| EURAUD | D1 | turtle55 | FX | 126 | -0.1557 | -0.1649 | -0.1325 | 2/8 | -0.1587 | -0.1502 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| GBPNZD | H1 | tp6 | FX | 2537 | -0.1561 | -0.1062 | -0.2549 | 0/8 | -0.1694 | -0.199 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| NZDCAD | H1 | tp6 | FX | 2831 | -0.1638 | -0.1279 | -0.2387 | 0/8 | -0.2155 | -0.288 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| CADCHF | H1 | tp6 | FX | 2907 | -0.1687 | -0.134 | -0.2447 | 1/8 | -0.2356 | -0.3379 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| AUDNZD | H1 | tp8 | FX | 2363 | -0.1724 | -0.1396 | -0.2416 | 0/8 | -0.2278 | -0.2076 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| AUDCAD | H1 | tp6 | FX | 2819 | -0.1738 | -0.1454 | -0.236 | 0/8 | -0.1996 | -0.2201 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| GBPCAD | H1 | tp8 | FX | 2597 | -0.1745 | -0.2004 | -0.1145 | 0/8 | -0.2061 | -0.1484 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| AUDUSD | D1 | turtle55 | FX | 137 | -0.1791 | -0.0758 | -0.4128 | 2/8 | -0.1815 | -0.0783 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| EURCHF | H1 | tp6 | FX | 2807 | -0.1888 | -0.2181 | -0.1274 | 0/8 | -0.2285 | -0.1594 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| NZDCHF | H1 | tp8 | FX | 2510 | -0.1889 | -0.1102 | -0.3479 | 1/8 | -0.2497 | -0.4062 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| NZDCHF | H1 | tp6 | FX | 2819 | -0.1914 | -0.1197 | -0.336 | 1/8 | -0.2521 | -0.3329 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| NZDCAD | H1 | tp8 | FX | 2441 | -0.1937 | -0.1452 | -0.2964 | 0/8 | -0.2459 | -0.2635 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| GBPNZD | H1 | tp8 | FX | 2216 | -0.1949 | -0.1581 | -0.2709 | 0/8 | -0.2082 | -0.1397 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| EURNZD | D1 | turtle55 | FX | 126 | -0.201 | -0.204 | -0.1934 | 4/8 | -0.2133 | -0.1693 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| CADCHF | H1 | tp8 | FX | 2564 | -0.2045 | -0.1574 | -0.3064 | 1/8 | -0.272 | -0.3517 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| GBPUSD | D1 | turtle55 | FX | 125 | -0.2058 | -0.1899 | -0.2467 | 2/8 | -0.2073 | -0.1455 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| EURCHF | H1 | tp8 | FX | 2479 | -0.2074 | -0.2459 | -0.1221 | 0/8 | -0.2479 | -0.1733 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| NZDJPY | D1 | turtle55 | FX | 116 | -0.2194 | -0.0545 | -0.5327 | 3/8 | -0.224 | -0.1396 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| AUDCHF | D1 | turtle55 | FX | 127 | -0.2494 | -0.312 | -0.0783 | 2/8 | -0.2564 | -0.2153 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| EURCAD | D1 | turtle55 | FX | 121 | -0.2516 | -0.1974 | -0.3657 | 1/8 | -0.2556 | -0.1788 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| AUDCAD | D1 | turtle55 | FX | 125 | -0.2851 | -0.2405 | -0.3834 | 1/8 | -0.2901 | -0.2316 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| GBPCHF | D1 | turtle55 | FX | 121 | -0.2971 | -0.1781 | -0.5473 | 2/8 | -0.3045 | -0.3304 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| NZDCAD | D1 | turtle55 | FX | 106 | -0.3194 | -0.1937 | -0.5636 | 1/8 | -0.3292 | -0.1238 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| GBPNZD | D1 | turtle55 | FX | 87 | -0.3232 | -0.2349 | -0.5093 | 2/8 | -0.3259 | -0.2976 | 0.0 | trades_100, full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| EURGBP | D1 | turtle55 | FX | 128 | -0.3261 | -0.1245 | -0.711 | 1/8 | -0.3319 | -0.3388 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| EURCHF | D1 | turtle55 | FX | 116 | -0.3903 | -0.2786 | -0.5871 | 2/8 | -0.3993 | -0.0787 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| GBPCAD | D1 | turtle55 | FX | 117 | -0.4022 | -0.3469 | -0.5219 | 3/8 | -0.4079 | -0.3045 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| NZDCHF | D1 | turtle55 | FX | 110 | -0.4261 | -0.4107 | -0.4622 | 1/8 | -0.4374 | -0.3821 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| CADCHF | D1 | turtle55 | FX | 130 | -0.484 | -0.4998 | -0.451 | 0/8 | -0.4962 | -0.3328 | 0.0 | full_pos, first70_pos, last30_pos, wf_6of8, cost2x_pos, dsr_95, broker_pos |
| EURUSD | D1 | turtle55 | FX | 2 | None | None | None | None/8 | None | None | None | **PASS** |

Skipped (history too short or corrupt): BTCUSD (dense from 2017, suspects 320), DJI30 (dense from 2019, suspects 28), ETHUSD (dense from 2018, suspects 7), FTSE100 (dense from 2024, suspects 1), GER40 (dense from 2022, suspects 1), JP225 (dense from 2024, suspects 1), NDX100 (dense from 2019, suspects 18), SPX500 (dense from 2019, suspects 9), UKOIL (dense from 2025, suspects 1), USOIL (dense from 2025, suspects 3)
