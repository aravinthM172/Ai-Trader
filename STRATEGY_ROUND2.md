# Strategy round 2 (classic open-source families) -- research candidates vs live momentum_rsi_mtf

Realistic costs, swap NOT included (applies to all equally). Pass rule pre-registered in `backtest/strategy_round2.py`.

| strategy | BTC trades | BTC exp R | BTC FINAL | BTC WF+ | Bitstamp unseen exp R | t | yrs + | XAU exp R | XAU WF+ | corr w/ live | PASS | USEFUL |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|:-:|
| momentum_rsi_mtf | 1243 | 0.2214 | 0.188 | 8/8 | 0.2894 | 14.31 | 10/10 | 0.1634 | 8/8 | - | live | - |
| supertrend_flip | 277 | -0.0812 | 0.1046 | 3/8 | 0.0501 | 1.16 | 6/10 | 0.1602 | 6/8 | 0.36 | no | no |
| donchian_supertrend | 601 | 0.0772 | -0.0096 | 7/8 | 0.1886 | 6.68 | 9/10 | 0.3805 | 7/8 | 0.682 | no | no |
| ema_cross_200 | 240 | -0.0492 | 0.2462 | 4/8 | 0.0721 | 1.65 | 7/10 | 0.0178 | 5/8 | 0.299 | no | no |
| macd_trend | 253 | -0.032 | -0.0233 | 3/8 | 0.0568 | 1.38 | 8/10 | -0.048 | 4/8 | 0.07 | no | no |
| squeeze_breakout | 353 | 0.0141 | -0.0539 | 5/8 | 0.1169 | 3.38 | 8/10 | 0.0516 | 7/8 | 0.432 | no | no |
| rsi2_pullback | 630 | -0.081 | -0.0667 | 1/8 | -0.0475 | -1.73 | 4/10 | -0.1351 | 1/8 | 0.144 | no | no |

Failed checks per candidate:

- **supertrend_flip**: btc_combined_pos, btc_wf_6of8, bitstamp_t_2_75
- **donchian_supertrend**: btc_final_pos
- **ema_cross_200**: btc_combined_pos, btc_wf_6of8, bitstamp_t_2_75
- **macd_trend**: btc_combined_pos, btc_final_pos, btc_wf_6of8, xau_combined_pos, xau_wf_5of8, bitstamp_t_2_75
- **squeeze_breakout**: btc_final_pos, btc_wf_6of8
- **rsi2_pullback**: btc_combined_pos, btc_final_pos, btc_wf_6of8, bitstamp_pos, bitstamp_majority_years, xau_combined_pos, xau_wf_5of8, bitstamp_t_2_75
