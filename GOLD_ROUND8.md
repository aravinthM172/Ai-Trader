# Gold round 8

N for the deflated Sharpe: 2091. Look-ahead check failed for: none.

| idea | data | years | trades/yr | mean R | t | R/yr | first 70 % | last 30 % | last 2 y | slices | 2x cost | max DD (R) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pmfix_long | H1 Dukascopy | 21.5 | 258.9 | 0.0365 | 3.42 | 9.4 | 0.0387 | 0.0312 | 0.0485 | 6/8 | 0.0298 | 38.5 | works |
| asia_uptrend | H1 Dukascopy | 21.5 | 161.5 | 0.0277 | 3.46 | 4.5 | 0.0337 | 0.0136 | 0.0346 | 6/8 | 0.021 | 16.5 | works |
| ny_cont_1r | H1 Dukascopy | 21.5 | 172.2 | -0.0352 | -3.29 | -6.1 | -0.0364 | -0.0325 | -0.0477 | 0/8 | -0.0437 | 133.1 | - |
| ny_cont_run | H1 Dukascopy | 21.5 | 201.0 | -0.0098 | -0.81 | -2.0 | -0.007 | -0.0162 | -0.0336 | 5/8 | -0.0179 | 58.0 | - |
| round50_break | H1 Dukascopy | 21.5 | 73.9 | 0.0106 | 0.25 | 0.8 | -0.0279 | 0.0606 | 0.1433 | 3/8 | 0.0019 | 94.9 | - |
| round50_fade | H1 Dukascopy | 21.5 | 29.2 | -0.0391 | -0.83 | -1.1 | -0.0025 | -0.0862 | -0.1227 | 3/8 | -0.052 | 33.6 | - |
| round100_break | H1 Dukascopy | 21.5 | 44.3 | 0.0207 | 0.38 | 0.9 | 0.0329 | 0.0073 | 0.1065 | 5/8 | 0.0123 | 62.6 | - |
| round100_fade | H1 Dukascopy | 21.5 | 15.6 | -0.0118 | -0.18 | -0.2 | -0.0748 | 0.0645 | 0.0403 | 3/8 | -0.0249 | 15.8 | - |
| gap_fade | H1 Dukascopy | 21.5 | 2.7 | 0.0732 | 0.58 | 0.2 | 0.095 | 0.0424 | 0.1321 | 6/8 | 0.0647 | 4.7 | - |
| gap_follow | H1 Dukascopy | 21.5 | 3.8 | -0.1418 | -1.42 | -0.5 | -0.1816 | -0.081 | -0.1939 | 2/8 | -0.1502 | 13.7 | - |
| silver_lead | H1 FundingPips | 17.1 | 48.9 | -0.0219 | -0.65 | -1.1 | -0.0226 | -0.0205 | -0.0092 | 2/8 | -0.0332 | 42.1 | - |
| eur_lead | H1 FundingPips | 17.1 | 118.6 | 0.0409 | 1.89 | 4.8 | 0.0507 | 0.0134 | 0.0114 | 4/8 | 0.0295 | 82.0 | weak |
| h4_live | H4 FundingPips | 22.3 | 40.6 | 0.1261 | 2.15 | 5.1 | 0.0831 | 0.2325 | 0.3897 | 7/8 | 0.1209 | 43.2 | weak |
| h4_live_up | H4 FundingPips | 22.3 | 19.8 | 0.2364 | 2.72 | 4.7 | 0.1869 | 0.3378 | 0.6525 | 5/8 | 0.231 | 36.5 | weak |
| nfp_follow | H1 Dukascopy | 21.5 | 9.0 | -0.0195 | -0.24 | -0.2 | -0.0172 | -0.0255 | 0.0216 | 4/8 | -0.0292 | 15.6 | - |
| fomc_follow | H1 Dukascopy | 21.5 | 6.2 | 0.6463 | 3.65 | 4.0 | 0.7641 | 0.3998 | 0.0435 | 8/8 | 0.6375 | 5.6 | works |
| tnx_fade | D1 futures | 26.1 | 208.6 | -0.0016 | -0.41 | -0.3 | 0.0009 | -0.0062 | -0.017 | 3/8 | -0.003 | 23.6 | - |
| silver_fade | D1 futures | 26.1 | 211.2 | 0.0045 | 1.08 | 0.9 | 0.0061 | 0.0014 | 0.0012 | 4/8 | 0.0031 | 18.7 | - |
| gold_reversal | D1 futures | 26.1 | 210.6 | 0.0028 | 0.68 | 0.6 | 0.0058 | -0.0028 | 0.0066 | 4/8 | 0.0014 | 14.8 | - |
| january_long | D1 futures | 26.1 | 15.6 | 0.044 | 3.11 | 0.7 | 0.0417 | 0.0478 | 0.1059 | 7/8 | 0.0426 | 3.6 | works |
| friday_long | D1 futures | 26.1 | 42.0 | 0.0316 | 3.3 | 1.3 | 0.0386 | 0.0185 | 0.0023 | 8/8 | 0.0302 | 5.1 | works |
| scalp_m5 | M5 FundingPips | 1.4 | 659.6 | -0.0334 | -0.74 | -22.0 | -0.0974 | 0.1181 | -0.0334 | 3/8 | -0.0657 | 92.4 | - |
| scalp_m15 | M15 FundingPips | 4.2 | 305.4 | -0.0629 | -1.63 | -19.2 | -0.0683 | -0.0458 | -0.0481 | 2/8 | -0.0919 | 113.0 | - |

## Reading the table

- **works** = positive on the whole sample, the first 70 %, the last 30 %, at double cost, in >= 6 of 8 slices, t >= 2.75,
  >= 100 trades -- everything except the deflated Sharpe (N = 2,091 tests so far), which nothing here clears.
- R is the stop distance: 3 ATR for the session rules (about $50-60 per 0.01 lot in 2026), 2 ATR elsewhere.

## Follow-up on fomc_follow (post-hoc, for understanding only)

- Second data source, FundingPips H1 2010-: 83 trades, +0.51R, t 2.6 (Dukascopy 2005-: 133 trades, +0.65R, t 3.6).
- Sells +1.02R (t 3.5), buys +0.36R (t 1.7).  Median trade -0.04R: the profit is a few long runs (best 9.7R, 9.4R);
  without the best five, +0.40R (t 2.8).
- By period (Dukascopy): 2005-12 +0.47R, 2013-19 +1.04R, 2020-26 +0.39R; **2023-26 together about +0.4R in total** -- faded.
- Placebo: same rule the day before FOMC -0.25R (31 trades), the day after 0.00R (28), a week before +0.32R (48).
- Other exits: 6 h +0.19R, 12 h +0.28R, 48 h +0.78R, 24 h with a 6 ATR target +0.55R -- not sensitive to the exit.
- The 2 ATR stop on the announcement bar is $26-75 per 0.01 lot (2025-26): 0.5-1.5 % of a $5k account.

## Sources for the ideas

- Overnight / fix: Blose, Gondhalekar & Kort 2018 (J. Economics and Finance) -- overnight returns positive, day returns
  negative; Univ. of Zurich trading-rule study -- short London open to PM fix, long PM fix to next London open;
  Abrantes-Metz & Metz (PM fix moves), Fertig's LBMA rebuttal.
- Seasonality: Baur 2013 "autumn effect"; Potrykus & Augustynowicz 2024 (reversed into a winter effect); Kohli 2012.
- Retail methods (London / New York breakout, EMA 9 / 21 scalping, round numbers): MQL5 and TradingView material.
