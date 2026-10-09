# TEST 1 -- daily EMA regime filter

| data | exit | filter | trades | mean R | total R | max DD R | first70 | last30 | folds better | per-year R>0 |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| XAUUSD duka 2005- | tp3 | none (live) | 6154 | +0.0283 | +174.4 | -64.1 | +0.0333 | +0.0174 | - | 16/22 |
| XAUUSD duka 2005- | tp3 | stack | 2202 | +0.0809 | +178.2 | -38.6 | +0.0918 | +0.0573 | 6/8 | 14/22 |
| XAUUSD duka 2005- | tp3 | side200 | 3199 | +0.0691 | +221.0 | -51.4 | +0.0804 | +0.0446 | 7/8 | 16/22 |
| XAUUSD duka 2005- | tp6 | none (live) | 3764 | +0.0721 | +271.4 | -63.3 | +0.0797 | +0.0552 | - | 17/22 |
| XAUUSD duka 2005- | tp6 | stack | 1473 | +0.1154 | +169.9 | -46.0 | +0.1110 | +0.1251 | 5/8 | 14/22 |
| XAUUSD duka 2005- | tp6 | side200 | 2103 | +0.1102 | +231.8 | -53.3 | +0.1243 | +0.0791 | 6/8 | 17/22 |
| XAUUSD FP 2010- | tp3 | none (live) | 4898 | +0.0019 | +9.2 | -67.6 | -0.0015 | +0.0093 | - | 9/17 |
| XAUUSD FP 2010- | tp3 | stack | 1930 | +0.0613 | +118.3 | -41.1 | +0.0453 | +0.0896 | 6/8 | 10/17 |
| XAUUSD FP 2010- | tp3 | side200 | 2777 | +0.0434 | +120.6 | -58.6 | +0.0218 | +0.0867 | 6/8 | 9/17 |
| XAUUSD FP 2010- | tp6 | none (live) | 2981 | +0.0387 | +115.3 | -65.4 | +0.0301 | +0.0572 | - | 11/17 |
| XAUUSD FP 2010- | tp6 | stack | 1283 | +0.0993 | +127.4 | -41.9 | +0.0675 | +0.1581 | 5/8 | 12/17 |
| XAUUSD FP 2010- | tp6 | side200 | 1818 | +0.0827 | +150.4 | -56.7 | +0.0428 | +0.1680 | 5/8 | 13/17 |
| BTCUSD bitstamp 2014- | tp3 | none (live) | 5007 | +0.0540 | +270.4 | -69.3 | +0.0877 | -0.0160 | - | 10/13 |
| BTCUSD bitstamp 2014- | tp3 | stack | 2165 | +0.0641 | +138.8 | -69.1 | +0.1075 | -0.0344 | 6/8 | 10/13 |
| BTCUSD bitstamp 2014- | tp3 | side200 | 2849 | +0.0687 | +195.6 | -52.1 | +0.1115 | -0.0184 | 5/8 | 11/13 |
| BTCUSD bitstamp 2014- | tp6 | none (live) | 3234 | +0.1592 | +514.7 | -44.2 | +0.1958 | +0.0862 | - | 12/13 |
| BTCUSD bitstamp 2014- | tp6 | stack | 1437 | +0.1821 | +261.7 | -37.7 | +0.2358 | +0.0609 | 6/8 | 11/13 |
| BTCUSD bitstamp 2014- | tp6 | side200 | 1894 | +0.1722 | +326.1 | -42.4 | +0.2393 | +0.0400 | 5/8 | 12/13 |
| GER40 duka 2014- | tp3 | none (live) | 2166 | -0.0108 | -23.4 | -88.9 | -0.0525 | +0.0226 | - | 4/8 |
| GER40 duka 2014- | tp3 | stack | 674 | +0.0018 | +1.2 | -40.6 | -0.1952 | +0.0621 | 1/8 | 4/7 |
| GER40 duka 2014- | tp3 | side200 | 909 | +0.0209 | +19.0 | -26.4 | -0.0560 | +0.0493 | 3/8 | 4/7 |
| GER40 duka 2014- | tp6 | none (live) | 1295 | -0.0112 | -14.5 | -78.8 | -0.0590 | +0.0291 | - | 4/8 |
| GER40 duka 2014- | tp6 | stack | 426 | +0.0034 | +1.5 | -34.9 | -0.1419 | +0.0510 | 2/8 | 5/7 |
| GER40 duka 2014- | tp6 | side200 | 567 | +0.0586 | +33.2 | -24.8 | +0.0190 | +0.0736 | 3/8 | 4/7 |

Pass check (pre-registered, gold Dukascopy, both exits):
- tp3 stack: PASS  {'full': np.True_, 'first70': np.True_, 'last30': np.True_, 'folds': True, 'second_src': np.True_, 'total_kept': np.True_}
- tp3 side200: PASS  {'full': np.True_, 'first70': np.True_, 'last30': np.True_, 'folds': True, 'second_src': np.True_, 'total_kept': np.True_}
- tp6 stack: fail  {'full': np.True_, 'first70': np.True_, 'last30': np.True_, 'folds': False, 'second_src': np.True_, 'total_kept': np.False_}
- tp6 side200: PASS  {'full': np.True_, 'first70': np.True_, 'last30': np.True_, 'folds': True, 'second_src': np.True_, 'total_kept': np.True_}

Gold (Dukascopy, tp3, live) trades split by the daily regime at entry:
                   count   mean      sum
reg          side                       
stacked down BUY     632 -0.025  -15.684
             SELL    691  0.041   28.050
stacked up   BUY    1716  0.105  180.545
             SELL   1406 -0.042  -58.913
tangled      BUY     897  0.057   51.428
             SELL    812 -0.014  -11.075

# TEST 2 -- sizing, XAU + BTC + GER40 portfolio, live exit tp3, 2014-2026, $10,000 account at today's prices

today's min-lot (0.01) risk as % of $10k, by symbol (median / p90):
  BTCUSD: 0.14 % / 0.30 %
  GER40: 0.38 % / 0.70 %
  XAUUSD: 0.21 % / 0.35 %

| sizing | trades taken | total R | total $ | $ per R | max DD $ | worst day $ | prop pass / fail | XAU / BTC / GER40 avg risk $ |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| live 0.25 % + min-lot allowance | 10883 | +287.3 | +3,547 | 12.3 | -4,396 | -404 | 56% / 41% | 24 / 22 / 42 |
| skip if min lot > 0.31 % | 8402 | +381.1 | +8,558 | 22.5 | -1,698 | -126 | 70% / 14% | 21 / 20 / 26 |
| equal 0.25 % (ideal) | 10949 | +274.9 | +6,872 | 25.0 | -3,560 | -202 | 61% / 32% | 25 / 25 / 25 |
| equal 0.40 % | 10949 | +274.9 | +10,995 | 40.0 | -5,697 | -324 | 48% / 48% | 40 / 40 / 40 |

Per symbol, live exit tp3, 2014-2026: mean R
        count   mean      sum
sym                          
BTCUSD   5007  0.054  270.411
GER40    2166 -0.011  -23.441
XAUUSD   3776  0.007   27.915

NOTE (2026-10-08): TEST 2 assumes a $10,000 account.  The live FundingPips account is $5,000, where the 0.01 lot of gold risks ~0.4-0.7 % at current volatility, so the 'skip if min lot > 0.31 %' rule would block nearly every gold trade.  The real-account combined test (scratchpad combined_bt.py) favoured the 1 % cap + 6 ATR target + gold side200 + no GER40.  Read TEST 2 as: GER40 at the minimum lot is what hurts.