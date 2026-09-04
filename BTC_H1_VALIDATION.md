# BTCUSD.vx H1 Validation

- Valetax / BTCUSD.vx **H1**, 22,729 bars, 2024-01-26 to 2026-08-30 (947 days / 2.59 years)
- Realistic costs: spread $29.76, slippage $1.0/side -> round-trip $31.76
- Round-trip cost is ~6% of median H1 ATR (vs ~72% on M5) -- the reason to test H1
- Look-ahead probe: **no look-ahead detected**

## Strategies (realistic costs)
| strategy | trades | net $ | PF | exp R | DD % | WF +/8 | OOS exp R |
|---|--:|--:|--:|--:|--:|--:|--:|
| ema_trend | 1473 | 5573.6 | 1.0917 | 0.0588 | 28.82 | 7/8 | 0.0592 |
| ema_rsi_trend | 1054 | 3617.53 | 1.0941 | 0.0606 | 23.18 | 7/8 | -0.0424 |
| donchian_breakout | 783 | 4790.44 | 1.1755 | 0.0953 | 18.97 | 7/8 | 0.0666 |
| momentum_rsi_mtf | 1243 | 60693.48 | 1.4109 | 0.2214 | 14.61 | 8/8 | 0.188 |
| mean_reversion_range | 303 | -2109.95 | 0.6934 | -0.1898 | 43.62 | 0/8 | -0.1567 |

**Best (weakest-link score): `momentum_rsi_mtf`**

## Edge authenticity (best strategy)
- Look-ahead probe: **no look-ahead detected**
- Direction split: 648 long / 595 short; long expectancy 0.2333 R, short 0.2084 R — **both directions profitable: True**
- Random-direction null (200 runs, same entry bars): observed 0.2214 R vs null -0.0339 ± 0.0329 R → **100.0th percentile**, direction carries information: **True**
- Always-long expectancy -0.0431 R, always-short -0.0871 R — **both lose on the same entries: True**
- Per calendar year: 2024 +0.253 R (434 tr), 2025 +0.161 R (486 tr), 2026 +0.269 R (323 tr) — all positive: **True**
- Exit mix (target/stop/time/eq-stop): {0: 616, 1: 626, 2: 1, 3: 0}; avg hold 14.3 bars
- **Caveat:** 2.59 years of H1 covers ONE BTC macro cycle (2024 accumulation -> 2025 bull -> 2026). The edge has NOT been observed through a prolonged bear/chop regime, and live execution/slippage on Valetax is unverified.

## Walk-forward (best)
8 folds, 8 positive, mean 0.2212 R, worst 0.0849 R, single-fold profit share 0.17, majority positive: True.

## Cost sensitivity (best)
| scenario | round-trip $ | trades | net $ | PF | exp R |
|---|--:|--:|--:|--:|--:|
| optimistic | 16.0 | 1252 | 76458.72 | 1.4587 | 0.2385 |
| realistic | 31.76 | 1243 | 60693.48 | 1.4109 | 0.2214 |
| conservative | 44.0 | 1224 | 49164.68 | 1.3825 | 0.21 |
| stressed | 68.0 | 1203 | 36938.14 | 1.3376 | 0.1895 |

## Parameter robustness
9/9 nearby configs positive (min 0.1901, mean 0.2206 R). Cluster: True; single combination only: False; **stable region: True**.

## Monte Carlo (best)
- P(negative final): 0.0
- P(>50% drawdown): 0.0
- P(ruin / equity stop): 0.0
- Expected max DD $146.38, p95 $296.7
- Worst losing streak: 11

## $100 account feasibility (H1)
- BTC $78027.13; H1 ATR now $211.8 / median $494.5 / p90 $849.8
- 0.01 lot risk: calm 5.47%, median 9.89%, p90 17.0%, broker-min stop 0.3%
- margin for 0.01 lot: $7.8; stops executable: True
- can survive worst MC streak (11 losses): False
- account for a clean 1% ATR stop: ~$989 (median vol) to ~$1700 (p90 vol)
- **Verdict:** H1 ATR is large ($494 median), so a proper 1%-risk ATR stop at 0.01 lot needs ~$989 (median vol) to ~$1699 (p90 vol). On $100, 0.01 lot risks 5.5%/9.9%/17.0% (calm/median/p90) -- above the 1% target and often above the 2% ceiling, so the sizer would SKIP most H1 setups. Margin ($7.80) is not the constraint; the stop size vs a $100 account is.
