# XAUUSD.vx H1 Validation

- Valetax / XAUUSD.vx **H1**, 69,936 bars, 2015-01-02 to 2026-08-28 (11.65 years)
- Same engine / strategies / method as `BTC_H1_VALIDATION.md` -- directly comparable
- Realistic costs: spread $0.3, slippage $0.1/side -> round-trip $0.50
- Valetax H1 records spread on ~13% of bars (rest 0); floored at a realistic gold spread, bracketed by 4 scenarios -- never set to 0.
- Look-ahead probe: signal identical pre-truncation = **True**

## Strategies (realistic costs)
| strategy | trades | net $ | PF | exp R | DD % | WF +/8 | OOS exp R |
|---|--:|--:|--:|--:|--:|--:|--:|
| ema_trend | 1840 | -3500.24 | 0.8959 | -0.0684 | 70.39 | 3/8 | 0.0793 |
| ema_rsi_trend | 1438 | -3500.57 | 0.8646 | -0.088 | 70.95 | 3/8 | 0.0779 |
| donchian_breakout | 2012 | 821709.84 | 1.3271 | 0.2696 | 42.84 | 6/8 | 0.0961 |
| momentum_rsi_mtf | 3739 | 1496403.12 | 1.4088 | 0.1634 | 25.06 | 8/8 | 0.2313 |
| mean_reversion_range | 637 | -2865.91 | 0.7948 | -0.1499 | 65.26 | 1/8 | -0.1694 |

**Best (cross-regime score): `momentum_rsi_mtf`** -- max of (3*worst-calendar-year-expR + combined-expR + fraction-of-years-positive); punishes strategies that ride a single multi-year trend

## Edge authenticity (best strategy)
- Look-ahead: **no look-ahead detected**
- Direction split: 2045 long / 1694 short; long 0.2152 R, short 0.1008 R -- **both directions profitable: True**
- Random-direction null (150 runs): observed 0.1634 R vs null -0.0622 +/- 0.0395 R -> **100.0th percentile**, direction carries information: **True**
- Always-long -0.0477 R, always-short -0.1339 R -- **both lose on the same entries: True**
- Per calendar year (12y): 2015 -0.061 R, 2016 +0.095 R, 2017 +0.143 R, 2018 +0.051 R, 2019 +0.266 R, 2020 +0.368 R, 2021 +0.116 R, 2022 +0.158 R, 2023 +0.249 R, 2024 +0.162 R, 2025 +0.180 R, 2026 +0.298 R -- all positive: **False**

## Walk-forward (best)
8 folds, 8 positive, mean 0.1641 R, worst 0.0327 R, majority positive: True.

## Cost sensitivity (best)
| scenario | round-trip $ | trades | net $ | PF | exp R |
|---|--:|--:|--:|--:|--:|
| optimistic | 0.25 | 3778 | 3241087.3 | 1.4385 | 0.182 |
| realistic | 0.5 | 3739 | 1496403.12 | 1.4088 | 0.1634 |
| conservative | 0.9 | 3679 | 421561.69 | 1.3786 | 0.1339 |
| stressed | 1.8 | 3543 | 22248.25 | 1.208 | 0.0659 |

## Parameter robustness
9/9 nearby configs positive (min 0.1446, mean 0.1676 R). Single combination only: False; **stable region: True**.

## Regime breakdown (best)
| regime | trades | exp R | net $ |
|---|--:|--:|--:|
| ranging_2015_2018 | 1298 | 0.0577 | 4243.69 |
| gold_bull_2019_2023 | 1491 | 0.2301 | 228474.85 |
| recent_2024_2026 | 950 | 0.2029 | 1263684.58 |

Years positive: **11/12**; worst year exp R **-0.061**; all regimes positive: **True**.

## Monte Carlo (best, fixed-fractional on $100)
- P(negative final): 0.0
- P(>50% drawdown): 0.0002
- P(ruin / equity stop): 0.0
- Worst losing streak: 12

## $100 account feasibility (H1)
- Gold $4455.0; H1 ATR median $4.63 / p10 $2.2 / p90 $11.87
- 0.01 lot risk: calm 4.39%, median 9.25%, p90 23.74%, broker-min 0.31%
- Account for a clean 1% ATR stop: ~$925 (median vol) to ~$2374 (p90 vol)
- **Verdict:** XAU H1 ATR ~$4; 0.01 lot risks 4.4%/9.2%/23.7% of $100 (calm/median/p90). A clean 1% ATR-stop position needs ~$925-$2374. Same account constraint as BTC H1.

## Verdict
**PROMISING CROSS-REGIME H1 EDGE -- forward-test alongside BTC H1** -- edge_found = **True**

PF 1.4088, combined exp R 0.1634, OOS FINAL exp R 0.2313, WF 8/8, stressed exp R 0.0659, conservative exp R 0.1339.

Even if every research condition passes, `LIVE_TRADING` stays **false**: live execution + reconciliation is not implemented. See `tools/xau_h1_live_gate.py`.
