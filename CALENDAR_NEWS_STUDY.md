# Calendar + news study -- 20 years of daily data, every Valetax market

45 symbols, 1030 (symbol, effect) tests. Discovery 2005-2018 (|t| >= 3.0), validation 2019-01-01..today (same sign, |t| >= 2.0). Pre-registered in `backtest/calendar_news_study.py`.

**Direction candidates: 6** (about 2.8 expected by pure chance) -- **confirmed on unseen years: 1**

## Direction effects found in discovery

| symbol | effect | disc t | disc mean z | val t | val mean z | val avg return (bp) | cost (bp) | confirmed |
|---|---|--:|--:|--:|--:|--:|--:|:-:|
| GBPUSD.vx | month_Apr | 3.75 | +0.218 | 0.23 | +0.019 | +1.4 | 1.9 | no |
| GBPCAD.vx | NFP_day+0 | -3.32 | -0.328 | -2.30 | -0.260 | -10.0 | 2.2 | **yes** |
| CADJPY.vx | NFP_day+0 | 3.16 | +0.275 | -1.43 | -0.186 | -5.6 | 4.2 | no |
| GBPUSD.vx | weekday_Mon | -3.15 | -0.159 | -1.50 | -0.090 | -3.8 | 1.9 | no |
| AUDCAD.vx | month_May | -3.13 | -0.195 | -0.50 | -0.034 | -1.2 | 3.9 | no |
| XAUUSD.vx | weekday_Fri | 3.01 | +0.135 | 0.76 | +0.048 | +8.9 | 1.1 | no |

## Volatility on news days (|move| vs a normal day; 1.00 = no difference)

| group | FOMC -1 | FOMC 0 | FOMC +1 | NFP 0 | NFP +1 |
|---|--:|--:|--:|--:|--:|
| Crypto | 0.99 | 1.16 | 1.09 | 1.05 | 0.57 |
| Energies | 1.06 | 0.93 | 1.19 | 1.06 | 1.24 |
| FX Crosses | 0.87 | 0.99 | 1.12 | 1.12 | 1.23 |
| FX Majors | 0.82 | 0.93 | 1.49 | 1.14 | 1.32 |
| Indexes | 0.88 | 0.96 | 1.25 | 1.19 | 1.17 |
| Metals | 0.97 | 1.25 | 1.20 | 1.26 | 1.09 |

## Weekday / turn-of-month direction by group (validation years, mean z vs other days; t in brackets)

| group | Mon | Tue | Wed | Thu | Fri | turn of month |
|---|--:|--:|--:|--:|--:|--:|
| Crypto | +0.071 (+1.1) | -0.077 (-1.3) | +0.120 (+1.9) | -0.144 (-2.2) | -0.002 (-0.0) | +0.012 (+0.2) |
| Energies | +0.009 (+0.1) | -0.026 (-0.4) | +0.005 (+0.1) | +0.010 (+0.2) | +0.003 (+0.0) | -0.060 (-0.8) |
| FX Crosses | -0.036 (-0.6) | +0.079 (+1.5) | +0.017 (+0.3) | -0.011 (-0.2) | -0.048 (-0.8) | -0.069 (-1.1) |
| FX Majors | -0.044 (-0.7) | +0.071 (+1.4) | +0.011 (+0.2) | -0.001 (-0.0) | -0.037 (-0.6) | -0.041 (-0.6) |
| Indexes | -0.011 (-0.2) | +0.026 (+0.5) | +0.063 (+1.1) | -0.043 (-0.7) | -0.036 (-0.6) | +0.010 (+0.2) |
| Metals | -0.063 (-1.0) | -0.020 (-0.3) | +0.012 (+0.2) | +0.018 (+0.3) | +0.054 (+0.8) | +0.052 (+0.7) |
