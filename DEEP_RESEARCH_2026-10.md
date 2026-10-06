# Deep research, October 2026: all pairs, 20 years, strategies, calendar and news

Every test below fixed its pass rule **before** it was run, used broker spreads, slippage and swap,
and was checked on years it had not seen.  Every idea is counted in `research/ledger.csv`.

## 1. Data

| source | what | years |
|---|---|---|
| Yahoo Finance (`data/yahoo_D1`) | daily OHLC, 45 tradable symbols | 2005-2026 (indices back to 1985) |
| Broker MT5 (`data/mt5_D1`, `data/mt5_H1`) | daily + hourly, every Valetax symbol | gold 2004/2009-, others ~2018- |
| Federal Reserve website (`data/events_history.csv`) | 174 FOMC decision times | 2005-2026 |
| BLS rule (`news/history.py`) | 264 NFP release times | 2005-2026 (placebo-checked: 2.0x volatility on the computed hour vs 1.0x a week before/after) |
| Dukascopy (`tools/fetch_dukascopy.py`) | 20-year hourly | **incomplete** -- the server rate-limited us (HTTP 503). EURUSD 2005-2026 done; resume with `python -m tools.fetch_dukascopy` |

## 2. Strategies from the classic books (`STRATEGY_ROUND3.md`)

Seven book strategies (Turtle, Clenow, time-series momentum, Connors RSI(2), Williams, Crabel, turn of month)
on 49 symbols, 2005-2026.

- **Trend works on crypto and gold, is flat on indices, loses on FX and energies** -- even at zero cost on FX.
  Your live bot already trades exactly the markets where trend works.
- The 3 pairs that passed did not survive a replay on broker hourly prices.
- **Connors RSI(2) on indices** is the one robust new edge:
  2005-2026 positive on 9/10 indices; **unseen 1985-2004: 505 trades, +0.097 R, t = 4.6 (PASS)**.
  Added to the live bot (backtest 2019-2026): CAGR 69 -> 74 %, max DD -11.0 -> -10.4 %, prop pass 98.8 -> 100 %.
  Now in paper (dry-run) testing: `execution/paper_daily.py`.

## 3. How markets move by day, month and news (`CALENDAR_NEWS_STUDY.md`)

1,030 (symbol, effect) tests on 45 markets; found on 2005-2018, confirmed on 2019-2026.

- **Weekday / month / turn-of-month direction: noise.** 6 effects found where ~3 are expected by luck;
  1 confirmed (GBPCAD falls on NFP days -- a minor pair, likely chance).  No "buy Friday / sell in May" rule survives.
- **News days are more volatile, not directional.** Day after FOMC up to 1.5x a normal day's move (FX majors);
  NFP days 1.1-1.3x.  It changes how often stops are hit, not which way the market goes.

## 4. Your live strategy in context (`backtest/live_context_study.py`)

8,713 backtest trades of the live H1 momentum on BTC, XAUUSD, XAUEUR, DAX.

| finding | numbers | decision |
|---|---|---|
| Entries 12-16 UTC are weaker | +0.02 R (2009-21), +0.15 R (2022-26) vs +0.19 R overall | passed the statistical rule, **rejected**: still profitable, skipping them cuts CAGR 37 -> 29 % and prop pass 95 -> 90 % |
| Entries 20-24 UTC are strongest | +0.44 R / +0.32 R | not acted on -- the backtest uses median spread, real rollover spreads (21-22 UTC) are wider, so part of this may be a cost-model artefact |
| Entries within 2 h after FOMC/NFP | -0.06 R in 2022-26 (t -1.9), not in 2009-21 | did not pass; the live news filter already blocks 15 min before -> 30 min after |
| Fridays | -0.03 R in 2009-21, normal in 2022-26 | did not pass (faded) |

## 5. Bottom line

1. **Keep the live strategy and its 4 symbols unchanged.** 20 years and 49 markets confirm it trades the right markets;
   no weekday, hour or news filter makes it earn more.
2. **RSI(2) on indices is the strongest addition** -- different market, uncorrelated (-0.02), passed on unseen data.
   Paper-test it; promote after 30 paper trades with mean R > 0 and owner approval.
3. **Don't trade calendar or news direction patterns** -- they are noise after 2018.
4. **Finish the 20-year hourly download** (one command, resumable) to re-run the live strategy on 2005-2026 hourly data
   for every pair; expect it to confirm, not change, the symbol list (the daily and 7-year hourly tests already agree).
