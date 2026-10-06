# Operating rules — unattended live trading

Fixed in advance (2026-10-05) so no decision is made under drawdown stress.
Everything here is enforced by code unless marked **manual**.

## 1. What trades
- Only `momentum_rsi_mtf`, BTCUSD.vx, H1, frozen parameters (`strategy/btc_h1_signal.py`).
- XAU stays on paper (failed its 4-week forward test).
- No LLM or agent generates signals, sizes positions, or touches `execution/live.py`.
  `.claude/settings.json` blocks agent edits to the live files, `.env` and `state/`.

## 2. Costs verified
| | Spread + slippage | Swap (Valetax, 2026-10-05) | Edge after swap | Swap break-even |
|---|---|---|---|---|
| BTCUSD.vx | $31.76 round trip | −10 %/yr both sides, triple Wednesday | ≈ +0.21 R | ≈ 178 %/yr |
| XAUUSD.vx | $0.50 round trip | long −2.1 %/yr, short 0 | ≈ +0.15 R (backtest) | ≈ 38 %/yr |

Re-check with `python -m tools.fetch_swap` monthly; re-run `python -m backtest.swap_sensitivity`
if a rate moves above 30 %/yr.

## 3. Automatic stops (no human needed)
| Trigger | Action | Code |
|---|---|---|
| Balance 35 % below its peak | KILL_SWITCH | `execution/live.py` drawdown breaker |
| Daily realised loss limit | entries blocked for the day | `execution/safety.py` |
| Live mean R below the backtest's 2.5th percentile (edge cut by 0.10 R), from 20 trades on | KILL_SWITCH | `tools/edge_monitor.py` (hourly via watchdog) |
| Losing streak ≥ 15 (backtest worst 11 + 3) | KILL_SWITCH | `tools/edge_monitor.py` |
| Last 100 trades mean R ≤ 0 | KILL_SWITCH | `tools/edge_monitor.py` |
| Position without broker SL | SL re-applied, else closed | `execution/live.py` |
| Telegram `/kill` | KILL_SWITCH | `tools/watchdog.py` |

Retirement thresholds for the live mean R: −0.42 R after 20 trades, −0.26 after 40,
−0.20 after 60, −0.13 after 100, −0.05 after 200.

A KILL_SWITCH only blocks new entries; open positions keep their broker SL/TP.

## 4. Restart after a stop — **manual, on purpose**
There is no remote resume. To restart: read `reports/edge_monitor.json` and `logs/btc_live.log`,
decide, then delete `state/KILL_SWITCH` on the trading machine. An edge-monitor retirement
means the strategy goes back to research, not straight back on.

## 5. Position size (lot) by balance
Risk per trade at 0.01 lot is about $6–13 (2 ATR stop at median–p90 volatility).
| Balance | Lot | Approx. risk/trade |
|---|---|---|
| < $300 | no trading (the 5 % cap blocks every trade) | — |
| $300 – $1,299 | 0.01 | 2–4 % |
| $1,300 – $2,599 | 0.02 | 1–2 % |
| $2,600 – $3,899 | 0.03 | ≈ 1 % |
| each further $1,300 | +0.01 | ≈ 1 % |
Step up only on a new balance high; never step up during a drawdown. **Manual** until a
balance-based sizer is approved for `execution/live.py`.

## 6. Going from demo to real — **manual, once**
Requirements before setting `LIVE_ACCOUNT_MODE=real` in `.env`:
1. Watchdog running with Telegram alerts confirmed working (`/status` replies).
2. Balance ≥ $300 on the real account.
3. MT5 Algo Trading ON; PC on mains power with sleep off, or MT5 on the Oracle VM.
4. At least one week of demo dry-run/live logs with no errors.

## 7. Multi-symbol trader (`execution/live_multi.py`) — added 2026-10-04
- Same frozen rule on the symbols that passed `backtest/multi_symbol_scan.py` (55 scanned,
  pre-registered rule): **DAX40, XAUUSD, XAUEUR, BTCUSD**. BTCUSD stays with `live.py`.
  XAUUSD and XAUEUR are the same market — trade one (set `MULTI_SYMBOLS` in `.env`).
- Sends only if `MULTI_LIVE_TRADING=true` AND `LIVE_TRADING=true` AND the account type matches
  `LIVE_ACCOUNT_MODE`. Risk 0.5 % of equity per trade; skipped if the minimum lot risks > 1 %.
- Guard: max 6 positions, 3 % total open risk, 1 per group (`MULTI_MAX_PER_GROUP=1` in `.env`,
  2026-10-06), 4 % daily loss stop, 8 % drawdown pause. Every group except Metals holds one symbol,
  so this only stops XAUUSD + XAUEUR being open together. Backtest 2 → 1 per group: CAGR 49 → 37 %,
  max DD 15.2 → 14.3 %, prop pass (all 3 stages) 89 → 94 %, phase-1 median 113 → 147 days.
- News: no entries 15 min before → 30 min after high-impact events in the symbol's currencies.
- Portfolio backtest (`PORTFOLIO_SIM.md`): CAGR ~49 %, max DD −15 %, 414 days longest under water,
  89 % prop-challenge pass. **Backtest numbers; at +0.10 R/trade live expect ~25–30 %/yr.**
- Watchdog monitors it and runs its own edge monitor against `reports/multi_reference_trades.csv`.

## 8. Prop-firm accounts (wired 2026-10-04, owner approved)
Active in `execution/live_multi.py` whenever `PROP_CHALLENGE` is set (logic in `risk/prop_controls.py`):
| Trigger | Action |
|---|---|
| Daily equity loss ≥ 70% of the firm's daily limit (3% limit → 2.1%, 5% → 3.5%) of the day's baseline (higher of opening balance/equity; day starts 21:00 UTC) | no new entries today |
| Daily equity loss ≥ 85% of the limit (3% → 2.55%, 5% → 4.25%) | close all bot positions, halt until the next trading day |
| Equity ≥ 7% below the phase starting balance | no new entries |
| Equity ≥ 8.5% below the phase starting balance | close all, write KILL_SWITCH |
| Phase target reached (+8% phase 1, +5% phase 2) | no new entries (pass locked in) |
| One trade idea's floating loss ≥ 0.9% of account size | close it (stays under the 1% strike trigger) |
| A losing trade closed < 10 minutes ago | no new entry (avoids "trade idea" grouping) |
| Little daily room left | risk per trade shrinks so all open stops fit inside the remaining room |
Firm limits: 3% or 5% daily (`PROP_DAILY_LOSS_LIMIT`, default 0.03 since 2026-10-06), 10% static max loss, 30-day inactivity (tracked), funded-only news window.
Reward cycle on the funded account: **Bi-weekly 80%** (the 35% consistency rule blocks Monthly in ~2 of 3 months).

## 9. FundingPips 2-Step Standard -- official help centre, read 2026-10-06 (Chrome)
Sources: help.fundingpips.com articles "2 Step Standard" and "Trading Conduct and Security Standards".

| Rule | Official text (summary) | What it means for this bot |
|---|---|---|
| Targets / days | Phase 1 8 %, Phase 2 5 %, min 3 trading days each (none with the 3 % daily add-on) | already in challenge_tracker |
| Daily loss | 5 % (or **3 % add-on**) of the higher of opening balance / equity; floating counts; resets 00:00 UTC+3 | bot buffers scale: block 70 %, flatten 85 % of `PROP_DAILY_LOSS_LIMIT` (default 3 %) |
| Max loss | 10 % below starting size, equity or balance, any time | in prop_guard |
| Inactivity | breach after 30 days without a completed trade | tracked |
| EAs | own EA = full automation allowed **with proof of ownership** (source code, version-control history, dev environment, or explain the logic on a call) | keep git history; commit regularly |
| **VPN / VPS** | **connecting to the account through a VPN or VPS is not permitted**; IP region must stay consistent | run on the home PC only; it must never sleep |
| Forbidden | gap trading, HFT, server spamming, latency/reverse arbitrage, toxic flow, hedging, tick scalping, churning, copy trading in, third-party management | none apply |
| News (eval) | no restriction, but purposely trading news is prohibited | live news filter stays on |
| News (Master) | profits of trades opened or closed within +-5 min of red news (+-10 min of speeches) may be deducted, unless opened >= 5 h before | soft breach only |
| Holding (Master) | without the Swing add-on, every position is auto-closed daily 20:45-21:00 UTC (summer) / 21:45-22:00 UTC (winter) | backtest: mean R +0.160 -> +0.165 (29 % of trades cut, no swap) -> Swing add-on not needed |
| Instruments | 41: XAUUSD, XAGUSD, GER40, NDX100, SPX500, DJI30, JP225, FTSE100, STX50, BTCUSD, ETHUSD, oils, 28 FX. **No XAUEUR** | MULTI_SYMBOLS on FundingPips = BTCUSD, XAUUSD, GER40 (exact names from the terminal) |
| Commission | FX / metals $5 per lot, indices/energies 0, **crypto lot x price x 0.04 %** | BTC edge +0.20 R -> about +0.13-0.17 R |
| Leverage | eval: metals 1:30, indices 1:20, crypto 1:2; Master: crypto 1:1, dynamic leverage on metals/indices | margin fine at $5k |
| Lot limit | 20 lots per trade, crypto 1 lot | fine |
| Rewards | On Demand 90 % needs 35 % consistency; Monthly 100 % needs 7 days >= 0.5 % and lowers the strike trigger to 1 % | 0.9 % idea-loss close already matches |
