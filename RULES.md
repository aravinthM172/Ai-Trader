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
- Guard: max 6 positions, 3 % total open risk, 2 per group, 4 % daily loss stop, 8 % drawdown pause.
- News: no entries 15 min before → 30 min after high-impact events in the symbol's currencies.
- Portfolio backtest (`PORTFOLIO_SIM.md`): CAGR ~49 %, max DD −15 %, 414 days longest under water,
  89 % prop-challenge pass. **Backtest numbers; at +0.10 R/trade live expect ~25–30 %/yr.**
- Watchdog monitors it and runs its own edge monitor against `reports/multi_reference_trades.csv`.

## 8. Prop-firm accounts (wired 2026-10-04, owner approved)
Active in `execution/live_multi.py` whenever `PROP_CHALLENGE` is set (logic in `risk/prop_controls.py`):
| Trigger | Action |
|---|---|
| Daily equity loss ≥ 3.5% of the day's baseline (higher of opening balance/equity; day starts 21:00 UTC) | no new entries today |
| Daily equity loss ≥ 4.25% | close all bot positions, halt until the next trading day |
| Equity ≥ 7% below the phase starting balance | no new entries |
| Equity ≥ 8.5% below the phase starting balance | close all, write KILL_SWITCH |
| Phase target reached (+8% phase 1, +5% phase 2) | no new entries (pass locked in) |
| One trade idea's floating loss ≥ 0.9% of account size | close it (stays under the 1% strike trigger) |
| A losing trade closed < 10 minutes ago | no new entry (avoids "trade idea" grouping) |
| Little daily room left | risk per trade shrinks so all open stops fit inside the remaining room |
Firm limits: 5% daily, 10% static max loss, 30-day inactivity (tracked), funded-only news window.
Reward cycle on the funded account: **Bi-weekly 80%** (the 35% consistency rule blocks Monthly in ~2 of 3 months).
