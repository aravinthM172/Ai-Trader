# BTC H1 go-live runbook

Live: **BTCUSD.vx, H1, `momentum_rsi_mtf` (frozen)**, 0.01 lot, broker-side SL (2 ATR) / TP (3 ATR),
96 h time exit, one position at a time.  Code: `execution/live.py`, `run_live.py`, `start_live.bat`.
**XAU is NOT live**: its 4-week forward test failed (31 trades, -0.14 R). It stays on paper.

## Evidence (2026-09-26)
| Test | Trades | Avg R/trade | PF |
|---|---|---|---|
| Valetax backtest 2024-01 → 2026-08 | 1,257 | +0.22 | 1.43 |
| Bitstamp 2014-2023, never seen (`backtest/btc_longhistory_validation.py`) | 4,330 | +0.29 | 1.58 |
| **4-week forward, real Valetax feed** | **39** | **+0.30** | **1.51** |

Expect losing streaks of 6-8 trades. The edge has been getting smaller over the years (+0.41 R in 2014, +0.13 R in 2025).

## Risk at 0.01 lot (the minimum)
About $6-13 per trade. **$300 → ~2-4.4 % per trade, $500 → ~1.2-2.6 %.**
With the balance still at $100 the 5 % cap (`BTC_MAX_RISK_PER_TRADE=0.05`) blocks every trade.

## Before Monday
1. Deposit to **$300-500**.
2. MT5: stay logged in to ValetaxIntl-Live2 and turn the **Algo Trading** button **ON** (it is off now).
3. PC: plugged in. `run_live.py` blocks idle sleep while it runs. Pause Windows Update restarts for the week.
4. Sunday: leave the dry run going. In `logs/btc_live.log`, look for `DRY RUN -- would BUY/SELL ...`
   near the top of each hour when the strategy fires. Each one is an order it would have sent.

## Monday: switch on
1. Stop the dry-run loop.
2. In `.env`, set `LIVE_TRADING=true`.
3. Double-click `start_live.bat`. It logs to `logs/live_console.log` and restarts itself if Python crashes.
4. The first line should read `LIVE_TRADING=True  KILL_SWITCH=clear`.

## Watching it
- `reports/btc_live_status.json`: mode, balance, peak, drawdown floor, open position, last decision.
- `logs/btc_live.log`: every `LIVE OPEN` / `TRADE CLOSED` / error.
- MT5 → Trade tab: positions carry magic **26092601** and comment `btc-h1-live`, with SL and TP on the broker side.
  They are protected even if the PC dies.

## Stopping
- **Emergency:** create an empty file `state\KILL_SWITCH`. No new entries after that. Open positions keep their broker SL/TP;
  close them by hand in MT5 if you want out now.
- **Automatic:** balance 35 % below its peak (`LIVE_MAX_DRAWDOWN_FRAC`) creates the kill switch itself.
  Daily realised loss over 15 % blocks entries for the rest of that UTC day.
- To resume after a kill switch: delete `state\KILL_SWITCH`, but only after reviewing why it tripped.

## Review points
- After **20 live trades**: compare with the table above. Live results below 0 R after 30+ trades → stop and investigate.
- Fills vs. plan: slippage per trade is in the `order_send` events in `state/btc_live_H1.sqlite`.
- Oracle paper replay keeps running as a shadow. Live and paper should broadly agree.

## Known limits
- If the PC is off or MT5 is disconnected at the top of the hour, that bar is skipped.
  The bot never chases price more than 15 min into a bar.
- The Valetax history CSVs label winter bars 1 h early (single-offset UTC conversion in `mt5/gateway.py`).
  Live trading is unaffected, because the offset is re-detected on every connect, but CSV exports mix seasons.

---

# Real-account checklist -- multi-symbol trader (added 2026-10-06)

The first live week (demo) lost money mostly to problems, not the strategy: DAX40 sized 11x
(fixed), MT5/PC down ~19 h with missed trades, XAUUSD + XAUEUR losing together.  The preflight
checks catch each of these. **Run it and get `READY` before switching to real money.**

    venv\Scripts\python -m tools.go_live_preflight --balance <planned deposit>

It also runs every day at 07:00 (scheduled task "GoldAI Preflight") and sends a Telegram message if a check FAILs.

1. **Account size** (retail-broker view; a FundingPips account is bought at a fixed size and $5k fits all symbols): the 0.01 minimum lot must fit under the 1 % cap.  On 2026-10-06: BTC < $1,000,
   DAX40 ~$1,420, XAUEUR ~$2,610, XAUUSD ~$2,950.  **All four need ~$3,000; use $3,500+.**
   Below that, the preflight lists which symbols would be skipped.
2. **.env**: `MULTI_MAX_PER_GROUP=1`, `MULTI_SYMBOLS` = the FundingPips names of BTCUSD, XAUUSD and GER40 (FundingPips has no XAUEUR), and `LIVE_ACCOUNT_MODE` = the type the preflight reports for the FundingPips account (prop accounts often show as DEMO).
   Restart the trader after any `.env` change. The preflight fails if the running trader is older than `.env`.
3. **Run 24/7 on the home PC** (FundingPips forbids VPN/VPS access): no trader gap over 1 h in the last 7 days. The laptop must never enter Modern Standby (plugged in, lid action "Do nothing", sleep "Never") and Windows Update must not restart it,
   with MT5 logged in and Algo Trading ON.
4. **Sizing**: every symbol shows PASS. The broker's own loss-at-stop matches the plan (x1.00).
5. **Symbols**: real accounts can use different symbol names or contract specs.  A missing symbol shows up as a sizing FAIL.
6. After switching on, read the first few `LIVE OPEN` lines in `logs/multi_live.log` against the preflight plan.
