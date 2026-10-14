# FundedNext: VPS + automated trading permission - evidence

Collected 2026-10-14 ~08:30 UTC from FundedNext's own systems (FundedNext MCP, signed in as the account owner).

## 1. The account itself (FundedNext dashboard data)
- Account: login **14329885**, internal id 4478796, "Stellar 2-Step Challenge P1 | 15K", MT5, created 2026-10-13 22:36.
- `get_account_applicable_rules` returned, among others:
  - `VPS` - "Addons (VPS)" = **1**, `is_account_override: true`  <- the VPS add-on is recorded on this account
  - `ANT` - "News Trading Allowed (add-on)" = 1
  - `WH` - Weekend Holding = 1
  - DLL 5 %, MLL 10 %, PT 8 %, MTD 5

## 2. Published policy (help centre, quoted short)
- Stellar 2-Step Challenge, section "EAs and Indicators" - EAs and custom indicators are allowed on account sizes
  below $50,000 when their settings fit the account's parameters.
  https://help.fundednext.com/en/articles/17255124-stellar-2-step-challenge
- "Can I use a VPN or VPS?" - VPS use is allowed; prohibited only for "Account Management" services or "Group
  Trading"; flagged VPS users may be asked for information about their VPS use.
  https://help.fundednext.com/en/articles/8223809-can-i-use-a-vpn-or-vps

## 3. How this bot fits
- Own code, own account, one account only (no copy / group trading, no account management).
- Runs on the trader's own Windows VPS (Ezerhost), dedicated IP, MT5 + Python MetaTrader5 API.
- 1-hour candles, a few trades a week, 2 ATR stop / 6 ATR target, max 96 h hold (no HFT, no tick scalping,
  no trades under 30 s, no latency or news arbitrage). Risk 0.28 % per trade, notional cap 60 % on crypto.

## 4. Open point
The help centre says "EA"; this bot uses MT5's official Python API rather than an MQL5 EA. Written confirmation
from support that this counts the same is still worth having -> message below. Save their reply here.

## Message to support (send from the dashboard live chat, logged in)
> Hello, my Stellar 2-Step $15K challenge (MT5 login 14329885) has the VPS add-on. I trade it with my own
> automated system: Python using MetaTrader 5's official Python API (not a purchased or third-party EA), running
> on my own Windows VPS with a dedicated IP. It trades only this one account, on 1-hour candles, a few trades a
> week, every trade with a stop-loss and take-profit, held from hours to 4 days. There is no copy trading, group
> trading, account management, HFT, tick scalping or news/latency arbitrage. Can you please confirm in writing
> that this is allowed on this account in the challenge and the funded stage, and tell me if any setting must
> change? Thank you.

## Reply
(paste FundedNext's answer here with date, agent name and ticket / chat id)
