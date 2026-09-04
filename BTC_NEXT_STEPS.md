# BTC (+ XAU) — Next Steps

_Updated after the M5 validation (negative) and the H1 validation (research-grade edge found
on **both** BTCUSD.vx and XAUUSD.vx)._

## XAUUSD.vx H1 — same frozen strategy, stronger cross-regime evidence

`momentum_rsi_mtf` (identical frozen params) was put through the **same** validation
machinery on 11.7 years of Valetax XAUUSD.vx H1 (`backtest/xau_h1_validation.py`,
`tools/xau_h1_live_gate.py`, `XAU_H1_VALIDATION.md`):

| | BTC H1 | **XAU H1** |
|---|---|---|
| History | 2.6 years / 1 macro regime | **11.7 years / 3 macro regimes** |
| Combined (realistic costs) | PF 1.41, +0.22 R | PF 1.41, **+0.16 R**, DD 25 % |
| Walk-forward | 8/8 folds | **8/8 folds** (weakest +0.033 R) |
| Cost sensitivity | positive through "stressed" | positive through "stressed" (**+0.066 R / PF 1.21**) |
| Parameter robustness | 9/9 | **9/9, stable_region = true** |
| Regime split | n/a (one regime) | ranging '15–'18 **+0.058 R**, bull '19–'23 **+0.230 R**, recent '24–'26 **+0.203 R** |
| Calendar years positive | 3/3 | **11/12** (worst −0.061 R, 2015) |
| Monte Carlo ($100, 1 % ff) | P(ruin) 0 % | **P(ruin) 0 %, P(DD>50 %) 0.02 %** |
| Live gate | 9/11 → `LIVE_CANDIDATE = false` | **9/11 → `LIVE_CANDIDATE = false`** |

The **same** frozen momentum rule showing a cost-surviving, cross-regime edge on two
unrelated instruments is the strongest evidence so far that it is real momentum
persistence, not a curve-fit. XAU is blocked by the **same two** conditions as BTC:
$100 can't size an H1 stop (0.01 lot risks 9 % median / 24 % p90), and there is no
forward record yet.

## Where we are

| | M5 | H1 |
|---|---|---|
| Data | 100,000 bars / 348 days | **22,729 bars / 2.59 years** (2024-01-26 → 2026-08-30, dense hourly) |
| Round-trip cost vs ATR | ~72 % of median ATR | **~6 %** of median ATR |
| Best strategy | `breakout_donchian` — negative at every cost level | **`momentum_rsi_mtf`** (RSI≥60 + 8-bar momentum + price>EMA96, all from bar i-1) |
| Combined (realistic costs) | PF 0.68, expectancy −0.26 R | **PF 1.41, expectancy +0.22 R, DD 14.6 %** |
| Out-of-sample (FINAL split) | −0.17 R | **+0.19 R** |
| Walk-forward | 0 / 8 folds positive | **8 / 8 folds positive** (weakest +0.085 R) |
| Cost sensitivity | negative even "optimistic" | **positive through "stressed"** ($68 round-trip → +0.19 R) |
| Parameter robustness | 0 / 9, not stable | **9 / 9 positive, `stable_region = true`** |
| Edge authenticity | n/a | look-ahead clean; **both directions profitable** (L +0.23 R, S +0.21 R); **beats 200 random-direction runs (100th pctile)**; always-long AND always-short both lose; positive in 2024, 2025, 2026 |
| Monte Carlo (1 % fixed-fractional, $100) | P(neg) 98 %, P(ruin) 25 % | P(neg) 0 %, P(ruin) 0 %, P(DD>50 %) 0 % |
| Live gate | 4 / 14 → `LIVE_CANDIDATE = false` | **9 / 11 → `LIVE_CANDIDATE = false`** |

**M5 has no edge. H1 shows a statistically defensible momentum edge that survives realistic
Valetax costs.** It is NOT yet a live candidate — two research conditions and one hard rule
still block it.

## Why H1 is not `LIVE_CANDIDATE` yet

1. **$100 account cannot size H1.** Median H1 ATR ≈ $494 → a 2×ATR stop ≈ $989 → the minimum
   0.01 lot risks ≈ **10 % of $100** (17 % at p90 volatility). The sizer correctly rejects
   almost every H1 setup on $100. A clean 1 %-risk position needs **≈ $1,000 (median vol) to
   ≈ $1,700 (p90 vol)**. Margin ($7.80) is not the constraint — the stop size is.
2. **No forward / paper confirmation.** The edge is in-sample + walk-forward only; it has not
   been observed live.
3. **Live execution not implemented** (hard rule): `order_send` + post-fill reconciliation
   do not exist. `LIVE_TRADING` stays `false` regardless of the gate.
4. **One macro regime.** 2.59 years is 2024 accumulation → 2025 bull → 2026. The edge has not
   been tested through a prolonged bear / chop market.

## Recommended next actions (in order)

1. **Forward-paper the H1 momentum strategy for ≥ 4 weeks.**  Implemented — the paper
   architecture is now timeframe-selectable and `--timeframe H1` auto-selects the frozen
   `momentum_rsi_mtf` strategy (params `rsi_buy=60, rsi_sell=40, mom_win=8, ema_htf=24`).
   Run it (real Valetax spread, all safety gates, no `order_send`):

   ```powershell
   .\venv\Scripts\python.exe run_btc.py --paper --timeframe H1 --paper-balance 1500 --loop 3600
   ```

   Records go to `state\btc_paper_H1.sqlite` and `reports\btc_paper_status_H1.json`.
   Require ≥ 20 hypothetical trades and net > 0 before the gate condition #10 flips.
   (`--paper-balance 1500` sizes the *hypothetical* position for the intended account —
   the real $100 account and every live gate are untouched; omit it to see the $100
   account correctly refuse H1 trades.)
2. **Plan for a ≥ $1,500 account** (not $100) so H1 positions can be sized at 1 % risk.
   Re-run `backtest/btc_h1_validation.py` with `RISK.initial_balance = 1500` to confirm
   drawdown and ruin on the *real* target account.
3. **Keep collecting fresh out-of-sample H1 data.** Re-run `python -m backtest.btc_h1_dataset`
   monthly and re-run the validation + gate on the appended data.
4. **Only after 1–3 look good:** implement `order_send` behind the gate, with a dry→live
   toggle, order-fill reconcile, and the existing safety controller in front of it.
   Then run `python -m tools.btc_h1_live_gate` on fresh data and require **11 / 11**.

## What NOT to do

- Do not enable `LIVE_TRADING`. Do not implement `order_send` before 1–3.
- Do not keep adding M5 strategies — M5 on this broker is conclusively dead (spread ≈ 0.7 × ATR).
- Do not tune the H1 momentum parameters further — the current region is already stable
  (9/9 nearby configs positive); more tuning only adds overfitting risk.

## Commands

```bash
# BTC
python -m backtest.btc_h1_dataset                 # refresh H1 dataset + report
python -m backtest.btc_h1_validation              # full H1 validation
python -m tools.btc_h1_live_gate                  # 11-condition H1 gate
python run_btc.py --paper --timeframe H1 --paper-balance 1500 --loop 3600

# XAU (same machinery)
python -m backtest.xau_h1_validation              # full H1 validation -> XAU_H1_VALIDATION.md
python -m tools.xau_h1_live_gate                  # 11-condition H1 gate

# MT5-free forward test on Oracle Cloud -- see ORACLE_DEPLOY.md
# Mode A (PC available weekly): export the real Valetax feed
python -m tools.export_h1 --asset BTC --bars 24000
python -m tools.export_h1 --asset XAU --bars 24000
# Mode B (PC off): fetch a free public feed ON the cloud box, no PC needed
python -m tools.fetch_h1_public --asset ALL
# then, either mode:
python run_btc.py --paper --timeframe H1            --from-csv data/export/btcusd_vx_H1_export.csv --paper-balance 1500
python run_btc.py --paper --timeframe H1 --asset XAU --from-csv data/export/xauusd_vx_H1_export.csv --paper-balance 1500
```
