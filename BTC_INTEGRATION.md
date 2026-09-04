# BTC Integration (Valetax `BTCUSD.vx`)

Status: **DRY-RUN READY**. `LIVE_TRADING=false`. `order_send()` is never called anywhere
in the codebase (`grep -rn order_send` → nothing).

## What was added (modular, XAUUSD untouched)

| File | Purpose |
|---|---|
| `config/assets.py` | `AssetConfig` dataclass + `btc_config()` / `xau_config()`; per-asset `.env` prefixes (`BTC_*`, `XAU_*`). Legacy `ASSETS` dict preserved. |
| `mt5/gateway.py` | The only BTC module that imports `MetaTrader5`. `getattr`-safe spec reads (`trade_exemode`, not `trade_execution`), chunked history (terminal rejects >~50k), broker-server timezone detection, `order_calc_margin` / `order_calc_profit` / `order_check` wrappers. No `order_send`. |
| `data/quality.py` | OHLC validation: NaN, `high<low`, close/open outside range, non-positive price, duplicate timestamps, impossible jumps (`>N·ATR`), gaps/missing candles, staleness, insufficient history, zero-volume, spread availability. Never repairs — rejects + logs. |
| `strategy/btc_features.py` | Strictly causal features (SMA-seeded Wilder EMA/RSI/ATR matching `backtest/btc_engine_core.py`). Returns/log-returns, ATR/nATR, rolling vol, RSI, EMA/SMA, MACD, candle body/range/wicks/close-loc, momentum, distance-from-EMA, trend strength, spread, spread/ATR. |
| `strategy/stops.py` | ATR-based dynamic SL/TP with **broker minimum enforced** (2976 pts = $29.76, ×`broker_stop_buffer`). Reward/Risk measured from the **actual fill price** (spread-aware). Rejects cost-dominated / absurdly-wide / low-R:R trades. Spread filter (`spread/ATR`). |
| `risk/sizing.py` | Volume from **risk %, not leverage**. Uses `order_calc_profit` for exact loss-at-SL and `order_calc_margin` for exact margin. Normalises to `volume_step`, clamps `[volume_min, volume_max]`, checks free-margin-after. Small-account allowance: takes `volume_min` if the risk-derived size rounds to 0 **and** `volume_min` risk ≤ `max_risk_per_trade`. |
| `execution/safety.py` | Kill switch (`state/KILL_SWITCH` file or `KILL_SWITCH=1`), `LIVE_TRADING` master switch, max total/per-symbol positions, duplicate-position block, daily-loss limit, cooldown, stale-price, spread, margin, trading-session. Cannot be loosened by config. |
| `execution/order_validator.py` | Builds the MT5 `TRADE_ACTION_DEAL` request, runs `order_check()` (READ-ONLY), renders the `## BTC TRADE VALIDATION` block. Never sends. |
| `execution/state.py` | JSON store: per-day realised P/L, trade count, last-entry time (cooldown). |
| `execution/pipeline.py` | The end-to-end chain (see below). Returns one dict. Never sends. |
| `strategy/btc_signal.py` | Transparent rule-based BUY/SELL/HOLD (NOT optimised, NOT an ML model). Conservative — HOLD unless multi-factor trend alignment. |
| `strategy/model.py` + `models/README.md` | ML interface stub. No model exists. XAU/BTC kept separate; NaN/shape/feature-name mismatch rejected. |
| `tools/btc_diagnostic.py` | `python -m tools.btc_diagnostic` — full readiness check. Never sends. |
| `run_btc.py` | `python run_btc.py` — SAFE/DRY-RUN entry point. Never sends. |
| `backtest/btc_valetax_backtest.py` | Real-`BTCUSD.vx` history + real contract/costs, chronological split + walk-forward, full metrics. |
| `common/logging_setup.py` | Console + rotating file logs in `logs/`, secret redaction. |
| `tests/test_btc_integration.py`, `tests/test_xauusd_regression.py`, `tests/conftest.py` | 35 tests (unit + live-MT5 + XAU regression). |

Modified: `backtest/engine_core.py` — fixed a **pre-existing** `NameError` (`slippage` / `commission_per_lot`
undefined in `run_window_logged`) so the XAUUSD engine-parity regression passes. Backup at
`backtest/engine_core.py.pre-btc-integration`. `.env` — appended `BTC_*` / `XAU_*` / `LIVE_TRADING`
keys and corrected `MT5_*_SYMBOL` to the real `.vx` names.

## Pipeline

```
MT5 connect → account → symbol spec (dynamic) → tick → timeframe support →
history (chunked) → data quality → features → ATR → signal → spread filter →
stop plan (broker-min enforced) → position sizing (order_calc_*) →
safety controller → order_check (READ-ONLY) → decision
```

## Commands

```bash
# BTC readiness diagnostic (no order sent)
python -m tools.btc_diagnostic
python -m tools.btc_diagnostic --buy            # evaluate the BUY side
python -m tools.btc_diagnostic --buy --ignore-spread   # TEST ONLY: exercise order_check

# BTC bot, SAFE / DRY-RUN
python run_btc.py
python run_btc.py --loop 300                    # repeat every 5 min (still dry-run)

# Real BTCUSD.vx backtest
python -m backtest.btc_valetax_backtest --download --bars 50000

# Tests
python -m pytest tests/ -q

# Emergency stop (blocks all trading immediately)
:> state/KILL_SWITCH        # create the file
```

## To ever go live (NOT done here, intentionally)

1. `LIVE_TRADING=true` in `.env`
2. Enable the **AutoTrading** button in the MT5 terminal (`terminal.trade_allowed` is currently `False`)
3. Remove `state/KILL_SWITCH`
4. Implement the actual `order_send` step (deliberately absent) with a post-send reconcile
5. Only after a forward **paper** test shows the strategy is not negative

Current research verdict: no directional edge on BTC M5, and the live spread ($29.76 ≈ 0.7× M5 ATR)
makes M5 BTC trading structurally unattractive on this broker. The bot correctly refuses to trade.

## Strategy-validation pipeline (Phases 1–12)

| command | purpose |
|---|---|
| `python -m backtest.btc_dataset` | download + validate M5/M15/H1 → `reports/btc_dataset.json` |
| `python -m backtest.btc_validate` | baseline + 4 strategies + cost sensitivity + walk-forward + parameter robustness + Monte Carlo + $100 feasibility → `reports/btc_strategy_validation.{json,md}` and 3 more JSONs |
| `python -m tools.btc_live_gate` | 14-condition readiness gate → `reports/btc_live_gate.json` (`LIVE_CANDIDATE`) |
| `python run_btc.py --paper [--loop 300]` | paper / forward test → `state/btc_paper.sqlite` + `reports/btc_paper_status.json` |

New modules: `backtest/btc_lab.py` (audited engine + walk-forward + cost sensitivity + Monte Carlo),
`backtest/btc_strategies.py`, `backtest/btc_dataset.py`, `backtest/btc_validate.py`,
`execution/paper.py`, `tools/btc_live_gate.py`.  See `BTC_NEXT_STEPS.md`.

## Timeframe-selectable paper mode (H1 forward test)

`run_btc.py` and the pipeline take `--timeframe {M5,M15,H1,H4}`. **H1/H4 auto-select the
validated `momentum_rsi_mtf` strategy** (frozen params); M5 keeps the placeholder rule.
The pipeline fetches candles for the selected timeframe, computes the signal on the last
**completed** bar (fill = next bar open), and uses **that timeframe's ATR** for stops/sizing.
A hard guard aborts the paper pass if the pipeline ever runs on a different timeframe.

```powershell
.\venv\Scripts\python.exe run_btc.py --paper --timeframe H1 --loop 3600                 # $100 account (refuses H1 trades)
.\venv\Scripts\python.exe run_btc.py --paper --timeframe H1 --paper-balance 1500 --loop 3600   # intended account size
```

Per-timeframe state: `state/btc_paper_<TF>.sqlite`, `reports/btc_paper_status_<TF>.json`.
Every paper record carries: opened_utc, timeframe, strategy, direction, signal_bar_utc,
entry, sl, tp, volume, atr, spread, est_risk_usd, est_cost_usd, reason_entry, and on close
exit / exit_reason / pnl_usd / r_multiple / bars_held.  `order_send` is still never called;
`LIVE_TRADING=false`; `KILL_SWITCH` still blocks every path.
