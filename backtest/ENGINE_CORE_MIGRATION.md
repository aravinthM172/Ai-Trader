# engine_core migration — freeze record & status

## Phase 1 freeze (2026-08-29)

Repo is **not** a git repository, so no SHA. SHA-256 of the files at freeze time:

| file | sha256 |
|---|---|
| `backtest/realistic_xauusd.py` (pre-migration) | `fe1692fd9f6746b96ab8c1ac77f5c4ccaffe86da592dbbbdb15f681a453b6c38` |
| `backtest/optimizer_realistic_numba.py` (pre-migration) | `a6015a1c62a7b01188a45cc62fdb93df932fe2c4ae46faa864254135dcd02c1d` |
| `backtest/optimizer_realistic_numba.py.bak` (untouched) | `875bccd3dc28d21e1e3f703a189e06c3dfefe50562ba4964431e33c1eb8f225a` |
| `reports/xauusd_realistic_optimization.checkpoint.csv` (untouched, 17,280 rows) | `c6144ab06886ac3a5c7c85c968e8cd089249d9a8cfd8cfc70dd7dc176b13622b` |
| `data/xauusd_m5_5y.csv` (untouched, 100,000 bars) | `0a7c821b9daefec3b749f014fe3a06992741f3e0b621bda56a1362b49d36ac2d` |

The 17,280-config checkpoint is **INVALID for strategy selection** (pre-migration
optimizer was not simulation-equivalent) but is **kept on disk untouched**.
The new engine changes simulation semantics, so any real run of the optimizer
must use `--fresh` (the checkpoint is position-keyed only; it has no engine /
grid / data fingerprint).

## What changed

* **NEW** `backtest/engine_core.py` — one source of truth. `ema` / `rsi` / `atr`
  / `_backtest` ported verbatim from `realistic_xauusd.py`; `_backtest` gains
  only optional trade-log side-writes + `gross_profit/gross_loss/n_recorded` in
  the return tuple (no control-flow / arithmetic change). Plus `load_ohlc`,
  `prepare_spread`, `compute_indicators`, `run_window_fast` (njit),
  `run_window_logged` (python), `summary_dict`, and all cost/risk/contract
  constants.
* **MODIFIED** `backtest/realistic_xauusd.py` — deletes its local
  `ema/rsi/atr/_backtest`, imports them from `engine_core`, sources cost/risk
  constants from `engine_core`, calls `run_window_fast`. Keeps the training-chosen
  strategy params (`FAST=10 … TP_MULT=2.0`) and all CLI/CSV output. Numbers
  **unchanged** (1678 / 58 / 271 trades; PF 0.926 / 0.569 / 0.831).
* **MODIFIED** `backtest/optimizer_realistic_numba.py` — deletes its local
  `ema/rsi/atr/floor_lot/run_backtest`, routes `run_chunk` through
  `engine_core.run_window_fast`. Fixes carried by the shared engine: ATR seed,
  spread floor + full round-trip spread, `SLIPPAGE 0.02 → 0.01`, trend filter
  `close>trend → slowEMA>trendEMA`, `RSI_BUY 55 → 60` (**pinned**, not yet a grid
  dimension), min-stop floor, margin cap, equity kill-switch. Grid unchanged at
  17,280 (RSI_PERIOD still optimized).
* **NEW** `tests/test_engine_parity.py` — 3-path parity check, all exact.

## Parity result (EMA 10/80/200, RSI 7 60/40, ATR 20, SL 1.5, TP 2.0, bars [0,70000))

| metric | standalone | opt matrices | opt run_chunk kernel |
|---|---|---|---|
| trades | 1678 | 1678 | 1678 |
| net P/L | -32.64283012 | -32.64283012 | -32.64283012 |
| win rate | 0.4195470799 | 0.4195470799 | 0.4195470799 |
| profit factor | 0.9264425024 | 0.9264425024 | 0.9264425024 |
| gross profit / loss | 411.1301527 / 443.7729828 | same | — |
| max DD $ / % | 53.67611491 / 0.4435037399 | same | 53.67611491 / — |
| commission | 0 | 0 | — |

All 1678 individual trades (entry idx, exit idx, direction, lots, entry px,
exit px, P/L) identical between standalone and optimizer wiring. Tolerance 1e-9.

## NOT modified
`optimizer_realistic_numba.py.bak`, the checkpoint, the data, all other
`backtest/*.py`, all `reports/*` except `xauusd_realistic_results.csv`
(rewritten by re-running the standalone — identical content).

---

## Phase 2 — checkpoint fingerprinting (2026-08-29, later same day)

### State of the on-disk checkpoint when Phase 2 started
`reports/xauusd_realistic_optimization.checkpoint.csv` on disk is **17,280 rows,
sha256 `5b50b4a2…`, mtime 22:44** — this is **not** the pre-migration checkpoint
recorded in the Phase-1 freeze table (`c6144ab0…`). Two independent rows were
verified against the current engine to full float precision:

| config | checkpoint row | current engine |
|---|---|---|
| `10,80,200,7,20,1.5,2.0` | train 1678 / PF 0.9264425024 / dd 53.67611491 | identical (parity test) |
| `5,30,100,7,10,1.0,1.5` | train 2337 / PF 0.9276563175 / val 157 / -13.2477 | identical (`--smoke` row 0) |

So a **full post-migration sweep already ran** (RSI 60/40 pinned, shared engine)
after the Phase-1 doc was written, and the original pre-migration checkpoint was
overwritten (the old `--fresh` did `os.remove`). That file still has **no
fingerprint**, so it is treated as unverifiable and is **not** resumed.

Defensive copy taken:
`reports/xauusd_realistic_optimization.checkpoint.PRESERVED-2026-08-29.csv`
(byte-identical, sha256 `5b50b4a2…`).

### What changed in Phase 2
* **`engine_core.py`**: added `ENGINE_VERSION = "2026-08-29.1"` and
  `warmup_for(fast, slow, trend, rsi_period, atr_period)` (the `max(...)+5`
  rule, now single-source). `compute_indicators` calls it. No semantic change —
  parity still exact.
* **`optimizer_realistic_numba.py`**:
  * `configs_to_arrays` / `report` now call `ec.warmup_for` instead of an inline
    `max(...)+5` — the last piece of duplicated numeric logic is gone. The
    optimizer has **no private simulation code**: `run_chunk` → `run_window_fast`
    → `engine_core._backtest`; indicators via `engine_core.ema/rsi/atr`.
  * **Checkpoint fingerprinting.** New sidecar
    `reports/xauusd_realistic_optimization.checkpoint.fingerprint.json` holds a
    canonical JSON of everything that can move a checkpoint metric + its
    sha256: engine version + `engine_core.py` source hash; full grid
    (`FAST/SLOW/TREND/RSI_PERIOD/ATR_PERIOD/SL/TP`) + `config_count`; pinned
    `RSI_BUY_TH/RSI_SELL_TH`; `TRAIN_END/VAL_END/TOTAL/n_bars`; execution costs
    (`DEFAULT_SPREAD, SLIPPAGE, COMMISSION_PER_LOT, MIN_STOP_TICKS,
    MIN_STOP_DIST, TICK_SIZE, TICK_VALUE, VALUE_PER_POINT`); risk/lot/margin
    (`INITIAL_BALANCE, RISK_PER_TRADE, EQUITY_STOP_FRACTION, EQUITY_STOP,
    MAX_MARGIN_FRACTION, LEVERAGE, CONTRACT_SIZE, MIN_LOT, LOT_STEP, MAX_LOT`);
    the checkpoint column schema; and the data file identity (SHA-256, with
    size+mtime fallback).
  * **Resume rule**: a bare run (no `--fresh`) with a checkpoint present is
    allowed **only** if the sidecar exists and its `fingerprint_sha256` matches
    the current config. Otherwise it prints a field-level diff and
    `sys.exit(2)` — the checkpoint is not read, moved or deleted.
  * **`--fresh`**: archives `checkpoint.csv` (+ sidecar if any) to
    `*.superseded-<UTC>.csv` via `os.rename` (**never `os.remove`**), writes a
    fresh sidecar, starts from config 0. Old and new rows are never in one file.
  * **`--smoke N`**: unchanged guarantee — `checkpoint_path=None`, so it never
    reads, writes, resumes or archives the checkpoint/sidecar and never runs the
    full sweep. Verified: checkpoint sha256 + mtime identical before/after.
* **`tests/test_engine_parity.py`**: unchanged. Still `PARITY: PASS`.

### Verification (Phase 2)
* `python tests/test_engine_parity.py` → **PARITY: PASS** (A=B=C, 1678 trades,
  net -32.64283012, PF 0.9264425024).
* `python backtest/optimizer_realistic_numba.py --smoke 10` → 10 configs, engine
  banner shows engine_version `2026-08-29.1`, RSI 60/40 pinned, spread floor
  $0.35, slippage $0.01, 50% margin cap, $20 equity stop. Checkpoint untouched.
* Bare run (no flag) → refuses with the "no fingerprint sidecar" message, exit 2,
  nothing on disk changed.
* `check_checkpoint_fingerprint` unit-checked for match / field-diff / missing.

### Next command (real sweep — NOT yet run, awaiting approval)
```
python backtest/optimizer_realistic_numba.py --fresh
```
