# models/

**Empty on purpose.** There is no trained ML model in this project.

- The legacy decision source is `claude/analyzer.py` (an LLM API call).
- BTC research (`backtest/btc_signal_discovery.py`, `backtest/btc_phase3_confirmation.py`,
  `backtest/btc_valetax_backtest.py`) found **no robust directional edge** in BTC M5 OHLC,
  so no BTC model was trained.
- The BTC live path uses the transparent rule in `strategy/btc_signal.py` (conservative,
  not optimised) as a placeholder so the pipeline is testable end to end.

## Adding a model later
- Keep **XAUUSD and BTC models separate** — one trained on gold is not valid for BTC.
- Store, next to the model file: exact feature list + order, the fitted scaler, the format.
- Register it in `strategy/model.py::REGISTRY`; the pipeline calls `strategy.model.predict()`.
- `strategy/model.py` already rejects NaN / shape / feature-name mismatch.
