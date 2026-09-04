from pathlib import Path
import sys
import json
import time
import traceback

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

ROOT = Path(".").resolve()
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

print("=" * 90)
print("BTC MT5 PAPER-VALIDATION SETUP — 20 STEPS")
print("=" * 90)

# ---------------------------------------------------------------------------
# 1. CONNECT TO MT5
# ---------------------------------------------------------------------------
print("\n[01/20] MT5 connection")
if not mt5.initialize():
    raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
print("OK")

# ---------------------------------------------------------------------------
# 2. ACCOUNT
# ---------------------------------------------------------------------------
print("\n[02/20] Account")
account = mt5.account_info()
if account is None:
    raise RuntimeError(f"account_info failed: {mt5.last_error()}")
print(f"Server       : {account.server}")
print(f"Balance      : {account.balance}")
print(f"Equity       : {account.equity}")
print(f"Leverage     : 1:{account.leverage}")
print(f"Trade allowed: {account.trade_allowed}")

# ---------------------------------------------------------------------------
# 3. FIND BTC SYMBOLS
# ---------------------------------------------------------------------------
print("\n[03/20] BTC symbols")
symbols = mt5.symbols_get()
btc_symbols = sorted(
    [s.name for s in symbols if "BTC" in s.name.upper()]
)
print(btc_symbols)

if not btc_symbols:
    raise RuntimeError("No BTC symbol found in MT5.")

# Prefer BTCUSD, otherwise first BTC symbol.
preferred = [
    "BTCUSD",
    "BTCUSDm",
    "BTCUSD.a",
    "BTCUSD.",
]
symbol = "BTC"

print(f"Selected symbol: {symbol}")

# ---------------------------------------------------------------------------
# 4. SELECT SYMBOL
# ---------------------------------------------------------------------------
print("\n[04/20] Symbol selection")
if not mt5.symbol_select(symbol, True):
    raise RuntimeError(f"Could not select {symbol}: {mt5.last_error()}")
print("OK")

# ---------------------------------------------------------------------------
# 5. SYMBOL INFO
# ---------------------------------------------------------------------------
print("\n[05/20] BTC contract specification")
s = mt5.symbol_info(symbol)
if s is None:
    raise RuntimeError(f"symbol_info failed: {mt5.last_error()}")

spec = {
    "symbol": s.name,
    "digits": s.digits,
    "point": s.point,
    "trade_tick_size": s.trade_tick_size,
    "trade_tick_value": s.trade_tick_value,
    "trade_tick_value_profit": s.trade_tick_value_profit,
    "trade_tick_value_loss": s.trade_tick_value_loss,
    "contract_size": s.trade_contract_size,
    "volume_min": s.volume_min,
    "volume_step": s.volume_step,
    "volume_max": s.volume_max,
    "trade_stops_level": s.trade_stops_level,
    "trade_freeze_level": s.trade_freeze_level,
    "trade_mode": s.trade_mode,
    "currency_base": s.currency_base,
    "currency_profit": s.currency_profit,
    "currency_margin": s.currency_margin,
}

for k, v in spec.items():
    print(f"{k:25s}: {v}")

# ---------------------------------------------------------------------------
# 6. CURRENT TICK
# ---------------------------------------------------------------------------
print("\n[06/20] Live BTC tick")
tick = mt5.symbol_info_tick(symbol)
if tick is None:
    raise RuntimeError(f"symbol_info_tick failed: {mt5.last_error()}")

bid = float(tick.bid)
ask = float(tick.ask)
spread_price = ask - bid
spread_points = spread_price / s.point if s.point else float("nan")

print(f"Bid            : {bid}")
print(f"Ask            : {ask}")
print(f"Spread price   : {spread_price}")
print(f"Spread points  : {spread_points}")

# ---------------------------------------------------------------------------
# 7. P/L CALCULATOR
# ---------------------------------------------------------------------------
print("\n[07/20] BTC P/L identity")
if s.trade_tick_size and s.trade_tick_value:
    for move in [s.trade_tick_size, 1.0, 10.0, 100.0]:
        pnl = (move / s.trade_tick_size) * s.trade_tick_value
        print(f"${move:g} move @ 1 lot -> ${pnl:.6f}")
else:
    print("Tick value unavailable.")

# ---------------------------------------------------------------------------
# 8. MARGIN CHECK
# ---------------------------------------------------------------------------
print("\n[08/20] Margin check")
try:
    margin_001 = mt5.order_calc_margin(
        mt5.ORDER_TYPE_BUY,
        symbol,
        float(s.volume_min),
        ask,
    )
    print(f"Estimated margin for minimum lot: {margin_001}")
except Exception as exc:
    print(f"Margin calculation unavailable: {exc}")

# ---------------------------------------------------------------------------
# 9. SESSION / MARKET STATE
# ---------------------------------------------------------------------------
print("\n[09/20] Market/session state")
print(f"visible          : {s.visible}")
print(f"trade_mode       : {s.trade_mode}")
print(f"session_deals    : {s.session_deals}")
print(f"session_buy_orders  : {s.session_buy_orders}")
print(f"session_sell_orders : {s.session_sell_orders}")

# ---------------------------------------------------------------------------
# 10. BROKER JSON
# ---------------------------------------------------------------------------
print("\n[10/20] Save broker specification")
spec_out = REPORTS / "btc_broker_spec.json"

snapshot = {
    "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
    "symbol": symbol,
    "account": {
        "server": account.server,
        "leverage": account.leverage,
        "balance": account.balance,
        "equity": account.equity,
        "trade_allowed": account.trade_allowed,
    },
    "symbol": spec,
    "tick": {
        "bid": bid,
        "ask": ask,
        "spread_price": spread_price,
        "spread_points": spread_points,
    },
}

spec_out.write_text(
    json.dumps(snapshot, indent=2, default=str),
    encoding="utf-8",
)
print(f"Saved: {spec_out}")

# ---------------------------------------------------------------------------
# 11. DOWNLOAD M5 HISTORY
# ---------------------------------------------------------------------------
print("\n[11/20] BTC M5 history")
rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 10000)

if rates is None:
    raise RuntimeError(f"copy_rates_from_pos failed: {mt5.last_error()}")

hist = pd.DataFrame(rates)
hist["time"] = pd.to_datetime(hist["time"], unit="s", utc=True)

print(f"Rows downloaded: {len(hist)}")
print(f"From: {hist['time'].min()}")
print(f"To  : {hist['time'].max()}")

# ---------------------------------------------------------------------------
# 12. SAVE RAW HISTORY
# ---------------------------------------------------------------------------
print("\n[12/20] Save BTC M5 history")
hist_path = ROOT / "data" / "btc_m5_live_snapshot.csv"
hist_path.parent.mkdir(exist_ok=True)
hist.to_csv(hist_path, index=False)
print(f"Saved: {hist_path}")

# ---------------------------------------------------------------------------
# 13. HISTORY SPREAD ANALYSIS
# ---------------------------------------------------------------------------
print("\n[13/20] Historical spread analysis")

if "spread" in hist.columns:
    hs = hist["spread"].astype(float) * float(s.point)
    print(hs.quantile([0, .25, .50, .75, .90, .95, .99, 1]).to_string())

    print("\nSpread counts:")
    for x in [0.5, 1, 2, 5, 10, 20, 50]:
        print(f"<= ${x:g}: {(hs <= x).sum()}")

# ---------------------------------------------------------------------------
# 14. DATA QUALITY
# ---------------------------------------------------------------------------
print("\n[14/20] Data quality")

required = ["open", "high", "low", "close", "tick_volume"]
missing = [c for c in required if c not in hist.columns]
print(f"Missing columns: {missing}")

for c in ["open", "high", "low", "close"]:
    print(
        f"{c:6s}: NaN={hist[c].isna().sum()} "
        f"finite={np.isfinite(hist[c].to_numpy()).all()}"
    )

print(
    "Duplicate timestamps:",
    int(hist["time"].duplicated().sum())
)

# ---------------------------------------------------------------------------
# 15. PRICE SANITY
# ---------------------------------------------------------------------------
print("\n[15/20] Price sanity")

bad_ohlc = (
    (hist["high"] < hist["low"]) |
    (hist["open"] > hist["high"]) |
    (hist["open"] < hist["low"]) |
    (hist["close"] > hist["high"]) |
    (hist["close"] < hist["low"])
)

print(f"Bad OHLC rows: {int(bad_ohlc.sum())}")

# ---------------------------------------------------------------------------
# 16. VOLATILITY
# ---------------------------------------------------------------------------
print("\n[16/20] Volatility snapshot")

ret = hist["close"].pct_change()
print("Return quantiles:")
print(ret.quantile([.01,.05,.25,.50,.75,.95,.99]).to_string())

# ---------------------------------------------------------------------------
# 17. BUILD PAPER CONFIG
# ---------------------------------------------------------------------------
print("\n[17/20] Paper configuration")

paper_config = {
    "symbol": symbol,
    "timeframe": "M5",
    "mode": "PAPER_ONLY",
    "live_orders": False,
    "risk_per_trade": 0.01,
    "rsi_buy": 60.0,
    "rsi_sell": 40.0,
    "strategy_candidates": [
        {
            "name": "BTC_C01",
            "ema_fast": 20,
            "ema_slow": 60,
            "ema_trend": 150,
            "rsi_period": 20,
            "atr_period": 20,
            "sl": 2.0,
            "tp": 3.5,
        },
        {
            "name": "BTC_C02",
            "ema_fast": 15,
            "ema_slow": 80,
            "ema_trend": 200,
            "rsi_period": 20,
            "atr_period": 20,
            "sl": 2.5,
            "tp": 3.5,
        },
    ],
}

paper_path = REPORTS / "btc_paper_config.json"
paper_path.write_text(
    json.dumps(paper_config, indent=2),
    encoding="utf-8",
)
print(f"Saved: {paper_path}")

# ---------------------------------------------------------------------------
# 18. NO-ORDER SAFETY CHECK
# ---------------------------------------------------------------------------
print("\n[18/20] LIVE ORDER SAFETY CHECK")
LIVE_ORDER_FUNCTIONS = [
    "order_send",
    "order_check",
]
print("Live order functions are NOT called.")
print("PAPER MODE:", paper_config["live_orders"] is False)

# ---------------------------------------------------------------------------
# 19. CURRENT SIGNAL PREVIEW
# ---------------------------------------------------------------------------
print("\n[19/20] Current signal data preview")

# Pure indicator preview; does not place trades.
close = hist["close"].to_numpy(np.float64)

def ema(x, period):
    alpha = 2.0 / (period + 1.0)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out

fast = ema(close, 20)
slow = ema(close, 60)
trend = ema(close, 150)

print(f"Latest close : {close[-1]}")
print(f"EMA20        : {fast[-1]}")
print(f"EMA60        : {slow[-1]}")
print(f"EMA150       : {trend[-1]}")

if fast[-1] > slow[-1] > trend[-1]:
    trend_state = "BULLISH"
elif fast[-1] < slow[-1] < trend[-1]:
    trend_state = "BEARISH"
else:
    trend_state = "MIXED"

print(f"EMA state    : {trend_state}")

# ---------------------------------------------------------------------------
# 20. FINAL STATUS
# ---------------------------------------------------------------------------
print("\n[20/20] STATUS")
print("=" * 90)
print("BTC PAPER VALIDATION SETUP COMPLETE")
print("=" * 90)
print(f"Symbol              : {symbol}")
print(f"Current spread      : ${spread_price:.8f}")
print(f"Tick size           : {s.trade_tick_size}")
print(f"Tick value          : {s.trade_tick_value}")
print(f"Contract size       : {s.trade_contract_size}")
print(f"Min / step / max    : {s.volume_min} / {s.volume_step} / {s.volume_max}")
print(f"Leverage            : 1:{account.leverage}")
print(f"History rows        : {len(hist)}")
print("Live orders         : DISABLED")
print("XAU experiments     : UNTOUCHED")
print("=" * 90)

mt5.shutdown()
